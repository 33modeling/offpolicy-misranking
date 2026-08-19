"""Canonical artifact-only predicates shared by judge and readout tools."""

from __future__ import annotations

import json
from pathlib import Path

from select_rules import overlap_under_independent_ties, topk_count


ONE_SIDED_DROP = 0.15
CAUSAL_CUT = "0.5"
SCORE_PROTOCOL_SCHEMA = "offpolicy-score-validation-split/v1"
ORACLE_PROTOCOL_SCHEMA = "offpolicy-oracle-validation-split/v1"
HYBRID_PROTOCOL_SCHEMA = "offpolicy-hybrid-validation-split/v1"


def load_json(path: Path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def has_valid_score_protocol(run: Path) -> bool:
    protocol = load_json(run / "score_protocol.json")
    try:
        return bool(
            protocol
            and protocol.get("schema") == SCORE_PROTOCOL_SCHEMA
            and int(protocol.get("generation_validation", {}).get("validated_rows", 0)) > 0
        )
    except (AttributeError, TypeError, ValueError):
        return False


def has_valid_oracle_protocol(run: Path) -> bool:
    protocol = load_json(run / "oracle_protocol.json")
    try:
        return bool(
            protocol
            and protocol.get("schema") == ORACLE_PROTOCOL_SCHEMA
            and int(protocol.get("generation_validation", {}).get("validated_rows", 0)) > 0
        )
    except (AttributeError, TypeError, ValueError):
        return False


def has_valid_analysis_protocol(run: Path) -> bool:
    return has_valid_score_protocol(run) and has_valid_oracle_protocol(run)


def canonical_gate_report(run: Path, frac: float = 0.10, seed: int = 0) -> dict | None:
    """Prefer recomputation from raw scores; fall back to a stored report."""
    if not has_valid_analysis_protocol(run):
        return None
    stored = load_json(run / "report.json")
    oracle = load_json(run / "scores_oracle.json")
    off = load_json(run / "scores_offpolicy.json")
    halves = load_json(run / "scores_splithalf.json")
    if not (oracle and off and halves):
        return stored

    oracle_scores = {str(i): float(value["score"]) for i, value in oracle.items()}
    half_scores = {str(i): value for i, value in halves.items()}
    common = set(oracle_scores) & set(half_scores)
    if not common:
        return stored
    k = topk_count(len(common), frac)
    floor = overlap_under_independent_ties(
        {i: float(half_scores[i]["a"]) for i in common},
        {i: float(half_scores[i]["b"]) for i in common},
        k,
        seed=seed,
    ).mean
    report = dict(stored or {})
    report.update({"noise_floor": floor, "k": k, "_recomputed": True})
    for estimator in ("g00", "g10", "g01", "g11"):
        values = off.get(estimator)
        if not values:
            continue
        scores = {
            str(i): float(value["score"])
            for i, value in values.items()
            if str(i) in common
        }
        if set(scores) != common:
            continue
        overlap = overlap_under_independent_ties(
            {i: oracle_scores[i] for i in common}, scores, k, seed=seed
        )
        report[estimator] = {"precision": overlap.mean}
    return report


def one_sided_failures(report: dict, drop: float = ONE_SIDED_DROP) -> dict[str, bool]:
    floor = float(report["noise_floor"])
    return {
        estimator: (
            estimator in report
            and float(report[estimator]["precision"]) <= floor - drop
        )
        for estimator in ("g10", "g01")
    }


def hybrid_precisions(
    oracle: dict,
    cells: dict,
    *,
    frac: float = 0.25,
    seed: int = 0,
) -> dict[str, float]:
    required = {"bb", "bp", "pb", "pp"}
    if not required.issubset(cells):
        missing = sorted(required - set(cells))
        raise ValueError(f"hybrid cells missing: {missing}")
    normalized_oracle = {str(i): float(value["score"]) for i, value in oracle.items()}
    sub = set(map(str, cells["bb"])) & set(normalized_oracle)
    if not sub:
        raise ValueError("hybrid subset has no oracle scores")
    k = topk_count(len(sub), frac)
    truth = {i: normalized_oracle[i] for i in sub}
    result: dict[str, float] = {}
    for cell in sorted(required):
        scores = {str(i): float(value) for i, value in cells[cell].items() if str(i) in sub}
        if set(scores) != sub:
            raise ValueError(f"hybrid cell {cell} does not cover the common subset")
        result[cell] = overlap_under_independent_ties(truth, scores, k, seed=seed).mean
    return result


def evaluate_causal_run(
    run: Path,
    drop: float = ONE_SIDED_DROP,
    causal_cut: str = CAUSAL_CUT,
) -> dict:
    """Evaluate one run at the pre-registered causal cut."""
    report = canonical_gate_report(run)
    axis_failures = one_sided_failures(report, drop) if report else None
    joint_failure = bool(axis_failures and all(axis_failures.values()))
    oracle = load_json(run / "scores_oracle.json")
    hybrid_results = []
    if oracle:
        for path in sorted(run.glob("scores_hybrid_*.json")):
            cells = load_json(path)
            try:
                cut = path.stem.split("_")[-1]
                protocol = load_json(run / f"hybrid_protocol_{cut}.json")
                if not protocol or protocol.get("schema") != HYBRID_PROTOCOL_SCHEMA:
                    raise ValueError("corrected hybrid validation protocol is missing")
                precision = hybrid_precisions(oracle, cells or {})
                recovery = {
                    "g10": precision["pp"] > precision["pb"],
                    "g01": precision["pp"] > precision["bp"],
                }
                hybrid_results.append(
                    {
                        "path": path,
                        "cut": cut,
                        "eligible": cut == causal_cut,
                        "precision": precision,
                        "recovery": recovery,
                        "joint_recovery": all(recovery.values()),
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                hybrid_results.append({"path": path, "error": str(exc)})
    witnesses = [
        result
        for result in hybrid_results
        if joint_failure
        and result.get("eligible") is True
        and result.get("joint_recovery") is True
    ]
    return {
        "report": report,
        "axis_failures": axis_failures,
        "joint_failure": joint_failure,
        "causal_cut": causal_cut,
        "hybrid_results": hybrid_results,
        "witnesses": witnesses,
    }
