#!/usr/bin/env python3
"""Classify stale selectors as effective, ineffective, or inconclusive.

The primary outcome is utility under a sample-disjoint fresh half, not exact
top-k overlap alone.  The other fresh half is an independent current-policy
ranker.  This lets a stale selector receive credit when it captures useful
prompts without reproducing every noisy boundary decision.

Usage:
    python src/regime_map.py RUN_DIR [RUN_DIR ...] --output-dir RESULTS_DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from gate_rules import has_valid_analysis_protocol
from run_provenance import generation_commit, partition_by_generation
from score_artifacts import load_complete_score_artifacts
from select_rules import jittered_topk, overlap_under_independent_ties, topk_count

SCHEMA = "offpolicy-regime-map/v4"
ESTIMATORS = ("g00", "g10", "g01", "g11")
DEFAULT_RETENTION = 0.50
DEFAULT_REPLICATION = 0.80
MIN_FINAL_BOOTSTRAP = 10_000


def _analysis_commit() -> str:
    repository = Path(__file__).resolve().parents[1]
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--", "src"],
        text=True,
    ).strip()
    if dirty:
        raise ValueError("analysis source is dirty; commit it before aggregation")
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _score_margin(scores: Mapping[int, float], k: int) -> tuple[float, float]:
    """Return raw and scale-normalized top-k boundary margins."""
    ordered = sorted(float(value) for value in scores.values())
    if k >= len(ordered):
        return float("inf"), float("inf")
    raw = ordered[-k] - ordered[-k - 1]
    scale = statistics.pstdev(ordered)
    return raw, raw / scale if scale > 0 else 0.0


def _selection_metrics(
    selector_scores: Mapping[int, float],
    fresh_scores: Mapping[int, float],
    fresh_high_budget_scores: Mapping[int, float],
    truth_scores: Mapping[int, float],
    half_a_scores: Mapping[int, float],
    half_b_scores: Mapping[int, float],
    *,
    frac: float,
    seed: int,
) -> dict[str, float | bool | None]:
    ids = sorted(truth_scores)
    if any(
        set(scores) != set(ids)
        for scores in (
            selector_scores,
            fresh_scores,
            fresh_high_budget_scores,
            half_a_scores,
            half_b_scores,
        )
    ):
        raise ValueError("selector, ranking, reference, and half score IDs must match")
    k = topk_count(len(ids), frac)
    tie_pairs = 20

    def selected_utility(scores: Mapping[int, float], offset: int) -> float:
        return _mean(
            _mean(
                truth_scores[idx]
                for idx in jittered_topk(scores, k, seed + offset + pair * 7_919)
            )
            for pair in range(tie_pairs)
        )

    random_utility = _mean(truth_scores[idx] for idx in ids)
    stale_utility = selected_utility(selector_scores, 0)
    fresh_utility = selected_utility(fresh_scores, 17)
    fresh_high_budget_utility = selected_utility(fresh_high_budget_scores, 17)
    best_utility = selected_utility(truth_scores, 31)
    stale_gain = stale_utility - random_utility
    fresh_gain = fresh_utility - random_utility
    fresh_high_budget_gain = fresh_high_budget_utility - random_utility
    retention = stale_gain / fresh_gain if fresh_gain > 1e-12 else None
    floor = overlap_under_independent_ties(
        half_a_scores, half_b_scores, k, seed=seed + 17, pairs=tie_pairs
    ).mean
    precision = overlap_under_independent_ties(
        selector_scores, truth_scores, k, seed=seed, pairs=tie_pairs
    ).mean
    chance = k / len(ids)
    margin, normalized_margin = _score_margin(selector_scores, k)
    truth_margin, normalized_truth_margin = _score_margin(truth_scores, k)
    uniform_error = max(
        abs(float(selector_scores[idx]) - float(truth_scores[idx])) for idx in ids
    )
    # This is an observed-reference diagnostic, not a population certificate:
    # the averaged A/B score remains a noisy realization of the latent target.
    observed_margin_condition = truth_margin > 2.0 * uniform_error
    observed_error_margin_ratio = (
        uniform_error / truth_margin if truth_margin > 0 else None
    )

    return {
        "n": len(ids),
        "k": k,
        "chance": chance,
        "floor": floor,
        "measurable_point": floor >= 2.0 * chance and fresh_gain > 0,
        "topk_precision": precision,
        "stale_utility": stale_utility,
        "random_utility": random_utility,
        "fresh_utility": fresh_utility,
        "fresh_high_budget_utility": fresh_high_budget_utility,
        "best_utility": best_utility,
        "utility_gain": stale_gain,
        "fresh_gain": fresh_gain,
        "fresh_high_budget_gain": fresh_high_budget_gain,
        "utility_retention": retention,
        "regret_to_fresh": fresh_utility - stale_utility,
        "regret_to_best": best_utility - stale_utility,
        "boundary_margin": margin,
        "normalized_boundary_margin": normalized_margin,
        "truth_boundary_margin": truth_margin,
        "normalized_truth_boundary_margin": normalized_truth_margin,
        "uniform_score_error": uniform_error,
        "observed_error_margin_ratio": observed_error_margin_ratio,
        "observed_margin_condition": observed_margin_condition,
    }


def exact_topk_margin_implication(
    truth_scores: Mapping[int, float],
    estimate_scores: Mapping[int, float],
    k: int,
    *,
    seed: int = 0,
) -> tuple[bool, bool]:
    """Return (sufficient condition holds, selected sets are equal)."""
    if set(truth_scores) != set(estimate_scores):
        raise ValueError("score ID sets must match")
    truth_margin, _ = _score_margin(truth_scores, k)
    error = max(abs(truth_scores[idx] - estimate_scores[idx]) for idx in truth_scores)
    sufficient = truth_margin > 2.0 * error
    exact = jittered_topk(truth_scores, k, seed) == jittered_topk(
        estimate_scores, k, seed
    )
    return sufficient, exact


def _behavior_rates(run: Path, expected_ids: set[int]) -> dict[int, float]:
    rewards: dict[int, list[float]] = defaultdict(list)
    path = run / "rollouts_behavior_train.jsonl"
    for line in path.open():
        row = json.loads(line)
        rewards[int(row["prompt_idx"])].append(float(row["reward"]))
    if set(rewards) != expected_ids:
        raise ValueError(
            f"{run.name}: behavior prompt coverage differs from score artifacts"
        )
    return {idx: _mean(values) for idx, values in rewards.items()}


def _strata(rates: Mapping[int, float]) -> dict[str, list[int]]:
    ids = sorted(rates)
    return {
        "all": ids,
        "mixed_reward": [idx for idx in ids if 0.0 < rates[idx] < 1.0],
        "identical_reward": [idx for idx in ids if rates[idx] in (0.0, 1.0)],
    }


def _subset(scores: Mapping[int, float], ids: list[int]) -> dict[int, float]:
    return {idx: float(scores[idx]) for idx in ids}


def _load_divergence(run: Path) -> dict[str, float | None]:
    paths = [run / "divergence_stats.json"]
    if not paths[0].exists():
        paths = sorted(run.glob("divergence_stats.shard*.json"))
    docs = [json.loads(path.read_text()) for path in paths]
    if not docs:
        return {
            "token_kl_beta_pi": None,
            "traj_ess_frac_g11": None,
            "clipfrac_g11": None,
        }

    def weighted(key: str, weight: str) -> float | None:
        available = [doc for doc in docs if key in doc and int(doc.get(weight, 0)) > 0]
        total = sum(int(doc[weight]) for doc in available)
        if not total:
            return None
        return sum(float(doc[key]) * int(doc[weight]) for doc in available) / total

    # A merged exact ESS is preferred.  Shard-local ESS fractions cannot be
    # averaged into a global ESS, so omit the field when only shards exist.
    exact_ess = (
        float(docs[0]["traj_ess_frac_g11"])
        if len(docs) == 1 and ("traj_ess_frac_g11" in docs[0])
        else None
    )
    return {
        "token_kl_beta_pi": weighted("token_kl_beta_pi", "tokens"),
        "traj_ess_frac_g11": exact_ess,
        "clipfrac_g11": weighted("clipfrac_g11", "rollouts"),
    }


def analyze_run(
    run: Path,
    frac: float = 0.10,
    *,
    first_bootstrap: int = 0,
    retention_threshold: float = DEFAULT_RETENTION,
) -> list[dict]:
    if not has_valid_analysis_protocol(run):
        raise ValueError(f"{run}: corrected score/oracle protocols are required")
    artifacts = load_complete_score_artifacts(run)
    config = json.loads((run / "run_config.json").read_text())
    run_generation_git = generation_commit(run)
    ids = set(artifacts.oracle)
    rates = _behavior_rates(run, ids)
    divergence = _load_divergence(run)
    rows: list[dict] = []
    strata = _strata(rates)
    # R ranks independently; A and B form the held-out averaged reference and
    # independently measure split-half reliability.
    if any(
        "r" not in halves or "r_high_budget" not in halves
        for halves in artifacts.splithalf.values()
    ):
        raise ValueError(
            f"{run}: scores_splithalf.json lacks the matched R or high-budget R+ split"
        )
    fresh_all = {idx: halves["r"] for idx, halves in artifacts.splithalf.items()}
    fresh_high_budget_all = {
        idx: halves["r_high_budget"] for idx, halves in artifacts.splithalf.items()
    }
    half_a_all = {idx: halves["a"] for idx, halves in artifacts.splithalf.items()}
    half_b_all = {idx: halves["b"] for idx, halves in artifacts.splithalf.items()}
    truth_all = {
        idx: (half_a_all[idx] + half_b_all[idx]) / 2.0 for idx in artifacts.splithalf
    }
    policies = {f"stale_{name}": artifacts.offpolicy[name] for name in ESTIMATORS}
    policies["passrate_beta"] = {idx: -abs(rate - 0.5) for idx, rate in rates.items()}
    intervals = {}
    if first_bootstrap:
        from first_interval import bootstrap_regime_intervals

        intervals = bootstrap_regime_intervals(
            run,
            strata,
            policies,
            frac=frac,
            samples=first_bootstrap,
            seed=int(config.get("seed", 0)) + 20_260_824,
            tie_seed=int(config.get("seed", 0)) + 1_000,
            retention_threshold=retention_threshold,
        )
    for stratum, stratum_ids in strata.items():
        if len(stratum_ids) < 20:
            continue
        fresh = _subset(fresh_all, stratum_ids)
        fresh_high_budget = _subset(fresh_high_budget_all, stratum_ids)
        half_a = _subset(half_a_all, stratum_ids)
        half_b = _subset(half_b_all, stratum_ids)
        truth = _subset(truth_all, stratum_ids)
        for policy, scores in policies.items():
            metrics = _selection_metrics(
                _subset(scores, stratum_ids),
                fresh,
                fresh_high_budget,
                truth,
                half_a,
                half_b,
                frac=frac,
                seed=int(config.get("seed", 0)) + 1_000,
            )
            interval = intervals.get(stratum)
            ceiling_interval = None
            if interval:
                from measurement_ceiling import gaussian_ceiling_interval

                ceiling_interval = gaussian_ceiling_interval(
                    interval["lower_two_sided_95"],
                    metrics["floor"],
                    interval["upper_two_sided_95"],
                    metrics["n"],
                    metrics["k"],
                )
            interval_final = bool(
                interval
                and int(interval.get("samples", 0)) >= MIN_FINAL_BOOTSTRAP
            )
            floor_lower = interval["lower_one_sided_95"] if interval else None
            fresh_gain_lower = (
                interval["fresh_gain"].get("lower_one_sided_95") if interval else None
            )
            selector_interval = interval["selectors"].get(policy) if interval else None
            gain_lower = (
                selector_interval["gain"].get("lower_one_sided_95")
                if selector_interval
                else None
            )
            gain_upper = (
                selector_interval["gain"].get("upper_one_sided_95")
                if selector_interval
                else None
            )
            retention_margin_lower = (
                selector_interval["retention_margin"].get("lower_one_sided_95")
                if selector_interval
                else None
            )
            measurable = (
                floor_lower >= 2.0 * metrics["chance"] and fresh_gain_lower > 0
                if floor_lower is not None and fresh_gain_lower is not None
                else metrics["measurable_point"]
            )
            retention = metrics["utility_retention"]
            positive_evidence = measurable and (
                gain_lower > 0
                if gain_lower is not None
                else metrics["utility_gain"] > 0
            )
            effective_evidence = positive_evidence and (
                retention_margin_lower >= 0.0
                if retention_margin_lower is not None
                else retention is not None and retention >= retention_threshold
            )
            ineffective_evidence = measurable and (
                gain_upper <= 0
                if gain_upper is not None
                else metrics["utility_gain"] <= 0
            )
            if not measurable:
                point_status = "inconclusive"
            elif ineffective_evidence:
                point_status = (
                    "ineffective_candidate"
                    if interval_final
                    else "provisional_ineffective_candidate"
                )
            elif effective_evidence:
                point_status = (
                    "effective_candidate"
                    if interval_final
                    else "provisional_effective_candidate"
                )
            else:
                point_status = "inconclusive"
            rows.append(
                {
                    "run": run.name,
                    "generation_git": run_generation_git,
                    "model": str(
                        config.get("model_resolved", config.get("model", "?"))
                    ),
                    "dataset": str(config.get("dataset", "?")),
                    "seed": int(config.get("seed", 0)),
                    "drift": int(config.get("drift", config.get("drift_steps", 0))),
                    "stratum": stratum,
                    "policy": policy,
                    "retention_threshold": retention_threshold,
                    "behavior_mixed_fraction": sum(
                        0.0 < rates[idx] < 1.0 for idx in stratum_ids
                    )
                    / len(stratum_ids),
                    **divergence,
                    **metrics,
                    "floor_lower_one_sided_95": floor_lower,
                    "fresh_gain_lower_one_sided_95": fresh_gain_lower,
                    "utility_gain_lower_one_sided_95": gain_lower,
                    "utility_gain_upper_one_sided_95": gain_upper,
                    "retention_margin_lower_one_sided_95": retention_margin_lower,
                    "gaussian_ceiling": (
                        ceiling_interval["ceiling"] if ceiling_interval else None
                    ),
                    "gaussian_ceiling_lower_two_sided_95": (
                        ceiling_interval["ceiling_lower_two_sided_95"]
                        if ceiling_interval
                        else None
                    ),
                    "gaussian_ceiling_upper_two_sided_95": (
                        ceiling_interval["ceiling_upper_two_sided_95"]
                        if ceiling_interval
                        else None
                    ),
                    "gaussian_ceiling_rho_half": (
                        ceiling_interval["rho_half"] if ceiling_interval else None
                    ),
                    "gaussian_ceiling_schema": (
                        ceiling_interval["schema"] if ceiling_interval else None
                    ),
                    "first_bootstrap_samples": interval["samples"] if interval else 0,
                    "final_resampling": interval_final,
                    "measurable": measurable,
                    "positive_evidence": positive_evidence,
                    "effective_evidence": effective_evidence,
                    "ineffective_evidence": ineffective_evidence,
                    "decision_basis": (
                        f"FIRST-{int(interval['samples'])}-final"
                        if interval_final
                        else f"FIRST-{int(interval['samples'])}-provisional"
                        if interval
                        else "point-floor-provisional"
                    ),
                    "point_status": point_status,
                }
            )
    return rows


def summarize_regimes(
    rows: list[dict],
    *,
    retention_threshold: float = DEFAULT_RETENTION,
    replication_fraction: float = DEFAULT_REPLICATION,
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["generation_git"],
                row["model"],
                row["dataset"],
                row["drift"],
                row["stratum"],
                row["policy"],
            )
        ].append(row)

    out = []
    for key, group in sorted(
        grouped.items(), key=lambda item: tuple(map(str, item[0]))
    ):
        n = len(group)
        required = max(1, math.ceil(replication_fraction * n))
        measurable = [row for row in group if row["measurable"]]
        positive = [row for row in measurable if row["positive_evidence"]]
        retained = [
            row
            for row in measurable
            if row["positive_evidence"]
            and (
                (
                    row["retention_margin_lower_one_sided_95"] is None
                    and row["utility_retention"] is not None
                    and row["utility_retention"] >= retention_threshold
                )
                or (
                    row["retention_margin_lower_one_sided_95"] is not None
                    and row["retention_margin_lower_one_sided_95"] >= 0.0
                )
            )
        ]
        nonpositive = [row for row in measurable if row["ineffective_evidence"]]
        final_basis = all(row.get("final_resampling") is True for row in group)
        if n >= 3 and len(measurable) >= required and len(retained) >= required:
            status = "effective" if final_basis else "provisional_effective"
        elif n >= 3 and len(measurable) >= required and len(nonpositive) >= required:
            status = "ineffective" if final_basis else "provisional_ineffective"
        else:
            status = "inconclusive"
        retentions = [
            row["utility_retention"]
            for row in group
            if row["utility_retention"] is not None
        ]
        gain_lowers = [
            row["utility_gain_lower_one_sided_95"]
            for row in group
            if row["utility_gain_lower_one_sided_95"] is not None
        ]
        gain_uppers = [
            row["utility_gain_upper_one_sided_95"]
            for row in group
            if row["utility_gain_upper_one_sided_95"] is not None
        ]
        retention_margin_lowers = [
            row["retention_margin_lower_one_sided_95"]
            for row in group
            if row["retention_margin_lower_one_sided_95"] is not None
        ]
        ceiling_points = [
            row["gaussian_ceiling"]
            for row in group
            if row["gaussian_ceiling"] is not None
        ]
        ceiling_lowers = [
            row["gaussian_ceiling_lower_two_sided_95"]
            for row in group
            if row["gaussian_ceiling_lower_two_sided_95"] is not None
        ]
        ceiling_uppers = [
            row["gaussian_ceiling_upper_two_sided_95"]
            for row in group
            if row["gaussian_ceiling_upper_two_sided_95"] is not None
        ]
        out.append(
            {
                "generation_git": key[0],
                "model": key[1],
                "dataset": key[2],
                "drift": key[3],
                "stratum": key[4],
                "policy": key[5],
                "retention_threshold": retention_threshold,
                "seeds": n,
                "required_replicates": required,
                "measurable_seeds": len(measurable),
                "positive_seeds": len(positive),
                "retained_seeds": len(retained),
                "nonpositive_seeds": len(nonpositive),
                "median_floor": statistics.median(row["floor"] for row in group),
                "median_floor_lower": statistics.median(
                    row["floor_lower_one_sided_95"]
                    for row in group
                    if row["floor_lower_one_sided_95"] is not None
                )
                if any(row["floor_lower_one_sided_95"] is not None for row in group)
                else None,
                "median_gain": statistics.median(row["utility_gain"] for row in group),
                "median_gain_lower": statistics.median(gain_lowers)
                if gain_lowers
                else None,
                "median_gain_upper": statistics.median(gain_uppers)
                if gain_uppers
                else None,
                "median_retention": statistics.median(retentions)
                if retentions
                else None,
                "median_retention_margin_lower": (
                    statistics.median(retention_margin_lowers)
                    if retention_margin_lowers
                    else None
                ),
                "median_gaussian_ceiling": (
                    statistics.median(ceiling_points) if ceiling_points else None
                ),
                "median_gaussian_ceiling_lower": (
                    statistics.median(ceiling_lowers) if ceiling_lowers else None
                ),
                "median_gaussian_ceiling_upper": (
                    statistics.median(ceiling_uppers) if ceiling_uppers else None
                ),
                "median_mixed_fraction": statistics.median(
                    row["behavior_mixed_fraction"] for row in group
                ),
                "median_current_margin": statistics.median(
                    row["normalized_truth_boundary_margin"] for row in group
                ),
                "median_error_margin_ratio": statistics.median(
                    row["observed_error_margin_ratio"]
                    for row in group
                    if row["observed_error_margin_ratio"] is not None
                )
                if any(row["observed_error_margin_ratio"] is not None for row in group)
                else None,
                "median_kl": statistics.median(
                    row["token_kl_beta_pi"]
                    for row in group
                    if row["token_kl_beta_pi"] is not None
                )
                if any(row["token_kl_beta_pi"] is not None for row in group)
                else None,
                "status": status,
            }
        )
    return out


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(summary: list[dict]) -> str:
    retention_threshold = (
        float(summary[0].get("retention_threshold", DEFAULT_RETENTION))
        if summary
        else DEFAULT_RETENTION
    )
    lines = [
        "# Stale-Rollout Regime Map",
        "",
        "Rows from different generation commits are partitioned and never pooled.",
        "",
        (
            "`effective` requires at least three seeds, FIRST-measurable fresh halves in at "
            "least 80% of seeds, positive utility gain over random in those seeds, and at "
            f"least {retention_threshold:.0%} retention of the independent fresh ranker's gain. With resampling, "
            "the retention test uses the one-sided 95% lower endpoint of the paired "
            f"contrast `stale_gain - {retention_threshold:g} * fresh_gain`. `ineffective` requires a non-positive "
            "gain upper endpoint with the same replication rule; all other results are "
            "`inconclusive`. Labels are prefixed with `provisional_` unless each run "
            f"uses at least {MIN_FINAL_BOOTSTRAP:,} prespecified bootstrap replicates."
        ),
        "",
        "| generation | model | data | drift | pool | selector | seeds | KL | margin | e/margin | floor [LB] | Gaussian ceiling [L,U] | gain [L,U] | retention [margin LB] | status |",
        "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['generation_git'][:12]} | {Path(row['model']).name} | "
            f"{row['dataset']} | {row['drift']} | "
            f"{row['stratum']} | {row['policy']} | {row['seeds']} | "
            f"{_fmt(row['median_kl'])} | {_fmt(row['median_current_margin'])} | "
            f"{_fmt(row['median_error_margin_ratio'])} | "
            f"{_fmt(row['median_floor'])} [{_fmt(row['median_floor_lower'])}] | "
            f"{_fmt(row['median_gaussian_ceiling'])} "
            f"[{_fmt(row['median_gaussian_ceiling_lower'])},"
            f"{_fmt(row['median_gaussian_ceiling_upper'])}] | "
            f"{_fmt(row['median_gain'])} [{_fmt(row['median_gain_lower'])},"
            f"{_fmt(row['median_gain_upper'])}] | "
            f"{_fmt(row['median_retention'])} "
            f"[{_fmt(row['median_retention_margin_lower'])}] | **{row['status']}** |"
        )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk-frac", type=float, default=0.10)
    parser.add_argument("--retention", type=float, default=DEFAULT_RETENTION)
    parser.add_argument("--replication", type=float, default=DEFAULT_REPLICATION)
    parser.add_argument(
        "--first-bootstrap",
        type=int,
        default=0,
        help="hierarchical FIRST replicates per run (0=provisional point floor)",
    )
    args = parser.parse_args(argv)
    if not 0 < args.retention <= 1 or not 0 < args.replication <= 1:
        parser.error("retention and replication must be in (0, 1]")

    rows: list[dict] = []
    try:
        analysis_git = _analysis_commit()
        generation_partitions = partition_by_generation(args.runs)
        for run in args.runs:
            rows.extend(
                analyze_run(
                    run,
                    args.topk_frac,
                    first_bootstrap=args.first_bootstrap,
                    retention_threshold=args.retention,
                )
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[regime-map-abort] {exc}", file=sys.stderr)
        return 1
    summary = summarize_regimes(
        rows,
        retention_threshold=args.retention,
        replication_fraction=args.replication,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "analysis_git": analysis_git,
        "generation_commits": sorted(generation_partitions),
        "topk_frac": args.topk_frac,
        "retention_threshold": args.retention,
        "replication_fraction": args.replication,
        "first_bootstrap": args.first_bootstrap,
        "minimum_final_bootstrap": MIN_FINAL_BOOTSTRAP,
        "rows": rows,
        "summary": summary,
    }
    (args.output_dir / "REGIME.json").write_text(json.dumps(payload, indent=1))
    _write_csv(args.output_dir / "REGIME.csv", rows)
    _write_csv(args.output_dir / "REGIME_SUMMARY.csv", summary)
    report = render_report(summary)
    (args.output_dir / "FINAL_REPORT.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
