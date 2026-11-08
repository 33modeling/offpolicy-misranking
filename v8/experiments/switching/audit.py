"""E0 provenance audit: paper cell -> run -> checkpoint -> raw evaluation -> cost receipts.

Never overwrites reported numbers: it emits a reconciliation report with one of
verified / corrected_with_evidence / incomplete_provenance / inconsistent /
missing_raw_evaluation / partial_cost per cell. Gemma's progress/final mismatch and
Llama's executed checkpoint/denominator are separate audit cases.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .common import read_json, sha256_file

ARMS = ("random", "sr", "on_policy", "switch")
EVIDENCE = {
    "olmo_math": ("2026-10-01/math-final-seeds5-9.csv", "counts"),
    "olmo_mbpp": ("2026-10-01/mbpp-final-seeds5-9.csv", "counts"),
    "qwen_math": ("2026-10-14/qwen-math-seeds5-9.csv", "percent"),
    "qwen_mbpp": ("2026-10-19/qwen-mbpp-seeds5-9.csv", "percent"),
    "gemma_math": ("2026-11-02/gemma-math-seeds5-9.csv", "percent"),
    "gemma_mbpp": ("2026-11-03/gemma-mbpp-seeds5-9.csv", "percent"),
}
SPECIAL = {
    "gemma_mbpp": "progress display vs final table differ for Switch seeds 5/8 (93.7/94.7 vs 93.0/92.7); compare "
                  "the endpoint JSON and progress records of the same run ID (v7/EXPERIMENTS.md 2026-11-03)",
    "gemma_math": "IT model identity confirmed from the preflight path; recheck run ID linkage",
    "llama_math": "configured model name vs loaded checkpoint hash; endpoint horizon and denominator; GPU-hour lower bounds",
    "llama_mbpp": "configured model name vs loaded checkpoint hash; endpoint horizon and denominator; GPU-hour lower bounds",
}


def paper_cells(manuscript_root: Path):
    ev = Path(manuscript_root) / "v7" / "evidence"
    cells = []
    for cohort, (rel, kind) in EVIDENCE.items():
        path = ev / rel
        if not path.exists():
            continue
        digest = sha256_file(path)
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            seed = int(r["seed"])
            if kind == "counts":
                cells.append({"cohort": cohort, "seed": seed, "arm": r["arm"],
                              "reward_fraction": int(r["correct"]) / int(r["responses"]), "correct": int(r["correct"]),
                              "responses": int(r["responses"]), "precision": "exact_counts",
                              "switch_update": None, "source": f"v7/evidence/{rel}", "source_sha256": digest})
            else:
                for arm in ARMS:
                    value = r.get(f"{arm}_percent")
                    if value in (None, ""):
                        continue
                    su = r.get("switch_after_update") if arm == "switch" else None
                    cells.append({"cohort": cohort, "seed": seed, "arm": arm, "reward_fraction": float(value) / 100,
                                  "correct": None, "responses": None, "precision": "one_decimal_percent",
                                  "switch_update": int(su) if su not in (None, "") else None,
                                  "never_switched": arm == "switch" and r.get("switch_decision") == "never_switched",
                                  "source": f"v7/evidence/{rel}", "source_sha256": digest})
    return cells


def _endpoint(run_root: Path, seed: int, arm: str):
    p = Path(run_root) / f"seed-{seed}" / f"{arm}-endpoint.json"
    return (read_json(p), p) if p.exists() else (None, p)


def audit_cell(cell, run_root: Path | None):
    out = dict(cell, status=None, checks={}, special_case=SPECIAL.get(cell["cohort"]))
    if run_root is None:
        out.update(status="missing_raw_evaluation", reason="run root not available on this machine")
        return out
    endpoint, path = _endpoint(run_root, cell["seed"], cell["arm"])
    if endpoint is None:
        out.update(status="missing_raw_evaluation", reason=f"no {path.name}")
        return out
    from .runner import integer_counts
    per_q = endpoint.get("per_question_reward") or {}
    try:
        counts = integer_counts(per_q, 8)
        recomputed = sum(counts.values()) / (8 * len(counts)) if counts else None
    except ValueError as exc:
        out.update(status="inconsistent", reason=f"per-question rewards not integer counts: {exc}")
        return out
    out["checks"] = {"endpoint": str(path), "n_questions": len(per_q), "recomputed_reward": recomputed,
                     "endpoint_reward": endpoint.get("reward"), "total_updates": endpoint.get("total_updates"),
                     "switched_at": endpoint.get("switched_at"),
                     "prefix_checkpoint_sha256": endpoint.get("prefix_checkpoint_sha256"),
                     "cost_measurement_complete": endpoint.get("cost_measurement_complete"),
                     "implementation_sha256": endpoint.get("implementation_sha256")}
    prefix_ready = Path(run_root) / f"seed-{cell['seed']}" / "prefix-ready.json"
    linked = prefix_ready.exists() and read_json(prefix_ready).get("checkpoint_sha256") == endpoint.get("prefix_checkpoint_sha256")
    tol = 1e-12 if cell["precision"] == "exact_counts" else 0.0005 + 1e-12
    problems = []
    if len(per_q) != 300:
        problems.append(f"question count {len(per_q)} != 300")
    if endpoint.get("total_updates") != 275:
        problems.append(f"horizon {endpoint.get('total_updates')} != 275")
    if recomputed is None or abs(recomputed - cell["reward_fraction"]) > tol:
        problems.append(f"recomputed {recomputed} vs reported {cell['reward_fraction']}")
    if cell.get("switch_update") is not None and endpoint.get("switched_at") != cell["switch_update"]:
        problems.append(f"switch {endpoint.get('switched_at')} vs reported {cell['switch_update']}")
    if problems:
        out.update(status="inconsistent", reason="; ".join(problems))
    elif not linked:
        out.update(status="incomplete_provenance", reason="prefix checkpoint hash not linked to prefix-ready.json")
    elif not endpoint.get("cost_measurement_complete"):
        out.update(status="partial_cost", reason="unfinished phase receipts; reward verified")
    else:
        out.update(status="verified")
    return out


def split_overlap(bundle_paths, code_root=None):
    from . import coderepo
    coderepo.activate(code_root)
    from srgc_rebuttal.plan import validate_inputs
    rows = []
    for p in bundle_paths:
        d = json.loads(Path(p).read_text())
        try:
            validate_inputs(d, require_cache=False)
            ok, err = True, None
        except ValueError as exc:
            ok, err = False, str(exc)
        ranking, ev, pool = set(d["ranking_validation_ids"]), set(d["evaluation_ids"]), set(d["validation_pool_ids"])
        dev = pool - ranking
        rows.append({"bundle": str(p), "sha256": sha256_file(p), "validate_inputs": ok, "error": err,
                     "ranking_vs_final_overlap": len(ranking & ev), "dev_vs_final_overlap": len(dev & ev),
                     "dev_vs_ranking_overlap": len(dev & ranking), "candidates_vs_final_overlap":
                     len(set(d["candidate_ids"]) & ev), "n_dev_eval": len(dev), "n_final": len(ev),
                     "n_ranking_reference": len(ranking), "cache_present": bool(d.get("cached_rewards"))})
    return rows


def run_audit(manuscript_root, *, run_roots=None, bundles=(), code_root=None):
    cells = paper_cells(manuscript_root)
    run_roots = run_roots or {}
    audited = [audit_cell(c, Path(run_roots[c["cohort"]]) if c["cohort"] in run_roots else None) for c in cells]
    overlap = split_overlap(bundles, code_root) if bundles else []
    status = {}
    for a in audited:
        status.setdefault(a["cohort"], {}).setdefault(a["status"], 0)
        status[a["cohort"]][a["status"]] += 1
    return {"cells": audited, "status_counts": status, "split_overlap": overlap,
            "special_cases": SPECIAL, "verified_runs": [a for a in audited if a["status"] in ("verified", "partial_cost")],
            "policy": "audit never overwrites manuscript numbers; differences are reported only"}
