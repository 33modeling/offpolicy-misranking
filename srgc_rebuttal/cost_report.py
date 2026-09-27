"""Compare measured per-seed arm costs without scaling one refresh by update count."""

import argparse
import json
from pathlib import Path
import statistics

from .cost_ledger import PhaseLedger
from .plan import DEFAULT_PLAN, digest, input_path, load_plan
from .runtime import matches, run_root


def measured(directory):
    report = PhaseLedger(directory).totals()
    if not report["recorded_phases"]:
        report["complete"] = False
        report["total_gpu_seconds"] = None
    return report


def seed_costs(folder, bundle_path, arms):
    scopes = {p.name: measured(p) for p in sorted((folder / "cost-receipts").glob("*")) if p.is_dir()}
    sessions = {p.name: measured(p) for p in sorted((folder / "invocations").glob("*")) if p.is_dir()}
    rows = {}
    marker_path = folder / "run.json"
    marker = json.loads(marker_path.read_text()) if marker_path.exists() else None
    for arm in arms:
        ledger = scopes.get(arm, measured(folder / "cost-receipts" / arm))
        endpoint_path = folder / f"{arm}-endpoint.json"
        endpoint = json.loads(endpoint_path.read_text()) if endpoint_path.exists() else {}
        expected = {k: marker[k] for k in ("seed", "plan_sha256", "input_sha256", "implementation_sha256")} if marker else {}
        if endpoint and marker is not None and not matches(endpoint, expected):
            raise ValueError("cost endpoint belongs to a different experiment identity")
        complete = ledger["complete"] and endpoint.get("cost_measurement_complete", False)
        relevant_sessions = [sessions[s] for s in (arm, "all") if s in sessions]
        complete = complete and bool(relevant_sessions) and all(s["complete"] for s in relevant_sessions)
        values = ledger["known_gpu_seconds"]
        rows[arm] = {
            "complete": complete,
            "selection_gpu_seconds": values.get("selection_gpu_seconds", 0.0),
            "training_gpu_seconds": values.get("training_gpu_seconds", 0.0),
            "preparation_gpu_seconds": values.get("preparation_gpu_seconds", 0.0),
            "evaluation_gpu_seconds": values.get("evaluation_gpu_seconds", 0.0),
            "checkpoint_gpu_seconds": sum(values.get(f"{p}_gpu_seconds", 0.0)
                                          for p in ("checkpoint_save", "checkpoint_load")),
            "startup_gpu_seconds": values.get("startup_gpu_seconds", 0.0),
            "continuation_phases_gpu_seconds": ledger["total_gpu_seconds"] if complete else None,
            "selection_training_preparation_gpu_seconds": sum(values.get(f"{p}_gpu_seconds", 0.0)
                for p in ("selection", "training", "preparation")) if complete else None,
            "exclusive_stages": ledger["exclusive_stages"], "counts": ledger["counts"],
            "unfinished_phases": ledger["unfinished_phases"]}
        rows[arm]["known_completed_phase_gpu_seconds"] = values
        if not complete:
            for key in ("selection_gpu_seconds", "training_gpu_seconds", "preparation_gpu_seconds",
                        "evaluation_gpu_seconds", "checkpoint_gpu_seconds", "startup_gpu_seconds"):
                rows[arm][key] = None
    cache_path = bundle_path.with_suffix(".cache") / "cost-summary.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else None
    if cache is not None and cache["bundle_sha256"] != digest(bundle_path):
        raise ValueError("cache timings belong to a different input bundle")
    all_complete = (bool(scopes) and bool(sessions) and "shared-prefix" in scopes and
                    all(r["complete"] for r in [*scopes.values(), *sessions.values(), *rows.values()]))
    phase_sum = sum(sum(r["known_gpu_seconds"].values()) for r in scopes.values())
    session_sum = sum(sum(r["known_gpu_seconds"].values()) for r in sessions.values())
    difference = session_sum - phase_sum
    if all_complete and difference < -max(0.01, session_sum * 1e-6):
        raise ValueError("phase totals exceed inclusive invocation costs")
    cache_complete = cache is not None and cache["complete"]
    cache_total = cache["invocations"]["total_gpu_seconds"] if cache_complete else None
    return {"arms": rows, "shared_prefix": scopes.get("shared-prefix"),
            "shared_all_task_startup_and_load": scopes.get("all"),
            "cache_build": cache, "invocations": sessions,
            "experiment_accounting": {
                "complete": all_complete,
                "phase_gpu_seconds": phase_sum,
                "inclusive_invocation_gpu_seconds": session_sum if all_complete else None,
                "cache_inclusive_gpu_seconds": cache_total,
                "experiment_including_cache_gpu_seconds": session_sum + cache_total
                    if all_complete and cache_complete else None,
                "orchestration_and_timer_gpu_seconds": max(0.0, difference) if all_complete else None}}


def compare(plan_path):
    plan = load_plan(plan_path)
    root = run_root(plan_path, plan)
    for seed in plan["seeds"]:
        marker_path = root / f"seed-{seed}" / "run.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            if marker["plan_sha256"] != digest(plan_path):
                raise ValueError("cost receipts belong to a different plan")
    rows = [{"seed": seed, **seed_costs(root / f"seed-{seed}", input_path(plan_path, plan, seed), plan["arms"])}
            for seed in plan["seeds"]]
    statistics_by_arm, contrasts = {}, {}
    for arm in plan["arms"]:
        values = [row["arms"][arm]["selection_training_preparation_gpu_seconds"] for row in rows]
        statistics_by_arm[arm] = {"per_seed_gpu_seconds": values,
            "mean_gpu_seconds": statistics.mean(values) if all(v is not None for v in values) else None,
            "sample_sd_gpu_seconds": statistics.stdev(values) if len(values) > 1 and all(v is not None for v in values) else None}
    for baseline in ("random", "sr", "on_policy"):
        deltas = [None if a is None or b is None else a - b for a, b in zip(
            statistics_by_arm["switch"]["per_seed_gpu_seconds"], statistics_by_arm[baseline]["per_seed_gpu_seconds"])]
        contrasts[f"switch_minus_{baseline}"] = {"per_seed_gpu_seconds": deltas,
            "mean_gpu_seconds": statistics.mean(deltas) if all(d is not None for d in deltas) else None}
    return {"units": "synchronized elapsed seconds and allocated GPU-seconds; not CUDA kernel active time",
            "zero": "an absent stage in a completed measured scope did not run; missing or interrupted totals are null",
            "accounting": "exclusive stages sum to their phase; phases are contained in invocations, never added to them; cache and shared prefix are separate shared costs",
            "per_seed": rows, "arm_statistics": statistics_by_arm, "paired_cost_differences": contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    print(json.dumps(compare(args.plan), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
