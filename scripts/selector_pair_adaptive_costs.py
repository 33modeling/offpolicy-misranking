#!/usr/bin/env python3
"""Export saved Pair endpoint costs and checkpoint-linked target times on CPU."""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import selector_pair_gpu as pair_gpu
from paper_result_text import write_export


ARMS = (("on_policy", "selection_full", "fixed_on"),
        ("cached", "selection_full", "fixed_sr"),
        ("adaptive", "selection_full", "sr_gc"),
        ("on_policy", "random_full", "random"))


def inclusive_branch_seconds(closed, scoring_phases):
    """Count deployment work plus selection scored on the reporting ledger."""
    return sum(finish["allocated_gpu_seconds"] for _, finish in closed
               if finish["ledger"] != "reporting" or finish["phase"] in scoring_phases)


def state_entries(root, seed, step):
    entries = {}
    for name in pair_gpu.BRANCHES:
        branch = root / "branches" / name
        child = pair_gpu.switch.child_root(branch, seed, step)
        protocol_path = child / "net_protocol.json"
        if not protocol_path.is_file():
            raise FileNotFoundError(f"state protocol not published: {protocol_path}")
        points = list(pair_gpu.base.entries(child))
        if len(points) != 1:
            raise ValueError(f"expected one saved Pair state: {child}")
        out = points[0]
        protocol = pair_gpu.core.read(protocol_path)
        if protocol.get("mode") != "test" or protocol.get("role") != "test":
            raise ValueError(f"not a held-out Pair state: {child}")
        entries[name] = (branch, out, pair_gpu.core.read(out / "contract.json"),
                         protocol, pair_gpu.core.read(child / "suite.json"))
    return pair_gpu.pair.matched_state([entry[2] for entry in entries.values()]), entries


def frozen_choice(root, barrier, identity, seed, step):
    state = f"s{seed}-t{step}"
    path = root / "sr-gc" / state / "decision.json"
    if pair_gpu.base.digest(path) != barrier["decisions"][state]:
        raise ValueError(f"frozen SR-GC decision hash changed: {state}")
    value = pair_gpu.core.read(path)
    selector = value.get("selector")
    if (value.get("method") != "SR-GC" or value.get("state_id") != identity or
            selector not in {"on_policy", "cached"} or
            not math.isfinite(value.get("d", math.nan)) or
            selector != ("cached" if value["d"] < 0 else "on_policy") or
            not all(math.isfinite(value.get(key, math.nan)) and value[key] >= 0 for key in
                    ("new_measurement_gpu_seconds", "reused_ranking_gpu_seconds",
                     "diagnosis_gpu_seconds")) or
            not math.isclose(value["diagnosis_gpu_seconds"],
                             value["new_measurement_gpu_seconds"] +
                             value["reused_ranking_gpu_seconds"], abs_tol=1e-6)):
        raise ValueError(f"invalid frozen SR-GC decision: {state}")
    return value


def measure(entry, arm, label, target, diagnosis):
    _, out, _, protocol, _ = entry
    directory = out / arm
    result_path = directory / "result.json"
    row = {"arm": label, "path": str(directory), "reward_percent": None,
           "branch_gpu_hours": None, "diagnosis_gpu_hours": diagnosis / 3600,
           "inclusive_endpoint_gpu_hours": None, "target_status": "unavailable",
           "inclusive_target_gpu_hours": None, "reason": None}
    if not result_path.is_file():
        row["reason"] = "endpoint_result_missing"
        return row
    try:
        result = pair_gpu.switch.runtime.validate_result(out, protocol, arm)
        if not result["complete"]:
            raise ValueError("endpoint result is incomplete")
        _, events = pair_gpu.base.read_cost_events(directory)
        closed = pair_gpu.pair.finished_events(events)
        branch_seconds = inclusive_branch_seconds(closed, pair_gpu.switch.SCORING_PHASES)
        if not math.isfinite(branch_seconds) or branch_seconds < 0:
            raise ValueError("invalid inclusive branch cost")
        row["reward_percent"] = statistics.fmean(result["rewards"].values()) * 100
        row["branch_gpu_hours"] = branch_seconds / 3600
        row["inclusive_endpoint_gpu_hours"] = (branch_seconds + diagnosis) / 3600
        if not (directory / "curve.json").is_file():
            row["reason"] = "curve_summary_missing"
            return row
        # measured_curve otherwise recovers this receipt by writing it. This
        # exporter is read-only and leaves that separate recovery to the runner.
        if not (directory / "policy/curve-cost/final.json").is_file():
            row["reason"] = "final_cost_receipt_missing"
            return row
        curve = pair_gpu.measured_curve(entry, arm)
        outcome = pair_gpu.pair.crossing(curve["points"], target, diagnosis=diagnosis,
                                         observed_gpu_seconds=curve["observed_gpu_seconds"])
        row["target_status"] = outcome["status"]
        if outcome["status"] == "reached":
            row["inclusive_target_gpu_hours"] = outcome["gpu_seconds"] / 3600
        elif outcome["status"] == "right_censored":
            row["observed_through_gpu_hours"] = outcome["observed_through_gpu_seconds"] / 3600
        return row
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        row.update(target_status="invalid", reason=f"{type(exc).__name__}: {exc}")
        return row


def collect(root, *, seed=None, step=None):
    root = Path(root).resolve()
    protocol = pair_gpu.manifest(root, bind_runtime=False)
    barrier = pair_gpu.core.read(root / "test-decisions.json")
    expected = {f"s{s}-t{t}" for s in pair_gpu.pair.TEST_SEEDS for t in pair_gpu.pair.STEPS}
    if (barrier.get("protocol_id") != protocol["protocol_id"] or
            set(barrier.get("decisions", {})) != expected):
        raise ValueError("frozen Pair decision barrier changed")
    rows = []
    for s in pair_gpu.pair.TEST_SEEDS:
        if seed is not None and s != seed:
            continue
        for t in pair_gpu.pair.STEPS:
            if step is not None and t != step:
                continue
            identity, entries = state_entries(root, s, t)
            choice = frozen_choice(root, barrier, identity, s, t)
            for branch, arm, label in ARMS:
                name = "adaptive-" + choice["selector"] if branch == "adaptive" else branch
                diagnosis = choice["diagnosis_gpu_seconds"] if branch == "adaptive" else 0.
                row = measure(entries[name], arm, label, protocol["target_reward"], diagnosis)
                rows.append({"seed": s, "step": t, "sr_gc_selector": choice["selector"], **row})
    return {"schema": "offpolicy-selector-pair/adaptive-cost-export-v1",
            "scope": "Saved held-out Pair branch endpoint costs and validated checkpoint target costs. "
                     "Selection scored on the reporting ledger is included. The SR-GC branch "
                     "adds its frozen acquisition cost once. Shared prefix and initial cache "
                     "construction are excluded. Independent t25/t50/t100 starts are not a "
                     "within-run switch. Missing curves do not imply non-attainment.",
            "source_root": str(root), "target_reward": protocol["target_reward"], "rows": rows}


def table(report):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(("state", "branch", "choice", "reward_%", "endpoint_GPU_h",
                     "target_status", "target_GPU_h", "reason"))
    for row in report["rows"]:
        writer.writerow((f"s{row['seed']}-t{row['step']}", row["arm"], row["sr_gc_selector"],
                         "" if row["reward_percent"] is None else f"{row['reward_percent']:.3f}",
                         "" if row["inclusive_endpoint_gpu_hours"] is None else
                         f"{row['inclusive_endpoint_gpu_hours']:.3f}",
                         row["target_status"],
                         "" if row["inclusive_target_gpu_hours"] is None else
                         f"{row['inclusive_target_gpu_hours']:.3f}", row["reason"] or ""))
    return output.getvalue()


def main():
    work = Path(os.environ.get("OM_WORK", f"/group-volume/{os.environ.get('OM_USER', 'minsoo3.kim')}/offpolicy-misranking"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=work / "runs/selector-pair-v1")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, choices=(3, 4))
    parser.add_argument("--step", type=int, choices=(25, 50, 100))
    args = parser.parse_args()
    report = collect(args.root, seed=args.seed, step=args.step)
    write_export("selector-pair-adaptive-costs", report, table(report), args.out)
    print(f"[pair-costs] endpoints={sum(r['reward_percent'] is not None for r in report['rows'])}/"
          f"{len(report['rows'])}; checkpoint-target costs="
          f"{sum(r['target_status'] == 'reached' for r in report['rows'])}; "
          "unavailable values remain blank")


if __name__ == "__main__":
    main()
