#!/usr/bin/env python3
"""Explicit experiment dispatch; never start an entire suite implicitly."""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Experiment:
    runner: str
    scope: str | None
    tasks_per_dataset: int
    purpose: str


EXPERIMENTS = {
    "mechanism": Experiment("run_srgc_mechanism.sh", None, 5, "Requested stage-wise GRPO interventions"),
    "qwen": Experiment("run_srgc_qwen35.sh", None, 20, "V6: another online backbone; preparation has 5 seed tasks"),
    "timing": Experiment("run_srgc_switch_validation.sh", "timing", 25, "V6: common-prefix switching times"),
    "rules": Experiment("run_srgc_switch_validation.sh", "rules", 10, "V6: temporal confirmation"),
    "support": Experiment("run_srgc_support.sh", None, 15, "V6: matched SR refresh and retention controls"),
    "pool": Experiment("run_srgc_sr_refresh.sh", "pool", 5, "V6: full-pool SR refresh"),
    "switch_repeat": Experiment("run_srgc_sr_refresh.sh", "switch_repeat", 5, "V6: repeated transitions and costs"),
    "direction": Experiment("run_srgc_sr_refresh.sh", "direction", 15, "Partial reference-direction controls"),
    "fixed200": Experiment("run_srgc_sr_refresh.sh", "switch_fixed200", 5, "Existing fixed control, reused by timing"),
    "replicate": Experiment("run_srgc_sr_refresh.sh", "replicate", 20, "Supplementary post-prefix sampling repeats"),
    "candidates": Experiment("run_srgc_sr_refresh.sh", "candidates", 5, "Legacy candidate SR refresh"),
    "sr_hold": Experiment("run_srgc_sr_refresh.sh", "sr_hold", 5, "Cached-SR retained batch control"),
    "p0": Experiment("run_srgc.sh", None, 20, "Original additional-seed arms; collect existing results first"),
}


def commands(dataset, experiment, action):
    if dataset not in {"math", "mbpp", "all"} or experiment not in EXPERIMENTS:
        raise ValueError("choose math|mbpp|all and one explicit experiment")
    if action not in {"run", "status", "results", "json", "costs"}:
        raise ValueError("unknown action")
    spec = EXPERIMENTS[experiment]
    runner = ["sh", str(REPO / "scripts" / spec.runner)]
    if action == "costs":
        if experiment != "p0":
            raise ValueError("costs is a P0 action; other experiments include costs in results")
        return [[*runner, dataset, "costs"]]
    if spec.runner == "run_srgc_sr_refresh.sh":
        if action == "run":
            return [[*runner, dataset, spec.scope]]
        if action == "status":
            return [[*runner, dataset, "status", spec.scope]]
        datasets = ("math", "mbpp") if dataset == "all" else (dataset,)
        if action == "json" and dataset == "all":
            raise ValueError("JSON stdout requires math or mbpp; use 'all results' to collect both files")
        flags = ["--json"] if action == "json" else []
        return [[*runner, name, "results", *flags] for name in datasets]
    if experiment in {"timing", "rules"}:
        if action == "run":
            return [[*runner, dataset, spec.scope]]
        if action == "status":
            return [["sh", str(REPO / "scripts/run_srgc_sr_refresh.sh"), dataset, "status", spec.scope]]
    if action == "json":
        if experiment == "support":
            return [["sh", str(REPO / "scripts/run_srgc_sr_refresh.sh"), dataset, "support_json"]]
        if experiment in {"qwen", "p0"}:
            raise ValueError(f"{experiment} uses its existing results exporter, not JSON stdout")
    return [[*runner, dataset] if action == "run" and experiment == "qwen" else [*runner, dataset, action]]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["list"]:
        print("experiment     tasks/dataset  purpose (one task per four-GPU node)")
        for name, spec in EXPERIMENTS.items():
            print(f"{name:14} {spec.tasks_per_dataset:>13}  {spec.purpose}")
        print("These are design maxima, not pending counts. Scopes overlap; do not add them.")
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("math", "mbpp", "all"))
    parser.add_argument("experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("action", nargs="?", default="run", choices=("run", "status", "results", "json", "costs"))
    args = parser.parse_args(argv)
    try:
        tasks = commands(args.dataset, args.experiment, args.action)
    except ValueError as exc:
        parser.error(str(exc))
    if len(tasks) == 1:
        os.execvpe(tasks[0][0], tasks[0], os.environ.copy())
    failed = False
    for command in tasks:
        # Reports only: a missing MATH export must not suppress a healthy MBPP export.
        code = subprocess.call(command, cwd=REPO)
        if code in (-2, -15, 130, 143):
            return 128 - code if code < 0 else code
        failed |= code != 0
    return int(failed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except OSError as exc:
        print(f"LAUNCH ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
