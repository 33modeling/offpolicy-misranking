#!/usr/bin/env python3
"""Validated, seed-paired comparisons for selection, retention and cache controls."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.srgc_replicate_worker import SUPPORT_ARMS, tasks_for
from scripts.srgc_sr_refresh import _endpoint, result_identity
from scripts.srgc_switch_validation_report import comparable, recorded_policy
from srgc_rebuttal.plan import load_plan

ARMS = ("on_policy", "random", "sr", "switch", *SUPPORT_ARMS)
CONTRASTS = (
    ("gradient ranking", "on_policy", "direction_removed"),
    ("SR batch retention", "sr_hold", "sr"),
    ("fresh SR rewards", "sr_refresh_matched", "sr_hold"),
    ("SR vs gradient at matched retention", "sr_hold", "on_policy"),
)


def summarize(rows):
    keys = [(row["dataset"], row["seed"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate training seed in support report")
    summaries = []
    for dataset in sorted({row["dataset"] for row in rows}):
        selected = [row for row in rows if row["dataset"] == dataset]
        identities = sorted({(v["implementation_sha256"], v["checkpoint_policy"]["attention"])
                             for row in selected for v in row["arms"].values() if comparable((v,))})
        for identity in identities or [("unknown", "unknown")]:
            for label, left, right in CONTRASTS:
                paired = []
                for row in selected:
                    a, b = row["arms"].get(left), row["arms"].get(right)
                    if a is None or b is None or not comparable((a, b)):
                        continue
                    if (a["implementation_sha256"], a["checkpoint_policy"]["attention"]) != identity:
                        continue
                    paired.append((row["seed"], a["reward_percent"] - b["reward_percent"]))
                values = [value for _, value in paired]
                summaries.append(dict(dataset=dataset, implementation_sha256=identity[0], attention=identity[1],
                    comparison=label, left=left, right=right, n=len(values),
                    seeds=[seed for seed, _ in paired], differences_pp=values,
                    mean_pp=statistics.mean(values) if values else None,
                    sample_sd_pp=statistics.stdev(values) if len(values) > 1 else None))
    return summaries


def collect(dataset):
    rows, errors, warnings, sources = [], [], [], {}
    datasets = ("math", "mbpp") if dataset == "all" else (dataset,)
    for name in datasets:
        try:
            tasks = tasks_for(name, "support")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        seen = set()
        for task in tasks:
            if task.seed in seen:
                continue
            seen.add(task.seed)
            row = dict(dataset=name, seed=task.seed, plan=str(task.plan), output=str(task.folder), arms={})
            rows.append(row)
            try:
                plan = load_plan(task.plan)
                paths = {arm: task.folder / f"{arm}-endpoint.json" for arm in ARMS}
                present = {arm: path for arm, path in paths.items() if path.is_file()}
                if not present:
                    continue
                verified = result_identity(task.plan, plan, task.seed, recorded=True)
                for arm, path in present.items():
                    try:
                        raw = {}
                        value = _endpoint(path, task.plan, plan, task.seed, task.folder, arm,
                                          verified=verified, raw_records=raw)
                        try:
                            value["checkpoint_policy"] = recorded_policy(
                                value, task.folder, arm, task.seed, plan["total_updates"])
                        except (ImportError, OSError, ValueError, TypeError, KeyError, RuntimeError,
                                EOFError, pickle.UnpicklingError) as exc:
                            value["checkpoint_policy"] = None
                            warnings.append(f"{path} attention: {exc}")
                        if value["checkpoint_policy"] is None:
                            warnings.append(f"{path}: attention unverified; reward shown, pairing excluded")
                        row["arms"][arm] = value
                        sources[str(path)] = raw[str(path)]
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        errors.append(f"{path}: {exc}")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f"{task.key}: {exc}")
    return dict(rows=rows, comparisons=summarize(rows), errors=errors, warnings=warnings, source_results=sources)


def number(value):
    return "-" if value is None else f"{value:.3f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("math", "mbpp", "all"), required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    from scripts.srgc_result_collection import publish, print_paths
    report = collect(args.dataset)
    try:
        publish(report, "support")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"COLLECTION ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print("SUPPORT results: validated endpoints; selection costs include all recorded selection work")
        print(f"{'dataset':7} {'seed':>4} {'arm':24} {'reward%':>9} {'select GPU-h':>12} {'train GPU-h':>12}")
        for row in report["rows"]:
            print(f"{row['dataset']} seed={row['seed']} output={row['output']}")
            for arm in ARMS:
                value = row["arms"].get(arm, {})
                hours = [value.get(key) / 3600 if value.get(key) is not None else None
                         for key in ("selection_gpu_seconds", "training_gpu_seconds")]
                print(f"{row['dataset']:7} {row['seed']:4} {arm:24} {number(value.get('reward_percent')):>9} "
                      f"{number(hours[0]):>12} {number(hours[1]):>12}")
                if value and value.get("cost_measurement_complete") is not True:
                    print("  cost measurements incomplete or unverified; not a complete execution total")
        print("PAIRED reward differences (left minus right, percentage points; sample SD across training seeds)")
        for item in report["comparisons"]:
            coverage = "COMPLETE" if item["n"] == 5 else "PARTIAL"
            print(f"{item['dataset']} implementation={item['implementation_sha256']} attention={item['attention']} "
                  f"{item['left']} - {item['right']}: n={item['n']}/5 "
                  f"mean={number(item['mean_pp'])} SD={number(item['sample_sd_pp'])} {coverage}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        for error in report["errors"]:
            print(f"ERROR: {error}")
    print_paths(report, json_output=args.json)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
