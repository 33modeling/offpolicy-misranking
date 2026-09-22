#!/usr/bin/env python3
"""Read complete step-zero selector trajectories, without a fixed end-step cap."""
import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import export_e5_checkpoint_curves as curves


def training_history(directory):
    policy = directory / "policy"
    manifests = [p for p in curves.files(policy) if p.name in
                 {"policy_train.json", "checkpoint_state.json"}] if policy.is_dir() else []
    records, sources, issues, log_paths = {}, {}, [], []
    for path in sorted(manifests):
        value = curves.read(path)
        sources[str(path)] = curves.digest(path)
        step = value.get("completed_steps")
        if not isinstance(step, int) or step < 0:
            issues.append(f"invalid completed step: {path}")
            continue
        records[str(path.parent)] = {"step": step, "metadata": value,
            "files": {name: {"path": str((path.parent / name).resolve()),
                "exists": (path.parent / name).is_file()}
                for name in ("adapter_model.safetensors", "optimizer.pt")}}
        stats = path.parent / "grpo_stats.jsonl"
        if stats.is_file():
            log_paths.append((stats, value))
    final_stats = policy / "grpo_stats.jsonl"
    if final_stats.is_file() and all(path != final_stats for path, _ in log_paths):
        log_paths.append((final_stats, {}))
    rows, conflict_steps = {}, set()
    for path, manifest in log_paths:
        raw = path.read_bytes()
        sha = curves.hashlib.sha256(raw).hexdigest()
        expected = manifest.get("grpo_stats_sha256")
        if expected and expected != sha:
            issues.append(f"log hash mismatch: {path}")
            continue
        sources[str(path)] = sha
        # Ignore an unfinished final line while the server is appending to a log.
        lines = raw.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            issues.append(f"unfinished final line omitted: {path}")
            lines = lines[:-1]
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            step = row.get("step")
            if not isinstance(step, int) or step <= 0:
                raise ValueError(f"invalid training step: {path}")
            if manifest.get("completed_steps") is not None and step > manifest["completed_steps"]:
                raise ValueError(f"log extends beyond checkpoint: {path}")
            for key in ("reward_mean", "step_seconds", "grad_norm"):
                number = row.get(key)
                if not isinstance(number, (int, float)) or not math.isfinite(number):
                    raise ValueError(f"invalid {key}, step {step}: {path}")
            if not 0 <= row["reward_mean"] <= 1 or row["step_seconds"] < 0 or row["grad_norm"] < 0:
                raise ValueError(f"invalid metric range, step {step}: {path}")
            world = manifest.get("world_size")
            if world is not None and (not isinstance(world, int) or world < 1):
                raise ValueError(f"invalid world size: {path}")
            item = {"metrics": row, "world_size": world, "sources": [str(path)]}
            if step in rows:
                previous = rows[step]
                if previous["metrics"] != row or (previous["world_size"] is not None and world is not None
                                                 and previous["world_size"] != world):
                    conflict_steps.add(step)
                    continue
                previous["sources"].append(str(path))
                if previous["world_size"] is None:
                    previous["world_size"] = world
            else:
                rows[step] = item
    for step in conflict_steps:
        rows.pop(step, None)
        issues.append(f"conflicting archived/final records excluded at step {step}")
    values = []
    for step, item in sorted(rows.items()):
        item = {**item["metrics"], "world_size": item["world_size"], "sources": item["sources"]}
        item["timed_update_gpu_seconds"] = (item["step_seconds"] * item["world_size"]
                                            if item["world_size"] is not None else None)
        values.append(item)
    return {"training_log": values, "checkpoints": list(records.values()),
            "last_logged_step": max(rows, default=0), "issues": issues, "source_sha256": sources,
            "reward_scope": "training-batch reward; not held-out accuracy",
            "cost_scope": "timed updates only; scoring/setup/retry cost is not inferred",
            "tensor_scope": "model/optimizer locations only; tensors are not loaded or copied"}


def as_of(arm, step):
    points = [p for p in arm["points"] if p["step"] <= step]
    logs = [r for r in arm["history"]["training_log"] if r["step"] <= step]
    checkpoints = [c for c in arm["checkpoints"] if c["step"] <= step]
    return {"decision_step": step, "points": points, "training_log": logs,
            "checkpoints": checkpoints,
            "cutoff_scope": "step-filtered replay; publication time is not independently certified"}


def export(root, decisions):
    root = root.resolve()
    report = curves.export(root, starts={0})
    if not report["experiments"]:
        raise ValueError("no step-zero continuations found; nonzero starts are not stitched together")
    for experiment in report["experiments"]:
        experiment["arms"] = {name: arm for name, arm in experiment["arms"].items()
                              if name in ("on_policy", "cached")}
        for name, arm in experiment["arms"].items():
            arm["history"] = training_history(root / experiment["path"] / arm["selector"])
            # Every available step is retained. These are observation cutoffs, not stopping rules.
            cutoffs = sorted(set(decisions) | {p["step"] for p in arm["points"]}
                             | {c["step"] for c in arm["checkpoints"]})
            arm["as_of"] = [as_of(arm, step) for step in cutoffs]
    report.update(schema="continuous-selector-history/v1", end_step_limit=None,
        branch_start_step=0, decision_steps=sorted(set(decisions)),
        scope="Independent on-policy and SR trajectories from step zero through all saved updates. "
              "Decision cutoffs do not create new branches or limit training. "
              "Different later policies are not a matched-current-state switch experiment. "
              "No H forecast, training-reward substitution, fitting or future-point interpolation is performed.")
    return report


def write_outputs(report, output):
    output = output.resolve()
    root = Path(report["root"])
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("output must be separate from the input tree")
    output.mkdir(parents=True, exist_ok=False)
    (output / "continuous-history.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "evaluation-curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("experiment", "seed", "selector", "step", "reward_pct", "eval_k"))
        for experiment in report["experiments"]:
            for name, arm in experiment["arms"].items():
                for point in arm["points"]:
                    writer.writerow((experiment["path"], experiment["seed"], name, point["step"],
                                     point["reward_pct"], point["eval_k"]))
    with (output / "training-log.csv").open("w", newline="") as handle:
        columns = ("experiment", "seed", "selector", "step", "reward_mean", "grad_norm", "loss",
                   "step_seconds", "world_size", "timed_update_gpu_seconds")
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for experiment in report["experiments"]:
            for name, arm in experiment["arms"].items():
                for row in arm["history"]["training_log"]:
                    writer.writerow({**row, "experiment": experiment["path"], "seed": experiment["seed"],
                                     "selector": name})


def main():
    work = Path(os.environ.get("OM_WORK", "/group-volume/minsoo3.kim/offpolicy-misranking"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=work / "runs/e5-reduced")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decision-steps", type=int, nargs="+", default=[25, 50, 100])
    args = parser.parse_args()
    if any(t < 0 for t in args.decision_steps):
        parser.error("decision steps must be nonnegative")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or args.root.resolve().parent.parent / "exports" / f"continuous-history-{stamp}"
    try:
        report = export(args.root, args.decision_steps)
        write_outputs(report, output)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(f"JSON: {output.resolve() / 'continuous-history.json'}")
    for experiment in report["experiments"]:
        for name, arm in experiment["arms"].items():
            history = arm["history"]
            print(f"{experiment['path']} {name}: last logged step={history['last_logged_step']}; "
                  f"evaluations={len(arm['points'])}; missing checkpoint evaluations={len(arm['needs_evaluation'])}")
    print("Read-only export. No end-step cap, GPU work, experiment writes or predictor fitting.")


if __name__ == "__main__":
    main()
