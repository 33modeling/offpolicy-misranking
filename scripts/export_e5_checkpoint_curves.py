#!/usr/bin/env python3
"""Read-only E5 checkpoint/evaluation export; no training, evaluation or interpolation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ARMS = {"passrate_beta": "cached", "fresh_r": "on_policy", "random": "random"}
STEP = re.compile(r"(?:checkpoint-|step-|policy_step_)(\d+)$")
SKIP = {"discards", "waivers", "quarantine", ".git", "node_modules"}


def files(root):
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith(".")
                         and not Path(directory, d).is_symlink())
        for name in sorted(names):
            path = Path(directory, name)
            if not path.is_symlink():
                yield path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def number(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("reward must be a finite fraction in [0, 1]")
    return value


def step_from(path):
    for part in reversed(Path(path).parts):
        match = STEP.fullmatch(part)
        if match:
            return int(match[1])
    return None


def sharded_reward(directory, prompts, k, provenance):
    if not prompts or not k:
        raise ValueError("missing evaluation prompt count or k")
    rewards = defaultdict(list)
    for shard in range(4):
        path = directory / f"shard-{shard}.jsonl"
        done = path.with_suffix(".done.json")
        contract = path.with_suffix(".contract.json")
        receipt = read(done)
        binding = read(contract)
        if receipt.get("binding") != binding:
            raise ValueError(f"evaluation binding mismatch: {directory}")
        if receipt.get("sha256", receipt.get("rollouts_sha256")) != digest(path):
            raise ValueError(f"evaluation checksum mismatch: {path}")
        if "k" in binding and binding["k"] != k:
            raise ValueError("evaluation k differs")
        expected = set(range(prompts * shard // 4, prompts * (shard + 1) // 4))
        for line in path.read_text().splitlines():
            row = json.loads(line)
            idx = row["prompt_idx"]
            if idx not in expected:
                raise ValueError("evaluation prompt index outside shard")
            rewards[idx].append(number(row["reward"]))
        provenance.extend([path, done, contract])
    if set(rewards) != set(range(prompts)) or any(len(v) != k for v in rewards.values()):
        raise ValueError("incomplete or duplicate evaluation responses")
    return statistics.fmean(statistics.fmean(v) for v in rewards.values())


def export(root):
    root = root.resolve(strict=True)
    all_files = list(files(root))
    candidates = sorted({p.parent for p in all_files if p.name in
                         {"experiment.json", "downstream_results.csv"}})
    experiments, errors, sources = [], [], {}

    def source(path):
        key = str(path.relative_to(root))
        sources[key] = digest(path)
        return key

    for directory in candidates:
        match = re.fullmatch(r"s(\d+)", directory.name)
        branch = re.fullmatch(r"(.+)-d(\d+)(.*)", directory.parent.name)
        if not match or not branch:
            continue
        contract_path = directory / "experiment.json"
        contract = read(contract_path) if contract_path.is_file() else {}
        start, seed = int(branch[2]), int(match[1])
        if contract.get("drift", start) != start or contract.get("seed", seed) != seed:
            raise ValueError(f"folder/experiment identity mismatch: {directory}")
        if contract:
            source(contract_path)
        evaluation = directory / "evaluation.json"
        eval_hash = source(evaluation) if evaluation.is_file() else None
        eval_identity = sources[eval_hash] if eval_hash else None
        count, k = contract.get("eval_prompts"), contract.get("eval_k")
        horizon = contract.get("steps")
        csv_path = directory / "downstream_results.csv"
        endpoints = {}
        if csv_path.is_file():
            source(csv_path)
            endpoints = {row["selector"]: row for row in csv.DictReader(io.StringIO(csv_path.read_text()))}
        exp = {"path": str(directory.relative_to(root)), "group": str(directory.parent.relative_to(root)),
               "dataset": branch[1], "start_updates": start, "seed": seed, "arms": {}}
        for arm, label in ARMS.items():
            arm_dir = directory / arm
            arm_files = [p for p in all_files if p.is_relative_to(arm_dir)]
            points, checkpoints, issues = {}, [], []

            def add(step, reward, origin, eval_k=k):
                step = int(step)
                if step < start:
                    raise ValueError("checkpoint precedes starting policy")
                reward = number(reward)
                key = (step, eval_k)
                item = {"step": step, "updates": step-start, "reward": reward,
                        "reward_pct": reward*100, "eval_k": eval_k,
                        "evaluation_sha256": eval_identity, "sources": [source(origin)]}
                if key in points:
                    if abs(points[key]["reward"]-reward) > 1e-9:
                        raise ValueError(f"conflicting rewards for step {step}, k={eval_k}")
                    points[key]["sources"] = sorted(set(points[key]["sources"] + item["sources"]))
                else:
                    points[key] = item

            for path in arm_files:
                if path.name != "checkpoint_state.json":
                    continue
                try:
                    state = read(path)
                    step = int(state["completed_steps"])
                    if step_from(path.parent) not in (None, step):
                        raise ValueError("checkpoint folder/metadata step mismatch")
                    source(path)
                    checkpoints.append({"step": step, "updates": step-start,
                        "path": str(path.parent.relative_to(root)),
                        "adapter_present": (path.parent / "adapter_model.safetensors").is_file(),
                        "adapter_sha256_recorded": state.get("adapter_sha256")})
                except (ValueError, KeyError, OSError) as exc:
                    issues.append(f"{path.relative_to(root)}: {exc}")
            # Also inventory adapter-only archives, without loading model tensors.
            known = {c["path"] for c in checkpoints}
            for path in arm_files:
                if path.name != "adapter_model.safetensors" or str(path.parent.relative_to(root)) in known:
                    continue
                step = step_from(path.parent)
                if step is not None:
                    checkpoints.append({"step": step, "updates": step-start,
                        "path": str(path.parent.relative_to(root)), "adapter_present": True,
                        "adapter_sha256_recorded": None})
            try:
                if arm in endpoints:
                    row = endpoints[arm]
                    if int(row["seed"]) != seed or int(row["drift"]) != start:
                        raise ValueError("CSV/folder identity mismatch")
                    add(start, row["reward_before"], csv_path)
                    manifest_path = arm_dir / "policy/policy_train.json"
                    manifest = read(manifest_path) if manifest_path.is_file() else {}
                    end = manifest.get("completed_steps")
                    if end is None and horizon is not None:
                        end = start + int(horizon)
                    if end is not None:
                        add(end, row["reward_after"], csv_path)
                        if manifest:
                            source(manifest_path)
                    else:
                        issues.append("final reward exists but final step is unknown; not plotted")
                for path in arm_files:
                    if path.name == "curve.json":
                        curve = read(path)
                        for step, point in curve.get("points", {}).items():
                            if int(point["updates"]) != int(step)-start:
                                raise ValueError("curve absolute/relative step mismatch")
                            add(step, point["reward"], path, curve.get("k", k))
                    elif path.suffix == ".json" and (path.name.startswith("eval-") or
                            path.name in {"evaluation.json", "eval.json", "summary.json"}):
                        summary = read(path)
                        if not isinstance(summary, dict) or "mean_reward" not in summary:
                            continue
                        step = step_from(summary.get("adapter") or "")
                        if step is None:
                            step = step_from(path.parent)
                        if step is None:
                            issues.append(f"{path.relative_to(root)}: evaluation step unknown")
                            continue
                        if count and summary.get("prompts") != count:
                            raise ValueError("evaluation summary prompt count differs")
                        add(step, summary["mean_reward"], path, summary.get("k", k))
                eval_dirs = sorted({p.parent for p in arm_files if p.name == "shard-0.jsonl"
                                    and step_from(p.parent) is not None})
                for target in eval_dirs:
                    provenance = []
                    step = step_from(target)
                    try:
                        binding = read(target / "shard-0.contract.json")
                        if "step" in binding and int(binding["step"]) != step:
                            raise ValueError("evaluation step binding differs")
                        eval_k = binding.get("k", k)
                        reward = sharded_reward(target, count, eval_k, provenance)
                    except (ValueError, KeyError, OSError) as exc:
                        issues.append(f"{target.relative_to(root)}: {exc}")
                        continue
                    add(step, reward, target / "shard-0.done.json", eval_k)
                    for path in provenance:
                        points[(step, eval_k)]["sources"].append(source(path))
            except (ValueError, KeyError, OSError) as exc:
                # Conflicting evidence invalidates this arm rather than choosing a favorable source.
                points.clear()
                issues.append(str(exc))
            values = sorted(points.values(), key=lambda p: (p["step"], p["eval_k"] or 0))
            measured = {p["step"] for p in values}
            for c in checkpoints:
                c["evaluation_available"] = c["step"] in measured
            stop = start + int(horizon) if horizon is not None else max(measured, default=start)
            intermediate = sum(start < p["step"] < stop for p in values)
            exp["arms"][label] = {"selector": arm, "points": values,
                "checkpoints": sorted(checkpoints, key=lambda c: (c["step"], c["path"])),
                "intermediate_points": intermediate,
                "needs_evaluation": [c["path"] for c in checkpoints if not c["evaluation_available"]],
                "issues": issues}
            errors.extend(f"{exp['path']}/{arm}: {issue}" for issue in issues)
        experiments.append(exp)
    if not experiments:
        raise ValueError("no *-dN/sN experiment.json or downstream_results.csv found")
    return {"schema": "e5-checkpoint-curves/v1", "root": str(root),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "interpolated_points": 0, "training_rewards_used": False,
            "experiments": experiments, "issues": errors, "source_sha256": sources}


def write_outputs(report, out):
    out = out.resolve()
    if out.is_relative_to(Path(report["root"])):
        raise ValueError("output must be outside the source run tree")
    csv_path = out.with_suffix(".csv")
    if out == csv_path or out.exists() or csv_path.exists():
        raise ValueError("use a new .json output path; existing exports are never overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with csv_path.open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "dataset", "seed", "start_updates", "selector",
                         "step", "updates", "reward", "reward_pct", "eval_k", "sources"])
        for exp in report["experiments"]:
            for arm in exp["arms"].values():
                for p in arm["points"]:
                    writer.writerow([exp["path"], exp["dataset"], exp["seed"], exp["start_updates"],
                                     arm["selector"], p["step"], p["updates"], p["reward"],
                                     p["reward_pct"], p["eval_k"], "|".join(p["sources"])])
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    work = Path(os.environ.get("OM_WORK", "/group-volume/minsoo3.kim/offpolicy-misranking"))
    parser.add_argument("--root", type=Path, default=work / "runs/e5-reduced",
                        help="parent directory; auto-discovers math400-d0, math500-d400, etc.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parser.add_argument("--out", type=Path, default=work / "exports" / f"e5-checkpoints-{stamp}.json")
    args = parser.parse_args()
    try:
        report = export(args.root)
        csv_path = write_outputs(report, args.out)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Export failed: {exc}\n")
    for exp in report["experiments"]:
        for name, arm in exp["arms"].items():
            print(f"{exp['path']} {name}: {len(arm['points'])} evaluated points, "
                  f"{arm['intermediate_points']} intermediate, "
                  f"{len(arm['needs_evaluation'])} checkpoints need evaluation")
    print(f"JSON: {args.out}\nCSV: {csv_path}\nIssues: {len(report['issues'])}")
    for issue in report["issues"][:20]:
        print(f"WARNING: {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
