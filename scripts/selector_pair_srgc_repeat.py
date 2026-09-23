"""Measure SR-GC every 25 steps on the single t25 On trajectory.

Results export only aggregates saved projections. The explicit measure command
can generate missing current-policy projections, never train a policy. Each
Pair starting state remains a separate trajectory.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import selection_gate as core
import selection_gate_gpu as base
import selector_pair_srgc as srgc
import selector_pair_srgc_score as score

SCHEMA = "offpolicy-selector-pair/sr-gc-repeat-v2"
LEGACY_SCHEMA = "offpolicy-selector-pair/sr-gc-repeat-v1"
DEFAULT_INTERVAL = 25
DEFAULT_START_STEP = 25
SCOPE = (
    "Repeated SR-GC on the saved t25 fixed-On trajectory for each seed: measure all "
    "available checks regardless of D sign. No switch decision, reward inputs, "
    "regression, interpolated D, joining different Pair starting states, or "
    "executed switched-policy reward curves. Missing projections remain unmeasured."
)


def positive_int(value):
    if type(value) is not int or value <= 0:
        raise ValueError("SR-GC interval must be a positive integer")
    return value


def read(path, root):
    from selector_pair_results import read_source
    return read_source(path, root)[0]


def trajectory(root, seed, start):
    return root / f"branches/on_policy/states/s{seed}-t{start}/points/view-{start}"


def output_dir(root, seed, start, interval):
    return root / f"sr-gc-repeat/every-{interval}/s{seed}-t{start}"


def inventory(root, seed, start):
    """Only this On branch's published checkpoints, never another t or arm."""
    policy = trajectory(root, seed, start) / "selection_full/policy"
    found = {}
    for folder, pattern, prefix in ((policy / "curve-checkpoints", "step-*", "step-"),
                                    (policy, "checkpoint-*", "checkpoint-")):
        for path in sorted(folder.glob(pattern)):
            suffix = path.name.removeprefix(prefix)
            if not suffix.isdigit() or int(suffix) <= start:
                continue
            step = int(suffix)
            if not path.resolve().is_relative_to(root):
                raise ValueError("checkpoint escaped its Pair root")
            # Publication is atomic; incomplete temporary directories are excluded.
            if (path / "checkpoint_state.json").is_file() and (path / "adapter_model.safetensors").is_file():
                found.setdefault(step, path)
    return found


def checkpoint_reference(root, seed, start, step, checkpoint, initial, interval):
    origin = root / f"sr-gc/s{seed}-t{start}/reference.json"
    source = read(origin, root)
    if base.digest(origin) != initial["reference_sha256"]:
        raise ValueError("initial SR-GC reference changed")
    out = trajectory(root, seed, start)
    contract = read(out / "contract.json", root)
    if (base.digest(out / "contract.json") != source["contract_sha256"]
            or source["sets"] != initial["sets"] or source["state_id"] != initial["state_id"]):
        raise ValueError("repeated SR-GC belongs to a different starting state")
    state = read(checkpoint / "checkpoint_state.json", root)
    subset = out / "subsets/subset-selection_full.json"
    expected = {"schema": "offpolicy-grpo-checkpoint/v2", "seed": seed,
                "start_step": start, "completed_steps": step, "training_objective": "grpo",
                "world_size": 4, "prompts_sha256": base.digest(subset)}
    if (any(state.get(key) != value for key, value in expected.items())
            or Path(state["resume_adapter"]).resolve() != Path(source["parent"]).resolve()
            or state["adapter_sha256"] != base.digest(checkpoint / "adapter_model.safetensors")):
        raise ValueError("checkpoint identity, parent, subset, or adapter binding differs")
    selected = read(subset, root)
    prompts = core.read(Path(source["prompts"]))
    if selected["train"] != [prompts["train"][i] for i in initial["sets"]["on_policy"]]:
        raise ValueError("saved On trajectory trained on a different subset")
    if base.digest(Path(source["prompts"])) != source["prompts_sha256"]:
        raise ValueError("SR-GC prompt pool changed")
    cache = Path(source["parent"]).parent / "rollouts_behavior_train.jsonl"
    if base.digest(cache) != initial["cache_sha256"]:
        raise ValueError("initial SR cache changed")
    reference = {**source, "config": {**source["config"], "drift": step},
                 "parent": str(checkpoint), "adapter_sha256": state["adapter_sha256"],
                 "sampling_seed": 701000003 + seed * 1000003 + step * 7919,
                 "repeat": {"schema": SCHEMA, "seed": seed, "start_step": start,
                            "step": step, "interval": interval,
                            "initial_reference_sha256": initial["reference_sha256"],
                            "checkpoint_state_sha256": base.digest(checkpoint / "checkpoint_state.json"),
                            "subset_policy": "retain the fixed On subset until SR; initial SR cache stays fixed"}}
    return reference, contract


def saved_reference(directory, root, expected):
    actual = read(directory / "reference.json", root)
    legacy = {**expected, "repeat": {**expected["repeat"], "schema": LEGACY_SCHEMA}}
    if actual != expected and actual != legacy:
        raise ValueError("repeated SR-GC reference differs from this checkpoint")
    return actual


def check_projections(directory, root, expected):
    saved_reference(directory, root, expected)
    # Bounded regular-file reads precede the existing projection validator.
    for stage in score.STAGES:
        for shard in range(4):
            for suffix in (".json", ".done.json"):
                read(directory / f"{stage}-{shard}{suffix}", root)
    value = srgc.reference_contrast(directory, expected["sets"])
    costs = base.cost(directory) if (directory / "cost.jsonl").exists() else None
    cost_complete = bool(costs and costs["complete"])
    return {**value, "reference_sha256": base.digest(directory / "reference.json"),
            "adapter_sha256": expected["adapter_sha256"],
            "measurement_gpu_seconds": sum(v["gpu_seconds"] for v in costs["ledgers"].values()) if cost_complete else None,
            "measurement_cost_complete": cost_complete,
            "reference_shards": {f"{stage}-{i}.json": base.digest(directory / f"{stage}-{i}.json")
                                 for stage in score.STAGES for i in range(4)}}


def measure_point(directory, reference, contract, protocol, devices, cap):
    """Same A/B gradient measurement as Pair; no full-pool reranking or training."""
    if len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError("missing projections require four distinct allocated GPUs")
    base.bind(directory / "reference.json", reference)
    for stage in score.STAGES:
        commands = [([sys.executable, str(Path(score.__file__).resolve()), "--root", str(directory),
                      "--stage", stage, "--shard", str(i)], devices[i])
                    for i in range(4) if not (directory / f"{stage}-{i}.done.json").exists()]
        if commands:
            remaining = cap - base.spent(directory)
            if remaining <= 0:
                raise ValueError("repeated SR-GC measurement cap exhausted")
            base.meter(directory, "sr-gc-repeat-" + stage, protocol["gpu_type"],
                       commands=commands, env=srgc.worker.environment(contract),
                       timeout=remaining / 4, ledger="research")


def scan_state(root, initial, interval, *, protocol=None, devices=None, cap=None, through_step=None):
    seed, start = initial["seed"], initial["step"]
    state_name = f"s{seed}-t{start}"
    initial_choice = srgc.choose(initial["d_a"], initial["d_b"])
    if initial_choice["selector"] != initial["selector"] or not math.isclose(
            initial_choice["d"], initial["d"], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("invalid initial SR-GC decision")
    first = {**initial_choice, "step": start, "updates": 0,
             "reference_sha256": initial["reference_sha256"], "source": "initial_parent"}
    result = {"state": state_name, "state_id": initial["state_id"], "seed": seed,
              "start_step": start, "interval": interval, "decisions": [first],
              "first_negative_step": start if first["d"] < 0 else None,
              "status": "diagnostic_through_checked_step",
              "on_through_step": start, "next_check_step": start + interval,
              "pending": [], "errors": [], "executed_switch": False,
              "switched_policy_rewards": None}
    checkpoints = inventory(root, seed, start)
    if not checkpoints:
        result.update(status="awaiting_checkpoint")
        return result
    last_step = min(max(checkpoints), through_step) if through_step is not None else max(checkpoints)
    for step in range(start + interval, last_step + 1, interval):
        result["next_check_step"] = step
        if step not in checkpoints:
            result["pending"].append({"step": step, "reason": "checkpoint_not_saved"})
            result["status"] = "awaiting_checkpoint"
            break
        directory = output_dir(root, seed, start, interval) / f"step-{step}"
        try:
            expected, contract = checkpoint_reference(root, seed, start, step, checkpoints[step], initial, interval)
            if devices is not None:
                from selector_pair_parallel import checked
                checked(directory)
                with srgc.worker.pair_lease(directory / ".measurement.lock"):
                    reference = (saved_reference(directory, root, expected)
                                 if (directory / "reference.json").exists() or
                                 (directory / "reference.json").is_symlink() else expected)
                    measure_point(directory, reference, contract, protocol, devices, cap)
            value = check_projections(directory, root, expected)
            item = {**value, "step": step, "updates": step - start, "source": "saved_on_checkpoint"}
            result["decisions"].append(item)
            if devices is not None:
                print(f"[SR-GC repeat] {state_name} step={step} D={value['d']:.6g} "
                      "diagnostic only", flush=True)
            if value["d"] < 0 and result["first_negative_step"] is None:
                result["first_negative_step"] = step
            result.update(status="diagnostic_through_checked_step",
                          on_through_step=step, next_check_step=step + interval)
        except srgc.worker.PairLockBusy:
            result["pending"].append({"step": step, "reason": "measurement_owned_by_peer"})
            result["status"] = "awaiting_projections"
            break
        except FileNotFoundError as exc:
            result["pending"].append({"step": step, "reason": "current_gradient_inputs_missing",
                                      "path": str(exc.filename)})
            result["status"] = "awaiting_projections"
            break
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            result["errors"].append({"step": step, "error": str(exc)})
            result["status"] = "invalid"
            break
    result["available_checkpoint_steps"] = sorted(checkpoints)
    return result


def collect(root, initial, interval=DEFAULT_INTERVAL, *, start_step=DEFAULT_START_STEP,
            protocol=None, devices=None, cap=None, seed=None, through_step=None):
    positive_int(interval)
    if through_step is not None and (type(through_step) is not int or
                                     through_step < start_step or
                                     (through_step - start_step) % interval):
        raise ValueError("through-step must be a scheduled check at or after the start step")
    report = {"schema": SCHEMA, "interval": interval, "scope": SCOPE,
              "start_step": start_step,
              "seed_filter": seed, "through_step": through_step,
              "threshold": 0., "diagnostic_only": True, "trajectories": [], "errors": []}
    if initial.get("status") not in {"validated", "partial"}:
        report["status"] = "initial_decisions_unavailable"
        return report
    selected = [value for value in initial["decisions"]
                if value["step"] == start_step and (seed is None or value["seed"] == seed)]
    if not selected:
        report["status"] = "initial_decisions_unavailable"
        return report
    for value in sorted(selected, key=lambda v: v["seed"]):
        try:
            report["trajectories"].append(scan_state(root, value, interval,
                protocol=protocol, devices=devices, cap=cap, through_step=through_step))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            report["errors"].append({"state": f"s{value['seed']}-t{value['step']}", "error": str(exc)})
    report["status"] = "invalid" if report["errors"] or any(
        row["errors"] for row in report["trajectories"]) else "recorded"
    report["checked_points"] = sum(len(row["decisions"]) for row in report["trajectories"])
    report["negative_checks"] = sum(
        point["d"] < 0 for row in report["trajectories"] for point in row["decisions"])
    return report


def table(report):
    output = io.StringIO()
    output.write(f"\nSR-GC D DIAGNOSTICS: t={report['start_step']}, every {report['interval']} updates; no switch executed\n")
    writer = csv.writer(output)
    writer.writerow(("state", "step", "updates", "d_a", "d_b", "d", "sign", "first_negative_step", "status"))
    for row in report["trajectories"]:
        for point in row["decisions"]:
            writer.writerow((row["state"], *(point[key] for key in ("step", "updates", "d_a", "d_b", "d")),
                             "negative" if point["d"] < 0 else "nonnegative",
                             row["first_negative_step"], row["status"]))
        if row["pending"]:
            output.write(f"{row['state']}: {row['pending'][0]['reason']} at step {row['pending'][0]['step']}\n")
    return output.getvalue()


def main():
    from selector_pair_results import srgc_results
    from paper_result_text import write_export
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("results", "measure"), nargs="?", default="results")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--seed", type=int, help="measure only one seed's t25 path")
    parser.add_argument("--through-step", type=int, help="stop this measurement pass after a scheduled check")
    parser.add_argument("--max-gpu-seconds-per-checkpoint", type=float, default=14400.)
    args = parser.parse_args()
    root = args.root.resolve()
    positive_int(args.interval)
    cap = args.max_gpu_seconds_per_checkpoint
    if not math.isfinite(cap) or cap <= 0:
        parser.error("measurement cap must be finite and positive")
    initial = srgc_results(root)
    devices = None
    protocol = None
    if args.command == "measure":
        from light_selection_gate_gpu import install_signal_handlers
        install_signal_handlers()
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(devices) != 4 or len(set(devices)) != 4 or not all(devices):
            parser.error("measure requires four allocated CUDA_VISIBLE_DEVICES; results never launches GPU work")
        protocol = read(root / "pair.json", root)
    report = collect(root, initial, args.interval, protocol=protocol, devices=devices,
                     cap=cap, seed=args.seed, through_step=args.through_step)
    write_export("selector-pair-srgc-repeat", report, table(report), args.out)
    if report["status"] in {"invalid", "initial_decisions_unavailable"} or (
            args.command == "measure" and any(row["pending"] for row in report["trajectories"])):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
