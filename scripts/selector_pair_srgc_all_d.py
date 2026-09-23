"""Measure or export every 25-step D on saved t25 On-policy trajectories.

This is a diagnostic series, not an adaptive rollout or an executed switch.
It reuses completed SR-GC projections and never trains a policy.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
from pathlib import Path

import selector_pair_srgc as srgc
import selector_pair_srgc_repeat as repeat
from selector_pair_srgc_uncertainty import estimate
from paper_result_text import write_export
from selector_pair_results import srgc_results

SCHEMA = "offpolicy-selector-pair/sr-gc-all-d-v1"


def add_uncertainty(point, directory, sets, check_index):
    try:
        value = estimate(directory, sets, check_index)
        if not math.isclose(value["d"], point["d"], rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("uncertainty contrast differs from validated D")
        point["uncertainty"] = value
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        point["uncertainty"] = {"status": "unavailable", "error": str(exc)}


def scan_state(root, initial, interval=25, *, protocol=None, devices=None, cap=14400.):
    seed, start = initial["seed"], initial["step"]
    choice = srgc.choose(initial["d_a"], initial["d_b"])
    if choice["selector"] != initial["selector"] or not math.isclose(
            choice["d"], initial["d"], abs_tol=1e-9):
        raise ValueError("invalid frozen initial SR-GC contrast")
    checkpoints = repeat.inventory(root, seed, start)
    last = max(checkpoints, default=start)
    first = {"step": start, "status": "measured", "d_a": initial["d_a"],
             "d_b": initial["d_b"], "d": initial["d"], "source": "initial_parent"}
    add_uncertainty(first, root / f"sr-gc/s{seed}-t{start}", initial["sets"], 1)
    result = {"state": f"s{seed}-t{start}", "seed": seed,
              "start_step": start, "last_saved_checkpoint": last,
              "points": [first], "pending": [], "errors": []}
    for step in range(start + interval, last + 1, interval):
        checkpoint = checkpoints.get(step)
        if checkpoint is None:
            result["pending"].append({"step": step, "reason": "checkpoint_not_saved"})
            if devices is not None:
                print(f"[SR-GC all-D] seed={seed} start_step={start} "
                      f"check_step={step} STOP checkpoint_not_saved", flush=True)
                break
            continue
        directory = repeat.output_dir(root, seed, start, interval) / f"step-{step}"
        try:
            if devices is not None:
                print(f"[SR-GC all-D] seed={seed} start_step={start} "
                      f"check_step={step} begin", flush=True)
            expected, contract = repeat.checkpoint_reference(
                root, seed, start, step, checkpoint, initial, interval)
            if devices is not None:
                from selector_pair_parallel import checked
                checked(directory)
                with srgc.worker.pair_lease(directory / ".measurement.lock"):
                    reference = (repeat.saved_reference(directory, root, expected)
                                 if (directory / "reference.json").exists() or
                                 (directory / "reference.json").is_symlink() else expected)
                    repeat.measure_point(directory, reference, contract, protocol, devices, cap)
            value = repeat.check_projections(directory, root, expected)
            point = {"step": step, "status": "measured",
                     "d_a": value["d_a"], "d_b": value["d_b"],
                     "d": value["d"], "source": "saved_on_checkpoint",
                     "measurement_gpu_seconds": value["measurement_gpu_seconds"],
                     "measurement_cost_complete": value["measurement_cost_complete"]}
            add_uncertainty(point, directory, initial["sets"], 1 + (step - start) // interval)
            result["points"].append(point)
            if devices is not None:
                print(f"[SR-GC all-D] seed={seed} start_step={start} "
                      f"check_step={step} D={value['d']:.6g} complete", flush=True)
        except srgc.worker.PairLockBusy:
            result["pending"].append({"step": step, "reason": "measurement_owned_by_peer"})
            if devices is not None:
                print(f"[SR-GC all-D] seed={seed} start_step={start} "
                      f"check_step={step} STOP measurement_owned_by_peer", flush=True)
                break
        except FileNotFoundError as exc:
            result["pending"].append({"step": step, "reason": "projection_missing",
                                      "path": str(exc.filename)})
            if devices is not None:
                print(f"[SR-GC all-D] seed={seed} start_step={start} "
                      f"check_step={step} STOP projection_missing: {exc.filename}", flush=True)
                break
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            result["errors"].append({"step": step, "error": str(exc)})
            if devices is not None:
                print(f"[SR-GC all-D] seed={seed} start_step={start} "
                      f"check_step={step} STOP {type(exc).__name__}: {exc}", flush=True)
                break
    result["scheduled_steps"] = list(range(start, last + 1, interval))
    return result


def collect(root, initial, interval=25, *, start_step=25, protocol=None, devices=None,
            cap=14400., seed=None):
    repeat.positive_int(interval)
    report = {"schema": SCHEMA, "interval": interval, "scope":
              "D at every scheduled checkpoint on each saved t25 fixed-On trajectory; "
              "negative D does not stop this diagnostic or imply an executed switch.",
              "trajectories": [], "errors": []}
    if initial.get("status") not in {"validated", "partial"}:
        report["status"] = "initial_decisions_unavailable"
        return report
    selected = [item for item in initial["decisions"]
                if item["step"] == start_step and (seed is None or item["seed"] == seed)]
    if not selected:
        report["status"] = "initial_decisions_unavailable"
        return report
    for item in sorted(selected, key=lambda value: value["seed"]):
        try:
            report["trajectories"].append(scan_state(
                root, item, interval, protocol=protocol, devices=devices, cap=cap))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            report["errors"].append({"state": f"s{item['seed']}-t{start_step}", "error": str(exc)})
    report["scheduled_points"] = sum(len(row["scheduled_steps"]) for row in report["trajectories"])
    report["measured_points"] = sum(len(row["points"]) for row in report["trajectories"])
    report["status"] = ("invalid" if report["errors"] or any(row["errors"] for row in report["trajectories"])
                        else "complete" if report["trajectories"] and
                        report["scheduled_points"] == report["measured_points"] else "partial")
    return report


def table(report):
    output = io.StringIO()
    output.write("SR-GC D DIAGNOSTICS: saved t25 On trajectories, every 25 steps\n")
    writer = csv.writer(output)
    writer.writerow(("state", "step", "d_a", "d_b", "d", "upper", "confirmed_sr", "status"))
    for row in report["trajectories"]:
        for point in row["points"]:
            uncertainty = point["uncertainty"]
            writer.writerow((row["state"], point["step"], point["d_a"],
                             point["d_b"], point["d"], uncertainty.get("upper", ""),
                             uncertainty.get("confirmed_sr", ""), point["status"]))
        for item in row["pending"]:
            writer.writerow((row["state"], item["step"], "", "", "", "", "", item["reason"]))
        for item in row["errors"]:
            writer.writerow((row["state"], item["step"], "", "", "", "", "", "invalid"))
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("results", "measure"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--interval", type=int, default=25)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-gpu-seconds-per-checkpoint", type=float, default=14400.)
    args = parser.parse_args()
    if not math.isfinite(args.max_gpu_seconds_per_checkpoint) or args.max_gpu_seconds_per_checkpoint <= 0:
        parser.error("measurement cap must be finite and positive")
    root = args.root.resolve()
    devices = None
    protocol = None
    if args.command == "measure":
        from light_selection_gate_gpu import install_signal_handlers
        install_signal_handlers()
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(devices) != 4 or len(set(devices)) != 4 or not all(devices):
            parser.error("measurement requires four allocated GPUs")
        protocol = repeat.read(root / "pair.json", root)
    report = collect(root, srgc_results(root), args.interval, protocol=protocol,
                     devices=devices, cap=args.max_gpu_seconds_per_checkpoint,
                     seed=args.seed)
    write_export("selector-pair-srgc-all-d", report, table(report), args.out)
    if report["status"] in {"invalid", "initial_decisions_unavailable"} or (
            args.command == "measure" and report["status"] != "complete"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
