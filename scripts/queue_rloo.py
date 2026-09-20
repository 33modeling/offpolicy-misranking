"""Isolate branch failures without changing frozen RLOO training sources."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rloo_experiment as experiment


def admit_after_failure(root):
    import selection_nccl_preflight
    os.environ.update(selection_nccl_preflight.preflight(root))


def record(directory, state, error=""):
    directory.mkdir(parents=True, exist_ok=True)
    experiment.ed.atomic_json(directory / "queue-attempt.json", {
        "state": state, "error": error, "updated": time.time(), "pid": os.getpid()})


def evaluation_busy(directory):
    """An evaluation child may survive its controller; do not launch it twice."""
    for shard in range(4):
        try:
            handle = (directory / 'evaluation' / f'shard-{shard}.lock').open('rb')
        except FileNotFoundError:
            continue
        with handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
    return False


def run(root, seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("phase timeout must be positive and finite")
    outs = [root / f"math500-d{drift}" / f"s{seed}" for drift, seed in experiment.POINTS]
    for out in outs:
        experiment.validate(out)
    failures, busy = 0, 0
    for out in outs:
        for arm in ("before", *experiment.ARMS):
            directory = out / arm
            with ExitStack() as stack:
                try:
                    stack.enter_context(experiment.lock(directory / ".worker.lock"))
                except BlockingIOError:
                    busy += 1
                    continue
                try:
                    if evaluation_busy(directory):
                        busy += 1
                        print(f"[rloo-peer] {directory}: evaluation shard lease held; existing work preserved", flush=True)
                        continue
                    if not experiment.complete(out, arm):
                        experiment.run_arm(out, arm, seconds)
                    if not experiment.complete(out, arm):
                        raise ValueError("evaluation incomplete; not DONE")
                    record(directory, "DONE")
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                    failures += 1
                    error = f"{type(exc).__name__}: {exc}"
                    record(directory, "FAILED", error)
                    print(f"[rloo-failed] {directory}: {error}; saved work preserved", flush=True)
                    if isinstance(exc, (OSError, RuntimeError)):
                        # A dead CUDA/NCCL worker is not authority to dispatch
                        # another GPU task until this allocation passes a probe.
                        try:
                            admit_after_failure(root)
                        except (OSError, ValueError, RuntimeError) as admission:
                            print(f"[blocked] RLOO node admission failed: {admission}", flush=True)
                            return 78
                    continue
        with ExitStack() as stack:
            try:
                stack.enter_context(experiment.lock(out / ".report.lock"))
            except BlockingIOError:
                busy += 1
                continue
            try:
                if all(experiment.complete(out, arm) for arm in ("before", *experiment.ARMS)):
                    experiment.report(out)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                failures += 1
                record(out, "FAILED", f"report: {type(exc).__name__}: {exc}")
                print(f"[rloo-report-failed] {out}: {exc}; continuing other points", flush=True)
    print(f"[rloo-queue] failed={failures} peer-owned={busy}; completion is determined by saved evaluations", flush=True)
    return 1 if failures else 75 if busy else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-phase-seconds", type=float, required=True)
    args = parser.parse_args()
    from light_selection_gate_gpu import install_signal_handlers
    install_signal_handlers()
    try:
        return run(args.root.resolve(), args.max_phase_seconds)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"[rloo-error] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
