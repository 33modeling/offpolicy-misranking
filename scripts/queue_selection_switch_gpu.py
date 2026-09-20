#!/usr/bin/env python3
"""Run one original switch worker pass, returning peer waits to the node queue.

Idle peer waits yield to the node controller. MBPP convergence fits wait for
their required curves before validation; exhausted saved policies get separate
posthoc evaluations. Training, leases, checkpoints, scoring and frozen scientific
code stay in the original driver. Active owned work is never interrupted.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import selection_switch_gpu as worker
import mbpp_budget_recovery as recovery


def yield_to_node_queue(busy, *, last_progress, idle_timeout):
    if busy:
        print(f"[queue-yield] {len(busy)} peer-owned tasks; returning to the shared node queue", flush=True)
    return False


def fit_when_ready(root, original):
    manifest = root / "switch.json"
    if not manifest.is_file() or (root / "model.json").exists():
        return original(root)
    protocol = worker.core.read(manifest)
    if protocol.get("dataset") != "mbpp" or protocol.get("gate") != "convergence":
        return original(root)
    # This is only a readiness filter; the original fit still validates all
    # policies, results, curves and lineage before publishing a model.
    for seed in worker.rule.DEV_SEEDS:
        for step in worker.rule.STEPS:
            child = worker.child_root(root, seed, step)
            if not (child / "suite.json").is_file():
                return False
            out = next(worker.base.entries(child), None)
            if out is None:
                return False
            for arm in worker.rule.DEV_ARMS:
                if any(not (out / arm / name).is_file() for name in ("result.json", "curve.json")):
                    return False
    return original(root)


def finish_exhausted(p, directory, original):
    # Called by the original queue only while it owns the branch task lease.
    if original(p, directory):
        return True
    if not recovery.required(p, directory):
        return False
    import additive_experiment as ae
    try:
        prepared = recovery.prepare(p, directory)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        worker.core.atomic_json(directory / "budget-recovery/review.json", {
            "canonical_complete": False, "error": str(exc), "training_restarted": False})
        print(f"[WAIT] {directory}: budget exhausted; saved-policy review: {exc}", flush=True)
        return True
    cfg = worker.core.read(directory.parent / "contract.json")["config"]
    recovery.recover(p, directory, os.environ["CUDA_VISIBLE_DEVICES"].split(","), ae.model_environment(cfg),
                     prepared=prepared)
    # Recovered measurements never satisfy canonical result/fit requirements.
    return True


def run():
    original_wait, original_fit = worker.wait_for_peers, worker.fit_once
    original_blocked = worker.mbpp_resume_blocked
    worker.wait_for_peers = yield_to_node_queue
    worker.fit_once = lambda root: fit_when_ready(root, original_fit)
    worker.mbpp_resume_blocked = lambda p, directory: finish_exhausted(p, directory, original_blocked)
    try:
        return worker.main()
    finally:
        worker.wait_for_peers, worker.fit_once = original_wait, original_fit
        worker.mbpp_resume_blocked = original_blocked


if __name__ == "__main__":
    if sys.argv[1:2] != ["run"]:
        raise SystemExit("queue worker accepts only run; use the original driver for other commands")
    sys.exit(run())
