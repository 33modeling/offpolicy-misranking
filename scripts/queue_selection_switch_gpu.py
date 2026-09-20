#!/usr/bin/env python3
"""Run one original switch worker pass, returning peer waits to the node queue.

Idle peer waits yield to the node controller. MBPP convergence fits wait for
their required curves before validation; exhausted saved policies get separate
posthoc evaluations. Training, leases, checkpoints, scoring and frozen scientific
code stay in the original driver. Active owned work is never interrupted.
"""

import os
import sys
from contextlib import ExitStack
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


def finish_exhausted(p, directory, original, *, deferred=None):
    # Called by the original queue only while it owns the branch task lease.
    if original(p, directory):
        return True
    if not recovery.required(p, directory):
        return False
    if deferred is not None:
        deferred[directory] = p
        print(f"[budget-stop] {directory}: no training retry; yielding to other runnable branches before recovery", flush=True)
        return True
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


def only_review_dependencies(root, reviewed, only=None):
    """No runnable work remains when an unavailable dev result blocks the gate."""
    if not reviewed:
        return False
    try:
        p = worker.manifest(root)
        if p.get("dataset") != "mbpp" or (root / "model.json").exists():
            return False
        pending, development_review = [], False
        for seed in (*worker.rule.DEV_SEEDS, *worker.rule.TEST_SEEDS):
            if only and seed not in only["seeds"]:
                continue
            for step in worker.rule.STEPS:
                if not (worker.prefix_dir(root, seed) / f"prefix-{step}.json").is_file():
                    return False
                child = worker.child_root(root, seed, step)
                protocol = worker.protocol(child)
                out = next(worker.base.entries(child))
                for arm in protocol["arms"]:
                    if only and arm not in only["arms"]:
                        continue
                    directory = out / arm
                    if worker.branch_finished(p, directory):
                        continue
                    if directory in reviewed:
                        development_review |= seed in worker.rule.DEV_SEEDS and arm in worker.rule.DEV_ARMS
                    elif arm != "gated":
                        return False
                    pending.append(directory)
        if not development_review:
            return False
        # A peer may have claimed a reviewed branch after this worker released it.
        # Never turn that live evaluation/recovery into a terminal queue decision.
        with ExitStack() as locks:
            for directory in pending:
                locks.enter_context(worker.base.lease(directory / ".task.lock"))
            return not (root / "model.json").exists()
    except (OSError, ValueError, KeyError, StopIteration):
        return False


def run():
    original_wait, original_fit = worker.wait_for_peers, worker.fit_once
    original_blocked = worker.mbpp_resume_blocked
    original_work = worker.work
    reviewed = set()
    deferred = {}

    def blocked(p, directory):
        result = finish_exhausted(p, directory, original_blocked, deferred=deferred)
        if result:
            reviewed.add(directory)
        return result

    def work(root, *, idle_timeout=600., only=None):
        reviewed.clear()
        deferred.clear()
        result = original_work(root, idle_timeout=idle_timeout, only=only)
        # The original pass has tried every independent task. Only now may a
        # stopped branch use this node for reporting-only checkpoint recovery.
        for directory, p in deferred.items():
            try:
                with worker.base.lease(directory / ".task.lock"):
                    if worker.branch_finished(p, directory):
                        continue
                    if not finish_exhausted(p, directory, original_blocked):
                        reviewed.discard(directory)
                        result = 1
            except BlockingIOError:
                reviewed.discard(directory)
                result = 1
                print(f"[queue-yield] recovery owned by a peer: {directory}", flush=True)
            except Exception as exc:
                reviewed.discard(directory)
                result = 1
                worker.record_failure(directory / "budget-recovery", exc)
        if result == 1 and only_review_dependencies(root, reviewed, only):
            print("[WAIT] only MBPP review branches and dependent gate tasks remain; "
                  "releasing this worker without another GPU admission; NOT complete", flush=True)
            return 80
        return result

    worker.wait_for_peers = yield_to_node_queue
    worker.fit_once = lambda root: fit_when_ready(root, original_fit)
    worker.mbpp_resume_blocked = blocked
    worker.work = work
    try:
        return worker.main()
    finally:
        worker.wait_for_peers, worker.fit_once = original_wait, original_fit
        worker.mbpp_resume_blocked = original_blocked
        worker.work = original_work


if __name__ == "__main__":
    if sys.argv[1:2] != ["run"]:
        raise SystemExit("queue worker accepts only run; use the original driver for other commands")
    sys.exit(run())
