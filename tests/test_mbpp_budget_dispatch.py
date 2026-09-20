"""Exhausted branches must yield allocation to independent runnable work."""
import pytest

from test_mbpp_node_queue import queue_worker as queue
from test_selection_switch_gpu import simulated_queue


def setup(root, monkeypatch):
    worker = queue.worker
    calls, _ = simulated_queue(root, monkeypatch)
    p = worker.manifest(root)
    p.update(dataset="mbpp", gate="final")
    for seed in (*worker.rule.DEV_SEEDS, *worker.rule.TEST_SEEDS):
        for step in worker.rule.STEPS:
            worker.core.atomic_json(worker.prefix_dir(root, seed) / f"prefix-{step}.json", {})
            worker.publish_state(root, seed, step)
    blocked = next(worker.base.entries(worker.child_root(root, 0, 25))) / "selection_reduced"
    monkeypatch.setattr(queue.recovery, "required", lambda p, directory: directory == blocked)
    monkeypatch.setattr(worker, "mbpp_resume_blocked", lambda *a: False)
    monkeypatch.setattr(worker, "main", lambda: worker.work(root, idle_timeout=0))
    return worker, calls, blocked


def test_exhausted_branch_recovery_cannot_delay_other_assignments(tmp_path, monkeypatch):
    worker, calls, blocked = setup(tmp_path, monkeypatch)
    observed = []

    def prepare(p, directory):
        observed.append(list(calls))
        raise ValueError("no valid saved continuation checkpoint")

    monkeypatch.setattr(queue.recovery, "prepare", prepare)
    assert queue.run() == 80
    assert observed and len(observed[0]) == 41
    assert len(calls) == 41
    assert (0, 25, "selection_reduced") not in calls
    assert not (blocked / "result.json").exists()
    assert not (tmp_path / "model.json").exists()


def test_recovery_error_does_not_block_independent_training(tmp_path, monkeypatch):
    worker, calls, blocked = setup(tmp_path, monkeypatch)
    worker.core.atomic_json(blocked.parent / "contract.json", {"seed": 0, "step": 25, "config": {}})
    monkeypatch.setattr(queue.recovery, "prepare", lambda *a: ({}, {}))
    observed = []

    def recover(*args, **kwargs):
        observed.append(len(calls))
        raise RuntimeError("evaluation unavailable")

    monkeypatch.setattr(queue.recovery, "recover", recover)
    assert queue.run() == 1
    assert observed == [41]
    assert len(calls) == 41
    assert not (blocked / "result.json").exists()


def test_peer_owns_deferred_recovery_so_node_does_not_wait(tmp_path, monkeypatch):
    worker, calls, blocked = setup(tmp_path, monkeypatch)
    lease = worker.base.lease
    owned = []

    def prepare(*a):
        pytest.fail("peer-owned recovery must not run")

    def claim(path, **kwargs):
        if path == blocked / ".task.lock":
            owned.append(path)
            if len(owned) > 1:
                raise BlockingIOError("peer claimed recovery")
        return lease(path, **kwargs)

    monkeypatch.setattr(worker.base, "lease", claim)
    monkeypatch.setattr(queue.recovery, "prepare", prepare)
    assert queue.run() == 1
    assert len(calls) == 41
