"""A gate blocked by reviewed development work must not retain idle GPU nodes."""

import pytest

from test_mbpp_node_queue import queue_worker
from test_selection_switch_gpu import simulated_queue

worker = queue_worker.worker


def review_queue(root, monkeypatch):
    calls, _ = simulated_queue(root, monkeypatch)
    p = worker.manifest(root)
    p.update(dataset="mbpp", gate="final")
    for seed in (*worker.rule.DEV_SEEDS, *worker.rule.TEST_SEEDS):
        for step in worker.rule.STEPS:
            worker.core.atomic_json(worker.prefix_dir(root, seed) / f"prefix-{step}.json", {})
            worker.publish_state(root, seed, step)
    blocked = next(worker.base.entries(worker.child_root(root, 0, 25))) / "selection_reduced"
    monkeypatch.setattr(queue_worker, "finish_exhausted", lambda p, path, original, **kwargs: path == blocked)
    monkeypatch.setattr(worker, "main", lambda: worker.work(root, idle_timeout=0))
    return blocked, calls


def test_reviewed_development_and_its_gate_dependents_exit_without_done(tmp_path, monkeypatch, capsys):
    blocked, calls = review_queue(tmp_path, monkeypatch)
    assert queue_worker.run() == 80
    assert len(calls) == 41
    assert not (blocked / "result.json").exists()
    assert not (tmp_path / "model.json").exists()
    assert not any(arm == "gated" for _, _, arm in calls)
    assert "dependent gate" in capsys.readouterr().out
    assert queue_worker.run() == 80
    assert len(calls) == 41


@pytest.mark.parametrize("other", ["failure", "peer", "prefix", "unpublished", "math"])
def test_review_does_not_hide_independent_unfinished_work(tmp_path, monkeypatch, other):
    blocked, calls = review_queue(tmp_path, monkeypatch)
    if other == "math":
        worker.manifest(tmp_path)["dataset"] = "math500"
    elif other == "prefix":
        (worker.prefix_dir(tmp_path, 4) / "prefix-100.json").unlink()
        monkeypatch.setattr(worker, "build_prefix", lambda *args: None)
    elif other == "unpublished":
        (worker.child_root(tmp_path, 4, 100) / "net_protocol.json").unlink()
        monkeypatch.setattr(worker, "publish_state", lambda *args: (_ for _ in ()).throw(ValueError("input missing")))
    else:
        original = worker.runtime.run_arm

        def run(out, suite, p, arm, devices, env):
            c = worker.core.read(out / "contract.json")
            if (c["seed"], c["step"], arm) == (4, 100, "random_full"):
                if other == "peer":
                    raise BlockingIOError("peer active")
                raise ValueError("independent failure")
            return original(out, suite, p, arm, devices, env)

        monkeypatch.setattr(worker.runtime, "run_arm", run)
    assert queue_worker.run() == 1


def test_worker_and_review_callbacks_restored_after_error(monkeypatch):
    original_work, original_review = worker.work, worker.mbpp_resume_blocked
    monkeypatch.setattr(worker, "main", lambda: (_ for _ in ()).throw(ValueError("failed")))
    with pytest.raises(ValueError):
        queue_worker.run()
    assert worker.work is original_work and worker.mbpp_resume_blocked is original_review


def test_live_peer_recovery_is_not_terminal(tmp_path, monkeypatch):
    blocked, calls = review_queue(tmp_path, monkeypatch)
    assert queue_worker.run() == 80
    with worker.base.lease(blocked / ".task.lock"):
        assert queue_worker.only_review_dependencies(tmp_path, {blocked}) is False
    assert queue_worker.only_review_dependencies(tmp_path, {blocked}) is True
