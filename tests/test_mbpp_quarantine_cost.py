"""Automatic recovery leaves incomplete MBPP training and costs untouched."""

import pytest

import selection_gate as core
import selection_gate_gpu as base
from test_selection_switch_cost import open_event, recovery


@pytest.mark.parametrize("receipt", [False, True])
def test_quarantined_cost_is_preserved_while_independent_recovery_proceeds(tmp_path, receipt):
    directory, start = open_event(tmp_path)
    core.atomic_json(tmp_path / "switch.json", {"dataset": "mbpp"})
    core.atomic_json(directory / "policy/grpo_stats.jsonl", {"step": 26})
    if receipt:
        core.atomic_json(directory / "cost-events/aborted.json", {
            **start, "state": "finished", "seconds": 15., "allocated_gpu_seconds": 60., "exit_code": 130})
    other = tmp_path / "states/s4-t25/points/view-25/random_reduced"
    base.journal(other / "cost.jsonl", start)
    core.atomic_json(other / "progress.json", {**start, "seconds": 12., "updated": 112.})
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    for _ in range(2):
        rows = recovery.close_stale(tmp_path, now=5000, min_age=180)
        blocked = next(row for row in rows if row["directory"] == str(directory.relative_to(tmp_path)))
        assert blocked["status"] == "blocked" and "checkpoint" in blocked["reason"]
        assert {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()} == before
        assert not base.cost(directory)["complete"]
    assert base.cost(other)["complete"]
    assert base.spent(other) == (12 + recovery.STALE_MARGIN_SECONDS) * 4


def test_automatic_recovery_rechecks_checkpoint_under_task_lease(tmp_path, monkeypatch):
    directory, start = open_event(tmp_path)
    core.atomic_json(tmp_path / "switch.json", {"dataset": "mbpp"})
    original = recovery.checkpoint_quarantined
    calls = []

    def race(root, target):
        calls.append(target)
        if len(calls) == 1:
            # Simulate a policy becoming incomplete after the early check.
            core.atomic_json(directory / "policy/grpo_stats.jsonl", {"step": 26})
            return False
        return original(root, target)

    monkeypatch.setattr(recovery, "checkpoint_quarantined", race)
    before = (directory / "cost.jsonl").read_bytes()
    rows = recovery.close_stale(tmp_path, now=5000, min_age=180)
    assert rows[0]["status"] == "blocked"
    assert len(calls) >= 2
    assert (directory / "cost.jsonl").read_bytes() == before
    assert not (directory / ".cost.lock").exists()


def test_non_mbpp_recovery_behavior_is_unchanged(tmp_path):
    directory, _ = open_event(tmp_path)
    core.atomic_json(tmp_path / "switch.json", {"dataset": "math500"})
    core.atomic_json(directory / "policy/grpo_stats.jsonl", {"step": 26})
    rows = recovery.close_stale(tmp_path, now=5000, min_age=180)
    assert rows[0]["status"] == "recovered"
