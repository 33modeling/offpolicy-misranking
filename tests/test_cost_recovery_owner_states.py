"""A recent heartbeat is not a queue sleep or permission to close a live job."""

import pytest

import selection_gate as core
import selection_gate_gpu as base
from test_selection_switch_cost import open_event, recovery


def no_sleep(_):
    raise AssertionError("cost inspection must never sleep for the grace period")


@pytest.mark.parametrize("owner", ["remote", "unknown"])
def test_recent_unconfirmed_event_is_nonblocking_and_unchanged(tmp_path, monkeypatch, owner):
    directory, _ = open_event(tmp_path)
    before = (directory / "cost.jsonl").read_bytes()
    monkeypatch.setattr(recovery, "local_event_owner", lambda *args: owner)
    monkeypatch.setattr(recovery.time, "sleep", no_sleep)
    row = recovery.close_stale(tmp_path, min_age=180, now=122)[0]
    assert row["status"] == "skipped"
    assert "10s old (< 180s)" in row["reason"] and "queue continues (no sleep)" in row["reason"]
    assert (directory / "cost.jsonl").read_bytes() == before


def test_live_owner_or_detached_worker_is_active_without_changing_costs(tmp_path, monkeypatch):
    directory, _ = open_event(tmp_path)
    before = (directory / "cost.jsonl").read_bytes()
    monkeypatch.setattr(recovery, "local_event_owner", lambda *args: "live")
    for now in (122, 5000):
        rows = recovery.close_stale(tmp_path, min_age=180, now=now)
        assert rows[0]["status"] == "active"
        assert "active job(s) left running; queue continues" in recovery.brief("suite", rows, [object()])
    assert (directory / "cost.jsonl").read_bytes() == before


def test_live_peer_lease_is_active_even_when_heartbeat_is_stale(tmp_path, monkeypatch):
    directory, _ = open_event(tmp_path)
    before = (directory / "cost.jsonl").read_bytes()
    monkeypatch.setattr(recovery, "local_event_owner", lambda *args: pytest.fail("must honor peer lease first"))
    with base.lease(directory / ".task.lock"):
        row = recovery.close_stale(tmp_path, min_age=180, now=5000)[0]
    assert row["status"] == "active" and "lease in use" in row["reason"]
    assert (directory / "cost.jsonl").read_bytes() == before


def test_confirmed_local_stop_recovers_without_180_second_delay_or_refund(tmp_path, monkeypatch):
    directory, _ = open_event(tmp_path)
    before = (directory / "cost.jsonl").read_bytes()
    saved = directory / "policy/checkpoint-000005/checkpoint_state.json"
    core.atomic_json(saved, {"completed_steps": 5})
    saved_before = saved.read_bytes(), saved.stat().st_mtime_ns
    checked = []

    def stopped(*args):
        checked.append(args)
        return "stopped"

    monkeypatch.setattr(recovery, "local_event_owner", stopped)
    monkeypatch.setattr(recovery.time, "sleep", no_sleep)
    row = recovery.close_stale(tmp_path, min_age=180, now=122)[0]
    assert row["status"] == "recovered" and len(checked) == 2
    assert row["seconds"] == 12 + recovery.STALE_MARGIN_SECONDS
    assert row["allocated_gpu_seconds"] == base.spent(directory) == row["seconds"] * 4
    assert row["evidence"]["kind"] == "confirmed_local_stop_last_evidence"
    assert (directory / "cost.jsonl").read_bytes().startswith(before)
    assert saved_before == (saved.read_bytes(), saved.stat().st_mtime_ns)
    assert recovery.close_stale(tmp_path, min_age=180, now=122) == []


def test_owner_liveness_is_rechecked_under_locks_before_mutating(tmp_path, monkeypatch):
    directory, _ = open_event(tmp_path)
    before = (directory / "cost.jsonl").read_bytes()
    states = iter(("stopped", "live"))
    monkeypatch.setattr(recovery, "local_event_owner", lambda *args: next(states))
    row = recovery.close_stale(tmp_path, min_age=180, now=122)[0]
    assert row["status"] == "blocked" and "no longer confirmed" in row["reason"]
    assert (directory / "cost.jsonl").read_bytes() == before


def test_atomic_receipt_is_used_immediately_without_owner_guessing(tmp_path, monkeypatch):
    directory, start = open_event(tmp_path)
    finish = {**start, "state": "finished", "time": 115, "seconds": 15,
              "allocated_gpu_seconds": 60, "exit_code": 0}
    core.atomic_json(directory / "cost-events/aborted.json", finish)
    monkeypatch.setattr(recovery, "local_event_owner", lambda *args: pytest.fail("exact receipt takes precedence"))
    row = recovery.close_stale(tmp_path, min_age=180, now=122)[0]
    assert row["status"] == "recovered" and row["seconds"] == 15
    assert row["evidence"]["kind"] == "atomic_finish_receipt"
