"""An unrelated bad ledger or a curve sub-ledger must not strand saved work."""

import sys

import pytest

import selection_gate as core
import selection_gate_gpu as base
from test_selection_switch_cost import open_event, recovery


@pytest.mark.parametrize("damage", ["ledger", "progress", "outside"])
def test_bad_directory_does_not_block_other_stopped_branches(tmp_path, damage):
    root = tmp_path / "run"
    directory, start = open_event(root)
    bad = root / "states/s4-t100/points/view-100/random_reduced"
    bad.mkdir(parents=True)
    if damage == "outside":
        outside = tmp_path / "outside-cost.jsonl"
        outside.write_text("private unrelated record\n")
        (bad / "cost.jsonl").symlink_to(outside)
    elif damage == "ledger":
        (bad / "cost.jsonl").write_text("corrupt ledger\n")
    else:
        base.journal(bad / "cost.jsonl", start)
        core.atomic_json(bad / "progress.json", {**start, "gpus": 8})
    checkpoint = directory / "policy/checkpoint-000005/adapter_model.safetensors"
    core.atomic_json(checkpoint, {"saved": True})
    before = {p: p.read_bytes() for p in (bad / "cost.jsonl", checkpoint)}
    with pytest.raises((ValueError, TypeError)):
        recovery.inspect(root)
    rows = recovery.close_stale(root, min_age=180, now=5000)
    assert {row["status"] for row in rows} == {"blocked", "recovered"}
    assert base.cost(directory)["complete"]
    assert base.spent(directory) == (12 + recovery.STALE_MARGIN_SECONDS) * 4
    assert all(p.read_bytes() == data for p, data in before.items())
    assert "open count incomplete" in recovery.brief("run", rows, [])


def curve_event(root, parent=False, subledger="curve"):
    directory, start = open_event(root)
    # Leave a sealed final result untouched while recovering its reporting ledger.
    finish = {**start, "state": "finished", "seconds": 12., "allocated_gpu_seconds": 48., "exit_code": 0}
    base.journal(directory / "cost.jsonl", finish)
    core.atomic_json(directory / "result.json", {"complete": True})
    target = directory.parent / "curve-parent" if parent else directory / subledger
    curve = {**start, "event_id": "curve-interrupted", "phase": "curve", "ledger": "reporting"}
    base.journal(target / "cost.jsonl", curve)
    core.atomic_json(target / "progress.json", {**curve, "state": "running", "seconds": 12., "updated": 112.})
    return directory, target


@pytest.mark.parametrize("parent", [False, True])
@pytest.mark.parametrize("subledger", ["curve", "budget-recovery"])
def test_curve_recovery_uses_real_worker_lease_and_preserves_final_result(tmp_path, parent, subledger):
    directory, target = curve_event(tmp_path, parent, subledger)
    lock = target / ".point.lock" if parent else directory / ".task.lock"
    assert recovery.owner_lock_path(tmp_path, target) == lock
    before = {p: p.read_bytes() for p in (directory / "result.json", directory / "cost.jsonl", target / "cost.jsonl")}
    with base.lease(lock):
        assert recovery.close_stale(tmp_path, min_age=180, now=5000)[0]["status"] == "active"
    assert all(p.read_bytes() == data for p, data in before.items())
    row = recovery.close_stale(tmp_path, min_age=180, now=5000)[0]
    assert row["status"] == "recovered"
    assert base.cost(target)["complete"]
    assert base.spent(directory) == 48.
    for name in ("result.json", "cost.jsonl"):
        assert (directory / name).read_bytes() == before[directory / name]
    assert recovery.close_stale(tmp_path, min_age=180, now=5000) == []


def test_cli_reports_bad_ledger_after_recovering_valid_work(tmp_path, monkeypatch, capsys):
    directory, _ = open_event(tmp_path)
    bad = tmp_path / "states/bad/cost.jsonl"
    bad.parent.mkdir(parents=True)
    bad.write_text("broken\n")
    monkeypatch.setattr(sys, "argv", ["recover", "--root", str(tmp_path), "--stale", "--brief"])
    assert recovery.main() == 2
    text = capsys.readouterr().out
    assert "1 stale event(s) closed" in text and "open count incomplete" in text
    assert "states/bad" in text and "blocked" in text
    assert base.cost(directory)["complete"]
    assert bad.read_text() == "broken\n"


@pytest.mark.parametrize("damage", ["missing-time", "receipt-list", "progress-list"])
def test_bad_event_metadata_does_not_abort_later_healthy_branch(tmp_path, damage):
    directory, start = open_event(tmp_path)
    bad = tmp_path / "states/aaa/points/view-25/random_reduced"
    value = {key: val for key, val in start.items() if key != "time"} if damage == "missing-time" else start
    base.journal(bad / "cost.jsonl", value)
    if damage == "receipt-list":
        core.atomic_json(bad / "cost-events/aborted.json", [])
    elif damage == "progress-list":
        core.atomic_json(bad / "progress.json", [])
    before = {p: p.read_bytes() for p in bad.rglob("*") if p.is_file()}
    rows = recovery.close_stale(tmp_path, min_age=180, now=5000)
    assert [row["status"] for row in rows] == ["blocked", "recovered"]
    assert base.cost(directory)["complete"]
    assert all(p.read_bytes() == data for p, data in before.items())
