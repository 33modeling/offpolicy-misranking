import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

import selection_gate as core
import selection_gate_gpu as base

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recover_selection_switch_cost as recovery


def open_event(root):
    directory = root / "states/s0-t25/points/view-25/selection_reduced"
    start = {"event_id": "aborted", "state": "started", "phase": "fresh-r-validation",
             "ledger": "deployment", "gpus": 4, "gpu_type": "H100", "host": "stopped-node", "time": 100.}
    core.atomic_json(root / "switch.json", {"schema": "fixture"})
    base.journal(directory / "cost.jsonl", start)
    core.atomic_json(directory / "progress.json", {**start, "state": "running", "seconds": 12., "updated": 112.})
    return directory, start


def test_interrupted_finish_append_recovers_from_atomic_receipt(tmp_path, monkeypatch):
    directory = tmp_path / "prefixes/seed-0/segment-25"
    journal = base.journal

    def interrupt_finish(path, row):
        if row["state"] == "finished":
            raise OSError("interrupted finish append")
        return journal(path, row)

    monkeypatch.setattr(base, "journal", interrupt_finish)
    with pytest.raises(OSError, match="interrupted finish append"):
        base.meter(directory, "prefix-train", "H100", action=lambda: None)
    original = (directory / "cost.jsonl").read_bytes()
    event_id = base.cost(directory)["incomplete_events"][0]
    monkeypatch.setattr(base, "journal", journal)
    result = recovery.recover(tmp_path, directory, event_id)
    assert result["evidence"]["kind"] == "atomic_finish_receipt"
    assert base.spent(directory) == result["allocated_gpu_seconds"] > 0
    assert (directory / "cost.jsonl").read_bytes().startswith(original)
    repaired = (directory / "cost.jsonl").read_bytes()
    assert recovery.recover(tmp_path, directory, event_id)["status"] == "already_closed"
    assert (directory / "cost.jsonl").read_bytes() == repaired


def test_legacy_recovery_preserves_prior_cost_and_partial_outputs(tmp_path):
    directory, start = open_event(tmp_path)
    original = (directory / "cost.jsonl").read_bytes()
    core.atomic_json(directory / "partial.json", {"completed_prompt": 7})
    partial_sha = base.digest(directory / "partial.json")
    with pytest.raises(ValueError, match="lower bound"):
        recovery.recover(tmp_path, directory, start["event_id"])
    assert (directory / "cost.jsonl").read_bytes() == original
    result = recovery.recover(tmp_path, directory, start["event_id"], seconds=15., reason="scheduler termination log")
    assert result["allocated_gpu_seconds"] == base.spent(directory) == 60.
    assert base.cost(directory)["ledgers"]["deployment"]["failed_events"] == 1
    assert base.digest(directory / "partial.json") == partial_sha
    assert (directory / "cost.jsonl").read_bytes().startswith(original)


def test_resume_automatically_replays_completed_event_without_guessing(tmp_path):
    directory, start = open_event(tmp_path)
    original = (directory / "cost.jsonl").read_bytes()
    finish = {**start, "state": "finished", "time": 115., "seconds": 15.,
              "allocated_gpu_seconds": 60., "exit_code": 1}
    core.atomic_json(directory / "cost-events/aborted.json", finish)
    with base.lease(directory / ".cost.lock"), pytest.raises(BlockingIOError):
        base.spent(directory)
    assert (directory / "cost.jsonl").read_bytes() == original
    assert base.spent(directory) == 60.
    recovered = (directory / "cost.jsonl").read_bytes()
    assert recovered.startswith(original)
    assert base.spent(directory) == 60.
    assert (directory / "cost.jsonl").read_bytes() == recovered


@pytest.mark.parametrize("seconds", [0., 11., float("nan"), float("inf")])
def test_recovery_rejects_missing_or_underreported_duration(tmp_path, seconds):
    directory, start = open_event(tmp_path)
    original = base.digest(directory / "cost.jsonl")
    with pytest.raises(ValueError):
        recovery.recover(tmp_path, directory, start["event_id"], seconds=seconds, reason="termination log")
    assert base.digest(directory / "cost.jsonl") == original


@pytest.mark.parametrize("lock", [".task.lock", ".cost.lock"])
def test_recovery_cannot_modify_live_work(tmp_path, lock):
    directory, start = open_event(tmp_path)
    with base.lease(directory / lock), pytest.raises(BlockingIOError):
        recovery.recover(tmp_path, directory, start["event_id"], seconds=15., reason="termination log")
    assert not base.cost(directory)["complete"]


def test_live_legacy_owner_is_rejected(tmp_path):
    directory, start = open_event(tmp_path)
    start.update(host=socket.gethostname(), pid=os.getpid())
    (directory / "cost.jsonl").write_text(json.dumps(start) + "\n")
    (directory / "progress.json").unlink()
    with pytest.raises(ValueError, match="still alive"):
        recovery.recover(tmp_path, directory, start["event_id"], seconds=15., reason="termination log")


def test_recovery_rejects_changed_allocation_and_published_results(tmp_path):
    directory, start = open_event(tmp_path)
    finish = {**start, "state": "finished", "seconds": 15., "allocated_gpu_seconds": 120., "gpus": 8, "exit_code": 1}
    core.atomic_json(directory / "cost-events/aborted.json", finish)
    with pytest.raises(ValueError, match="allocation changed"):
        recovery.recover(tmp_path, directory, start["event_id"])
    core.atomic_json(directory / "result.json", {})
    with pytest.raises(ValueError, match="published result"):
        recovery.recover(tmp_path, directory, start["event_id"])


def test_recovery_rejects_outside_root(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        recovery.recover(tmp_path, tmp_path.parent, "aborted", seconds=15., reason="termination log")


def test_recover_cost_launcher_lists_events_without_modifying_them(tmp_path):
    directory, _ = open_event(tmp_path)
    original = {path: base.digest(path) for path in tmp_path.rglob("*") if path.is_file()}
    result = subprocess.run(["bash", "scripts/run_selection_switch.sh", "recover-cost"],
        cwd=base.ROOT, env={**os.environ, "SWITCH_ROOT": str(tmp_path), "SWITCH_PYTHON": sys.executable},
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    pending = json.loads(result.stdout)["open_events"]
    assert len(pending) == 1 and pending[0]["start"]["event_id"] == "aborted"
    assert pending[0]["progress"]["seconds"] == 12.
    assert {path: base.digest(path) for path in original} == original
    assert not base.cost(directory)["complete"]
