import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selection_switch_gpu as switch

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


def test_stale_closure_charges_last_observed_duration_and_keeps_recent_events(tmp_path):
    directory, start = open_event(tmp_path)
    log = directory / "fresh-r-validation-0.log"
    log.write_text("rank log\n")
    os.utime(log, (start["time"] + 40., start["time"] + 40.))
    assert recovery.close_stale(tmp_path, min_age=900., now=start["time"] + 100.)[0]["status"] == "skipped"
    assert not base.cost(directory)["complete"]
    outcome = recovery.close_stale(tmp_path, min_age=900., now=start["time"] + 5000.)
    assert outcome[0]["status"] == "recovered" and outcome[0]["seconds"] == 40.
    assert outcome[0]["evidence"]["kind"] == "stale_owner_last_evidence"
    assert base.cost(directory)["complete"]
    assert recovery.close_stale(tmp_path, min_age=900., now=start["time"] + 5000.) == []


def test_stale_closure_uses_receipt_first_and_leaves_live_local_owner(tmp_path):
    directory, start = open_event(tmp_path)
    finish = {**start, "state": "finished", "seconds": 33., "allocated_gpu_seconds": 132., "exit_code": 130}
    core.atomic_json(directory / "cost-events/aborted.json", finish)
    outcome = recovery.close_stale(tmp_path, min_age=900., now=start["time"] + 5000.)
    assert outcome[0]["status"] == "recovered" and outcome[0]["evidence"]["kind"] == "atomic_finish_receipt"
    live_dir = tmp_path / "states/s1-t25/points/view-25/random_reduced"
    live = {**start, "host": socket.gethostname(), "pid": os.getpid()}
    base.journal(live_dir / "cost.jsonl", live)
    core.atomic_json(live_dir / "progress.json", {**live, "state": "running", "seconds": 5., "updated": start["time"] + 5.})
    outcome = recovery.close_stale(tmp_path, min_age=900., now=start["time"] + 5000.)
    assert outcome[0]["status"] == "blocked" and "still alive" in outcome[0]["reason"]
    assert not base.cost(live_dir)["complete"]


def test_recover_cost_launcher_stale_flag_closes_only_silent_events(tmp_path):
    directory, start = open_event(tmp_path)
    result = subprocess.run(["bash", "scripts/run_selection_switch.sh", "recover-cost", "--stale"],
        cwd=base.ROOT, env={**os.environ, "SWITCH_ROOT": str(tmp_path), "SWITCH_PYTHON": sys.executable},
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["stale_closure"][0]["status"] == "recovered" and payload["open_events"] == []
    assert base.cost(directory)["complete"]


def pending_prefix(root, *, ledger="research"):
    directory = root / "prefixes/seed-0/segment-25"
    start = {"event_id": "aborted", "state": "started", "phase": "prefix-train", "ledger": ledger,
             "gpus": 4, "gpu_type": "H100", "host": "stopped-node", "time": 100.}
    base.journal(directory / "cost.jsonl", start)
    core.atomic_json(directory / "progress.json", {**start, "state": "running", "seconds": 12., "updated": 112.})
    return directory, start


def test_prefix_restarts_with_unknown_research_cost_and_preserves_evidence(tmp_path, monkeypatch):
    from types import SimpleNamespace
    directory, start = pending_prefix(tmp_path)
    original = (directory / "cost.jsonl").read_bytes()
    old_progress = core.read(directory / "progress.json")
    cfg = {"model": "model", "max_new_tokens": 64, "prompt_format": "fixture"}
    monkeypatch.setattr(switch, "manifest", lambda _: {"gpu_type": "H100", "prefix_timeout": 60.})
    monkeypatch.setattr(switch, "verify_source", lambda *a: {"config": cfg, "subset": {"train": []}})
    monkeypatch.setattr(switch, "validate_prefix", lambda *a: None)
    monkeypatch.setitem(sys.modules, "evidence_downstream", SimpleNamespace(
        train_args=lambda *a: ["train"], _expected_config=lambda c: {}, POLICY_FILES=("policy_train.json",)))
    monkeypatch.setitem(sys.modules, "train_policy_grpo", SimpleNamespace(validate_policy_lineage=lambda *a, **kw: None))
    meter, calls = base.meter, []
    def train(segment, name, gpu_type, **kwargs):
        calls.append(name)
        return meter(segment, name, gpu_type, ledger=kwargs["ledger"],
                     action=lambda: core.atomic_json(segment / "fresh_r/policy/policy_train.json", {"done": True}))
    monkeypatch.setattr(base, "meter", train)
    with base.lease(directory.parent / ".prefix.lock"):
        switch.build_prefix(tmp_path, 0, 25, list("0123"), {})
    assert calls == ["prefix-train"]
    assert (directory.parent / "prefix-25.json").exists()
    assert (directory / "cost.jsonl").read_bytes().startswith(original)
    archive = core.read(directory / "pending-costs/aborted.json")
    assert archive["start"] == start and archive["progress"] == old_progress
    assert archive["total_gpu_seconds"] is None
    assert not base.cost(directory)["complete"]
    report = switch.prefix_cost_report(tmp_path)
    assert not report["complete"] and report["total_gpu_seconds"] is None
    assert report["known_gpu_seconds"] > 0
    with pytest.raises(ValueError, match="unclosed cost event"):
        base.spent(directory)
    assert recovery.inspect(tmp_path)[0]["progress"] == old_progress
    with pytest.raises(ValueError):
        recovery.recover(tmp_path, directory, "aborted", seconds=11., reason="termination log")
    recovery.recover(tmp_path, directory, "aborted", seconds=15., reason="termination log")
    report = switch.prefix_cost_report(tmp_path)
    assert report["complete"] and report["total_gpu_seconds"] > 60.


def test_prefix_unknown_cost_is_idempotent_and_never_waives_deployment(tmp_path):
    directory, _ = pending_prefix(tmp_path)
    assert not switch.prefix_cost(directory, "H100")["complete"]
    before = {path: base.digest(path) for path in directory.rglob("*.json*")}
    assert not switch.prefix_cost(directory, "H100")["complete"]
    assert {path: base.digest(path) for path in before} == before
    other, _ = pending_prefix(tmp_path / "other", ledger="deployment")
    with pytest.raises(ValueError, match="unexpected prefix research allocation"):
        switch.prefix_cost(other, "H100")
    assert not (other / "pending-costs").exists()


def test_prefix_cost_cannot_bypass_active_meter(tmp_path):
    directory, _ = pending_prefix(tmp_path)
    with base.lease(directory / ".cost.lock"), pytest.raises(BlockingIOError):
        switch.prefix_cost(directory, "H100")
    assert not (directory / "pending-costs").exists()


def test_killed_prefix_owner_can_resume_without_inventing_its_duration(tmp_path):
    import time
    directory = tmp_path / "prefixes/seed-0/segment-25"
    script = """
import sys, time
from pathlib import Path
import selection_gate_gpu as b
directory = Path(sys.argv[1])
with b.lease(directory.parent / '.prefix.lock'):
    b.meter(directory, 'prefix-train', 'H100', action=lambda: time.sleep(60))
"""
    worker = subprocess.Popen([sys.executable, "-c", script, str(directory)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (directory / "progress.json").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (directory / "progress.json").exists()
        original = (directory / "cost.jsonl").read_bytes()
        with pytest.raises(BlockingIOError):
            switch.prefix_cost(directory, "H100")
        worker.kill()
        worker.communicate(timeout=5)
        with base.lease(directory.parent / ".prefix.lock"):
            assert not switch.prefix_cost(directory, "H100")["complete"]
            base.meter(directory, "prefix-train", "H100", action=lambda: None)
        assert (directory / "cost.jsonl").read_bytes().startswith(original)
        assert switch.prefix_cost_report(tmp_path)["total_gpu_seconds"] is None
    finally:
        if worker.poll() is None:
            worker.kill()
        worker.communicate(timeout=5)


def test_prefix_rejects_live_local_legacy_owner(tmp_path):
    directory, start = pending_prefix(tmp_path)
    start.update(host=socket.gethostname(), pid=os.getpid())
    (directory / "cost.jsonl").write_text(json.dumps(start) + "\n")
    (directory / "progress.json").unlink()
    with pytest.raises(ValueError, match="still alive"):
        switch.prefix_cost(directory, "H100")
