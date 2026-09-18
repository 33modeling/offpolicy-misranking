import importlib.util
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

import selection_gate as core
import selection_gate_gpu as base

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("waive_stalled_attempts", ROOT / "scripts/waive_stalled_attempts.py")
waive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(waive)


def event(event_id, phase, state, **extra):
    row = {"event_id": event_id, "phase": phase, "ledger": "deployment", "gpus": 4, "gpu_type": "H100",
           "host": "run282427-wss-4", "state": state, "time": 1789520386.0 + len(event_id)}
    if state == "finished":
        row.update({"seconds": extra.get("seconds", 1.0), "exit_code": extra.get("exit_code", 0)})
        row["allocated_gpu_seconds"] = 4*row["seconds"]
    return row


def branch(root, name, *, fault=True, exhausted=True, result=False):
    directory = root / "states/s3-t100/points/view-100" / name
    rows = [event("verify1", "verify-inputs", "started"), event("verify1", "verify-inputs", "finished"),
            event("train1", "train", "started"), event("train1", "train", "finished", seconds=7260.2, exit_code=1),
            event("verify2", "verify-inputs", "started"), event("verify2", "verify-inputs", "finished")]
    for row in rows:
        base.journal(directory / "cost.jsonl", row)
    log = "[grpo] step 165/100100 reward=0.438\n"
    if fault:
        log += "[2026-09-16 02:18:44] run282427-wss-4:1098:26108 [2] misc/strongstream.cc:333 NCCL WARN Cuda failure 'unspecified launch failure'\n"
    (directory / "train-0.log").write_text(log)
    if exhausted:
        core.atomic_json(directory / "failure.json", {"error": waive.EXHAUSTED, "host": "run282303-wss-5", "time": 1789542206.0})
    if result:
        core.atomic_json(directory / "result.json", {"complete": True})
    return directory


def test_waiver_returns_the_stalled_attempt_and_keeps_every_line(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    directory = branch(tmp_path, "random_reduced")
    assert base.spent(directory) > 29040
    message = waive.waive(tmp_path, directory, apply=False)
    assert "would waive train train1 on run282427-wss-4 (29041 GPU-s)" in message
    assert (directory / "failure.json").exists()
    message = waive.waive(tmp_path, directory, apply=True)
    assert "waived train train1" in message and "29041 GPU-s returned" in message
    assert not (directory / "failure.json").exists()
    assert base.spent(directory) < 10
    kept = [json.loads(l) for l in (directory / "cost.jsonl").read_text().splitlines()]
    assert {r["event_id"] for r in kept} == {"verify1", "verify2"}
    moved = [json.loads(l) for l in (directory / "cost-waived.jsonl").read_text().splitlines()]
    assert [r["state"] for r in moved] == ["started", "finished"] and moved[0]["event_id"] == "train1"
    receipt = core.read(directory / "waivers/train1.json")
    assert receipt["attempt"]["fault"]["line"].endswith("'unspecified launch failure'")
    assert receipt["attempt"]["exit_code"] == 1
    assert (directory / "train-0.log").exists()
    # Idempotent: nothing left to waive.
    assert "skipped" in waive.waive(tmp_path, directory, apply=True)


def test_waiver_refuses_results_missing_fault_signatures_and_live_branches(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    published = branch(tmp_path, "gated", result=True)
    assert "result already published" in waive.waive(tmp_path, published, apply=True)
    assert (published / "failure.json").exists()
    # Trained to a checkpoint, then a failed attempt with no fault evidence: an operator's call.
    clean = branch(tmp_path, "selection_full", fault=False)
    (clean / "policy/checkpoint-40").mkdir(parents=True)
    (clean / "policy/checkpoint-40/adapter_model.safetensors").write_bytes(b"x")
    assert "without fault evidence after training progress" in waive.waive(tmp_path, clean, apply=True)
    assert (clean / "failure.json").exists() and base.spent(clean) > 29040
    live = branch(tmp_path, "random_full")
    with base.lease(live / ".task.lock"):
        assert "a worker holds this branch" in waive.waive(tmp_path, live, apply=True)
    assert (live / "failure.json").exists()


def test_launcher_waive_mode_applies_to_every_exhausted_branch(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    a = branch(tmp_path, "random_reduced")
    b = branch(tmp_path, "gated")
    env = {"PATH": "/usr/bin:/bin", "SWITCH_ROOT": str(tmp_path), "SWITCH_PYTHON": sys.executable,
           "OM_WORK": str(tmp_path / "absent-work"), "HOME": str(tmp_path)}
    result = subprocess.run(["bash", "scripts/run_selection_switch.sh", "waive"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("[waive] states/") == 2 and "returned to the allocation" in result.stdout
    assert not (a / "failure.json").exists() and not (b / "failure.json").exists()
    assert (a / "waivers/train1.json").exists() and (b / "waivers/train1.json").exists()


def test_waiver_discards_the_attempt_checkpoints_so_the_retry_starts_from_the_parent(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    directory = branch(tmp_path, "random_reduced")
    (directory / "policy/policy_step_120").mkdir(parents=True)
    (directory / "progress.json").write_text("{}")
    waive.waive(tmp_path, directory, apply=True)
    assert not (directory / "policy").exists() and not (directory / "progress.json").exists()
    kept = sorted(p.name for p in (directory / "discarded").iterdir())
    assert len(kept) == 1 and (directory / "discarded" / kept[0] / "policy/policy_step_120").is_dir()
    receipt = core.read(directory / "waivers/train1.json")
    assert receipt["discarded_outputs"] == ["policy", "progress.json", "failure.json"]
    assert receipt["discarded_to"] == f"discarded/{kept[0]}"


def test_reset_waived_reruns_branches_whose_retry_resumed_a_waived_attempt(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    directory = branch(tmp_path, "random_reduced", exhausted=False, result=True)
    # The waiver as first shipped: ledger returned, no outputs discarded; the retry then resumed step 120.
    core.atomic_json(directory / "waivers/train1.json", {"schema": waive.SCHEMA})
    (directory / "policy/policy_step_266").mkdir(parents=True)
    (directory / "evaluation").mkdir()
    core.atomic_json(directory / "decision.json", {"action": "random"})
    core.atomic_json(directory / "execution.json", {"action": "random"})
    base.journal(directory / "cost.jsonl", event("train2", "train", "started"))
    base.journal(directory / "cost.jsonl", event("train2", "train", "finished", seconds=7000.0))
    clean = branch(tmp_path, "selection_reduced", exhausted=False, result=True)
    assert waive.resumed_after_waiver(tmp_path) == [directory]
    message = waive.reset_branch(tmp_path, directory, apply=False)
    assert message.startswith("[reset] states/s3-t100/points/view-100/random_reduced: would discard")
    assert (directory / "result.json").exists() and base.spent(directory) > 29040
    message = waive.reset_branch(tmp_path, directory, apply=True)
    assert "discarded 8 ledger line(s) and policy, evaluation, result.json" in message
    assert base.spent(directory) == 0 and (directory / "cost.jsonl").read_text() == ""
    for name in ("policy", "evaluation", "result.json"):
        assert not (directory / name).exists()
    assert (directory / "decision.json").exists() and (directory / "execution.json").exists()
    assert (directory / "train-0.log").exists()
    receipts = list((directory / "discards").glob("*.json"))
    assert len(receipts) == 1 and core.read(receipts[0])["ledger_lines_discarded"] == 8
    discarded = [json.loads(l) for l in (directory / "cost-discarded.jsonl").read_text().splitlines()]
    assert {r["event_id"] for r in discarded} == {"verify1", "train1", "verify2", "train2"}
    assert (directory / "discarded").is_dir() and waive.resumed_after_waiver(tmp_path) == []
    assert (clean / "result.json").exists() and base.spent(clean) > 29040


def test_reset_waived_refuses_a_branch_a_worker_holds(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    directory = branch(tmp_path, "random_reduced", exhausted=False)
    core.atomic_json(directory / "waivers/train1.json", {"schema": waive.SCHEMA})
    with base.lease(directory / ".task.lock"):
        message = waive.reset_branch(tmp_path, directory, apply=True)
    assert "skipped, a worker holds this branch; stop that node first" in message
    assert base.spent(directory) > 29040 and not (directory / "discards").exists()


def test_reset_waived_cli_reports_when_nothing_resumed(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    branch(tmp_path, "random_reduced")
    out = subprocess.run([sys.executable, str(ROOT / "scripts/waive_stalled_attempts.py"), "--root", str(tmp_path),
                          "--reset-waived"], capture_output=True, text=True, check=True).stdout
    assert "no waived branch resumed a discarded attempt" in out


def test_waiver_accepts_watchdog_stops_kills_and_attempts_that_bought_no_training(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    # Silent hang the watchdog stopped: no fault line in the log, stalled.json names the event.
    stalled = branch(tmp_path, "random_reduced", fault=False)
    base.journal(stalled / "cost.jsonl", event("train2", "train", "started"))
    base.journal(stalled / "cost.jsonl", event("train2", "train", "finished", seconds=1550.0, exit_code=-15))
    core.atomic_json(stalled / "stalled.json", {"event_id": "train2", "silent_seconds": 1550.0, "host": "run282666-wss-4"})
    message = waive.waive(tmp_path, stalled, apply=False)
    assert "train1" in message and "no-progress" in message and "train2" in message and "stall-watchdog" in message
    message = waive.waive(tmp_path, stalled, apply=True)
    assert "waived" in message and not (stalled / "failure.json").exists() and base.spent(stalled) < 100
    assert core.read(stalled / "waivers/train2.json")["attempt"]["fault"]["kind"] == "stall-watchdog"
    assert core.read(stalled / "waivers/train1.json")["attempt"]["fault"]["kind"] == "no-progress"
    # A retry that died at once because only a sliver of the allocation was left is a candidate too.
    sliver = branch(tmp_path, "selection_reduced", fault=False, exhausted=False)
    core.atomic_json(sliver / "failure.json", {"error": "train exceeded 82s allocation limit", "host": "h", "time": 1.0})
    assert sliver in waive.candidates(tmp_path)
    # A kill without a checkpoint counts even without stalled.json.
    killed = branch(tmp_path, "gated", fault=False, exhausted=False)
    base.journal(killed / "cost.jsonl", event("train3", "train", "started"))
    base.journal(killed / "cost.jsonl", event("train3", "train", "finished", seconds=900.0, exit_code=-9))
    (killed / "policy/checkpoint-5").mkdir(parents=True)
    (killed / "policy/checkpoint-5/adapter_model.safetensors").write_bytes(b"x")
    core.atomic_json(killed / "failure.json", {"error": waive.EXHAUSTED, "host": "h", "time": 1.0})
    # train1 (exit 1, no fault line) blocks the waiver once a checkpoint exists; train3 alone would qualify.
    _, found, unattributed = waive.stalled_attempts(killed)
    assert [f["event_id"] for f in found] == ["train3"] and found[0]["fault"] == {"kind": "killed", "signal": 9}
    assert [u["event_id"] for u in unattributed] == ["train1"]
    assert "needs an operator" in waive.waive(tmp_path, killed, apply=True) and (killed / "failure.json").exists()


def test_waiver_stops_after_max_rounds(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    directory = branch(tmp_path, "random_reduced")
    for i in range(waive.MAX_ROUNDS):
        core.atomic_json(directory / "waivers" / f"old{i}.json", {"schema": waive.SCHEMA, "discarded_to": f"discarded/tag{i}"})
    message = waive.waive(tmp_path, directory, apply=True)
    assert f"{waive.MAX_ROUNDS} waiver rounds already" in message and (directory / "failure.json").exists()
    assert base.spent(directory) > 29040


def test_waiver_accepts_events_closed_after_their_owner_vanished(tmp_path):
    """A node reclaimed mid-attempt: the stale closer charged the event (exit 130, recovery evidence)."""
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    directory = branch(tmp_path, "random_reduced", fault=False, exhausted=False)
    (directory / "policy/checkpoint-10").mkdir(parents=True)
    (directory / "policy/checkpoint-10/adapter_model.safetensors").write_bytes(b"x")
    rows = [json.loads(l) for l in (directory / "cost.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["event_id"] != "train1"]
    (directory / "cost.jsonl").write_text("")
    for r in rows:
        base.journal(directory / "cost.jsonl", r)
    base.journal(directory / "cost.jsonl", event("train1", "train", "started"))
    closed = event("train1", "train", "finished", seconds=9000.0, exit_code=130)
    closed["recovery"] = {"kind": "stale_owner_last_evidence", "silent_seconds": 48167.0}
    base.journal(directory / "cost.jsonl", closed)
    core.atomic_json(directory / "failure.json", {"error": waive.EXHAUSTED, "host": "h", "time": 1.0})
    _, found, unattributed = waive.stalled_attempts(directory)
    assert unattributed == [] and found[0]["fault"]["kind"] == "stale-closed"
    assert "waived" in waive.waive(tmp_path, directory, apply=True) and base.spent(directory) < 100


def test_every_failed_attempt_is_waived_at_once_not_only_after_exhaustion(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    early = branch(tmp_path, "random_reduced", exhausted=False)
    core.atomic_json(early / "failure.json", {"error": "train worker failed: [None, 1, None, None]", "host": "h", "time": 1.0})
    published = branch(tmp_path, "gated", exhausted=False, result=True)
    core.atomic_json(published / "failure.json", {"error": "stale", "host": "h", "time": 1.0})
    assert waive.candidates(tmp_path) == [early]
    assert "waived" in waive.waive(tmp_path, early, apply=True)
    assert not (early / "failure.json").exists() and base.spent(early) < 100


def test_waiver_keeps_a_checkpoint_and_returns_only_the_time_after_it(tmp_path):
    """A stalled attempt that wrote checkpoint-40: charged up to that checkpoint, resumed from it."""
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    directory = branch(tmp_path, "random_reduced")
    start = event("train1", "train", "started")["time"]
    checkpoint = directory / "policy/checkpoint-40"
    checkpoint.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "checkpoint_state.json"):
        (checkpoint / name).write_bytes(b"x")
        os.utime(checkpoint / name, (start + 3000, start + 3000))
    message = waive.waive(tmp_path, directory, apply=False)
    assert "3000s kept to its checkpoint" in message
    message = waive.waive(tmp_path, directory, apply=True)
    assert "resumes from the kept checkpoint" in message and "17041 GPU-s returned" in message, message
    assert checkpoint.is_dir() and not (directory / "discarded").exists() and not (directory / "failure.json").exists()
    rows = [json.loads(l) for l in (directory / "cost.jsonl").read_text().splitlines()]
    fin = next(r for r in rows if r["event_id"] == "train1" and r["state"] == "finished")
    assert fin["seconds"] == 3000 and fin["allocated_gpu_seconds"] == 12000 and fin["waiver"]["returned_seconds"] > 4260
    assert 12000 < base.spent(directory) < 12010
    receipt = core.read(directory / "waivers/train1.json")
    assert receipt["kept_seconds"] == 3000 and receipt["resume"] is True and receipt["discarded_to"] is None
    # The kept row is settled: nothing left to waive, and reset-waived does not mistake it for the old bug.
    assert "no failed attempt with fault evidence" in waive.waive(tmp_path, directory, apply=True)
    assert waive.resumed_after_waiver(tmp_path) == []


def test_waived_scoring_stage_discards_its_outputs_and_their_charges(tmp_path):
    """Three scoring shards done, the fourth faulted: the retry may not reuse the three for free."""
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    directory = tmp_path / "states/s3-t100/points/view-100/selection_full"
    rows = [event("verify1", "verify-inputs", "started"), event("verify1", "verify-inputs", "finished"),
            event("score1", "fresh-r-validation", "started"), event("score1", "fresh-r-validation", "finished", seconds=900.0),
            event("score2", "fresh-r-candidate", "started"), event("score2", "fresh-r-candidate", "finished", seconds=400.0, exit_code=1)]
    for row in rows:
        base.journal(directory / "cost.jsonl", row)
    (directory / "fresh-r").mkdir(parents=True)
    for name in ("validation-0.done.json", "validation-1.done.json", "candidate-0.done.json", "candidate-2.done.json", "scoring.json"):
        (directory / "fresh-r" / name).write_text("{}")
    (directory / "fresh-r-candidate-3.log").write_text("NCCL WARN Cuda failure 802 'system not yet initialized'\n")
    core.atomic_json(directory / "failure.json", {"error": "fresh-r-candidate worker failed: [None, None, None, 1]", "host": "h", "time": 1.0})
    message = waive.waive(tmp_path, directory, apply=True)
    assert "scoring outputs discarded" in message, message
    assert not (directory / "fresh-r").exists()
    kept = sorted(p.name for p in (directory / "discarded").iterdir())
    assert len(kept) == 1 and (directory / "discarded" / kept[0] / "fresh-r/validation-0.done.json").exists()
    rows = [json.loads(l) for l in (directory / "cost.jsonl").read_text().splitlines()]
    assert {r["event_id"] for r in rows} == {"verify1"} and base.spent(directory) == 4
    receipt = core.read(directory / "waivers/score2.json")
    assert receipt["scoring_reset"] is True and receipt["kept_seconds"] == 0


def test_automatic_waiver_preserves_partial_selection_even_after_gpu_fault(tmp_path):
    core.atomic_json(tmp_path / "switch.json", {"schema": "x"})
    directory = tmp_path / "states/s3-t100/points/view-100/selection_full"
    for state in ("started", "finished"):
        base.journal(directory / "cost.jsonl", event("score1", "fresh-r-candidate", state,
                     seconds=7095., exit_code=1))
    core.atomic_json(directory / "fresh-r/candidate/prompt-2.json", {"saved": "gradient"})
    core.atomic_json(directory / "fresh-r/validation-0.done.json", {"sha256": "saved shard"})
    (directory / "fresh-r/candidate-0.partial").write_text('saved complete prompt groups\n')
    (directory / "fresh-r-candidate-3.log").write_text("NCCL WARN Cuda failure 1\n")
    core.atomic_json(directory / "failure.json", {"error": "GPU phases need commands and a positive finite timeout"})
    before = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    for _ in range(8):
        message = waive.waive(tmp_path, directory, apply=True, automatic=True)
        assert "automatic scoring reset disabled" in message
        assert before == {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    assert not (directory / "discarded").exists()
    assert base.spent(directory) == 28380.


def test_no_training_checkpoint_is_not_evidence_that_selection_bought_no_work(tmp_path):
    directory = tmp_path / "states/s3-t100/points/view-100/selection_full"
    for state in ("started", "finished"):
        base.journal(directory / "cost.jsonl", event("score1", "fresh-r-candidate", state,
                     seconds=7095., exit_code=1))
    core.atomic_json(directory / "failure.json", {"error": "allocation limit"})
    _, found, unattributed = waive.stalled_attempts(directory)
    assert found == [] and len(unattributed) == 1
    assert "needs an operator" in waive.waive(tmp_path, directory, apply=True)
    assert base.spent(directory) == 28380.


@pytest.mark.parametrize("saved", [
    "policy/grpo_stats.jsonl", "policy/optimizer.pt", "policy/policy_train.json",
    "policy/checkpoint-000040/checkpoint_state.json", "policy/.checkpoint-000040.tmp/optimizer.pt",
    "policy/checkpoint-000040/adapter_model.safetensors", "policy/adapter_model.safetensors",
    "policy/budget_stop.json", "result.sha256.json", "curve.json", "curve.sha256.json",
    "evaluation/rewards.json", "discarded/old/policy/grpo_stats.jsonl", "discarded/old/result.json",
])
def test_automatic_waiver_never_resets_saved_training_or_completion_evidence(tmp_path, saved):
    directory = branch(tmp_path, "random_reduced")
    target = directory / saved
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"saved work; may need validation or recovery")
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    for _ in range(2):
        message = waive.waive(tmp_path, directory, apply=True, automatic=True)
        assert "preserved training/completion evidence and all costs" in message
        assert before == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert not (directory / "waivers").exists()


def test_automatic_waiver_preserves_successful_training_receipt_when_outputs_are_missing(tmp_path):
    directory = branch(tmp_path, "random_full")
    for state in ("started", "finished"):
        base.journal(directory / "cost.jsonl", event("completed", "train", state, seconds=3000.))
    before = (directory / "cost.jsonl").read_bytes()
    message = waive.waive(tmp_path, directory, apply=True, automatic=True)
    assert "preserved training/completion evidence and all costs" in message
    assert (directory / "cost.jsonl").read_bytes() == before
    assert (directory / "failure.json").exists()
    assert not (directory / "discarded").exists()


def test_automatic_waiver_rechecks_training_evidence_after_acquiring_locks(tmp_path, monkeypatch):
    directory = branch(tmp_path, "random_reduced")
    core.atomic_json(directory / "stalled.json", {"event_id": "train1", "silent_seconds": 2000})
    before = (directory / "cost.jsonl").read_bytes()
    real_lease = base.lease

    @contextmanager
    def publish_before_lock(path, **kwargs):
        if path.name == ".task.lock":
            core.atomic_json(directory / "policy/policy_train.json", {"completed_steps": 175})
        with real_lease(path, **kwargs):
            yield

    monkeypatch.setattr(base, "lease", publish_before_lock)
    message = waive.waive(tmp_path, directory, apply=True, automatic=True)
    assert "preserved training/completion evidence and all costs" in message
    assert core.read(directory / "policy/policy_train.json") == {"completed_steps": 175}
    assert (directory / "cost.jsonl").read_bytes() == before
    assert (directory / "failure.json").exists()
    assert not (directory / "discarded").exists()


@pytest.mark.parametrize("progress", [
    {"training_step": 165}, {"completed_steps": 165}, {"step": 165},
    {"phase": "train", "state": "finished", "exit_code": 0}, [],
])
def test_automatic_waiver_preserves_training_progress_and_invalid_progress(tmp_path, progress):
    directory = branch(tmp_path, "random_reduced")
    core.atomic_json(directory / "progress.json", progress)
    before = (directory / "cost.jsonl").read_bytes()
    message = waive.waive(tmp_path, directory, apply=True, automatic=True)
    assert "preserved training/completion evidence and all costs" in message
    assert (directory / "cost.jsonl").read_bytes() == before
    assert core.read(directory / "progress.json") == progress
    assert not (directory / "discarded").exists()


@pytest.mark.parametrize("broken_link", [False, True])
def test_automatic_waiver_does_not_assume_unvalidated_policy_is_disposable(tmp_path, broken_link):
    directory = branch(tmp_path, "random_reduced")
    if broken_link:
        (directory / "policy").symlink_to("missing-policy-directory", target_is_directory=True)
    else:
        (directory / "policy").mkdir()
    before = (directory / "cost.jsonl").read_bytes()
    message = waive.waive(tmp_path, directory, apply=True, automatic=True)
    assert "preserved training/completion evidence and all costs" in message
    assert (directory / "cost.jsonl").read_bytes() == before
    assert not (directory / "discarded").exists()


@pytest.mark.parametrize("old_cuda_log", [False, True])
def test_automatic_waiver_does_not_refund_no_checkpoint_or_unattributed_old_fault(tmp_path, old_cuda_log):
    directory = branch(tmp_path, "random_reduced", fault=old_cuda_log)
    core.atomic_json(directory / "failure.json", {"error": "train worker rejected invalid input"})
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    for _ in range(3):
        message = waive.waive(tmp_path, directory, apply=True, automatic=True)
        assert "without event-bound infrastructure evidence" in message
        assert before == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert not (directory / "waivers").exists()


@pytest.mark.parametrize("evidence_kind", ["watchdog", "signal", "stale"])
def test_automatic_waiver_still_recovers_event_bound_infrastructure_loss_without_saved_work(tmp_path, evidence_kind):
    directory = branch(tmp_path, "random_full", fault=True)
    rows = [json.loads(line) for line in (directory / "cost.jsonl").read_text().splitlines()]
    if evidence_kind == "watchdog":
        core.atomic_json(directory / "stalled.json", {"event_id": "train1", "silent_seconds": 2000})
    else:
        for row in rows:
            if row["event_id"] == "train1" and row["state"] == "finished":
                if evidence_kind == "signal":
                    row["exit_code"] = -15
                else:
                    row["recovery"] = {"kind": "stale_owner_last_evidence"}
        (directory / "cost.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    message = waive.waive(tmp_path, directory, apply=True, automatic=True)
    assert "waived train train1" in message
    receipt = core.read(directory / "waivers/train1.json")
    assert receipt["attempt"]["fault"]["kind"] == {"watchdog": "stall-watchdog", "signal": "killed", "stale": "stale-closed"}[evidence_kind]


def test_automatic_waiver_does_not_treat_a_failed_finish_receipt_as_a_lost_node(tmp_path):
    directory = branch(tmp_path, "random_reduced", fault=False)
    rows = [json.loads(line) for line in (directory / "cost.jsonl").read_text().splitlines()]
    for row in rows:
        if row["event_id"] == "train1" and row["state"] == "finished":
            row["recovery"] = {"kind": "atomic_finish_receipt"}
    (directory / "cost.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = (directory / "cost.jsonl").read_bytes()
    message = waive.waive(tmp_path, directory, apply=True, automatic=True)
    assert "without event-bound infrastructure evidence" in message
    assert (directory / "cost.jsonl").read_bytes() == before
    assert (directory / "failure.json").exists()
