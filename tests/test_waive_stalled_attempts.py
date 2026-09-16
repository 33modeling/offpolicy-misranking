import importlib.util
import json
from pathlib import Path
import subprocess
import sys

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
    clean = branch(tmp_path, "selection_full", fault=False)
    assert "no failed attempt with a GPU-fault signature" in waive.waive(tmp_path, clean, apply=True)
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
