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
