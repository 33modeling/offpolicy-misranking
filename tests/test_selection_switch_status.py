import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import selection_gate as core
import selection_switch as rule

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("switch_status", ROOT / "scripts/selection_switch_status.py")
status = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(status)


def prepared(root):
    core.atomic_json(root / "switch.json", {"schema": rule.SCHEMA, "code_hashes": {"old": "unchanged"}})


def point(root, seed=0, step=25):
    return root / f"states/s{seed}-t{step}/points/view-{step}"


def prefix(root, seed=0, step=25):
    return root / f"prefixes/seed-{seed}/segment-{step}"


def completed_prefix(root, seed=0, step=25):
    core.atomic_json(prefix(root, seed, step).parent / f"prefix-{step}.json", {"schema": rule.SCHEMA})


def published(directory):
    core.atomic_json(directory / "result.json", {"schema": rule.SCHEMA, "complete": True})
    core.atomic_json(directory / "result.sha256.json", {"sha256": hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()})


def running(directory, host, *, now, phase="prefix-train", pid=123):
    core.atomic_json(directory / "progress.json", {"state": "running", "host": host, "pid": pid,
        "phase": phase, "updated": now-2, "seconds": 1234., "timeout": 14400., "event_id": host})


def four_nodes(root, now):
    prepared(root)
    completed_prefix(root)
    running(point(root) / "selection_reduced", "node-1", now=now, phase="fresh-r-validation")
    core.atomic_json(point(root) / "selection_reduced/failure.json", {"error": "old failure", "time": now-1000})
    published(point(root) / "random_reduced")
    for seed in (1, 2, 3):
        running(prefix(root, seed), f"node-{seed+1}", now=now, pid=124+seed)
    core.atomic_json(prefix(root, 4) / "failure.json", {
        "error": "prefix-train worker failed: [1]\n[worker-log] prefix-train-0.log\nRuntimeError: actual training failure"})


def test_four_active_nodes_compact_matrix_and_real_failure(tmp_path):
    now = time.time()
    four_nodes(tmp_path, now)
    data = status.snapshot(tmp_path, now=now)
    assert data["active_nodes"] == 4 and data["stale_nodes"] == 0
    assert data["development_done"] == 1 and data["test_done"] == 0
    assert data["prefix_done"] == 1
    assert sum(data["branch_counts"].values()) == 48
    assert len([task for task in data["tasks"] if task["kind"] == "prefix"]) == 15
    output = status.render(data)
    assert "4 active" in output and "Dev 1/18" in output and "Test 0/30" in output
    assert "RuntimeError: actual training failure" in output and "old failure" not in output
    assert "CURRENT WORK" in output and "PREFIXES" in output and "CONTINUATIONS" in output
    assert "ALERTS  FAILED 1" in output
    assert "prefix 25 (fail)" in output
    print(output)


def test_stale_heartbeat_is_not_running_or_ready(tmp_path):
    prepared(tmp_path)
    now = time.time()
    running(prefix(tmp_path), "old-node", now=now-100)
    data = status.snapshot(tmp_path, now=now)
    assert data["active_nodes"] == 0 and data["stale_nodes"] == 1
    assert data["tasks"][0]["status"] == "STALE"
    assert "old-node" in status.render(data)


@pytest.mark.parametrize("seed,expected", [(0, "WAIT"), (3, "READY")])
def test_failed_diagnostic_blocks_dev_but_allows_held_out_fallback(tmp_path, seed, expected):
    prepared(tmp_path)
    completed_prefix(tmp_path, seed)
    core.atomic_json(tmp_path / "model.json", {"model_id": "published"})
    name = "measurement" if seed == 0 else "gate_measurement"
    core.atomic_json(point(tmp_path, seed) / name / "initial.json", {"status": "failed_no_retry"})
    data = status.snapshot(tmp_path)
    branches = [task for task in data["tasks"] if task["kind"] == "branch" and task["seed"] == seed and task["step"] == 25]
    assert all(task["status"] == expected for task in branches)


def test_running_diagnostic_has_owner_and_blocks_branch_barrier(tmp_path):
    prepared(tmp_path)
    completed_prefix(tmp_path)
    running(point(tmp_path) / "measurement", "diagnostic-node", now=time.time(), phase="diagnose")
    data = status.snapshot(tmp_path)
    assert data["active_nodes"] == 1
    branches = [task for task in data["tasks"] if task["kind"] == "branch" and task["seed"] == 0 and task["step"] == 25]
    assert all(task["reason"] == "diagnostic run" and task["status"] == "WAIT" for task in branches)


@pytest.mark.parametrize("receipt,expected", [(None, "SAVING"), ("wrong", "INVALID")])
def test_unfinished_or_invalid_result_publication_is_not_done(tmp_path, receipt, expected):
    prepared(tmp_path)
    directory = point(tmp_path) / "random_reduced"
    published(directory)
    path = directory / "result.sha256.json"
    if receipt is None:
        path.unlink()
    else:
        core.atomic_json(path, {"sha256": receipt})
    data = status.snapshot(tmp_path)
    assert data["development_done"] == 0
    assert next(task for task in data["tasks"] if task["directory"] == str(directory.relative_to(tmp_path)))["status"] == expected


def test_active_cost_is_not_flagged_as_interrupted_and_status_does_not_recover(tmp_path):
    prepared(tmp_path)
    directory = prefix(tmp_path)
    running(directory, "node-1", now=time.time())
    events = [{"event_id": event_id, "state": "started", "phase": "prefix-train", "ledger": "research",
               "gpus": 4, "gpu_type": "H100"} for event_id in ("old", "node-1")]
    (directory / "cost.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
    core.atomic_json(directory / "cost-events/old.json", {**events[0], "state": "finished",
        "seconds": 2., "allocated_gpu_seconds": 8., "exit_code": 1})
    before = (directory / "cost.jsonl").read_bytes()
    data = status.snapshot(tmp_path)
    assert data["cost_pending"] == [{"directory": str(directory.relative_to(tmp_path)), "event_id": "old", "scope": "prefix research"}]
    assert (directory / "cost.jsonl").read_bytes() == before


def test_bad_progress_does_not_hide_other_nodes(tmp_path):
    now = time.time()
    four_nodes(tmp_path, now)
    (prefix(tmp_path, 1) / "progress.json").write_text("{invalid")
    data = status.snapshot(tmp_path, now=now)
    assert data["active_nodes"] == 3
    assert data["notices"] and "READ WARNING" in status.render(data)


def test_fresh_waiting_launcher_is_separate_from_active_and_old_logs(tmp_path):
    now = time.time()
    four_nodes(tmp_path, now)
    logs = tmp_path / "logs"
    logs.mkdir()
    for host in ("node-1", "node-5", "old-node"):
        path = logs / f"launcher.{host}_.log"
        path.write_text("[waiting] no claimable task; retry in 15s\n")
        stamp = now-100 if host == "old-node" else now
        os.utime(path, (stamp, stamp))
    data = status.snapshot(tmp_path, now=now)
    assert data["active_nodes"] == 4
    assert [node["host"] for node in data["waiting_nodes"]] == ["node-5"]


@pytest.mark.parametrize("width", [80, 100, 120])
def test_terminal_layout_has_bounded_lines(tmp_path, width):
    four_nodes(tmp_path, time.time())
    output = status.render(status.snapshot(tmp_path), width=width)
    assert all(len(line) <= width for line in output.splitlines())
    assert len(output.splitlines()) < 65


def test_status_launcher_is_read_only_and_does_not_migrate_runtime(tmp_path):
    four_nodes(tmp_path, time.time())
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = subprocess.run(["bash", "scripts/run_selection_switch.sh", "status", "--json"], cwd=ROOT,
        env={**os.environ, "SWITCH_ROOT": str(tmp_path), "SWITCH_PYTHON": sys.executable},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["active_nodes"] == 4
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not list(tmp_path.glob(".*.lock"))
    assert not list(tmp_path.glob("*-runtime.json"))


def test_watch_can_stop_without_touching_workers(tmp_path, monkeypatch, capsys):
    prepared(tmp_path)
    monkeypatch.setattr(sys, "argv", ["status", "--root", str(tmp_path), "--watch", "1"])
    monkeypatch.setattr(status.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert status.main() == 130
    assert "PROGRESS" in capsys.readouterr().out


def test_unprepared_root_does_not_get_created(tmp_path):
    root = tmp_path / "not-started"
    assert status.render(status.snapshot(root)).startswith("NOT PREPARED")
    assert not root.exists()


def test_training_step_uses_last_complete_log_row_without_claiming_publication(tmp_path):
    prepared(tmp_path)
    directory = prefix(tmp_path)
    running(directory, "node-1", now=time.time())
    stats = directory / "fresh_r/policy/grpo_stats.jsonl"
    stats.parent.mkdir(parents=True)
    stats.write_bytes(b'{"step": 17}\n{"step": 18}\n{"step":')
    data = status.snapshot(tmp_path)
    assert data["tasks"][0]["training_step"] == 18
    assert data["prefix_done"] == 0
    assert status.last_training_step(stats) == 18


def test_invalid_prefix_blocks_dependents_and_is_not_counted_complete(tmp_path):
    prepared(tmp_path)
    completed_prefix(tmp_path)
    (prefix(tmp_path).parent / "prefix-25.json").write_text("not json")
    data = status.snapshot(tmp_path)
    assert data["prefix_done"] == 0
    assert data["tasks"][0]["status"] == "INVALID"
    assert all(task["status"] == "WAIT" for task in data["tasks"] if task["kind"] == "branch" and task["seed"] == 0)
