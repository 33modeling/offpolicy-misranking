import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import selection_gate as core
import selection_switch as rule

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mopps_status", ROOT / "scripts/mopps_comparison_status.py")
status = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(status)


def prepared(root, parent):
    core.atomic_json(root / "mopps.json", {"schema": "mopps-comparison/v1", "parent": str(parent),
                                           "seeds": [3, 4], "steps": [25, 50, 100], "arms": ["mopps", "random_online"]})
    core.atomic_json(parent / "switch.json", {"schema": rule.SCHEMA})


def prefix_done(parent, seed, step):
    core.atomic_json(parent / f"prefixes/seed-{seed}/prefix-{step}.json", {"schema": rule.SCHEMA})


def running(directory, host, *, now, phase="train", pid=123, seconds=1234.):
    core.atomic_json(directory / "progress.json", {"state": "running", "host": host, "pid": pid, "phase": phase,
                                                   "updated": now-2, "seconds": seconds, "timeout": 7200., "event_id": host})


def published(directory):
    core.atomic_json(directory / "result.json", {"schema": "mopps-comparison/v1", "complete": True})
    core.atomic_json(directory / "result.sha256.json",
                     {"sha256": hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()})


def fixture(root, parent, now):
    prepared(root, parent)
    prefix_done(parent, 3, 25)
    prefix_done(parent, 3, 50)
    core.atomic_json(parent / "prefixes/seed-3/segment-100/failure.json",
                     {"error": "prefix-train worker failed: [1]\nRuntimeError: Cuda failure 802 'system not yet initialized'"})
    running(parent / "prefixes/seed-4/segment-25", "node-2", now=now, phase="prefix-train", pid=77)
    core.atomic_json(parent / "states/s3-t25/points/view-25/gated/result.json", {"complete": True})
    state = root / "states/s3-t25"
    core.atomic_json(state / "import.done.json", {"contract_sha256": "x"})
    running(state / "mopps", "node-1", now=now, phase="train", pid=41, seconds=300.)
    published(state / "random_online")
    core.atomic_json(root / "states/s3-t50/random_online/failure.json",
                     {"error": "train worker failed: [1]\n[worker-log] x\nRuntimeError: old failure"})
    core.atomic_json(root / "node-preflight/node-3-abc/admission.json",
                     {"schema": "selection-nccl-admission/v1", "host": "node-3", "state": "failed",
                      "failure_kind": "cuda_system_not_ready", "attempts": [{}, {}, {}, {}], "overrides": {}})
    core.atomic_json(root / "node-preflight/node-1-def/admission.json",
                     {"schema": "selection-nccl-admission/v1", "host": "node-1", "state": "passed",
                      "attempts": [{}, {}], "overrides": {"NCCL_NVLS_ENABLE": "0"}})
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "launcher.node-4_.log").write_text("[launcher-start] ...\n[holding] node retained; next queue pass in 600s\n")


def by_key(data):
    return {(task["seed"], task["step"], task["arm"]): task for task in data["tasks"]}


def test_snapshot_mirrors_queue_semantics_and_names_parent_prerequisites(tmp_path):
    now = time.time()
    root, parent = tmp_path / "mopps", tmp_path / "switch"
    fixture(root, parent, now)
    data = status.snapshot(root, now=now)
    tasks = by_key(data)
    assert tasks[(3, 25, "mopps")]["status"] == "RUNNING" and tasks[(3, 25, "mopps")]["host"] == "node-1"
    assert tasks[(3, 25, "random_online")]["status"] == "DONE"
    assert tasks[(3, 25, "import")]["status"] == "DONE"
    assert tasks[(3, 50, "random_online")]["status"] == "FAILED"
    assert tasks[(3, 50, "random_online")]["reason"] == "RuntimeError: old failure"
    assert tasks[(3, 50, "mopps")]["status"] == "READY"
    assert tasks[(3, 100, "mopps")]["status"] == "BLOCKED"
    assert "prefix 100 failed" in tasks[(3, 100, "mopps")]["reason"] and "802" in tasks[(3, 100, "mopps")]["reason"]
    assert tasks[(4, 25, "mopps")]["status"] == "WAIT" and tasks[(4, 25, "mopps")]["reason"] == "prefix 25 (run)"
    assert tasks[(4, 100, "random_online")]["status"] == "WAIT" and tasks[(4, 100, "random_online")]["reason"] == "prefix 25 (run)"
    assert data["active_nodes"] == 1 and data["stale_nodes"] == 0
    assert data["waiting_nodes"] == [{"host": "node-4", "state": "HOLD", "reason": "node retained between queue passes"}]
    assert data["imports_done"] == 1 and data["branches_done"] == 1
    assert data["gate_results"]["s3-t25"] is True and data["gate_results"]["s4-t25"] is False
    assert [s["status"] for s in data["prefixes"]["3"]] == ["DONE", "DONE", "FAILED"]
    assert [s["status"] for s in data["prefixes"]["4"]] == ["RUNNING", "QUEUED", "QUEUED"]
    hosts = {item["host"]: item for item in data["admissions"]}
    assert hosts["node-3"]["state"] == "failed" and hosts["node-3"]["failure_kind"] == "cuda_system_not_ready"
    assert hosts["node-1"]["overrides"] == {"NCCL_NVLS_ENABLE": "0"}


def test_render_has_every_section_and_fits_the_terminal(tmp_path):
    now = time.time()
    root, parent = tmp_path / "mopps", tmp_path / "switch"
    fixture(root, parent, now)
    text = status.render(status.snapshot(root, now=now), width=120)
    for heading in ("MOPPS COMPARISON", "NODES  1 active  |  1 waiting  |  0 stale", "PROGRESS  Imports 1/6 states",
                    "CURRENT WORK", "NODE ADMISSION", "PARENT PREFIXES", "STATES", "ATTENTION"):
        assert heading in text, heading
    assert "node-1" in text and "s3/t25 MOPPS" in text and "train" in text
    assert "node-4" in text and "HOLD" in text
    assert "NODES (every host" in text and "THIS NODE GPUS" in text
    nodes = {item["host"]: item for item in status.snapshot(root, now=now)["nodes"]}
    assert nodes["node-4"]["state"] == "HOLD" and nodes["node-1"]["state"] == "RUN"
    assert "cuda_system_not_ready" in text and "NCCL_NVLS_ENABLE=0" in text
    assert "s3/t100" in text and "BLOCKED" in text and "prefix 100 failed" in text
    assert "prefix 25 (run)" in text
    assert all(len(line) <= 120 for line in text.splitlines())
    narrow = status.render(status.snapshot(root, now=now), width=80)
    assert all(len(line) <= 80 for line in narrow.splitlines())
    for width in (80, 100, 120):
        rendered = status.render(status.snapshot(root, now=now), width=width).splitlines()
        # Table rows must fit without wrapping: no continuation line starts with two spaces after a table row.
        for index, line in enumerate(rendered):
            if line.startswith(("HOST ", "STATE ", "SEED ", "NODE ", "s3/", "s4/", "node-")):
                assert index+1 == len(rendered) or not rendered[index+1].startswith("  "), (width, line)


def test_not_prepared_root_and_json_output(tmp_path):
    data = status.snapshot(tmp_path)
    assert data["prepared"] is False
    assert status.render(data).startswith("NOT PREPARED")
    root, parent = tmp_path / "mopps", tmp_path / "switch"
    fixture(root, parent, time.time())
    result = subprocess.run([sys.executable, str(ROOT / "scripts/mopps_comparison_status.py"), "--root", str(root), "--json"],
                            capture_output=True, text=True, timeout=30, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["prepared"] is True and payload["branches_done"] == 1


def test_launcher_status_uses_the_detailed_view_without_touching_the_run(tmp_path):
    root, parent = tmp_path / "mopps", tmp_path / "switch"
    fixture(root, parent, time.time())
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.rglob("*") if path.is_file()}
    result = subprocess.run(["bash", "scripts/run_mopps_comparison.sh", "status"], cwd=ROOT,
                            env={**os.environ, "MOPPS_ROOT": str(root), "SWITCH_ROOT": str(parent), "MOPPS_PYTHON": sys.executable},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "MOPPS COMPARISON" in result.stdout and "PARENT PREFIXES" in result.stdout
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == before
