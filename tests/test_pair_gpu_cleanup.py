"""Startup recovery must prove phase ownership, not guess from GPU usage or time."""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pair_gpu_cleanup", ROOT / "scripts/_pair_gpu_cleanup.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
EVENT = "a" * 32


def fixture_phase(tmp_path):
    root = tmp_path / "pair"
    phase = root / "states/s0/t25/selection_reduced/train"
    phase.mkdir(parents=True)
    (phase / ".cost.lock").touch()
    (phase / "progress.json").write_text(json.dumps({"event_id": EVENT, "state": "running",
                                                  "host": "duplicate", "updated": 99999999999}))
    return root, phase


def process(root, phase, event=EVENT):
    return SimpleNamespace(environ={"OUT_ROOT": str(root), f"OM_SELECTION_COST_{event}": "1"},
                           open_files={str(phase / "train-0.log")})


@pytest.mark.parametrize("mismatch", ["root", "event", "log", "record", "missing-root"])
def test_does_not_guess_ownership(tmp_path, mismatch):
    root, phase = fixture_phase(tmp_path)
    p = process(root, phase)
    if mismatch == "root":
        p.environ["OUT_ROOT"] += "-other"
    elif mismatch == "event":
        p.environ = {"OUT_ROOT": str(root), f"OM_SELECTION_COST_{'b' * 32}": "1"}
    elif mismatch == "log":
        p.open_files = {str(tmp_path / "other/train-0.log")}
    elif mismatch == "missing-root":
        p.environ.pop("OUT_ROOT")
    else:
        (phase / "progress.json").write_text("[]")
    assert not recovery.candidates(root, {123: p})


@pytest.mark.parametrize("held", [False, True])
def test_real_process_cleanup_preserves_live_phase_and_other_root(tmp_path, monkeypatch, held):
    root, phase = fixture_phase(tmp_path)
    (phase / "ready").unlink(missing_ok=True)
    command = [sys.executable, "-c", "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(90)",
               str(phase / "ready")]
    env = {**os.environ, "OUT_ROOT": str(root), f"OM_SELECTION_COST_{EVENT}": "1"}
    with (phase / "train-0.log").open("a") as log, (phase / ".cost.lock").open("r+") as lease:
        worker = subprocess.Popen(command, env=env, stdout=log, stderr=log)
        other = subprocess.Popen(["sleep", "90"], env={**env, "OUT_ROOT": str(root) + "-other"})
        try:
            deadline = time.monotonic() + 5
            while not (phase / "ready").exists():
                assert time.monotonic() < deadline
                time.sleep(.01)
            monkeypatch.setattr(recovery, "wait_release", lambda targets: None)
            if held:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            count = recovery.recover(root)
            if held:
                assert count == 0 and worker.poll() is None
            else:
                assert count == 1 and worker.wait(timeout=3) != 0
            assert other.poll() is None
            assert json.loads((phase / "progress.json").read_text())["state"] == "running"
        finally:
            worker.kill()
            other.kill()
            worker.wait(timeout=3)
            other.wait(timeout=3)


def test_missing_lease_cannot_authorize_cleanup(tmp_path, monkeypatch):
    root, phase = fixture_phase(tmp_path)
    (phase / ".cost.lock").unlink()
    monkeypatch.setattr(recovery.cleanup, "_snapshot", lambda: {123: process(root, phase)})
    monkeypatch.setattr(recovery.cleanup, "terminate", lambda *a, **k: pytest.fail("must not signal"))
    assert recovery.recover(root) == 0


def test_finished_event_receipt_survives_progress_replacement(tmp_path):
    root, phase = fixture_phase(tmp_path)
    (phase / "progress.json").write_text(json.dumps({"event_id": "b" * 32}))
    (phase / "cost-events").mkdir()
    (phase / "cost-events" / f"{EVENT}.json").write_text(json.dumps({"event_id": EVENT}))
    assert recovery.candidates(root, {123: process(root, phase)})


@pytest.mark.parametrize("report", ["123\n", "not available\n"])
def test_cuda_release_failure_blocks_restart(monkeypatch, report):
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=report))
    with pytest.raises(RuntimeError):
        recovery.wait_release([SimpleNamespace(pid=123)], timeout=0)


def test_hook_runs_before_node_admission_and_propagates_failure(tmp_path):
    result = subprocess.run(["bash", "-c", '''
source scripts/_e5_node.sh
PAIR_ROOT="$1"
PY=probe
probe() { printf '%s\n' "$@"; return 75; }
e5_acquire_node
''', "test", str(tmp_path / "pair")], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 75
    assert "_pair_gpu_cleanup.py" in result.stdout
    assert "[node]" not in result.stdout
