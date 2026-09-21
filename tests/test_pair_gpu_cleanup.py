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
    return SimpleNamespace(pid=os.getpid(), argv=(),
                           environ={"OUT_ROOT": str(root), f"OM_SELECTION_COST_{event}": "1"},
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


@pytest.mark.parametrize("shared", [False, True])
def test_recovery_runs_after_node_lock_and_failure_releases_it(tmp_path, shared):
    result = subprocess.run(["bash", "-c", '''
source scripts/_e5_node.sh
PAIR_ROOT="$1"
OM_LOCAL_LOCK_DIR="$2"
EXPERIMENTS_NODE_ID=fixture
if [ "$3" = 1 ]; then stat() { echo nfs; }; fi
e5_cleanup_lock_helpers() { :; }
PY=probe
probe() {
  case "$1" in
    */_pair_gpu_cleanup.py)
      [ ! -e /proc/self/fd/8 ] || return 99
      if flock -n "$LOCK_FILE" true; then return 98; fi
      printf 'recovery-after-lock\n'
      return 75 ;;
  esac
}
rc=0
e5_acquire_node || rc=$?
[ ! -e /proc/$$/fd/8 ] || exit 97
[ ! -e /proc/$$/fd/7 ] || exit 96
exit "$rc"
''', "test", str(tmp_path / "pair"), str(tmp_path / "locks"), str(int(shared))],
        cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode == 75
    assert "recovery-after-lock" in result.stdout
    assert result.stdout.index("node ownership acquired") < result.stdout.index("recovery-after-lock")


@pytest.mark.parametrize("shared", [False, True])
def test_duplicate_launch_never_calls_gpu_recovery(tmp_path, shared):
    lock = tmp_path / ("physical.fixture.lock" if shared else "primary.lock")
    with lock.open("a+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["bash", "-c", '''
source scripts/_e5_node.sh
PAIR_ROOT="$1/pair"
OM_LOCAL_LOCK_DIR="$1"
EXPERIMENTS_NODE_ID=fixture
if [ "$2" = 1 ]; then stat() { echo nfs; }; fi
e5_physical_node_id() { printf fixture; }
e5_cleanup_lock_helpers() { :; }
PY=probe
probe() {
  case "$1" in */_pair_gpu_cleanup.py) echo forbidden-recovery; return 99 ;; esac
}
e5_acquire_node
''', "test", str(tmp_path), str(int(shared))], cwd=ROOT,
            capture_output=True, text=True, timeout=10)
        assert result.returncode == 75, result.stdout + result.stderr
        assert "[busy]" in result.stdout
        assert "forbidden-recovery" not in result.stdout


def test_live_controller_preserved_even_if_phase_lock_is_free(tmp_path, monkeypatch, capsys):
    root, phase = fixture_phase(tmp_path)
    controller_script = tmp_path / "queue_selector_pair_gpu.py"
    controller_script.write_text("import time; time.sleep(90)\n")
    controller = subprocess.Popen([sys.executable, str(controller_script), "run", "--root", str(root)])
    with (phase / "train-0.log").open("a") as log:
        child = subprocess.Popen([sys.executable, "-c",
            "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); time.sleep(90)",
            str(phase / "ready")], stdout=log, stderr=log,
            env={**os.environ, "OUT_ROOT": str(root), f"OM_SELECTION_COST_{EVENT}": "1"})
        try:
            deadline = time.monotonic() + 5
            while not (phase / "ready").exists():
                assert time.monotonic() < deadline
                time.sleep(.01)
            assert recovery.candidates(root, recovery.cleanup._snapshot())
            monkeypatch.setattr(recovery.cleanup, "terminate", lambda *a, **k: pytest.fail("must not signal"))
            assert recovery.recover(root) == 0
            assert controller.poll() is None and child.poll() is None
            assert "live Pair controllers preserved" in capsys.readouterr().out
        finally:
            child.terminate()
            controller.terminate()
            child.wait(timeout=3)
            controller.wait(timeout=3)


@pytest.mark.parametrize("different", ["pid", "mnt", "cgroup", "unreadable"])
def test_other_or_unreadable_allocation_is_not_a_cleanup_candidate(tmp_path, monkeypatch, different):
    root, phase = fixture_phase(tmp_path)
    proc = tmp_path / "proc"
    for name in ("self", "42"):
        folder = proc / name
        (folder / "ns").mkdir(parents=True)
        (folder / "cgroup").write_text("0::/same\n")
    for namespace in ("pid", "mnt"):
        origin = proc / "self/ns" / namespace
        origin.touch()
        os.link(origin, proc / "42/ns" / namespace)
    if different in {"pid", "mnt"}:
        path = proc / "42/ns" / different
        path.unlink()
        path.touch()
    elif different == "cgroup":
        (proc / "42/cgroup").write_text("0::/other\n")
    else:
        (proc / "42/cgroup").unlink()
    assert not recovery.same_allocation(42, proc=proc)
    monkeypatch.setattr(recovery, "same_allocation", lambda pid: False)
    assert not recovery.candidates(root, {42: process(root, phase)})


def test_event_with_a_foreign_descendant_is_preserved(tmp_path, monkeypatch):
    root, phase = fixture_phase(tmp_path)
    local = process(root, phase)
    foreign = SimpleNamespace(pid=987654321)
    monkeypatch.setattr(recovery.cleanup, "_snapshot", lambda: {local.pid: local})
    monkeypatch.setattr(recovery.cleanup, "list_processes", lambda *a, **k: [local, foreign])
    monkeypatch.setattr(recovery, "same_allocation", lambda pid: pid == local.pid)
    monkeypatch.setattr(recovery.cleanup, "terminate", lambda *a, **k: pytest.fail("must not signal"))
    assert recovery.recover(root) == 0
