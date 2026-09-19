"""Real CPU-only owner crashes, child teardown, lock reuse and bystanders."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mbpp_guard", ROOT / "scripts/_mbpp_node_guard.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def wait_for(predicate, timeout=20):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.05)
    assert predicate(), "guard simulation timed out"


@pytest.fixture
def node(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text('''import os, subprocess, sys, time
from pathlib import Path
marker = Path(sys.argv[1])
if sys.argv[2] == "complete":
    marker.write_text("complete")
    raise SystemExit(0)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
marker.write_text(str(child.pid))
if sys.argv[2] == "fail":
    raise SystemExit(7)
time.sleep(120)
''')
    lock = tmp_path / "node.lock"
    owners = []
    logs = []
    def start(mode="wait"):
        marker = tmp_path / f"marker-{len(owners)}"
        log = tmp_path / f"log-{len(owners)}"
        handle = log.open("w")
        logs.append(handle)
        owner = subprocess.Popen([sys.executable, str(ROOT / "scripts/_mbpp_node_guard.py"),
            "--lock", str(lock), "--", sys.executable, str(controller), str(marker), mode],
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        owners.append(owner)
        return owner, marker, log
    yield start, lock
    for owner in owners:
        if owner.poll() is None:
            owner.terminate()
        owner.wait(timeout=20)
    receipt = lock.with_suffix(".owner.json")
    if receipt.exists():
        guard.reap(json.loads(receipt.read_text())["token"])
    for handle in logs:
        handle.close()


def test_guard_lease_is_not_inherited_and_dead_owner_releases_own_children(node):
    start, lock = node
    first, marker, log = start()
    wait_for(marker.exists)
    child = int(marker.read_text())
    assert guard.identity(child) is not None
    assert str(lock) not in [os.readlink(p) for p in Path(f"/proc/{child}/fd").iterdir()]
    first.kill()
    first.wait(timeout=5)
    assert guard.identity(child) is not None  # Reproduce orphan left by SIGKILL.
    replacement, completed, replacement_log = start("complete")
    assert replacement.wait(timeout=20) == 0, replacement_log.read_text()
    assert completed.read_text() == "complete"
    assert guard.identity(child) is None
    assert "[mbpp-recover]" in replacement_log.read_text()


@pytest.mark.parametrize("stop", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_guard_stop_reaps_detached_child_and_preserves_unrelated_job(node, stop):
    start, lock = node
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    try:
        owner, marker, log = start()
        wait_for(marker.exists)
        child = int(marker.read_text())
        owner.send_signal(stop)
        assert owner.wait(timeout=20) == 128 + stop, log.read_text()
        assert guard.identity(child) is None and bystander.poll() is None
        again, completed, again_log = start("complete")
        assert again.wait(timeout=20) == 0, again_log.read_text()
    finally:
        bystander.terminate()
        bystander.wait(timeout=5)


def test_repeated_start_leaves_owner_and_child_running(node):
    start, lock = node
    first, marker, _ = start()
    wait_for(marker.exists)
    child = int(marker.read_text())
    duplicate, duplicate_marker, log = start("complete")
    assert duplicate.wait(timeout=10) == 0, log.read_text()
    assert "[already running]" in log.read_text()
    assert not duplicate_marker.exists()
    assert first.poll() is None and guard.identity(child) is not None


def test_failed_controller_reaps_child_before_returning_original_failure(node):
    start, lock = node
    owner, marker, log = start("fail")
    assert owner.wait(timeout=20) == 7, log.read_text()
    assert guard.identity(int(marker.read_text())) is None
    again, completed, again_log = start("complete")
    assert again.wait(timeout=20) == 0, again_log.read_text()


def test_runtime_receipt_binds_exact_code_pid_and_process_start(tmp_path):
    lock = tmp_path / "node.lock"
    fingerprint = guard.runtime_fingerprint()
    assert len(fingerprint) == 64
    assert not guard.runtime_current(lock, os.getpid(), fingerprint)
    record = guard.runtime_record(lock, os.getpid(), fingerprint)
    guard.publish(lock.with_suffix(".runtime.json"), record)
    assert guard.runtime_current(lock, os.getpid(), fingerprint)
    assert not guard.runtime_current(lock, os.getpid(), "changed-code")
    assert not guard.runtime_current(lock, 999999999, fingerprint)
    record["guard_hash"] = "old-guard-code"
    guard.publish(lock.with_suffix(".runtime.json"), record)
    assert not guard.runtime_current(lock, os.getpid(), fingerprint)
    record = guard.runtime_record(lock, os.getpid(), fingerprint)
    record["start_time"] -= 1
    guard.publish(lock.with_suffix(".runtime.json"), record)
    assert not guard.runtime_current(lock, os.getpid(), fingerprint)


def test_cleanup_refuses_start_when_owned_process_survives(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(guard.cleanup, "terminate", lambda *args, **kwargs: [])
    monkeypatch.setattr(guard.cleanup, "list_processes", lambda *args, **kwargs: [SimpleNamespace(pid=123)])
    with pytest.raises(RuntimeError, match="refusing new GPU work.*123"):
        guard.reap("a" * 32)


def test_cleanup_failure_does_not_start_replacement_or_release_owner_receipt(tmp_path, monkeypatch):
    lock = tmp_path / "node.lock"
    owner_path = lock.with_suffix(".owner.json")
    owner = {"schema": "mbpp-node-owner-v1", "lock": str(lock.resolve()),
             "pid": 999999999, "start_time": 0, "token": "a" * 32, "state": "active"}
    guard.publish(owner_path, owner)
    def failed_reap(token):
        raise RuntimeError("owned CUDA process cannot exit")
    monkeypatch.setattr(guard, "reap", failed_reap)
    def no_spawn(*args, **kwargs):
        pytest.fail("cleanup failure must block the new controller")
    monkeypatch.setattr(guard.subprocess, "Popen", no_spawn)
    with pytest.raises(RuntimeError, match="cannot exit"):
        guard.run(lock, ["unused"])
    assert json.loads(owner_path.read_text()) == owner


@pytest.mark.parametrize("outcome", ["released", "still-owned", "driver-error", "invalid"])
def test_cleanup_checks_driver_release_without_touching_other_gpu_owners(monkeypatch, outcome):
    from types import SimpleNamespace
    target = SimpleNamespace(pid=123, environ={"CUDA_VISIBLE_DEVICES": "0"}, command="worker")
    def query(*args, **kwargs):
        assert args[0] == ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']
        if outcome == 'driver-error':
            raise subprocess.TimeoutExpired(args[0], 3)
        return SimpleNamespace(stdout={'released': '999\n', 'still-owned': '123\n999\n',
                                      'invalid': '[Not Supported]\n'}[outcome])
    monkeypatch.setattr(guard.subprocess, 'run', query)
    if outcome == 'released':
        guard.wait_gpu_release([target], timeout=0)
    else:
        with pytest.raises(RuntimeError, match='refusing restart'):
            guard.wait_gpu_release([target], timeout=0)


def test_cleanup_waits_for_delayed_cuda_release(monkeypatch):
    from types import SimpleNamespace
    target = SimpleNamespace(pid=123, environ={"CUDA_VISIBLE_DEVICES": "0"}, command="worker")
    outputs = iter(['123\n999\n', '999\n'])
    calls = []
    def query(*args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(stdout=next(outputs))
    monkeypatch.setattr(guard.subprocess, 'run', query)
    monkeypatch.setattr(guard.time, 'sleep', lambda _: None)
    guard.wait_gpu_release([target], timeout=1)
    assert len(calls) == 2
