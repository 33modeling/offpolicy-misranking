"""Real-process handoff regressions, without models, CUDA or remote processes.

Only the short-lived processes created by these fixtures are signalled. Runtime
compatibility may be stubbed for the lock-lifecycle test, but procfs, pidfds,
flocks, TERM delivery and launcher exit are genuine Linux operations.
"""

import contextlib
import fcntl
import importlib.util
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/selector_pair_handoff.py"

LEGACY_CONTROLLER = '''
import fcntl
import os
from pathlib import Path
import sys
import time

def manifest(root, bind_runtime=False):
    return {"code_hashes": {}}

def bind_startup_runtime(root, hashes):
    return None

if __name__ == "__main__":
    root = Path(sys.argv[sys.argv.index("--root") + 1])
    with (root / ".pair.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        print(os.getpid(), flush=True)
        time.sleep(8)
'''

LAUNCHER = '''#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
"$FIXTURE_PYTHON" src/selector_pair_gpu.py "${1:-run}" --root "$PAIR_ROOT"
'''

QUEUE_CAPABILITIES = '''
def queue_lease():
    pass

def distributed_stage():
    pass

def run_distributed():
    pass
'''


@pytest.fixture
def handoff():
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        pytest.skip("Linux pidfds are required for real-process handoff tests")
    spec = importlib.util.spec_from_file_location("selector_pair_handoff_live", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def legacy(tmp_path, handoff):
    repo, root = tmp_path / "repo", tmp_path / "pair"
    (repo / "src").mkdir(parents=True)
    (repo / "scripts").mkdir()
    root.mkdir()
    (repo / "src/selector_pair_gpu.py").write_text(LEGACY_CONTROLLER)
    (repo / "scripts/run_selector_pair.sh").write_text(LAUNCHER)
    (root / "pair.json").write_text('{"preserve": true}\n')
    (root / ".pair.lock").write_text("preserve inode\n")
    (root / "checkpoint.data").write_bytes(b"existing checkpoint")
    (root / "cost.jsonl").write_bytes(b'{"already_charged":true}\n')
    env = dict(os.environ, FIXTURE_PYTHON=sys.executable, PAIR_ROOT=str(root),
               PYTHONDONTWRITEBYTECODE="1")
    launcher = subprocess.Popen(["bash", str(repo / "scripts/run_selector_pair.sh")],
                                cwd=repo, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
    assert select.select([launcher.stdout], [], [], 4)[0], "fixture controller did not start"
    pid = int(launcher.stdout.readline().strip())
    owner_fd = os.pidfd_open(pid, 0)
    assert handoff.local_owners(root / ".pair.lock", Path("/proc")) == [pid]
    assert not handoff.shared_available(root / ".pair.lock")
    try:
        yield repo, root, launcher, pid
    finally:
        # pidfd refers solely to the process created immediately above, even
        # if its numeric PID is recycled after a completed handoff.
        if not select.select([owner_fd], [], [], 0)[0]:
            with contextlib.suppress(ProcessLookupError):
                signal.pidfd_send_signal(owner_fd, signal.SIGTERM)
        os.close(owner_fd)
        launcher.wait(timeout=4)


def snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_ino,
                                          path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def test_live_serial_restart_target_is_rejected_before_term(handoff, legacy):
    repo, root, launcher, pid = legacy
    before = snapshot(root)
    with pytest.raises(RuntimeError):
        handoff.handoff(root, repo, timeout=2)
    assert launcher.poll() is None
    assert handoff.local_owners(root / ".pair.lock", Path("/proc")) == [pid]
    assert not handoff.shared_available(root / ".pair.lock")
    assert snapshot(root) == before


@pytest.mark.parametrize('guard_compatible', [False, True])
def test_live_legacy_exclusive_controller_hands_off_to_staged_shared_queue(
        handoff, legacy, tmp_path, guard_compatible):
    """Copied handoff must start a queue, not immediately re-lock via old code."""
    repo, root, launcher, pid = legacy
    staged = tmp_path / "staged-runtime"
    (staged / "src").mkdir(parents=True)
    (staged / "scripts").mkdir()
    (staged / "src/selector_pair_gpu.py").write_text(
        LEGACY_CONTROLLER.replace("fcntl.LOCK_EX", "fcntl.LOCK_SH") + QUEUE_CAPABILITIES)
    (staged / "scripts/run_selector_pair.sh").write_text(LAUNCHER)
    (staged / 'scripts/queue_selector_pair_gpu.py').write_text(
        'def validate_receipts(root, protocol):\n'
        + ('    pass\n' if guard_compatible else '    raise ValueError("incompatible recovery receipt")\n'))
    original_runtime = snapshot(repo)
    before = snapshot(root)
    if not guard_compatible:
        with pytest.raises(RuntimeError, match='live controller was NOT stopped'):
            handoff.handoff(root, repo, timeout=2, launch_repo=staged)
        assert launcher.poll() is None
        assert handoff.local_owners(root / '.pair.lock', Path('/proc')) == [pid]
        assert snapshot(root) == before and snapshot(repo) == original_runtime
        return
    mode = handoff.handoff(root, repo, timeout=2, launch_repo=staged)
    assert launcher.wait(timeout=2) != 0
    env = dict(os.environ, FIXTURE_PYTHON=sys.executable, PAIR_ROOT=str(root),
               PYTHONDONTWRITEBYTECODE="1")
    queue = subprocess.Popen(["bash", str(staged / "scripts/run_selector_pair.sh"), mode],
                             cwd=staged, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
    assert select.select([queue.stdout], [], [], 4)[0], "queue fixture did not start"
    queue_pid = int(queue.stdout.readline().strip())
    queue_fd = os.pidfd_open(queue_pid, 0)
    try:
        assert handoff.shared_available(root / ".pair.lock")
        assert handoff.local_owners(root / ".pair.lock", Path("/proc")) == []
        with (root / ".pair.lock").open("rb") as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if not select.select([queue_fd], [], [], 0)[0]:
            signal.pidfd_send_signal(queue_fd, signal.SIGTERM)
        os.close(queue_fd)
        queue.wait(timeout=4)
    assert snapshot(repo) == original_runtime
    assert snapshot(root) == before


def test_real_term_waits_for_controller_and_launcher_then_releases_same_inode(
        handoff, legacy, monkeypatch):
    repo, root, launcher, pid = legacy
    before = snapshot(root)
    # This test isolates the actual process/lock lifecycle. The separate test
    # above intentionally retains real runtime validation for serial refusal.
    monkeypatch.setattr(handoff, "validate_runtime", lambda *args: None)
    monkeypatch.setattr(handoff, "require_queue_runtime", lambda *args: None, raising=False)
    assert handoff.handoff(root, repo, timeout=2) == "run"
    assert launcher.wait(timeout=2) != 0
    assert handoff.shared_available(root / ".pair.lock")
    assert handoff.local_owners(root / ".pair.lock", Path("/proc")) == []
    assert snapshot(root) == before


def test_noninheritable_flock_survives_fork_and_keeps_dead_owner_pid(handoff, tmp_path):
    """No local live owner is not proof that a real kernel lock is stale/free."""
    lock = tmp_path / ".pair.lock"
    lock.write_text("unchanged lock\n")
    before = (lock.read_bytes(), lock.stat().st_ino)
    controller = '''
import fcntl, os, sys, time
handle = open(sys.argv[1], "a+")
fcntl.flock(handle, fcntl.LOCK_EX)
print(os.getpid(), int(os.get_inheritable(handle.fileno())), flush=True)
child = os.fork()
if child:
    print(child, flush=True)
    os._exit(0)
time.sleep(0.7)
'''
    process = subprocess.Popen([sys.executable, "-c", controller, str(lock)],
                               stdout=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 3)[0]
        owner, inheritable = map(int, process.stdout.readline().split())
        child = int(process.stdout.readline())
        child_fd = os.pidfd_open(child, 0)
        try:
            process.wait(timeout=2)
            assert inheritable == 0
            assert not handoff.shared_available(lock)
            # Linux flock still attributes this inherited open-file-description
            # lock to the exited acquiring process, not the child retaining it.
            assert handoff.local_owners(lock, Path("/proc")) == [owner]
            assert select.select([child_fd], [], [], 3)[0], "fixture child did not exit"
            deadline = time.monotonic() + 1
            while not handoff.shared_available(lock) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert handoff.shared_available(lock)
        finally:
            os.close(child_fd)
    finally:
        process.wait(timeout=3)
    assert (lock.read_bytes(), lock.stat().st_ino) == before
