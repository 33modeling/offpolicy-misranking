"""A pair handoff may terminate only the verified local legacy controller.

The process table is synthetic, lock contention is real, and pidfd signals are
intercepted. No experiment, GPU task, or real process is stopped by these tests.
"""

import contextlib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/selector_pair_handoff.py"


@pytest.fixture
def handoff(monkeypatch):
    spec = importlib.util.spec_from_file_location("selector_pair_handoff_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Unit handoff fixtures deliberately have no complete frozen protocol; the
    # real runtime validator is exercised separately through its subprocess API.
    module._test_validate_runtime = getattr(module, "validate_runtime", None)
    monkeypatch.setattr(module, "validate_runtime", lambda *args: None, raising=False)
    monkeypatch.setattr(module, "stage_runtime", lambda repo: repo, raising=False)
    return module


@pytest.fixture
def repo(tmp_path):
    directory = tmp_path / "repo"
    (directory / "src").mkdir(parents=True)
    (directory / "scripts").mkdir()
    (directory / "src/selector_pair_gpu.py").write_text("# test fixture\n")
    (directory / "scripts/run_selector_pair.sh").write_text("# test fixture\n")
    (directory / "scripts/queue_selector_pair_gpu.py").write_text(
        "def validate_receipts(root, protocol): pass\n")
    return directory


@pytest.fixture
def root(tmp_path):
    directory = tmp_path / "pair"
    artifacts = {
        "pair.json": b'{"frozen":true}\n',
        "pair-wait-guard-runtime.json": b'{"preserve":true}\n',
        "development/s0-t25/result.json": b'{"done":true}\n',
        "branches/on_policy/cost.jsonl": b'{"charged":true}\n',
        "branches/random/policy/checkpoint-25/optimizer.pt": b"optimizer data",
        "branches/on_policy/policy/checkpoint-25/adapter_model.safetensors": b"model data",
        ".pair.lock": b"preserve lock inode and bytes\n",
    }
    for relative, payload in artifacts.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return directory


@pytest.fixture
def proc(tmp_path):
    directory = tmp_path / "proc"
    directory.mkdir()
    (directory / "locks").write_text("")
    return directory


def snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns,
                                          path.stat().st_ino)
            for path in root.rglob("*") if path.is_file() and not path.is_symlink()}


@contextlib.contextmanager
def lease(root, mode=fcntl.LOCK_EX):
    with (root / ".pair.lock").open("rb") as handle:
        fcntl.flock(handle, mode | fcntl.LOCK_NB)
        yield handle


def owner_record(lock, pid, *, waiting=False, kind="WRITE", inode=None):
    stat = lock.stat()
    device = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}"
    return (f"1: {'-> ' if waiting else ''}FLOCK  ADVISORY  {kind} {pid} "
            f"{device}:{stat.st_ino if inode is None else inode} 0 EOF\n")


def process(proc, pid, repo, root, *, mode="run", ppid=1, uid=None,
            argv=None, cwd=None, start=12345, children=()):
    directory = proc / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    if argv is None:
        argv = [sys.executable, "src/selector_pair_gpu.py", mode, "--root", str(root)]
    (directory / "cmdline").write_bytes(b"\0".join(os.fsencode(x) for x in argv) + b"\0")
    (directory / "environ").write_bytes(b"")
    (directory / "cwd").symlink_to(cwd or repo, target_is_directory=True)
    (directory / "exe").symlink_to(argv[0] if str(argv[0]).startswith("/") else sys.executable)
    user = os.getuid() if uid is None else uid
    (directory / "status").write_text(
        f"Name:\tpython\nState:\tS (sleeping)\nPPid:\t{ppid}\n"
        f"Uid:\t{user}\t{user}\t{user}\t{user}\n")
    fields = ["0"] * 52
    fields[0:4] = [str(pid), "(python)", "S", str(ppid)]
    fields[21] = str(start)
    (directory / "stat").write_text(" ".join(fields) + "\n")
    task = directory / "task" / str(pid)
    task.mkdir(parents=True)
    (task / "children").write_text(" ".join(str(child) for child in children))
    allocation(proc, pid)
    return directory


@pytest.fixture
def pidfds(monkeypatch):
    """Pipe FDs mimic pidfd readiness without ever addressing an actual PID."""
    class Pidfds:
        def __init__(self):
            self.opened = []
            self.signals = []
            self.pipes = {}
            self.inodes = {}
            self.on_term = lambda pid: None

        def open(self, pid, flags=0):
            read_fd, write_fd = os.pipe()
            self.pipes[read_fd] = (pid, write_fd)
            self.inodes[read_fd] = os.fstat(read_fd).st_ino
            self.inodes[write_fd] = os.fstat(write_fd).st_ino
            self.opened.append(pid)
            return read_fd

        def send(self, fd, sig, *args, **kwargs):
            pid, _ = self.pipes[fd]
            self.signals.append((pid, sig))
            assert sig == signal.SIGTERM, "handoff must never escalate to SIGKILL"
            self.on_term(pid)

        def finish(self, pid):
            for fd, (owner, writer) in tuple(self.pipes.items()):
                if owner == pid and writer is not None:
                    os.close(writer)
                    self.pipes[fd] = (owner, None)

        def close(self):
            for reader, (_, writer) in self.pipes.items():
                for fd in (reader, writer):
                    if fd is not None:
                        with contextlib.suppress(OSError):
                            if os.fstat(fd).st_ino == self.inodes[fd]:
                                os.close(fd)

    fixture = Pidfds()
    monkeypatch.setattr(os, "pidfd_open", fixture.open)
    monkeypatch.setattr(signal, "pidfd_send_signal", fixture.send)

    def unsafe_signal(*args, **kwargs):
        pytest.fail("process termination must use a verified pidfd, not a numeric PID or group")

    monkeypatch.setattr(os, "kill", unsafe_signal)
    monkeypatch.setattr(os, "killpg", unsafe_signal)
    yield fixture
    fixture.close()


def test_missing_root_is_not_created(handoff, proc, repo, tmp_path, pidfds):
    root = tmp_path / "unmounted-volume/runs/selector-pair-v1"
    with pytest.raises((ValueError, RuntimeError, OSError)):
        handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert not root.parent.parent.exists()
    assert pidfds.signals == []


@pytest.mark.parametrize("shared", [False, True])
def test_no_exclusive_owner_returns_run_without_signals_or_writes(
        handoff, root, repo, proc, pidfds, shared):
    before = snapshot(root)
    with lease(root, fcntl.LOCK_SH) if shared else contextlib.nullcontext():
        assert handoff.handoff(root, repo, timeout=0.05, proc=proc) == "run"
    assert snapshot(root) == before
    assert pidfds.signals == []


def test_exclusive_owner_invisible_locally_is_not_stopped(handoff, root, repo, proc, pidfds):
    before = snapshot(root)
    with lease(root):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert snapshot(root) == before
    assert pidfds.signals == []


def test_stale_owner_record_without_actual_exclusive_lock_does_not_stop_process(
        handoff, root, repo, proc, pidfds):
    pid = 90001
    process(proc, pid, repo, root)
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))
    before = snapshot(root)
    assert handoff.handoff(root, repo, timeout=0.05, proc=proc) == "run"
    assert pidfds.signals == []
    assert snapshot(root) == before


@pytest.mark.parametrize("record", ["waiting", "read", "wrong-inode"])
def test_kernel_waiters_readers_and_other_inodes_are_not_owners(
        handoff, root, repo, proc, pidfds, record):
    pid = 90001
    process(proc, pid, repo, root)
    lock = root / ".pair.lock"
    kwargs = {"waiting": True} if record == "waiting" else (
        {"kind": "READ"} if record == "read" else {"inode": lock.stat().st_ino + 1})
    (proc / "locks").write_text(owner_record(lock, pid, **kwargs))
    with lease(root):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert pidfds.signals == []


@pytest.mark.parametrize("invalid", ["other-user", "other-root", "other-repo", "other-script", "wrong-mode"])
def test_unrecognized_or_out_of_scope_owner_is_not_signalled(
        handoff, root, repo, proc, tmp_path, pidfds, invalid):
    pid = 90001
    kwargs = {}
    if invalid == "other-user":
        kwargs["uid"] = os.getuid() + 1
    elif invalid == "other-root":
        kwargs["argv"] = [sys.executable, "src/selector_pair_gpu.py", "run", "--root", str(tmp_path / "other-root")]
    elif invalid == "other-repo":
        other_repo = tmp_path / "other-repo"
        (other_repo / "src").mkdir(parents=True)
        (other_repo / "src/selector_pair_gpu.py").write_text("# not our checkout\n")
        kwargs["cwd"] = other_repo
    elif invalid == "other-script":
        kwargs["argv"] = [sys.executable, "src/run_mbpp.py", "run", "--root", str(root)]
    else:
        kwargs["mode"] = "prepare"
    process(proc, pid, repo, root, **kwargs)
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))
    before = snapshot(root)
    with lease(root):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert snapshot(root) == before
    assert pidfds.signals == []


@pytest.mark.parametrize("extra", [
    ["--root", "duplicate"],
    ["--unknown-option", "value"],
    ["test"],
])
def test_duplicate_or_unknown_controller_arguments_are_not_signalled(
        handoff, root, repo, proc, pidfds, extra):
    pid = 90001
    suffix = [str(root) if item == "duplicate" else item for item in extra]
    process(proc, pid, repo, root,
            argv=[sys.executable, "src/selector_pair_gpu.py", "run", "--root", str(root), *suffix])
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))
    with lease(root):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert pidfds.signals == []


@pytest.mark.parametrize("mode", ["run", "develop", "test", "freeze"])
def test_verified_owner_receives_only_term_and_preserves_experiment(
        handoff, root, repo, proc, pidfds, mode):
    pid = 90001
    process(proc, pid, repo, root, mode=mode)
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))
    before = snapshot(root)
    with lease(root) as handle:
        def finish(owner):
            assert owner == pid
            fcntl.flock(handle, fcntl.LOCK_UN)
            pidfds.finish(pid)
            (proc / "locks").write_text("")

        pidfds.on_term = finish
        assert handoff.handoff(root, repo, timeout=0.2, proc=proc) == mode
    assert pidfds.signals == [(pid, signal.SIGTERM)]
    assert snapshot(root) == before


def test_controller_exit_without_released_root_lock_refuses_restart(
        handoff, root, repo, proc, pidfds):
    pid = 90001
    process(proc, pid, repo, root)
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))
    pidfds.on_term = pidfds.finish
    before = snapshot(root)
    with lease(root):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert pidfds.signals == [(pid, signal.SIGTERM)]
    assert snapshot(root) == before


def test_controller_ignoring_term_is_not_killed_or_restarted(handoff, root, repo, proc, pidfds):
    pid = 90001
    process(proc, pid, repo, root)
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))
    before = snapshot(root)
    with lease(root):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert pidfds.signals == [(pid, signal.SIGTERM)]
    assert snapshot(root) == before


def test_descendant_still_cleaning_up_prevents_restart(handoff, root, repo, proc, pidfds):
    pid, child = 90001, 90002
    process(proc, pid, repo, root, children=(child,))
    process(proc, child, repo, root, ppid=pid,
            argv=[sys.executable, "src/selector_pair_score.py", "--root", str(root)])
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))
    before = snapshot(root)
    with lease(root) as handle:
        def finish(owner):
            assert owner == pid, "controller's children must be reaped by their normal lifecycle"
            fcntl.flock(handle, fcntl.LOCK_UN)
            pidfds.finish(pid)
            (proc / "locks").write_text("")

        pidfds.on_term = finish
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert child in pidfds.opened
    assert pidfds.signals == [(pid, signal.SIGTERM)]
    assert snapshot(root) == before


def test_controller_and_descendants_exit_before_restart(handoff, root, repo, proc, pidfds):
    pid, child, grandchild = 90001, 90002, 90003
    process(proc, pid, repo, root, children=(child,))
    process(proc, child, repo, root, ppid=pid, children=(grandchild,),
            argv=[sys.executable, "src/selector_pair_score.py", "--root", str(root)])
    process(proc, grandchild, repo, root, ppid=child,
            argv=[sys.executable, "src/selector_pair_score.py", "--root", str(root)])
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))
    before = snapshot(root)
    with lease(root) as handle:
        def finish(owner):
            assert owner == pid
            fcntl.flock(handle, fcntl.LOCK_UN)
            for member in (grandchild, child, pid):
                pidfds.finish(member)
            (proc / "locks").write_text("")

        pidfds.on_term = finish
        assert handoff.handoff(root, repo, timeout=0.2, proc=proc) == "run"
    assert {pid, child, grandchild}.issubset(pidfds.opened)
    assert pidfds.signals == [(pid, signal.SIGTERM)]
    assert snapshot(root) == before


def test_active_training_without_its_own_recoverable_checkpoint_refuses_term(
        handoff, root, repo, proc, pidfds):
    pid, child = 90001, 90002
    output = root / "branches/on_policy/states/s0-t25/points/view-25/selection_reduced/policy"
    (repo / "src/selector_pair_train.py").write_text("# test fixture\n")
    process(proc, pid, repo, root, children=(child,))
    process(proc, child, repo, root, ppid=pid,
            argv=[sys.executable, "src/selector_pair_train.py", "--output", str(output)])
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))
    before = snapshot(root)
    with lease(root):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert pidfds.signals == []
    assert snapshot(root) == before


def test_checkpoint_validation_failure_precedes_any_signal(
        handoff, root, repo, proc, pidfds, monkeypatch):
    pid, child = 90001, 90002
    process(proc, pid, repo, root, children=(child,))
    process(proc, child, repo, root, ppid=pid,
            argv=[sys.executable, "src/selector_pair_train.py", "--output", str(root / "policy")])
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))
    checked = []

    def reject(*args, **kwargs):
        checked.append((args, kwargs))
        raise ValueError("checkpoint identity mismatch")

    monkeypatch.setattr(handoff, "validate_training_checkpoints", reject)
    with lease(root):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert checked
    assert pidfds.signals == []


def test_runtime_validation_failure_precedes_any_signal(
        handoff, root, repo, proc, pidfds, monkeypatch):
    pid = 90001
    process(proc, pid, repo, root)
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))
    checked = []

    def reject(*args):
        checked.append(args)
        raise RuntimeError("frozen runtime mismatch")

    monkeypatch.setattr(handoff, "validate_runtime", reject)
    with lease(root):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert checked
    assert pidfds.signals == []


@pytest.mark.parametrize("host_alias", [False, True])
def test_active_receipt_not_hidden_by_newer_queue_heartbeats(
        handoff, root, repo, proc, host_alias):
    pid = 90001
    directory = process(proc, pid, repo, root)
    host = "run-node-alias" if host_alias else socket.gethostname()
    if host_alias:
        (directory / "environ").write_bytes(b"EXPERIMENTS_NODE_ID=run-node-alias\0")
    owner = handoff.process(proc, pid)
    workers = root / "queue-workers"
    workers.mkdir()
    for index in range(16):
        (workers / f"waiting-{index}.json").write_text(json.dumps({
            "updated": time.time() + index, "host": "peer", "state": "WAIT"}))
    progress = root / "branches/on_policy/states/s0-t25/points/view-25/selection_reduced/progress.json"
    progress.parent.mkdir(parents=True)
    progress.write_text(json.dumps({"updated": time.time() - 10, "host": host,
                                   "pid": pid, "state": "running", "event_id": "owned-event"}))
    rows = handoff.active_receipts(root, owner, {b"OM_SELECTION_COST_owned-event"})
    assert rows == [(progress.parent / "cost-events/owned-event.json", "owned-event")]


@pytest.mark.parametrize('phase', ['train', 'curve'])
def test_owned_phase_receipt_survives_large_admission_and_cost_history(
        handoff, root, repo, proc, phase):
    pid = 90001
    process(proc, pid, repo, root)
    admission = root / 'node-preflight'
    admission.mkdir()
    for index in range(1000):
        path = admission / f'old-{index}'
        path.mkdir()
        (path / 'progress.json').write_text('{"updated": 9999999999, "state": "finished"}')
    directory = root / 'branches/on_policy/states/s2-t50/points/view-50/selection_reduced'
    directory.mkdir(parents=True)
    costs = directory / 'cost-events'
    costs.mkdir()
    for index in range(5000):
        (costs / f'old-{index}.json').touch()
    if phase == 'curve':
        directory = directory / 'curve'
        directory.mkdir()
    (directory / 'progress.json').write_text(json.dumps({
        'updated': 1, 'host': socket.gethostname(), 'pid': pid,
        'state': 'running', 'event_id': 'owned-event', 'phase': phase}))
    rows = handoff.active_receipts(root, handoff.process(proc, pid),
                                  {b'OM_SELECTION_COST_owned-event'})
    assert rows == [(directory / 'cost-events/owned-event.json', 'owned-event')]


def test_process_python_preserves_original_virtualenv_path(handoff, root, repo, proc, tmp_path):
    interpreter = tmp_path / "worker-venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    pid = 90001
    process(proc, pid, repo, root,
            argv=[str(interpreter), "src/selector_pair_gpu.py", "run", "--root", str(root)])
    value = handoff.process(proc, pid)
    assert value["exe"] == Path(sys.executable).resolve()
    assert Path(handoff.process_python(value, {"VIRTUAL_ENV": str(interpreter.parent.parent)})) == interpreter


def test_checkpoint_validator_uses_original_python_and_hides_cuda(
        handoff, root, repo, proc, tmp_path, monkeypatch):
    interpreter = tmp_path / "worker-venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    pid = 90001
    output = root / "branches/on_policy/training/policy"
    directory = process(proc, pid, repo, root,
                        argv=[str(interpreter), "src/selector_pair_train.py", "--output", str(output)])
    (directory / "environ").write_bytes(
        b"CUDA_VISIBLE_DEVICES=0,1,2,3\0OM_SELECTION_COST_test-phase=1\0" +
        os.fsencode(f"VIRTUAL_ENV={interpreter.parent.parent}") + b"\0")
    calls = []

    def completed(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="[checkpoint] validated local saved update 25\n", stderr="")

    monkeypatch.setattr(subprocess, "run", completed)
    handoff.validate_training_checkpoints(root, repo, {pid: handoff.process(proc, pid)})
    assert len(calls) == 1
    (argv,), kwargs = calls[0]
    assert Path(argv[0]) == interpreter
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(repo / "src")
    assert not any(key.startswith("OM_SELECTION_COST_") for key in kwargs["env"])
    assert kwargs["start_new_session"] is True
    assert json.loads(argv[-1]) == ["--output", str(output)]


@pytest.mark.parametrize("returncode", [0, 2])
def test_runtime_validator_is_cpu_only_unmetered_and_uses_original_python(
        handoff, root, repo, proc, tmp_path, monkeypatch, returncode):
    interpreter = tmp_path / "worker-venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    pid = 90001
    directory = process(proc, pid, repo, root,
                        argv=[str(interpreter), "src/selector_pair_gpu.py", "run", "--root", str(root)])
    (directory / "environ").write_bytes(
        b"CUDA_VISIBLE_DEVICES=0,1,2,3\0OM_SELECTION_COST_test-phase=1\0" +
        os.fsencode(f"VIRTUAL_ENV={interpreter.parent.parent}") + b"\0")
    calls = []

    def completed(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=returncode, stdout="runtime check\n", stderr="error detail")

    monkeypatch.setattr(subprocess, "run", completed)
    original = handoff._test_validate_runtime
    assert original is not None
    if returncode:
        with pytest.raises(RuntimeError):
            original(root, repo, handoff.process(proc, pid))
    else:
        original(root, repo, handoff.process(proc, pid))
    assert len(calls) == 1
    (argv,), kwargs = calls[0]
    assert Path(argv[0]) == interpreter
    assert argv[-1] == str(root)
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert not any(key.startswith("OM_SELECTION_COST_") for key in kwargs["env"])
    assert kwargs["start_new_session"] is True


def test_verified_shell_parent_still_cleaning_up_prevents_restart(
        handoff, root, repo, proc, pidfds):
    parent, pid = 90000, 90001
    process(proc, parent, repo, root, children=(pid,),
            argv=["/bin/bash", "scripts/run_selector_pair.sh"])
    process(proc, pid, repo, root, ppid=parent)
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))
    before = snapshot(root)
    with lease(root) as handle:
        def finish(owner):
            assert owner == pid
            fcntl.flock(handle, fcntl.LOCK_UN)
            pidfds.finish(pid)
            (proc / "locks").write_text("")

        pidfds.on_term = finish
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert parent in pidfds.opened
    assert pidfds.signals == [(pid, signal.SIGTERM)]
    assert snapshot(root) == before


def test_pid_reuse_between_identity_read_and_pidfd_open_is_not_signalled(
        handoff, root, repo, proc, pidfds, monkeypatch):
    pid = 90001
    directory = process(proc, pid, repo, root)
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))
    original_open = pidfds.open

    def replace_pid(owner, flags=0):
        fd = original_open(owner, flags)
        fields = (directory / "stat").read_text().split()
        fields[21] = "99999"
        (directory / "stat").write_text(" ".join(fields))
        return fd

    monkeypatch.setattr(os, "pidfd_open", replace_pid)
    with lease(root):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert pidfds.signals == []


def test_pidfd_unavailable_refuses_numeric_pid_fallback(
        handoff, root, repo, proc, pidfds, monkeypatch):
    pid = 90001
    process(proc, pid, repo, root)
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))

    def unavailable(*args, **kwargs):
        raise OSError("pidfd unavailable")

    monkeypatch.setattr(os, "pidfd_open", unavailable)
    with lease(root):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert pidfds.signals == []


def test_owner_identity_changes_during_checkpoint_validation_refuses_term(
        handoff, root, repo, proc, pidfds, monkeypatch):
    pid = 90001
    directory = process(proc, pid, repo, root)
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))

    def changed_owner(*args, **kwargs):
        fields = (directory / "stat").read_text().split()
        fields[21] = "99999"
        (directory / "stat").write_text(" ".join(fields))

    monkeypatch.setattr(handoff, "validate_training_checkpoints", changed_owner)
    with lease(root):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert pidfds.signals == []


def test_new_child_during_checkpoint_validation_refuses_term(
        handoff, root, repo, proc, pidfds, monkeypatch):
    pid, child = 90001, 90002
    directory = process(proc, pid, repo, root)
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))

    def changed_children(*args, **kwargs):
        process(proc, child, repo, root, ppid=pid,
                argv=[sys.executable, "src/selector_pair_score.py"])
        (directory / "task" / str(pid) / "children").write_text(str(child))

    monkeypatch.setattr(handoff, "validate_training_checkpoints", changed_children)
    with lease(root):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert pidfds.signals == []


@pytest.mark.parametrize("finish", [False, True])
def test_owned_cost_event_must_have_finished_receipt_before_restart(
        handoff, root, repo, proc, pidfds, monkeypatch, finish):
    pid = 90001
    process(proc, pid, repo, root)
    (proc / "locks").write_text(owner_record(root / ".pair.lock", pid))
    receipt = root / "cost-events/active-event.json"
    receipt.parent.mkdir()
    progress = dict(event_id='active-event', phase='curve', ledger='reporting',
                    gpus=4, gpu_type='H100', host='local-owner')
    (root / 'progress.json').write_text(json.dumps(progress))
    receipt.write_text(json.dumps({"event_id": "active-event", "state": "running"}))
    monkeypatch.setattr(handoff, "active_receipts",
                        lambda *args: [(receipt, "active-event")])
    before = snapshot(root)
    with lease(root) as handle:
        def stopped(owner):
            assert owner == pid
            fcntl.flock(handle, fcntl.LOCK_UN)
            pidfds.finish(pid)
            (proc / "locks").write_text("")
            if finish:
                receipt.write_text(json.dumps({**progress, 'state': 'finished', 'exit_code': -15,
                                               'seconds': 3., 'allocated_gpu_seconds': 12., 'time': 123.}))

        pidfds.on_term = stopped
        if finish:
            assert handoff.handoff(root, repo, timeout=0.2, proc=proc) == "run"
        else:
            with pytest.raises((ValueError, RuntimeError, OSError)):
                handoff.handoff(root, repo, timeout=0.05, proc=proc)
    assert pidfds.signals == [(pid, signal.SIGTERM)]
    after = snapshot(root)
    if finish:
        before.pop("cost-events/active-event.json")
        after.pop("cost-events/active-event.json")
    assert after == before


def test_cli_missing_root_exits_without_launching_training(tmp_path, repo):
    root = tmp_path / "missing-volume/runs/selector-pair-v1"
    result = subprocess.run([sys.executable, str(SCRIPT), "--root", str(root),
                             "--repo", str(repo), "--timeout", "0.05"],
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode != 0
    assert not root.parent.parent.exists()


def test_main_restarts_exact_same_root_and_stage_only_after_successful_handoff(
        handoff, root, repo, tmp_path, monkeypatch):
    calls, launches = [], []
    staged = tmp_path / "reviewed-runtime"
    staged.mkdir()

    def approved(received_root, received_repo, timeout, *, launch_repo=None, restart_shared=False):
        assert restart_shared is True
        calls.append((received_root, received_repo, timeout, launch_repo))
        return "develop"

    monkeypatch.setattr(handoff, "handoff", approved)
    monkeypatch.setattr(handoff, "stage_runtime", lambda original: staged)
    monkeypatch.setattr(os, "execve", lambda *args: launches.append(args))
    monkeypatch.setenv("E5_FORCE", "1")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--root", str(root), "--repo", str(repo),
                                      "--timeout", "1"])
    handoff.main()
    assert calls == [(root, repo, 1, staged)]
    assert len(launches) == 1
    executable, argv, env = launches[0]
    assert executable == "/bin/bash"
    assert argv == ["bash", str(staged / "scripts/run_selector_pair.sh"), "develop"]
    assert env["PAIR_ROOT"] == str(root)
    assert env["E5_FORCE"] == "0"


def allocation(proc, pid):
    (proc / 'self/ns').mkdir(parents=True, exist_ok=True)
    (proc / str(pid) / 'ns').mkdir(exist_ok=True)
    for name in ('mnt', 'pid'):
        source = proc / 'self/ns' / name
        if not source.exists():
            source.write_bytes(b'namespace')
        target = proc / str(pid) / 'ns' / name
        if not target.exists():
            os.link(source, target)
    for directory in (proc / 'self', proc / str(pid)):
        (directory / 'cgroup').write_text('0::/allocation-one\n')


def test_explicit_shared_restart_stops_verified_local_controller_only(
        handoff, root, repo, proc, pidfds):
    pid = 90001
    process(proc, pid, repo, root)
    allocation(proc, pid)
    (proc / 'locks').write_text(owner_record(root / '.pair.lock', pid, kind='READ'))
    before = snapshot(root)
    with lease(root, fcntl.LOCK_SH):
        pidfds.on_term = lambda target: pidfds.finish(target)
        assert handoff.handoff(root, repo, timeout=.1, proc=proc, restart_shared=True) == 'run'
    assert pidfds.signals == [(pid, signal.SIGTERM)]
    assert snapshot(root) == before


@pytest.mark.parametrize('shell_python', [None, '/wrong-shell-venv/bin/python'])
def test_main_restarts_with_verified_owner_virtualenv_not_shell_default(
        handoff, root, repo, proc, pidfds, monkeypatch, tmp_path, shell_python):
    pid = 90001
    interpreter = tmp_path / 'owner-venv/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    process(proc, pid, repo, root,
            argv=[str(interpreter), 'src/selector_pair_gpu.py', 'develop', '--root', str(root)])
    (proc / str(pid) / 'environ').write_bytes(b'PAIR_PYTHON=/untrusted-environment-path\0')
    (proc / 'locks').write_text(owner_record(root / '.pair.lock', pid, kind='READ'))
    if shell_python is None:
        monkeypatch.delenv('PAIR_PYTHON', raising=False)
    else:
        monkeypatch.setenv('PAIR_PYTHON', shell_python)
    validated, launched = [], []
    monkeypatch.setattr(handoff, 'validate_runtime', lambda root, repo, owner:
        validated.append(handoff.process_python(owner, handoff.process_environment(owner))))
    actual_handoff = handoff.handoff
    monkeypatch.setattr(handoff, 'handoff', lambda *args, **kwargs:
        actual_handoff(*args, **kwargs, proc=proc))
    monkeypatch.setattr(os, 'execve', lambda *args: launched.append(args))
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--root', str(root), '--repo', str(repo), '--timeout', '.1'])
    with lease(root, fcntl.LOCK_SH):
        pidfds.on_term = lambda target: pidfds.finish(target)
        handoff.main()
    assert validated == [str(interpreter)]
    assert pidfds.signals == [(pid, signal.SIGTERM)]
    assert launched[0][2]['PAIR_PYTHON'] == str(interpreter)
    assert launched[0][2]['PAIR_ROOT'] == str(root)
    assert launched[0][1][-1] == 'develop'


def test_owner_interpreter_disappearing_during_validation_aborts_before_term(
        handoff, root, repo, proc, pidfds, monkeypatch, tmp_path):
    pid = 90001
    interpreter = tmp_path / 'owner-venv/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    process(proc, pid, repo, root,
            argv=[str(interpreter), 'src/selector_pair_gpu.py', 'run', '--root', str(root)])
    (proc / 'locks').write_text(owner_record(root / '.pair.lock', pid, kind='READ'))
    monkeypatch.setenv('PAIR_PYTHON', '/original-shell-python')
    monkeypatch.setattr(handoff, 'validate_runtime', lambda *args: interpreter.unlink())
    with lease(root, fcntl.LOCK_SH), pytest.raises((OSError, RuntimeError)):
        handoff.handoff(root, repo, timeout=.1, proc=proc, restart_shared=True)
    assert pidfds.signals == []
    assert os.environ['PAIR_PYTHON'] == '/original-shell-python'


@pytest.mark.parametrize('mismatch', ['namespace', 'cgroup', 'gpu', 'multiple', 'foreign-root'])
@pytest.mark.parametrize('shared', [False, True])
def test_shared_restart_ambiguous_or_foreign_allocation_is_not_stopped(
        handoff, root, repo, proc, pidfds, monkeypatch, mismatch, shared):
    pid = 90001
    process(proc, pid, repo, root if mismatch != 'foreign-root' else root / 'another-run')
    allocation(proc, pid)
    kind = 'READ' if shared else 'WRITE'
    record = owner_record(root / '.pair.lock', pid, kind=kind)
    if mismatch == 'namespace':
        path = proc / str(pid) / 'ns/mnt'
        path.unlink()
        path.write_text('different namespace')
    elif mismatch == 'cgroup':
        (proc / str(pid) / 'cgroup').write_text('0::/different-allocation\n')
    elif mismatch == 'gpu':
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
        (proc / str(pid) / 'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=4,5,6,7\0')
    elif mismatch == 'multiple':
        record += owner_record(root / '.pair.lock', pid + 1, kind=kind)
    (proc / 'locks').write_text(record)
    before = snapshot(root)
    with lease(root, fcntl.LOCK_SH if shared else fcntl.LOCK_EX), pytest.raises(RuntimeError):
        handoff.handoff(root, repo, timeout=.1, proc=proc, restart_shared=True)
    assert pidfds.signals == []
    assert snapshot(root) == before


@pytest.mark.parametrize('name', ['pair-status-runtime.json', 'pair-curve-progress-runtime.json',
                                'pair-branch-queue-runtime.json', 'pair-curve-spawn-runtime.json',
                                'pair-recollection-runtime.json'])
def test_runtime_validation_copies_new_receipts_before_checking(
        handoff, root, repo, proc, name):
    pid = 90001
    process(proc, pid, repo, root)
    (root / name).write_text('{"must_be_copied": true}\n')
    (repo / 'src/selector_pair_gpu.py').write_text(
        'import json\n'
        'def queue_lease(): pass\n'
        'def distributed_stage(): pass\n'
        'def run_distributed(): pass\n'
        'def manifest(root, bind_runtime=False): return {"code_hashes": {}}\n'
        'def bind_startup_runtime(root, hashes):\n'
        f'    assert json.loads((root / {name!r}).read_text()) == {{"must_be_copied": True}}\n')
    handoff._test_validate_runtime(root, repo, handoff.process(proc, pid))


@pytest.mark.parametrize('compatible', [False, True])
def test_runtime_validation_checks_operational_receipts_without_writes(
        handoff, root, repo, proc, compatible):
    pid = 90001
    process(proc, pid, repo, root)
    (root / 'pair-curve-shard-guard-runtime.json').write_text('{"guard": "frozen"}\n')
    (repo / 'src/selector_pair_gpu.py').write_text(
        'def queue_lease(): pass\n'
        'def distributed_stage(): pass\n'
        'def run_distributed(): pass\n'
        'def manifest(root, bind_runtime=False): return {"code_hashes": {}}\n'
        'def bind_startup_runtime(root, hashes): pass\n')
    (repo / 'scripts/queue_selector_pair_gpu.py').write_text(
        'import json\n'
        'def validate_receipts(root, protocol):\n'
        '    assert protocol == {"code_hashes": {}}\n'
        '    assert json.loads((root / "pair-curve-shard-guard-runtime.json").read_text()) == {"guard": "frozen"}\n'
        + ('    raise ValueError("incompatible operational receipt")\n' if not compatible else ''))
    before = snapshot(root)
    if compatible:
        handoff._test_validate_runtime(root, repo, handoff.process(proc, pid))
    else:
        with pytest.raises(RuntimeError, match='live controller was NOT stopped'):
            handoff._test_validate_runtime(root, repo, handoff.process(proc, pid))
    assert snapshot(root) == before


@pytest.mark.parametrize('corrupt', [False, True])
def test_shared_restart_verifies_existing_isolated_runtime_against_its_git_commit(
        handoff, root, repo, proc, pidfds, monkeypatch, corrupt):
    import selector_pair_deploy as deploy
    from test_selector_pair_deploy import git, commit

    git(repo, 'init', '-q')
    (repo / '.gitignore').write_text('.work/\n')
    old = commit(repo, 'old controller fixture')
    monkeypatch.setattr(deploy, 'PINNED_COMMIT', old)
    runtime = deploy.stage_runtime(repo)
    (repo / 'src/selector_pair_gpu.py').write_text('# newer queue fixture\n')
    new = commit(repo, 'new controller fixture')
    monkeypatch.setattr(deploy, 'PINNED_COMMIT', new)
    if corrupt:
        (runtime / 'src/selector_pair_gpu.py').write_text('# unverified modification\n')
    pid = 90001
    process(proc, pid, runtime, root)
    allocation(proc, pid)
    (proc / 'locks').write_text(owner_record(root / '.pair.lock', pid, kind='READ'))
    before = snapshot(root)
    with lease(root, fcntl.LOCK_SH):
        if corrupt:
            with pytest.raises(RuntimeError, match='changed'):
                handoff.handoff(root, repo, timeout=.1, proc=proc, restart_shared=True)
            assert pidfds.signals == []
        else:
            pidfds.on_term = lambda target: pidfds.finish(target)
            assert handoff.handoff(root, repo, timeout=.1, proc=proc, restart_shared=True) == 'run'
            assert pidfds.signals == [(pid, signal.SIGTERM)]
    assert snapshot(root) == before


@pytest.mark.parametrize('damage', ['phase', 'host', 'gpus', 'exit_code', 'seconds', 'allocated_gpu_seconds', 'time'])
def test_handoff_requires_bound_finite_finish_cost_receipt(handoff, damage):
    progress = dict(event_id='event', phase='curve', ledger='reporting',
                    gpus=4, gpu_type='H100', host='node')
    receipt = dict(progress, state='finished', exit_code=-15, seconds=3., allocated_gpu_seconds=12., time=123.)
    assert handoff.valid_finish_receipt(receipt, 'event', progress)
    receipt[damage] = float('nan') if damage in {'seconds', 'allocated_gpu_seconds', 'time'} else 'changed'
    assert not handoff.valid_finish_receipt(receipt, 'event', progress)


def test_restart_retains_verified_owner_gpu_selection(handoff, root, repo, proc, pidfds, monkeypatch):
    pid = 90001
    directory = process(proc, pid, repo, root)
    (directory / 'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=0,1,2,3\0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    (proc / 'locks').write_text(owner_record(root / '.pair.lock', pid, kind='READ'))
    with lease(root, fcntl.LOCK_SH):
        pidfds.on_term = lambda target: pidfds.finish(target)
        assert handoff.handoff(root, repo, timeout=.1, proc=proc, restart_shared=True) == 'run'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1,2,3'


def test_main_does_not_launch_after_handoff_refusal(handoff, root, repo, monkeypatch):
    launches = []

    def refused(*args, **kwargs):
        raise RuntimeError("remote owner not verified")

    monkeypatch.setattr(handoff, "handoff", refused)
    monkeypatch.setattr(handoff, "collect", lambda root: "small diagnostic report")
    monkeypatch.setattr(os, "execve", lambda *args: launches.append(args))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--root", str(root), "--repo", str(repo)])
    assert handoff.main() != 0
    assert launches == []


def test_runtime_staging_failure_does_not_stop_or_restart_controller(handoff, root, repo, monkeypatch):
    calls = []

    def staging_failed(*args, **kwargs):
        raise RuntimeError("reviewed runtime unavailable")

    monkeypatch.setattr(handoff, "stage_runtime", staging_failed)
    monkeypatch.setattr(handoff, "handoff", lambda *args, **kwargs: calls.append("handoff"))
    monkeypatch.setattr(handoff, "collect", lambda root: "small diagnostic report")
    monkeypatch.setattr(os, "execve", lambda *args: calls.append("launch"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--root", str(root), "--repo", str(repo)])
    assert handoff.main() != 0
    assert calls == []


def test_bash_entrypoint_runs_from_any_directory_and_refuses_missing_root(tmp_path, repo):
    root = tmp_path / "missing-volume/runs/selector-pair-v1"
    shell = SCRIPT.with_name("restart_selector_pair.sh")
    result = subprocess.run(["bash", str(shell), "--root", str(root), "--repo", str(repo),
                             "--timeout", "0.05"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode != 0
    assert "[handoff-abort]" in result.stderr
    assert not root.parent.parent.exists()


def test_copied_bash_works_offline_without_helper_files(tmp_path, repo):
    launcher = tmp_path / "copied-restart-selector-pair.sh"
    shutil.copyfile(SCRIPT.with_name("restart_selector_pair.sh"), launcher)
    binaries = tmp_path / "offline-bin"
    binaries.mkdir()
    (binaries / "python3").symlink_to(sys.executable)
    env = dict(os.environ, PATH=str(binaries))
    for name in ("PYTHONPATH", "PAIR_ROOT", "OM_WORK", "OM_USER"):
        env.pop(name, None)
    root = tmp_path / "missing-volume/runs/selector-pair-v1"
    assert not (repo / "scripts/selector_pair_handoff.py").exists()
    assert not (repo / "scripts/selector_pair_diagnostic.py").exists()
    before = snapshot(repo)
    result = subprocess.run(["/bin/bash", str(launcher), "--root", str(root),
                             "--repo", str(repo), "--timeout", "0.05"],
                            cwd=repo, env=env, capture_output=True, text=True,
                            timeout=10, check=False)
    assert result.returncode != 0
    assert "[handoff-abort]" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert "can't open file" not in result.stderr
    assert not root.parent.parent.exists()
    assert snapshot(repo) == before


def test_embedded_bash_payload_matches_maintained_sources():
    result = subprocess.run([sys.executable, str(SCRIPT.with_name("build_selector_pair_handoff.py")),
                             "--check"],
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
