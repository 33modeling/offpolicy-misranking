"""Real local processes: only a lock-waiting launch child of a coordinator ends."""

import fcntl
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

COORDINATOR = """
import os, subprocess, sys
mode, result = sys.argv[1], sys.argv[2]
child = subprocess.Popen(["sh", "-c", f"exec {sys.executable} -m srgc_research.dispatch.gemma_it_run " + mode],
                         start_new_session=True)
code = child.wait()
open(result, "w").write(str(128 - code if code < 0 else code))
"""

CHILD = """
import fcntl, os, subprocess, sys, time
mode = sys.argv[1]
lock = open(os.environ["FAKE_LOCK"], "a+")
if mode == "torchrun":
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "torch.distributed.run"])
if mode == "sleep":
    time.sleep(60)
else:
    fcntl.flock(lock, fcntl.LOCK_EX)
    time.sleep(60)
"""


@pytest.fixture
def fake(tmp_path):
    package = tmp_path / "pkg" / "srgc_research" / "dispatch"
    package.mkdir(parents=True)
    for folder in (package.parent, package):
        (folder / "__init__.py").write_text("")
    (package / "model_chain.py").write_text(textwrap.dedent(COORDINATOR))
    (package / "gemma_it_run.py").write_text(textwrap.dedent(CHILD))
    lock = tmp_path / ".start-prepare.lock"
    holder = open(lock, "a+")
    fcntl.flock(holder, fcntl.LOCK_EX)
    holders = [holder]
    environment = {**os.environ, "PYTHONPATH": str(tmp_path / "pkg"), "FAKE_LOCK": str(lock)}
    processes = []

    def coordinator(mode):
        child_env = dict(environment)
        if mode == 'session':
            path = tmp_path / '.gemma-it.session.lock'
            locked = open(path, 'a+')
            fcntl.flock(locked, fcntl.LOCK_EX)
            holders.append(locked)
            child_env['FAKE_LOCK'] = str(path)
        result = tmp_path / f"{mode}.code"
        processes.append(subprocess.Popen(
            [sys.executable, "-m", "srgc_research.dispatch.model_chain", mode, str(result)],
            env=child_env, cwd=tmp_path))
        return result

    coordinator.processes = processes
    yield coordinator
    # Only this test's own processes, by PID: each coordinator's child group.
    for process in processes:
        for path in Path("/proc").iterdir():
            try:
                if path.name.isdecimal() and int((path / "stat").read_text().rsplit(") ", 1)[1].split()[1]) == process.pid:
                    os.killpg(int(path.name), 9)
            except (OSError, ValueError, IndexError):
                continue
        if process.poll() is None:
            process.kill()
        process.wait()
    for handle in holders:
        handle.close()


def waiting_children(module, count, deadline=20):
    end = time.time() + deadline
    while time.time() < end:
        rows = module.candidates()
        if len(rows) >= count:
            return rows
        time.sleep(.2)
    return module.candidates()


@pytest.mark.parametrize('wait_mode', ['wait', 'session'])
def test_only_the_lock_waiting_launch_child_is_ended(fake, monkeypatch, wait_mode):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import srgc_launch_unstick as module
    finally:
        sys.path.pop(0)
    original = module.candidates
    # Scope even a broken recovery implementation to this test's processes.
    waiter = fake(wait_mode)
    sleeper = fake("sleep")
    training = fake("torchrun")
    coordinator_pids = {process.pid for process in fake.processes}
    monkeypatch.setattr(module, 'candidates', lambda: [row for row in original()
                        if row['coordinator'] in coordinator_pids])
    rows = waiting_children(module, 1)
    time.sleep(2)  # the sleeping and torchrun children are fully started
    rows = module.candidates()
    suffix = '.session.lock' if wait_mode == 'session' else '.start-prepare.lock'
    assert len(rows) == 1 and rows[0]["lock"].endswith(suffix)
    assert module.main([]) == 0
    end = time.time() + 20
    while not waiter.exists() and time.time() < end:
        time.sleep(.2)
    # The coordinator sees 137, which the model chain retries with new code.
    assert waiter.read_text() == "137"
    assert not sleeper.exists() and not training.exists()
    assert module.candidates() == []


def test_help_and_dry_run_never_send_a_signal(monkeypatch):
    from scripts import srgc_launch_unstick as module

    monkeypatch.setattr(module.os, 'killpg', lambda *args: pytest.fail('help/dry-run sent a signal'))
    monkeypatch.setattr(module, 'candidates', lambda: pytest.fail('help inspected processes'))
    with pytest.raises(SystemExit) as result:
        module.main(['--help'])
    assert result.value.code == 0
    monkeypatch.setattr(module, 'candidates', lambda: [{'pid': 123, 'lock': 'fixture.lock'}])
    assert module.main(['--dry-run']) == 0


def test_a_changed_process_identity_is_not_signalled(monkeypatch):
    from scripts import srgc_launch_unstick as module

    row = dict(pid=123, pgid=123, start_ticks='1', leader_start_ticks='1', coordinator=120,
               coordinator_start_ticks='1', lock='fixture.lock', wait='locks_lock_inode_wait')
    rows = iter([[row], [{**row, 'start_ticks': '2'}]])
    monkeypatch.setattr(module, 'candidates', lambda: next(rows))
    monkeypatch.setattr(module.os, 'killpg', lambda *args: pytest.fail('PID identity changed'))
    assert module.main([]) == 0


@pytest.mark.parametrize('waiting', [False, True])
def test_an_open_session_fd_is_not_a_wait_without_kernel_evidence(tmp_path, waiting):
    from scripts import srgc_launch_unstick as module

    proc = tmp_path/'proc'
    fd = proc/'123/fd/7'
    fd.parent.mkdir(parents=True)
    lock = tmp_path/'.gemma.session.lock'
    lock.touch()
    fd.symlink_to(lock)
    stat = lock.stat()
    device = f'{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}'
    (proc/'locks').write_text(f'1: {"-> " if waiting else ""}FLOCK ADVISORY READ 123 {device} 0 EOF\n')
    assert module.startup_lock(proc, 123) == (str(lock) if waiting else None)


def test_shell_wrapper_forwards_dry_run_instead_of_applying_it(tmp_path):
    executable = tmp_path/'python3'
    result = tmp_path/'arguments'
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$UNSTICK_ARGS_FILE"\n')
    executable.chmod(0o755)
    env = {**os.environ, 'PATH': str(tmp_path)+os.pathsep+os.environ.get('PATH', ''),
           'UNSTICK_ARGS_FILE': str(result)}
    subprocess.run(['sh', str(ROOT/'scripts/unstick_gemma_launch.sh'), '--dry-run'],
                   env=env, cwd=ROOT, check=True)
    assert result.read_text().splitlines() == ['scripts/srgc_launch_unstick.py', '--dry-run']
