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
    environment = {**os.environ, "PYTHONPATH": str(tmp_path / "pkg"), "FAKE_LOCK": str(lock)}
    processes = []

    def coordinator(mode):
        result = tmp_path / f"{mode}.code"
        processes.append(subprocess.Popen(
            [sys.executable, "-m", "srgc_research.dispatch.model_chain", mode, str(result)],
            env=environment, cwd=tmp_path))
        return result

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
    holder.close()


def waiting_children(module, count, deadline=20):
    end = time.time() + deadline
    while time.time() < end:
        rows = module.candidates()
        if len(rows) >= count:
            return rows
        time.sleep(.2)
    return module.candidates()


def test_only_the_lock_waiting_launch_child_is_ended(fake):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import srgc_launch_unstick as module
    finally:
        sys.path.pop(0)
    waiter = fake("wait")
    sleeper = fake("sleep")
    training = fake("torchrun")
    rows = waiting_children(module, 1)
    time.sleep(2)  # the sleeping and torchrun children are fully started
    rows = module.candidates()
    assert len(rows) == 1 and rows[0]["lock"].endswith(".start-prepare.lock")
    assert module.main() == 0
    end = time.time() + 20
    while not waiter.exists() and time.time() < end:
        time.sleep(.2)
    # The coordinator sees 137, which the model chain retries with new code.
    assert waiter.read_text() == "137"
    assert not sleeper.exists() and not training.exists()
    assert module.candidates() == []
