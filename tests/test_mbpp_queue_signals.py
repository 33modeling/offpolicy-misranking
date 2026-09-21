"""Queue entrypoint signals must unwind the real meter and its owned children."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

REPO = Path(__file__).resolve().parents[1]
CHILD = """
import fcntl, os, pathlib, sys, time
root = pathlib.Path(sys.argv[1])
with (root / 'shard-0.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    (root / 'child-pid.tmp').write_text(str(os.getpid()))
    (root / 'child-pid.tmp').replace(root / 'child-pid')
    time.sleep(60)
"""
CONTROLLER = """
import pathlib, runpy, sys
import selection_switch_gpu as worker
root, child = pathlib.Path(sys.argv[1]), sys.argv[2]
def main():
    with worker.base.lease(root / '.task.lock'):
        return worker.base.meter(root, 'cpu-signal-fixture', 'CPU fixture',
            commands=[([sys.executable, '-c', child, str(root)], '')],
            env={}, timeout=60, devices=0)
worker.main = main
sys.argv = [str(pathlib.Path.cwd() / 'scripts/queue_selection_switch_gpu.py'), 'run', '--root', str(root)]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_queue_signal_closes_cost_and_terminates_owned_shard(tmp_path, signum):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join((str(REPO / "src"), str(REPO / "scripts")))}
    process = subprocess.Popen([sys.executable, "-c", CONTROLLER, str(tmp_path), CHILD],
                               cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child_pid = None
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and process.poll() is None:
            if (tmp_path / "child-pid").exists():
                child_pid = int((tmp_path / "child-pid").read_text())
                break
            time.sleep(.01)
        assert child_pid is not None, "CPU meter child never acquired its shard lease"
        shard = tmp_path / "shard-0.lock"
        inode = shard.stat().st_ino
        with shard.open("rb") as handle, pytest.raises(BlockingIOError):
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode != 0
        events = [json.loads(line) for line in (tmp_path / "cost.jsonl").read_text().splitlines()]
        assert [row["state"] for row in events] == ["started", "finished"], (stdout, stderr)
        start, finish = events
        assert start["event_id"] == finish["event_id"]
        assert finish["exit_code"] != 0 and finish["seconds"] > 0
        receipt = tmp_path / "cost-events" / f"{start['event_id']}.json"
        assert json.loads(receipt.read_text()) == finish
        assert json.loads((tmp_path / "progress.json").read_text())["state"] == "failed"
        for path in (shard, tmp_path / ".task.lock", tmp_path / ".cost.lock"):
            with path.open("rb") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert shard.stat().st_ino == inode
        proc_stat = Path(f"/proc/{child_pid}/stat")
        assert not proc_stat.exists() or proc_stat.read_text().rsplit(") ", 1)[1].split()[0] == "Z"
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
