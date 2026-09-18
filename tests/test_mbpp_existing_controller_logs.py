"""Re-running MBPP follows its existing log without touching the live owner."""

import json
import os
import pty
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from test_experiments_safe_resume import controller as controller_fixture
from test_experiments_safe_resume import mbpp_controller as mbpp_controller_fixture

controller = controller_fixture
mbpp_controller = mbpp_controller_fixture
ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_experiments.sh"


def saved_bytes(paths):
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def test_interactive_existing_mbpp_run_follows_only_its_log_and_ctrl_c_preserves_workers(mbpp_controller, tmp_path):
    owner, env, pid_file, saved = mbpp_controller
    log = pid_file.with_name(pid_file.name.replace("launcher.", "console.").replace(".pid", ".log"))
    log.write_text("".join(f"MBPP entry {index:02d}\n" for index in range(60)))
    math_log = log.with_name(log.name.replace(".mbpp.", "."))
    math_log.write_text("UNRELATED MATH LOG\n")
    before = saved_bytes((*saved, pid_file, log, math_log))
    tail_calls = tmp_path / "tail-calls.json"
    tail = Path(env["PATH"].split(os.pathsep)[0]) / "tail"
    real_tail = shutil.which("tail")
    assert real_tail is not None
    tail.write_text(f"#!{sys.executable}\n"
                    "import json, os, sys\nfrom pathlib import Path\n"
                    "Path(os.environ['TAIL_CALLS']).write_text(json.dumps(sys.argv[1:]))\n"
                    "os.execv(os.environ['REAL_TAIL'], [os.environ['REAL_TAIL'], *sys.argv[1:]])\n")
    tail.chmod(0o755)
    master, slave = pty.openpty()
    viewer = subprocess.Popen(
        ["bash", str(LAUNCHER)], cwd=ROOT,
        env={**env, "EXPERIMENTS_MBPP_SUITE": "all", "TAIL_CALLS": str(tail_calls), "REAL_TAIL": real_tail},
        stdin=subprocess.DEVNULL, stdout=slave, stderr=slave, start_new_session=True,
    )
    os.close(slave)
    output = b""
    try:
        deadline = time.monotonic() + 8
        while b"MBPP entry 59" not in output and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                output += os.read(master, 65536)
        assert b"MBPP entry 59" in output, output.decode(errors="replace")
        assert b"MBPP entry 10" in output and b"MBPP entry 09" not in output
        assert b"UNRELATED MATH LOG" not in output
        assert json.loads(tail_calls.read_text()) == ["-n", "50", "-F", f"--pid={owner.pid}", str(log)]
        assert viewer.poll() is None and owner.poll() is None
        # The viewer owns a different process group from the live guard. Model
        # terminal Ctrl-C without ever signalling the guard or its children.
        os.killpg(viewer.pid, signal.SIGINT)
        viewer.wait(timeout=5)
        assert owner.poll() is None
        assert before == saved_bytes(before)
        assert all(marker not in output for marker in (b"[stop]", b"[detached]", b"[fault-reset]", b"[pass "))
    finally:
        if viewer.poll() is None:
            os.killpg(viewer.pid, signal.SIGTERM)
            viewer.wait(timeout=5)
        os.close(master)


def test_noninteractive_existing_mbpp_run_returns_without_following_or_mutating(mbpp_controller):
    owner, env, pid_file, saved = mbpp_controller
    before = saved_bytes((*saved, pid_file))
    result = subprocess.run(["bash", str(LAUNCHER), "run"], cwd=ROOT,
                            env={**env, "EXPERIMENTS_MBPP_SUITE": "all"}, capture_output=True,
                            text=True, timeout=5, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[already running] MBPP" in result.stdout and "[logs]" not in result.stdout
    assert owner.poll() is None and before == saved_bytes(before)
    assert "[stop]" not in result.stdout and "[detached]" not in result.stdout
