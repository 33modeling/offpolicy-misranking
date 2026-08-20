"""Stale v4 launchers and workers must not retain GPU children on restart."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from cleanup_run_processes import terminate


def assert_terminated(process: subprocess.Popen[bytes], label: str) -> None:
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise AssertionError(f"{label} survived cleanup")


with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    fake_scripts = tmp / "scripts"
    fake_scripts.mkdir()
    launcher = fake_scripts / "go_v4.sh"
    launcher.write_text("#!/usr/bin/env bash\nsleep 300\n")

    old_launcher = subprocess.Popen(["bash", str(launcher)])
    time.sleep(0.1)
    assert terminate(str(tmp / "runs" / "v4-"), timeout=1)
    assert_terminated(old_launcher, "stale v4 launcher")

    environment = os.environ.copy()
    environment["RUN_LABEL"] = "v4-cleanup-test"
    old_worker = subprocess.Popen(["sleep", "300"], env=environment)
    time.sleep(0.1)
    assert terminate(str(tmp / "runs" / "v4-"), timeout=1)
    assert_terminated(old_worker, "stale v4 worker")

print("PASS stale v4 launchers and workers are terminated")
