"""Real CPU processes reproduce late shell exports and inherited node locks."""

import fcntl
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from cleanup_run_processes import _read_process, matching_processes, terminate
from test_generalization_launcher import checkout

RUN_ID = "qwen35-9b-posttrained-math-code-grpo-v1"
PATTERN = "scripts/run_additional_experiments.sh --run qwen35 "


def wait_file(path):
    deadline = time.monotonic() + 5
    while not path.exists():
        assert time.monotonic() < deadline, path
        time.sleep(0.02)


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def test_restart_releases_real_locks_when_launcher_exported_work_after_start(tmp_path):
    repo, env = checkout(tmp_path)
    work = Path(env["TEST_WORK"])
    locks = Path(env["OM_LOCAL_LOCK_DIR"])
    locks.mkdir()
    legacy = tmp_path / "legacy/scripts/run_additional_experiments.sh"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('''#!/usr/bin/env bash
export OM_WORK="$TEST_WORK"
exec 9>"$OM_LOCAL_LOCK_DIR/additional-suite.lock"
flock 9
exec 8>"$OM_LOCAL_LOCK_DIR/primary.lock"
flock 8
sleep 120 &
echo $! > "$TEST_WORK/old-child.pid"
touch "$TEST_WORK/old-ready"
wait
''')
    environment = {key: value for key, value in env.items() if key != "OM_WORK"}
    old = subprocess.Popen(["bash", str(legacy), "--run", "qwen35"], env=environment,
                           start_new_session=True)
    other = subprocess.Popen(["sleep", "120"], env={**env, "OM_WORK": str(work),
        "REGIME_ROOT": str(work / "runs/olmo3-healthy")}, start_new_session=True)
    try:
        wait_file(work / "old-ready")
        assert "OM_WORK" not in _read_process(old.pid).environ
        kwargs = dict(run_prefix=str(work / "runs" / RUN_ID), command_patterns=(PATTERN,),
                      required_environment=(("OM_WORK", str(work)),))
        assert old.pid not in matching_processes(**kwargs)
        assert old.pid in matching_processes(**kwargs, launcher_environment_from_child=True)
        inodes = {name: (locks / name).stat().st_ino for name in ("primary.lock", "additional-suite.lock")}
        with (locks / "additional-suite.lock").open("a") as stream:
            with pytest.raises(BlockingIOError):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["bash", "scripts/run_qwen35_9b.sh", "restart-idle"],
                                cwd=repo, env=environment, capture_output=True, text=True, timeout=25)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"pid={old.pid}" in result.stdout
        assert "node locks are available" in result.stdout
        assert old.wait(timeout=5) != 0
        assert _read_process(int((work / "old-child.pid").read_text())) is None
        assert other.poll() is None
        assert len((work / "phases").read_text().splitlines()) == 1
        for name, inode in inodes.items():
            assert (locks / name).stat().st_ino == inode
            with (locks / name).open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        stop(old)
        stop(other)


def test_orphan_that_ignores_term_is_tracked_until_lock_release(tmp_path):
    lock = tmp_path / "additional-suite.lock"
    prefix = tmp_path / "runs" / RUN_ID
    child = tmp_path / "child.sh"
    child.write_text('''#!/usr/bin/env bash
trap '' TERM
echo "$BASHPID" > "$TEST_ROOT/child.pid"
sleep 120 &
wait
''')
    parent = subprocess.Popen(["bash", "-c", '''
exec 9>"$TEST_ROOT/additional-suite.lock"
flock 9
env -u REGIME_ROOT bash "$TEST_ROOT/child.sh" &
wait
'''], env={**os.environ, "TEST_ROOT": str(tmp_path), "REGIME_ROOT": str(prefix)},
        start_new_session=True)
    try:
        wait_file(tmp_path / "child.pid")
        child_pid = int((tmp_path / "child.pid").read_text())
        assert "REGIME_ROOT" not in _read_process(child_pid).environ
        selected = terminate(str(prefix), timeout=0.3)
        assert child_pid in {process.pid for process in selected}
        assert parent.wait(timeout=5) != 0
        assert _read_process(child_pid) is None
        with lock.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        stop(parent)


def test_late_exports_from_another_work_root_do_not_authorize_cleanup(tmp_path):
    script = tmp_path / "scripts/run_additional_experiments.sh"
    script.parent.mkdir()
    script.write_text('export OM_WORK="$TEST_WORK"\nsleep 120 &\ntouch "$TEST_WORK/ready"\nwait\n')
    work = tmp_path / "other-work"
    work.mkdir()
    env = {key: value for key, value in os.environ.items() if key != "OM_WORK"}
    process = subprocess.Popen(["bash", str(script), "--run", "qwen35"],
                               env={**env, "TEST_WORK": str(work)}, start_new_session=True)
    try:
        wait_file(work / "ready")
        assert process.pid not in matching_processes(
            str(tmp_path / "our-work/runs" / RUN_ID), (PATTERN,),
            (("OM_WORK", str(tmp_path / "our-work")),), launcher_environment_from_child=True)
        assert process.poll() is None
    finally:
        stop(process)


def test_old_unscoped_session_child_matches_only_qwen9_work_namespace(tmp_path):
    work = tmp_path / "work"
    session_prefix = str(work / "console-logs/additional-qwen35-run-")
    env = {**os.environ, "OM_WORK": str(work)}
    env.pop("REGIME_ROOT", None)
    orphan = subprocess.Popen(["sleep", "120"], env={**env, "SESSION_LOG": session_prefix + "old.log"},
                              start_new_session=True)
    other = subprocess.Popen(["sleep", "120"], env={**env,
        "SESSION_LOG": str(work / "console-logs/additional-qwen35_2b-run-old.log")}, start_new_session=True)
    try:
        kwargs = dict(run_prefix=str(work / "runs" / RUN_ID),
                      required_environment=(("OM_WORK", str(work)),))
        assert orphan.pid not in matching_processes(**kwargs)
        selected = terminate(**kwargs, timeout=0.3, session_log_prefix=session_prefix)
        assert orphan.pid in {process.pid for process in selected}
        assert orphan.wait(timeout=5) != 0
        assert other.poll() is None
    finally:
        stop(orphan)
        stop(other)
