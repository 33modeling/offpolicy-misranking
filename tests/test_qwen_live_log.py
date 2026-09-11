"""Phone-sized live-log command is shell-only and never enters a launch path."""

import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


def checkout(tmp_path, fake_tail=True):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("run_qwen35_9b.sh", "log_qwen35.sh"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    for name in ("setup_env.sh", "run_additional_experiments.sh"):
        (scripts / name).write_text('touch "$OM_WORK/unexpected-mutation"; exit 99\n')
    logs = tmp_path / "work with spaces/console-logs"
    logs.mkdir(parents=True)
    bins = tmp_path / "bin"
    bins.mkdir()
    commands = {"hostname": 'echo node-three', "git": 'echo unexpected-git >&2; exit 99'}
    if fake_tail:
        commands["tail"] = "printf '%s\\n' \"$@\""
    for name, command in commands.items():
        path = bins / name
        path.write_text("#!/bin/sh\n" + command + "\n")
        path.chmod(0o755)
    env = {**os.environ, "OM_WORK": str(logs.parent), "PATH": str(bins) + os.pathsep + os.environ["PATH"]}
    return repo, logs, env


def log_file(logs, name, host, mtime):
    path = logs / name
    path.write_text(f"[launch] profile=qwen35 mode=--run host={host} pid=123 git=test\ninitial-output\n")
    os.utime(path, (mtime, mtime))
    return path


@pytest.mark.parametrize("mode", ["log", "logs", "live"])
def test_short_command_follows_latest_local_session_without_setup_or_git(tmp_path, mode):
    repo, logs, env = checkout(tmp_path)
    log_file(logs, "additional-qwen35-run-old.log", "node-three", 100)
    expected = log_file(logs, "additional-qwen35-run-local.log", "node-three", 200)
    log_file(logs, "additional-qwen35-run-remote.log", "node-one", 300)
    log_file(logs, "additional-qwen35-prepare-ignore.log", "node-three", 400)
    log_file(logs, "additional-qwen35_2b-run-ignore.log", "node-three", 500)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in logs.iterdir()}
    result = subprocess.run(["bash", "scripts/run_qwen35_9b.sh", mode], cwd=repo, env=env,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[-5:] == ["-n", "80", "-F", "--", str(expected)]
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in logs.iterdir()}
    assert not (logs.parent / "unexpected-mutation").exists()
    assert not result.stderr


def test_login_node_follows_latest_shared_session(tmp_path):
    repo, logs, env = checkout(tmp_path)
    log_file(logs, "additional-qwen35-run-old.log", "node-one", 100)
    expected = log_file(logs, "additional-qwen35-run-new.log", "node-two", 200)
    result = subprocess.run(["bash", "scripts/run_qwen35_9b.sh", "log"], cwd=repo, env=env,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert "newest shared Qwen session" in result.stdout
    assert result.stdout.splitlines()[-1] == str(expected)


def test_missing_logs_and_extra_arguments_do_not_start_a_worker(tmp_path):
    repo, logs, env = checkout(tmp_path)
    for args, expected in [(["log"], 1), (["log", "extra"], 2)]:
        result = subprocess.run(["bash", "scripts/run_qwen35_9b.sh", *args], cwd=repo, env=env,
                                capture_output=True, text=True, timeout=5)
        assert result.returncode == expected
    assert not (logs.parent / "unexpected-mutation").exists()


def test_real_tail_streams_appends_and_interrupt_does_not_stop_experiment(tmp_path):
    repo, logs, env = checkout(tmp_path, fake_tail=False)
    path = log_file(logs, "additional-qwen35-run-active.log", "node-three", time.time())
    experiment = subprocess.Popen(["sleep", "120"], start_new_session=True)
    viewer = subprocess.Popen(["bash", "scripts/run_qwen35_9b.sh", "log"], cwd=repo, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    seen = b""
    def wait_output(token):
        nonlocal seen
        deadline = time.monotonic() + 5
        while token not in seen:
            assert time.monotonic() < deadline, seen
            ready, _, _ = select.select([viewer.stdout], [], [], 0.1)
            if ready:
                chunk = os.read(viewer.stdout.fileno(), 4096)
                assert chunk, seen
                seen += chunk
    try:
        wait_output(b"initial-output")
        with path.open("a") as stream:
            stream.write("new-live-progress\n")
        wait_output(b"new-live-progress")
        viewer.send_signal(signal.SIGINT)
        assert viewer.wait(timeout=5) == -signal.SIGINT
        assert experiment.poll() is None
    finally:
        for process in (viewer, experiment):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
