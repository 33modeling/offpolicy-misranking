"""A failed MBPP launch is not an endless WAIT or a successful terminal exit."""

import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _node_view as view
import mbpp_failure_summary as why
from test_mbpp_node_assignments import NOW, dashboard, roots_at

@pytest.mark.parametrize("rc,state,label", [(75, "BLOCKED", "BLOCK"), (78, "BLOCKED", "BLOCK"),
                                         (80, "BLOCKED", "BLOCK"), (2, "FAILED", "FAIL"),
                                         (0, "EXITED", "EXIT")])
def test_controller_exit_is_not_wait_and_keeps_cause(tmp_path, rc, state, label):
    root = roots_at(tmp_path)[0]
    logs = root.parent / "experiments/logs"
    logs.mkdir(parents=True)
    path = logs / "console.mbpp.failed-node_.log"
    path.write_text("[node-launcher-start] host=failed-node pid=123\n"
                    "[blocked] actual-current-failure\n"
                    "[mbpp-clean] previous owned processes stopped=0; remaining=0\n"
                    f"[node-launcher-exit] pid=123 rc={rc} owner=mbpp-guard\n")
    os.utime(path, (NOW - 5, NOW - 5))
    before = path.read_bytes()
    nodes = view.launcher_nodes(root, [], now=NOW, node_namespace="mbpp")
    assert nodes[0]["state"] == state
    rendered = " ".join(dashboard.render(dashboard.snapshot([root], now=NOW), width=240).split())
    assert f"failed-node 배정 없음 {label}" in rendered
    assert "failed-node 배정 없음 WAIT" not in rendered
    assert f"rc={rc}" in rendered
    if rc:
        assert "actual-current-failure" in rendered
    assert before == path.read_bytes()


def test_reason_does_not_reuse_a_previous_controller_failure():
    lines = ["[abort] previous failure", "[node-launcher-start] pid=456",
             "[mbpp-clean] cleanup complete", "[node-launcher-exit] pid=456 rc=2"]
    assert "previous failure" not in view.exit_detail(lines)
    assert view.exit_detail(lines) == "rc=2: controller failed"


def test_review_exit_explicitly_says_not_complete():
    assert "NOT complete" in view.exit_detail(["[node-launcher-exit] pid=2 rc=80"])
    assert view.classify("[WAIT] only review branches remain", node_launcher=True) == "WAIT"


def test_why_preserves_exit_cause_before_cleanup_noise(tmp_path):
    logs = tmp_path / "runs/experiments/logs"
    logs.mkdir(parents=True)
    (logs / "console.mbpp.node_.log").write_text(
        "[node-launcher-start] pid=123\n[abort] actual startup failure\n"
        + "[mbpp-clean] cleanup detail\n" * 30
        + "[node-launcher-exit] pid=123 rc=2 owner=mbpp-guard\n")
    text = why.report(tmp_path, [])
    assert "EXIT rc=2" in text and "actual startup failure" in text
    assert len(text.encode()) <= why.MAX_BYTES


def test_why_does_not_hide_failed_node_behind_two_active_workers(tmp_path):
    logs = tmp_path / "runs/experiments/logs"
    logs.mkdir(parents=True)
    for name, age, text in (("failed", 60, "[node-launcher-exit] pid=1 rc=78\n"),
                            ("busy-a", 2, "[gate] curve in progress\n"),
                            ("busy-b", 1, "[gate] curve in progress\n")):
        path = logs / f"console.mbpp.{name}_.log"
        path.write_text(text)
        os.utime(path, (NOW-age, NOW-age))
    text = why.report(tmp_path, [])
    assert "NODE console.mbpp.failed_.log" in text
    assert "GPU admission failed" in text
    assert "NODE console.mbpp.busy-b_.log" in text
    assert text.count("\nNODE ") == 2


def detached_fragment():
    source = (ROOT / "scripts/run_experiments.sh").read_text()
    start = source.index('if [ -t 1 ] && [ "${EXPERIMENTS_DETACHED:-0}" != 1 ]; then')
    stop = source.index('\nif [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ] && [ "${MBPP_GUARD_PID:-}"', start)
    return 'set -euo pipefail\n' + source[start:stop]


@pytest.mark.parametrize("rc", [0, 2, 75, 78, 80])
def test_interactive_launch_returns_actual_controller_exit(tmp_path, rc):
    controller = tmp_path / "controller.sh"
    controller.write_text(f'#!/usr/bin/env bash\necho "[node-launcher-exit] pid=$$ rc={rc}"\nexit {rc}\n')
    logs = tmp_path / "logs"
    env = {**os.environ, "LOG_DIR": str(logs), "CONSOLE_LOG": str(logs / "console.log"),
           "LAUNCHER_SELF": str(controller), "HOST": "test-node", "PID_FILE": str(logs / "pid"),
           "EXPERIMENTS_MBPP_SUITE": "all", "EXPERIMENTS_DETACHED": "0"}
    master, slave = pty.openpty()
    process = subprocess.Popen(["bash", "-c", detached_fragment()], env=env,
                               stdout=slave, stderr=slave, stdin=subprocess.DEVNULL, start_new_session=True)
    os.close(slave)
    output = b""
    try:
        deadline = time.monotonic() + 10
        while process.poll() is None and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], .1)
            if ready:
                try:
                    output += os.read(master, 65536)
                except OSError:
                    break
        assert process.wait(timeout=2) == rc, output.decode(errors="replace")
        assert f"rc={rc}" in (logs / "console.log").read_text()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        os.close(master)


def test_closing_detached_viewer_does_not_wait_for_or_kill_worker(tmp_path):
    marker = tmp_path / "child.pid"
    controller = tmp_path / "controller.sh"
    controller.write_text('#!/usr/bin/env bash\necho "$$" > "$MARKER"\nexec sleep 60\n')
    binaries = tmp_path / "bin"
    binaries.mkdir()
    tail = binaries / "tail"
    tail.write_text('#!/usr/bin/env bash\nfor i in {1..50}; do [ ! -f "$MARKER" ] || exit 130; sleep .02; done\nexit 1\n')
    tail.chmod(0o755)
    logs = tmp_path / "logs"
    env = {**os.environ, "LOG_DIR": str(logs), "CONSOLE_LOG": str(logs / "console.log"),
           "LAUNCHER_SELF": str(controller), "HOST": "test-node", "PID_FILE": str(logs / "pid"),
           "MARKER": str(marker), "EXPERIMENTS_MBPP_SUITE": "all", "EXPERIMENTS_DETACHED": "0",
           "PATH": str(binaries) + os.pathsep + os.environ["PATH"]}
    master, slave = pty.openpty()
    process = subprocess.Popen(["bash", "-c", detached_fragment()], env=env,
                               stdout=slave, stderr=slave, stdin=subprocess.DEVNULL, start_new_session=True)
    os.close(slave)
    try:
        assert process.wait(timeout=5) == 130
        os.kill(int(marker.read_text()), 0)
    finally:
        if marker.exists():
            os.killpg(int(marker.read_text()), signal.SIGTERM)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        os.close(master)
