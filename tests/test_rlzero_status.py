from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "src/rlzero_status.py"
TAG = "olmo3-test"


def status_command(root: Path, **overrides: int) -> list[str]:
    values = {
        "probe_seconds": 0,
        "stuck_seconds": 30,
        "worker_stale_seconds": 30,
        "expected_workers": 1,
        "log_lines": 5,
        "error_lines": 5,
        **overrides,
    }
    return [
        sys.executable,
        str(STATUS),
        "--profile",
        "h100",
        "--root",
        str(root),
        "--results",
        str(root.parent / "results"),
        "--model-tag",
        TAG,
        "--datasets",
        "math500",
        "--seeds",
        "0",
        "--drifts",
        "0",
        "--probe-seconds",
        str(values["probe_seconds"]),
        "--stuck-seconds",
        str(values["stuck_seconds"]),
        "--worker-stale-seconds",
        str(values["worker_stale_seconds"]),
        "--expected-workers",
        str(values["expected_workers"]),
        "--log-lines",
        str(values["log_lines"]),
        "--error-lines",
        str(values["error_lines"]),
    ]


def active_family(root: Path) -> tuple[Path, Path, object]:
    family = root / "family-math500-s0"
    run = family / f"{TAG}-s0-math500-d0"
    logs = run / "logs"
    logs.mkdir(parents=True)
    (root / ".families").mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    (root / ".families/math500-s0.owner.json").write_text(
        '{"worker":"worker-1","host":"node-1"}\n', encoding="utf-8"
    )
    (root / "logs/worker-1.log").write_text("worker running\n", encoding="utf-8")
    (logs / "fresh-shard0.log").write_text("rollout 1/100\n", encoding="utf-8")
    partial = run / "rollouts_fresh_train.shard0.partial"
    partial.write_text("{}\n", encoding="utf-8")
    lock = (root / ".families/math500-s0.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX)
    return run, partial, lock


def run_status(root: Path, **overrides: int) -> str:
    result = subprocess.run(
        status_command(root, **overrides),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_status_observes_real_progress_and_scans_all_active_logs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs" / TAG
    run, partial, lock = active_family(root)
    old_error = run / "logs/regime-attempt-1.log"
    old_error.write_text(
        "RuntimeError: CUDA error: unspecified launch failure\n", encoding="utf-8"
    )
    old = time.time() - 5
    os.utime(old_error, (old, old))

    def advance() -> None:
        time.sleep(0.2)
        with partial.open("a", encoding="utf-8") as stream:
            stream.write("{}\n")

    updater = threading.Thread(target=advance)
    updater.start()
    try:
        output = run_status(root, probe_seconds=1)
    finally:
        updater.join()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()

    assert "verdict=PROGRESSING reason=artifact_or_log_changed_during_probe" in output
    assert "observed_changes=rollouts_fresh_train.shard0.partial:" in output
    assert "error_evidence_from_all_checked_logs:" in output
    assert "unspecified launch failure" in output
    assert "worker=worker-1 state=CLAIMED claims=math500/s0" in output
    assert "workers_observed=1/1" in output
    assert "overall_verdict=RUNNING" in output


def test_status_distinguishes_alive_stuck_and_dead(tmp_path: Path) -> None:
    alive_root = tmp_path / "alive" / TAG
    _, _, alive_lock = active_family(alive_root)
    try:
        alive = run_status(alive_root)
    finally:
        fcntl.flock(alive_lock, fcntl.LOCK_UN)
        alive_lock.close()
    assert "verdict=ALIVE" in alive
    assert "overall_verdict=RUNNING" in alive

    stuck_root = tmp_path / "stuck" / TAG
    _, _, stuck_lock = active_family(stuck_root)
    old = time.time() - 120
    for path in (
        stuck_root / ".families/math500-s0.owner.json",
        stuck_root / "logs/worker-1.log",
        *list((stuck_root / "family-math500-s0").rglob("*")),
    ):
        if path.is_file():
            os.utime(path, (old, old))
    try:
        stuck = run_status(stuck_root, stuck_seconds=10, worker_stale_seconds=10)
    finally:
        fcntl.flock(stuck_lock, fcntl.LOCK_UN)
        stuck_lock.close()
    assert "verdict=STUCK reason=lock_held_without_activity_for_" in stuck
    assert "overall_verdict=STOPPED" in stuck

    dead_root = tmp_path / "dead" / TAG
    active_family(dead_root)[2].close()
    dead = run_status(dead_root)
    assert "math500/s0 stale-owner" in dead
    assert "verdict=DEAD" in dead
    assert "overall_verdict=STOPPED" in dead


def test_status_marks_missing_workers_as_degraded(tmp_path: Path) -> None:
    root = tmp_path / "runs" / TAG
    _, _, lock = active_family(root)
    try:
        output = run_status(root, expected_workers=3)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    assert "workers_observed=1/3" in output
    assert "overall_verdict=DEGRADED" in output
    assert (
        "recommended_action=inspect_STUCK_DEAD_families_and_missing_workers" in output
    )
