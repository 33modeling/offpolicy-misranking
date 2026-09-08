from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEARTBEAT = ROOT / "src/rlzero_heartbeat.py"


def test_heartbeat_refreshes_and_stops_with_signal(tmp_path: Path) -> None:
    path = tmp_path / "worker.json"
    process = subprocess.Popen(
        [
            sys.executable,
            str(HEARTBEAT),
            "--path",
            str(path),
            "--worker",
            "node-a-worker",
            "--host",
            "node-a",
            "--launcher-pid",
            str(os.getpid()),
            "--interval-seconds",
            "0.05",
        ]
    )
    try:
        deadline = time.monotonic() + 3
        while not path.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        first = json.loads(path.read_text(encoding="utf-8"))
        time.sleep(0.12)
        second = json.loads(path.read_text(encoding="utf-8"))
        assert second["heartbeat_at_ns"] > first["heartbeat_at_ns"]
        assert second["state"] == "running"
        assert second["worker"] == "node-a-worker"
        assert second["schema"] == "offpolicy-worker-heartbeat/v1"
    finally:
        process.terminate()
        process.wait(timeout=3)
    final = json.loads(path.read_text(encoding="utf-8"))
    assert final["state"] == "stopped"


def test_progress_line_reports_the_family_this_worker_holds(tmp_path: Path) -> None:
    """The matrix root holds every node's families, so a worker that has written
    nothing for hours still printed a healthy [progress] line because other nodes
    were writing (2026-09-07). The line must also judge this worker's own family."""
    sys.path.insert(0, str(ROOT / "src"))
    import rlzero_heartbeat

    root = tmp_path / "runs" / "tag"
    (root / ".families").mkdir(parents=True)
    (root / ".families/mbpp-s0.owner.json").write_text(
        json.dumps({"worker": "worker-1", "host": "node-1"}), encoding="utf-8"
    )
    # another node is writing into the same root right now
    busy = root / "family-math500-s3" / "tag-s3-math500-d0"
    (busy / "logs").mkdir(parents=True)
    (busy / "run_config.json").write_text("{}", encoding="utf-8")
    (busy / "rollouts_fresh_train.shard0.partial").write_bytes(b"x" * 4096)
    # this worker's own family wrote nothing for hours
    mine = root / "family-mbpp-s0" / "tag-s0-mbpp-d0"
    (mine / "logs").mkdir(parents=True)
    (mine / "run_config.json").write_text("{}", encoding="utf-8")
    (mine / "rollouts_fresh_train.shard0.partial").write_bytes(b"y" * 1024)
    old = time.time() - 4 * 3600
    for path in list(mine.rglob("*")) + [mine]:
        os.utime(path, (old, old))

    line = rlzero_heartbeat.report_progress(
        root,
        total_points=40,
        worker="worker-1",
        elapsed=10_000.0,
        not_started_grace=600.0,
        alerts_log=None,
        worker_log=None,
        terminal=None,
    )
    assert "this worker: mbpp-s0 NOT TRAINING" in line, line
    assert "[NOT TRAINING]" in line, line

    # a worker that holds no family says nothing extra about itself
    quiet = rlzero_heartbeat.report_progress(
        root,
        total_points=40,
        worker="worker-2",
        elapsed=10_000.0,
        not_started_grace=600.0,
        alerts_log=None,
        worker_log=None,
        terminal=None,
    )
    assert "this worker:" not in quiet, quiet
