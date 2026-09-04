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
