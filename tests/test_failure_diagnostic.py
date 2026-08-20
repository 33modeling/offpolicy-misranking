"""Failure diagnostics must expose errors from console and nested stage logs."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


root = Path(__file__).resolve().parents[1]
script = root / "scripts" / "diagnose_run_failure.sh"

with tempfile.TemporaryDirectory() as raw_tmp:
    tmp = Path(raw_tmp)
    run = tmp / "smoke"
    logs = run / "logs"
    logs.mkdir(parents=True)
    console = tmp / "console.log"
    console.write_text("[watchdog] restart\nCUDA out of memory\n")
    (logs / "drift.log").write_text("Traceback (most recent call last):\nRuntimeError: failed\n")

    proc = subprocess.run(
        ["bash", str(script), str(run), str(console), "137"],
        check=True,
        capture_output=True,
        text=True,
    )
    report = (run / "FAILURE_DIAGNOSTIC.txt").read_text()
    combined = proc.stdout + report
    assert "exit_code=137" in combined
    assert "CUDA out of memory" in combined
    assert "Traceback" in combined
    assert "missing report.json" in combined

print("PASS failure diagnostics expose nested root causes")
