"""Runtime probes must fail conservatively while workers run concurrently."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cleanup_run_processes import _read_process


@pytest.mark.parametrize(
    "rows, expected_rc, expected_peak",
    [
        ("0 101 C - -", 1, ""),
        ("0 101 C N/A -", 1, ""),
        ("0 101 C 0 0", 0, "0"),
        ("0 101 C 73 0", 0, "73"),
        ("0 101 C 0 0\n1 102 C - -", 1, ""),
        ("0 999 C - -\n1 101 C 0 0", 0, "0"),
        ("", 0, "0"),
    ],
)
def test_gpu_probe_rejects_unavailable_target_utilization(
    rows, expected_rc, expected_peak
):
    script = (ROOT / "scripts/run_matrix.sh").read_text()
    function = script.split("gpu_peak_util() {", 1)[1].split(
        "\nterminate_process_group()", 1
    )[0]
    harness = """
set -uo pipefail
WATCH_GPU_SAMPLES=1
ps() { printf '101 42\n102 42\n999 99\n'; }
timeout() { printf '# gpu pid type sm mem\n%s\n' "$TEST_PMON_ROWS"; }
""" + "gpu_peak_util() {" + function + "\ngpu_peak_util 42\n"
    result = subprocess.run(
        ["bash", "-c", harness],
        env={**os.environ, "TEST_PMON_ROWS": rows},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == expected_rc, result.stderr
    assert result.stdout.strip() == expected_peak


def test_process_snapshot_tolerates_descriptor_disappearing(monkeypatch):
    original = os.readlink

    def race(path, *args, **kwargs):
        if str(path).startswith(f"/proc/{os.getpid()}/fd/"):
            raise OSError(22, "Invalid argument")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", race)
    assert _read_process(os.getpid()) is None
