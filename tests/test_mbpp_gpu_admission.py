"""Run the real shell occupancy gate with a fake driver; no GPU access."""

import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode,expected", [
    ("ready", 0), ("busy", 75), ("driver-error", 78),
    ("invalid", 78), ("missing-gpu", 78), ("enumeration-error", 78),
])
def test_mbpp_driver_failure_is_admission_failure_not_a_holding_retry(tmp_path, mode, expected):
    driver = tmp_path / "nvidia-smi"
    driver.write_text('''#!/usr/bin/env bash
case "$*" in
  *query-gpu=index*)
    [ "$TEST_DRIVER_MODE" != enumeration-error ] || exit 1
    printf '0\\n1\\n2\\n3\\n'; exit 0 ;;
esac
case "$TEST_DRIVER_MODE" in
  ready) printf '0\\n0\\n0\\n0\\n' ;;
  busy) printf '10000\\n0\\n0\\n0\\n' ;;
  driver-error) echo 'CUDA driver unavailable' >&2; exit 1 ;;
  invalid) echo unknown ;;
  missing-gpu) printf '0\\n0\\n0\\n' ;;
esac
''')
    driver.chmod(0o755)
    source = (ROOT / "scripts/run_selection_switch.sh").read_text()
    block = source.split("\nGPU_ADMISSION_RC=2\n", 1)[1].split("\n# The allocation", 1)[0]
    result = subprocess.run(["bash", "-c", 'set -euo pipefail\nGPU_ADMISSION_RC=2\n' + block + '\necho admitted\n'],
                            env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
                                 "EXPERIMENTS_MBPP_SUITE": "all", "TEST_DRIVER_MODE": mode,
                                 "CUDA_VISIBLE_DEVICES": "" if mode == "enumeration-error" else "0,1,2,3"},
                            capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == expected, result.stdout + result.stderr
    assert ("admitted" in result.stdout) is (expected == 0)
