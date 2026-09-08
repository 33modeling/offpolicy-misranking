"""CPU regressions for scheduler waits, cleanup and exhausted retry budgets."""
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "scripts/run_matrix.sh"


def shell_function(source, name):
    start = source.index(name + "() {")
    return source[start:source.index("\n}\n", start) + 2]


def run_shell(script, tmp_path, timeout=5):
    process = subprocess.Popen(
        ["bash", "-c", script], cwd=tmp_path,
        env={**os.environ, "TEST_ROOT": str(tmp_path), "PY": sys.executable},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(process.args, process.returncode, out, err)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def test_completed_pipeline_does_not_wait_for_watchdog_interval(tmp_path):
    source = MATRIX.read_text()
    start = source.index("probe_exclude_pid() {")
    end = source.index("\nrollout_artifact_ready() {")
    result = run_shell(source[start:end] + '''
WATCH_INTERVAL_SECONDS=30
WATCH_KILL_GRACE_SECONDS=1
WATCH_GPU_SAMPLES=1
STALL_SECONDS=60
HARD_STALL_SECONDS=120
run_pipeline_watchdog "$TEST_ROOT" "$TEST_ROOT/logs/attempt.log" bash -c 'sleep 0.2'
''', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("attempts", [1, 2])
def test_last_point_failure_does_not_recover_or_sleep(tmp_path, attempts):
    (tmp_path / "logs").mkdir()
    source = shell_function(MATRIX.read_text(), "run_point")
    result = run_shell(source + f'\nMAX_RETRIES={attempts}\n' + '''
run_dir() { echo "$TEST_ROOT"; }
n_train_for_dataset() { echo 400; }
run_complete() { return 1; }
reenter_runtime_fields() { return 0; }
short_reason() { cat; }
run_pipeline_watchdog() { echo attempt; return 1; }
recover_cuda_rollout() { echo recovery; return 1; }
bash() { echo diagnosis; }
sleep() { echo pause; }
CONTRACT=''
DRIFTS=(0)
SEEDS=(0)
DATASETS=(math500)
run_point math500 0 0 '' '' ''
''', tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert lines.count("attempt") == attempts
    assert lines.count("recovery") == attempts - 1
    assert lines.count("pause") == attempts - 1
    assert lines.count("diagnosis") == attempts


def test_failed_deep_contract_preserves_artifacts_and_yields(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "checkpoint").write_text("preserve")
    source = shell_function(MATRIX.read_text(), "run_point")
    result = run_shell(source + '''
run_dir() { echo "$TEST_ROOT"; }
n_train_for_dataset() { echo 400; }
run_complete() { return 1; }
reenter_runtime_fields() { return 0; }
contract_run() {
  echo "$1" >> "$TEST_ROOT/contracts"
  [ "$1" != check-run ] || { echo 'wrong artifact hash' >&2; return 1; }
}
run_pipeline_watchdog() { echo attempt; return 0; }
MAX_RETRIES=3
CONTRACT=matrix.json
DRIFTS=(0)
SEEDS=(0)
DATASETS=(math500)
run_point math500 0 0 '' '' ''
''', tmp_path)
    assert result.returncode == 43, result.stdout + result.stderr
    assert result.stdout.splitlines().count("attempt") == 1
    assert (tmp_path / "contracts").read_text().splitlines() == ["prepare-run", "check-run"]
    assert "wrong artifact hash" in (tmp_path / "logs/supervisor.log").read_text()
    assert (tmp_path / "checkpoint").read_text() == "preserve"


def test_busy_fallback_matrix_yields_instead_of_waiting(tmp_path):
    source = MATRIX.read_text()
    start = source.index("failures=0\nwhile :;")
    end = source.index("\n# A suite-level supervisor", start)
    result = run_shell('''
REGIME_YIELD_WHEN_BUSY=1
QUEUE="$TEST_ROOT"
family_complete() { return 1; }
ordered_families() { echo 'math500 0'; }
flock() { return 1; }
sleep() { echo unexpected-wait; exit 99; }
''' + source[start:end], tmp_path)
    assert result.returncode == 75, result.stdout + result.stderr
    assert "unexpected-wait" not in result.stdout


def test_missing_gpu_telemetry_is_not_treated_as_idle(tmp_path):
    source = (ROOT / "scripts/run_additional_experiments.sh").read_text()
    result = run_shell(shell_function(source, "wait_for_gpu_release") + '''
ADDITIONAL_GPU_WAIT_SECONDS=1
timeout() { printf 'N/A\\nN/A\\nN/A\\nN/A\\n'; }
wait_for_gpu_release
''', tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr


def test_slow_gpu_probe_cannot_multiply_wait_budget(tmp_path):
    source = (ROOT / "scripts/run_additional_experiments.sh").read_text()
    fake = tmp_path / "nvidia-smi"
    fake.write_text('#!/bin/sh\nexec sleep 30\n')
    fake.chmod(0o755)
    result = run_shell(shell_function(source, "wait_for_gpu_release") + '''
PATH="$TEST_ROOT:$PATH"
ADDITIONAL_GPU_WAIT_SECONDS=1
wait_for_gpu_release
''', tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr


def test_unknown_current_failure_does_not_reuse_historical_cuda(tmp_path):
    source = (ROOT / "scripts/run_olmo3_rlzero.sh").read_text()
    logs = tmp_path / "point/logs"
    logs.mkdir(parents=True)
    (logs / "old.log").write_text('CUDA error: unspecified launch failure\n')
    (logs / "supervisor.log").write_text(
        '[2026-09-08] [point-failed] try 1/1 rc=1: no error line in regime-attempt-1.log\n'
    )
    result = run_shell(shell_function(source, "family_last_error") + '''
family_root() { echo "$TEST_ROOT"; }
family_last_error math500 0
''', tmp_path)
    assert result.returncode == 0
    assert "no error line" in result.stdout
    assert "CUDA" not in result.stdout
