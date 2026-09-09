"""Permanent failures yield useful work; completed rotations terminate."""

import os
from pathlib import Path
import subprocess

import pytest

from test_generalization_launcher import checkout


ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    start = source.index(name + "() {")
    return source[start:source.index("\n}\n", start) + 2]


@pytest.mark.parametrize("marker", ["config-abort", "code-abort", "permanent-contract", "regime-contract-abort"])
def test_permanent_point_error_does_not_repeat_gpu_work(tmp_path, marker):
    source = (ROOT / "scripts/run_matrix.sh").read_text()
    (tmp_path / "logs").mkdir()
    (tmp_path / "checkpoint.partial").write_bytes(b"preserve saved work")
    script = "\n".join(function(source, name) for name in ("run_point", "run_point_unlocked")) + r'''
run_dir() { echo "$TEST_ROOT"; }
n_train_for_dataset() { echo 400; }
run_complete() { return 1; }
reenter_runtime_fields() { :; }
short_reason() { cat; }
run_pipeline_watchdog() {
  echo attempt >> "$TEST_ROOT/attempts"
  echo "[$TEST_ERROR] existing artifacts do not match this configuration" > "$2"
  return 2
}
recover_cuda_rollout() { echo unexpected-recovery; return 1; }
sleep() { echo unexpected-sleep; }
CONTRACT=''
DRIFTS=(0 25)
SEEDS=(0)
DATASETS=(math500)
MAX_RETRIES=3
run_point math500 0 25 '' '' ''
'''
    result = subprocess.run(["bash", "-c", script], cwd=tmp_path,
        env={**os.environ, "TEST_ROOT": str(tmp_path), "TEST_ERROR": marker},
        text=True, capture_output=True, timeout=5)
    assert result.returncode == 43, result.stdout + result.stderr
    assert (tmp_path / "attempts").read_text().splitlines() == ["attempt"]
    assert "unexpected-" not in result.stdout
    assert marker in result.stdout
    assert "rc=43:" in (tmp_path / "logs/supervisor.log").read_text()
    assert (tmp_path / "checkpoint.partial").read_bytes() == b"preserve saved work"


def test_permanent_family_failure_still_runs_other_eligible_families(tmp_path):
    source = (ROOT / "scripts/run_matrix.sh").read_text()
    start = source.index("failures=0\nwhile :;")
    queue = source[start:source.index("# A suite-level supervisor", start)]
    script = r'''
QUEUE="$TEST_ROOT"
CONTROL_ONLY=0
ordered_families() { printf 'broken 0\nhealthy 0\n'; }
family_complete() { test -e "$TEST_ROOT/$1.done"; }
cleanup_active_pipeline() { :; }
run_family() {
  echo "$1" >> "$TEST_ROOT/attempts"
  [ "$1" != broken ] || return 43
  touch "$TEST_ROOT/$1.done"
}
sleep() { echo unexpected-sleep; exit 99; }
''' + queue + '\necho unexpected-collection\n'
    result = subprocess.run(["bash", "-c", script],
        env={**os.environ, "TEST_ROOT": str(tmp_path)},
        text=True, capture_output=True, timeout=5)
    assert result.returncode == 43, result.stdout + result.stderr
    assert (tmp_path / "attempts").read_text().splitlines() == ["broken", "healthy"]
    assert (tmp_path / "healthy.done").exists()
    assert not (tmp_path / "broken.done").exists()
    assert "unexpected-" not in result.stdout


def test_rotation_exits_after_every_selected_profile_completes(tmp_path):
    repo, env = checkout(tmp_path)
    runner = repo / "scripts/run_additional_experiments.sh"
    runner.write_text('echo "$2" >> "$TEST_WORK/rotation-attempts"\n')
    result = subprocess.run(["bash", "scripts/run_available_experiments.sh"], cwd=repo,
        env={**env, "OM_RLZERO_FALLBACK_PROFILES": "qwen35 qwen38"},
        text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all selected profiles complete" in result.stdout
    assert "retaining worker" not in result.stdout
    assert (Path(env["TEST_WORK"]) / "rotation-attempts").read_text().splitlines() == ["qwen35", "qwen38"]


def test_rotation_does_not_report_failed_profile_as_complete(tmp_path):
    repo, env = checkout(tmp_path)
    runner = repo / "scripts/run_additional_experiments.sh"
    runner.write_text('[ "$2" != qwen35 ]\n')
    result = subprocess.run(["bash", "scripts/run_available_experiments.sh", "--once"], cwd=repo,
        env={**env, "OM_RLZERO_FALLBACK_PROFILES": "qwen35 qwen38"},
        text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "profile=qwen35 unavailable/failed" in result.stdout
    assert "profile=qwen38 complete" in result.stdout
    assert "all selected profiles complete" not in result.stdout
