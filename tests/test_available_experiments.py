"""Independent fallback rotation without a real GPU or model process."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_failed_profile_yields_and_success_is_not_repeated(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(ROOT / "scripts/run_available_experiments.sh", scripts)
    (scripts / "run_additional_experiments.sh").write_text('''#!/usr/bin/env bash
set -eu
test "$1" = --run
test "$ADDITIONAL_MAX_RESTARTS" = 0
test "$ADDITIONAL_REGIME_MAX_RETRIES" = 1
test "$REGIME_YIELD_WHEN_BUSY" = 1
test "${OM_EXTERNAL_GPU_KEEPALIVE-unset}" = unset
test "$HF_HUB_OFFLINE" = 1
test "${OM_PIPELINE_REPO-unset}" = unset
test "${OM_GENERATION_GIT-unset}" = unset
test "${REGIME_SKIP_COLLECTION-unset}" = unset
test "${HF_TOKEN-unset}" = unset
echo "$2" >> "$TEST_CALLS"
test "$2" != olmo3_domains
''')
    calls = tmp_path / "calls"
    result = subprocess.run(
        ["bash", str(scripts / "run_available_experiments.sh"), "--once"],
        env={**os.environ, "OM_LOCAL_LOCK_DIR": str(tmp_path / "locks"),
             "OM_RLZERO_FALLBACK_PROFILES": "olmo3_domains qwen35_2b qwen35_2b",
             "TEST_CALLS": str(calls), "OM_PIPELINE_REPO": "/old/pinned",
             "OM_GENERATION_GIT": "old", "REGIME_SKIP_COLLECTION": "1", "HF_TOKEN": "secret",
             "OM_EXTERNAL_GPU_KEEPALIVE": "1"},
        text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls.read_text().splitlines() == ["olmo3_domains", "qwen35_2b"]
    assert "unavailable/failed rc=1" in result.stdout
    assert "profile=qwen35_2b complete" in result.stdout


def test_unknown_profile_is_rejected_before_any_execution(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_available_experiments.sh"), "--once"],
        env={**os.environ, "OM_RLZERO_FALLBACK_PROFILES": "not-registered",
             "OM_LOCAL_LOCK_DIR": str(tmp_path / "locks")},
        text=True, capture_output=True, timeout=5,
    )
    assert result.returncode == 2
    assert not (tmp_path / "locks").exists()


@pytest.mark.parametrize("args", [[], ["--first", "qwen35"]])
def test_explicit_snapshot_does_not_leak_to_independent_profiles(tmp_path, args):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(ROOT / "scripts/run_available_experiments.sh", scripts)
    (scripts / "run_additional_experiments.sh").write_text('''#!/usr/bin/env bash
printf '%s|%s\\n' "$2" "${OM_SNAPSHOT_PATH-unset}" >> "$TEST_CALLS"
exit 1
''')
    calls = tmp_path / "calls"
    result = subprocess.run(
        ["bash", str(scripts / "run_available_experiments.sh"), "--once", *args],
        env={**os.environ, "OM_LOCAL_LOCK_DIR": str(tmp_path / "locks"),
             "OM_RLZERO_FALLBACK_PROFILES": "qwen35 qwen35_2b",
             "OM_SNAPSHOT_PATH": "/uploaded/custom-9b", "TEST_CALLS": str(calls)},
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls.read_text().splitlines() == [
        "qwen35|/uploaded/custom-9b", "qwen35_2b|unset",
    ]


@pytest.mark.parametrize("args", [["--first"], ["--first", "not-registered"], ["--once", "extra"]])
def test_invalid_rotation_arguments_do_not_launch(tmp_path, args):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_available_experiments.sh"), *args],
        env={**os.environ, "OM_LOCAL_LOCK_DIR": str(tmp_path / "locks")},
        text=True, capture_output=True, timeout=5,
    )
    assert result.returncode == 2
    assert not (tmp_path / "locks").exists()


def test_permanent_completion_failure_is_shared_immediately(tmp_path):
    source = (ROOT / "scripts/run_olmo3_rlzero.sh").read_text()
    start = source.index("note_family_failure() {")
    function = source[start:source.index("\n}\n", start) + 2]
    script = function + '''
declare -A FAMILY_FAILURES=() FAMILY_CUDA_FAILURES=()
MAX_FAMILY_FAILURES=4
WORKER_ID=test
HOST_TAG=test
LOG="$TEST_ROOT/worker.log"
loop_marker() { echo "$TEST_ROOT/family.loop"; }
family_last_error() { echo 'old CUDA error'; }
failure_kind() { echo runtime; }
note_family_failure math500 0 43
test "$LAST_FAILURE_KIND" = other
'''
    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True,
                            env={**os.environ, "TEST_ROOT": str(tmp_path)}, timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = (tmp_path / "family.loop").read_text()
    assert "consecutive_failures=1 last_rc=43" in marker
    assert "[cuda-flaky]" not in result.stdout
    start = source.index('for marker in "$QUEUE"/*.loop; do')
    startup = source[start:source.index("\nfamily_looping()", start)]
    result = subprocess.run(
        ["bash", "-c", 'failure_kind() { echo runtime; }\n' + startup],
        env={**os.environ, "QUEUE": str(tmp_path), "OM_RLZERO_CLEAR_LOOPS": "0"},
        text=True, capture_output=True, timeout=5,
    )
    assert result.returncode == 0
    assert (tmp_path / "family.loop").read_text() == marker
