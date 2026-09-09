"""Preserve a blocked matrix and keep its allocation on useful registered work."""

import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

from test_generalization_launcher import checkout

RUN_ID = "qwen35-9b-posttrained-math-code-grpo-v1"


def matrix_files(env):
    work = Path(env["TEST_WORK"])
    root = work / "runs" / RUN_ID / "m1"
    (root / ".queue").mkdir(parents=True)
    (root / ".queue/generation.git").write_text("a" * 40 + "\n")
    config_id = hashlib.sha256(b"{}\n").hexdigest()[:16]
    contract = work / "contracts" / f"{RUN_ID}-m1-{config_id}.json"
    contract.parent.mkdir(exist_ok=True)
    contract.write_text('{"git": "old", "preserve": true}\n')
    return work, root, contract


def reset(repo, env):
    return subprocess.run(
        ["bash", "scripts/reset_qwen35_root.sh"], cwd=repo, env=env,
        text=True, capture_output=True, timeout=10,
    )


def test_contract_conflict_is_reported_before_any_gpu_smoke(tmp_path):
    repo, env = checkout(tmp_path)
    work, root, contract = matrix_files(env)
    original = contract.read_bytes()
    payload = root / "point/checkpoint.partial"
    payload.parent.mkdir()
    payload.write_text("expensive work")
    result = subprocess.run(
        ["bash", "scripts/run_additional_experiments.sh", "--run", "qwen35"],
        cwd=repo, env={**env, "TEST_CONTRACT_FAIL": "1"},
        text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "stage=matrix-contract-m1" in result.stdout + result.stderr
    assert "generation commit " + "a" * 40 in result.stdout
    assert not (work / "gpu-preflights").exists()
    assert not (work / "phases").exists()
    assert contract.read_bytes() == original
    assert payload.read_text() == "expensive work"
    lease = work / "locks" / f"{RUN_ID}-m1.lifecycle.lock"
    with lease.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_failed_qwen_contract_yields_to_another_real_launcher(tmp_path):
    repo, env = checkout(tmp_path)
    work, root, contract = matrix_files(env)
    original = contract.read_bytes()
    result = subprocess.run(
        ["bash", "scripts/run_available_experiments.sh", "--once", "--first", "qwen35"],
        cwd=repo,
        env={**env, "TEST_CONTRACT_FAIL_CONFIG": "qwen35_9b_grpo.json",
             "OM_RLZERO_FALLBACK_PROFILES": "qwen38"},
        text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "profile=qwen35 unavailable/failed rc=1" in result.stdout
    assert "profile=qwen38 complete" in result.stdout
    assert (work / "phases").read_text().startswith("qwen38-27b-posttrained-")
    assert contract.read_bytes() == original
    assert (root / ".queue/generation.git").read_text().strip() == "a" * 40
    assert (work / "gpu-preflights").read_text().splitlines() == ["fla", "smoke"]


@pytest.mark.parametrize("args", [[], ["run"], ["run-idle"]])
def test_qwen_entrypoint_does_not_rotate_to_other_models(tmp_path, args):
    repo, env = checkout(tmp_path)
    runner = repo / "scripts/run_additional_experiments.sh"
    runner.write_text('''set -eu
test "$OM_QWEN_IDLE_NODE" = "$(hostname)"
test "$OM_WAIT_PRIMARY" = 0
for name in OM_PIPELINE_REPO OM_PIPELINE_SCRIPT OM_GENERATION_GIT REGIME_SKIP_COLLECTION REGIME_MATRIX MODEL_PATH OM_EXTERNAL_GPU_KEEPALIVE; do
  test -z "${!name+x}"
done
printf 'args=%s\\n' "$*"
''')
    env.pop("OM_RLZERO_FALLBACK_PROFILES", None)
    env.update({name: "old-olmo-setting" for name in (
        "OM_PIPELINE_REPO", "OM_PIPELINE_SCRIPT", "OM_GENERATION_GIT",
        "REGIME_SKIP_COLLECTION", "REGIME_MATRIX", "MODEL_PATH", "OM_EXTERNAL_GPU_KEEPALIVE",
    )})
    env["OM_WAIT_PRIMARY"] = "1"
    result = subprocess.run(
        ["bash", "scripts/run_qwen35_9b.sh", *args], cwd=repo, env=env,
        text=True, capture_output=True, timeout=5,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "args=--run qwen35" in result.stdout
    assert "qwen35_2b" not in result.stdout
    assert "olmo3_domains" not in result.stdout
    assert "qwen38" not in result.stdout


@pytest.mark.parametrize("args", [["run", "extra"], ["run-idle", "extra"], ["restart-idle", "extra"]])
def test_qwen_run_rejects_extra_arguments(tmp_path, args):
    repo, env = checkout(tmp_path)
    runner = repo / "scripts/run_additional_experiments.sh"
    runner.write_text('touch "$TEST_WORK/unexpected-launch"\n')
    result = subprocess.run(
        ["bash", "scripts/run_qwen35_9b.sh", *args], cwd=repo, env=env,
        text=True, capture_output=True, timeout=5,
    )
    assert result.returncode == 2
    assert not (Path(env["TEST_WORK"]) / "unexpected-launch").exists()


@pytest.mark.parametrize("stage", ["matrix-contract-m1", "smoke-m1"])
def test_optional_failure_doctor_cannot_block_handoff(tmp_path, stage):
    repo, env = checkout(tmp_path)
    doctor = repo / "scripts/doctor_qwen35.sh"
    doctor.write_text('#!/bin/sh\necho started > "$TEST_WORK/doctor-started"\nexec sleep 30\n')
    doctor.chmod(0o755)
    script = '''set -Eeuo pipefail
source scripts/setup_env.sh
PROFILE=qwen35; MODE=--run
source scripts/launch_logging.sh
log_stage "$TEST_STAGE"
echo '[regime-contract-abort] matrix contract mismatch: model.revision differs'
exit 17
'''
    result = subprocess.run(
        ["bash", "-c", script], cwd=repo,
        env={**env, "TEST_STAGE": stage, "ADDITIONAL_FAILURE_DOCTOR_TIMEOUT": "1"},
        text=True, capture_output=True, timeout=8,
    )
    assert result.returncode == 17, result.stdout + result.stderr
    started = (Path(env["TEST_WORK"]) / "doctor-started").exists()
    assert started == (stage == "smoke-m1")
    assert "DIAGNOSIS: matrix contract" in result.stdout
    if started:
        assert "optional doctor stopped rc=124" in result.stdout


@pytest.mark.parametrize("name", [
    "point/run_config.json", "point/policy_step_25/optimizer.pt",
    "point/rollouts_fresh_train.jsonl.partial", "point/DONE", "orphan.pt",
])
def test_reset_preserves_unfinished_artifacts_too(tmp_path, name):
    repo, env = checkout(tmp_path)
    work, root, contract = matrix_files(env)
    payload = root / name
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text("preserve")
    result = reset(repo, env)
    assert result.returncode != 0
    assert "preserving existing point/artifact" in result.stdout
    assert payload.read_text() == "preserve"
    assert contract.exists()
    assert not (work / "quarantine").exists()


@pytest.mark.parametrize("kind", ["queue", "generation", "lifecycle", "primary", "contract"])
def test_reset_respects_live_locks_even_when_old_or_forced(tmp_path, kind):
    repo, env = checkout(tmp_path)
    work, root, contract = matrix_files(env)
    locks = {
        "queue": root / ".queue/math500-s0.lock",
        "generation": root / ".queue/generation.git.lock",
        "lifecycle": work / "locks" / f"{RUN_ID}-m1.lifecycle.lock",
        "primary": Path(env["OM_LOCAL_LOCK_DIR"]) / "primary.lock",
        "contract": contract.with_suffix(".json.lock"),
    }
    path = locks[kind]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH if kind == "lifecycle" else fcntl.LOCK_EX)
        os.utime(path, (time.time() - 7200,) * 2)
        result = reset(repo, {**env, "OM_FORCE": "1"})
    assert result.returncode != 0, result.stdout
    assert "[abort]" in result.stdout
    assert root.exists() and contract.exists()


def test_empty_reset_archives_metadata_without_moving_contract_lock(tmp_path):
    repo, env = checkout(tmp_path)
    work, root, contract = matrix_files(env)
    lock = contract.with_suffix(".json.lock")
    lock.touch()
    inode = lock.stat().st_ino
    original = contract.read_bytes()
    result = reset(repo, env)
    assert result.returncode == 0, result.stdout + result.stderr
    archives = list((work / "quarantine").iterdir())
    assert len(archives) == 1
    archive = archives[0]
    assert (archive / "run/.queue/generation.git").read_text().strip() == "a" * 40
    assert (archive / "contracts" / contract.name).read_bytes() == original
    assert not root.exists() and not contract.exists()
    assert lock.stat().st_ino == inode
    relaunched = subprocess.run(
        ["bash", "scripts/run_additional_experiments.sh", "--run", "qwen35"],
        cwd=repo, env=env, text=True, capture_output=True, timeout=15,
    )
    assert relaunched.returncode == 0, relaunched.stdout + relaunched.stderr
    assert (work / "phases").exists()
    assert "generation commit " + "a" * 40 not in relaunched.stdout


def test_reset_keeps_results_even_without_point_directories(tmp_path):
    repo, env = checkout(tmp_path)
    work, root, contract = matrix_files(env)
    report = work / "results" / RUN_ID / "m1/report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}")
    result = reset(repo, env)
    assert result.returncode != 0 and "preserving existing results" in result.stdout
    assert report.exists() and root.exists() and contract.exists()


def test_reset_fails_if_archiving_fails(tmp_path):
    repo, env = checkout(tmp_path)
    work, root, contract = matrix_files(env)
    fake_mv = tmp_path / "bin/mv"
    fake_mv.write_text("#!/bin/sh\nexit 21\n")
    fake_mv.chmod(0o755)
    result = reset(repo, env)
    assert result.returncode == 21
    assert "Next launch starts" not in result.stdout
    assert root.exists() and contract.exists()


def test_live_launcher_retains_shared_matrix_lease(tmp_path):
    repo, env = checkout(tmp_path)
    work = Path(env["TEST_WORK"])
    process = subprocess.Popen(
        ["bash", "scripts/run_additional_experiments.sh", "--run", "qwen35"],
        cwd=repo, env={**env, "TEST_MATRIX_WAIT": "1"},
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not (work / "phases").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (work / "phases").exists()
        lease = work / "locks" / f"{RUN_ID}-m1.lifecycle.lock"
        with lease.open("a") as stream:
            with pytest.raises(BlockingIOError):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = reset(repo, {**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "other-node")})
        assert result.returncode != 0 and "lifecycle lock" in result.stdout
        (work / "release-matrix").touch()
        out, err = process.communicate(timeout=10)
        assert process.returncode == 0, out + err
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
