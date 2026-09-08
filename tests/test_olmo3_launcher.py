"""Three-cluster assignment and commit-stable restart for the OLMo launcher."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def fixture_checkout(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    checkout = tmp_path / "checkout"
    shared = tmp_path / "shared"
    fake_bin = tmp_path / "bin"
    for path in (
        checkout / "scripts",
        checkout / "src",
        checkout / "configs",
        shared / "work/venv/bin",
        shared / "models/Olmo-3-1025-7B",
        shared / "datasets",
        fake_bin,
    ):
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts/run_olmo3_rlzero.sh", checkout / "scripts")
    shutil.copy2(ROOT / "src/regime_resume_commit.py", checkout / "src")
    shutil.copy2(ROOT / "src/cleanup_run_processes.py", checkout / "src")
    shutil.copy2(ROOT / "src/rlzero_status.py", checkout / "src")
    shutil.copy2(ROOT / "src/rlzero_heartbeat.py", checkout / "src")
    shutil.copy2(ROOT / "src/repair_run_config.py", checkout / "src")
    shutil.copy2(ROOT / "src/training_progress.py", checkout / "src")
    shutil.copy2(ROOT / "src/recovery_policy.py", checkout / "src")
    shutil.copy2(ROOT / "configs/olmo3_rlzero.json", checkout / "configs")
    shutil.copy2(ROOT / "configs/olmo3_rlzero_h100.json", checkout / "configs")
    (checkout / "requirements.txt").write_text("fixture\n")
    (shared / "models/Olmo-3-1025-7B/config.json").write_text("{}\n")
    executable(checkout / "scripts/run_point.sh", "#!/bin/sh\nexit 0\n")
    executable(checkout / "scripts/run_available_experiments.sh", '''#!/usr/bin/env bash
set -eu
exec 8>"$OM_LOCAL_LOCK_DIR/primary.lock"
flock -n 8 || { echo 'primary lock leaked into handoff'; exit 9; }
echo '[fixture-fallback] independent experiment admitted'
''')
    (checkout / "scripts/setup_env.sh").write_text(
        'export GROUP_VOLUME="$TEST_SHARED"\n'
        'export OM_WORK="$TEST_SHARED/work"\n'
        'export VENV_DIR="$TEST_SHARED/work/venv"\n'
        'export MODELS_DIR="$TEST_SHARED/models"\n'
        'export DATASETS_DIR="$TEST_SHARED/datasets"\n'
        'export PYTHONPATH="$OM_REPO/src${PYTHONPATH:+:$PYTHONPATH}"\n',
        encoding="utf-8",
    )
    executable(
        shared / "work/venv/bin/python",
        r'''#!/usr/bin/env bash
script=$1; shift
case "$script" in
  *bootstrap_math_verify.py) printf '%s/runtime-deps\n' "$TEST_SHARED" ;;
  *regime_resume_commit.py) exec python3 "$script" "$@" ;;
  *cleanup_run_processes.py) exec python3 "$script" "$@" ;;
  *rlzero_status.py|*rlzero_heartbeat.py) exec python3 "$script" "$@" ;;
  *repair_run_config.py) exec python3 "$script" "$@" ;;
  *model_matrix.py)
    case " $* " in
      *" field olmo3-7b-base path "*) printf '%s/models/Olmo-3-1025-7B\n' "$TEST_SHARED" ;;
      *" field olmo3-7b-base revision "*) printf 'a81bae42db3975be1671e27b9c9a56da1a9f980f\n' ;;
      *" field olmo3-7b-base lora_targets "*) printf 'q_proj,v_proj\n' ;;
      *" experiment-field datasets "*) printf 'math500 mbpp\n' ;;
      *" experiment-field seeds "*) printf '0 1 2 3 4\n' ;;
      *" experiment-field drifts "*) printf '0 25 100 400\n' ;;
      *" experiment-field n_val "*) printf '100\n' ;;
      *" experiment-field behavior_k "*) printf '8\n' ;;
      *" experiment-field fresh_k "*) printf '32\n' ;;
      *" experiment-field val_k "*) printf '8\n' ;;
      *" experiment-field micro_group "*) printf '4\n' ;;
      *" experiment-field max_new_tokens "*) printf '2048\n' ;;
      *" experiment-field proj_dim "*) printf '4096\n' ;;
      *" experiment-field grad_layers "*) printf '4\n' ;;
      *" experiment-field clip_cap "*) printf '10.0\n' ;;
      *" experiment-field topk_frac "*) printf '0.1\n' ;;
      *" experiment-field temperature "*) printf '1.0\n' ;;
      *" experiment-field first_bootstrap "*) printf '10000\n' ;;
      *" experiment-field attn "*) printf 'eager\n' ;;
      *" grpo-field world_size "*) printf '4\n' ;;
      *" grpo-field group_size "*) printf '8\n' ;;
      *" grpo-field clip_epsilon "*) printf '0.2\n' ;;
      *" grpo-field learning_rate "*) printf '1e-5\n' ;;
      *" grpo-field epochs_per_batch "*) printf '1\n' ;;
      *" grpo-field max_grad_norm "*) printf '1.0\n' ;;
      *" grpo-field advantage_epsilon "*) printf '1e-4\n' ;;
      *" grpo-field lora_rank "*) printf '16\n' ;;
      *" grpo-field lora_alpha "*) printf '32\n' ;;
      *" runtime-field generation_batch "*)
        case "$*" in *olmo3_rlzero_h100.json*) printf '8\n' ;; *) printf '4\n' ;; esac ;;
      *" runtime-field gradient_micro_batch "*)
        case "$*" in *olmo3_rlzero_h100.json*) printf '4\n' ;; *) printf '1\n' ;; esac ;;
      *" runtime-field logprob_micro_batch "*)
        case "$*" in *olmo3_rlzero_h100.json*) printf '4\n' ;; *) printf '1\n' ;; esac ;;
      *" runtime-field gradient_checkpointing "*) printf '1\n' ;;
      *" check olmo3-7b-base "*) printf '[model-check] passed\n' ;;
      *) printf 'unexpected model args: %s\n' "$*" >&2; exit 2 ;;
    esac ;;
  *qualify_domain_data.py|*qualify_rlzero_signal.py)
    if [[ "$script" == *qualify_rlzero_signal.py ]] && [ "${TEST_HANG_SIGNAL:-0}" = 1 ]; then
      trap 'exit 0' TERM INT
      while :; do /bin/sleep 1; done
    fi
    while [ $# -gt 0 ]; do
      if [ "$1" = --output ]; then mkdir -p "$(dirname "$2")"; printf '{}\n' > "$2"; break; fi
      shift
    done ;;
  *gpu_keepalive.py)
    printf 'pid=%s gpus=4\n' "$$" > "$OM_GPU_KEEPALIVE_READY_FILE"
    trap 'exit 0' TERM INT
    while :; do /bin/sleep 1; done ;;
  *regime_map.py)
    while [ $# -gt 0 ]; do
      if [ "$1" = --output-dir ]; then
        out=$2; mkdir -p "$out"
        for name in REGIME.json REGIME.csv REGIME_SUMMARY.csv FINAL_REPORT.md; do printf 'pass\n' > "$out/$name"; done
        break
      fi
      shift
    done ;;
  -c) exit 0 ;;
  -m)
    if [ "${TEST_HANG_SMOKE:-0}" = 1 ]; then
      trap 'exit 0' TERM INT
      while :; do /bin/sleep 1; done
    fi
    while [ $# -gt 0 ]; do
      if [ "$1" = --output ]; then
        out=$2; mkdir -p "$out"
        for name in policy_train.json adapter_config.json adapter_model.safetensors optimizer.pt grpo_stats.jsonl; do printf 'pass\n' > "$out/$name"; done
        break
      fi
      shift
    done ;;
  -)
    if [[ "$1" == *grpo_stats.jsonl || "${2:-}" == */src ]]; then exec python3 - "$@"; fi
    destination=$1; mkdir -p "$(dirname "$destination")"
    case "$destination" in
      *.owner.json) printf '{"worker":"%s"}\n' "$WORKER_ID" > "$destination" ;;
      *) printf '{"train":[{"question":"q","answer":"1"}],"val":[]}' > "$destination" ;;
    esac
    cat >/dev/null ;;
  *) printf 'unexpected python target: %s %s\n' "$script" "$*" >&2; exit 2 ;;
esac
''',
    )
    executable(
        checkout / "scripts/run_matrix.sh",
        r'''#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
[ -z "${HF_TOKEN+x}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN+x}" ] || exit 91
[ "$HF_HUB_OFFLINE" = 1 ] && [ "$TRANSFORMERS_OFFLINE" = 1 ] \
  && [ "$HF_DATASETS_OFFLINE" = 1 ] || exit 92
[[ ":$PYTHONPATH:" == *":$TEST_SHARED/runtime-deps:"* ]] || exit 94
key="$REGIME_DATASETS-s$REGIME_SEEDS"
if [ "${TEST_INTERRUPT_FAMILY:-}" = "$key" ]; then
  marker="$TEST_SHARED/work/interrupt-once-$key"
  mkdir -p "$REGIME_ROOT"
  if mkdir "$marker" 2>/dev/null; then
    printf 'durable partial\n' > "$REGIME_ROOT/interrupted.partial"
    printf '%s\n' "$key" > "$TEST_SHARED/work/interrupt-ready"
    trap 'exit 130' TERM INT HUP
    while :; do /bin/sleep 1; done
  fi
  [ -s "$REGIME_ROOT/interrupted.partial" ] || exit 95
  printf 'resumed\n' >> "$REGIME_ROOT/interrupted.partial"
fi
if [ "${TEST_FAIL_FAMILY:-}" = "$key" ]; then
  marker="$TEST_SHARED/work/fail-$key"
  mkdir -p "$(dirname "$marker")"
  if [ "${TEST_FAIL_ALWAYS:-0}" = 1 ] || [ ! -e "$marker" ]; then
    : > "$marker"
    exit "${TEST_FAIL_RC:-43}"
  fi
fi
if [ "${TEST_CUDA_FAIL_FAMILY:-}" = "$key" ]; then
  counter="$TEST_SHARED/work/cuda-fail-$key"
  n=$(cat "$counter" 2>/dev/null || echo 0)
  if [ "$n" -lt "${TEST_CUDA_FAIL_TIMES:-5}" ]; then
    echo $((n + 1)) > "$counter"
    mkdir -p "$REGIME_ROOT/point/logs"
    echo "RuntimeError: CUDA error: unspecified launch failure" >> "$REGIME_ROOT/point/logs/main.log"
    printf '[%s] [point-failed] try 1/1 rc=1: RuntimeError: CUDA error: unspecified launch failure\n' \
      "$(date '+%F %T')" >> "$REGIME_ROOT/point/logs/supervisor.log"
    exit 1
  fi
fi
if [ "${TEST_STALE_CUDA_FAIL_FAMILY:-}" = "$key" ]; then
  # an old CUDA line in a stage log, but the attempt that just failed says config-abort
  mkdir -p "$REGIME_ROOT/point/logs"
  echo "RuntimeError: CUDA error: unspecified launch failure" >> "$REGIME_ROOT/point/logs/main.log"
  echo "[2026-09-08 09:00:00] [point-failed] try 3/3 rc=2: [config-abort] existing artifacts use a different run config: ['gen_batch']" >> "$REGIME_ROOT/point/logs/supervisor.log"
  exit 1
fi
git=$(git -C "$OM_PIPELINE_REPO" rev-parse HEAD)
printf '%s|%s|%s|%s|%s\n' "$WORKER_ID" "$key" "$git" "${OM_GEN_BATCH:-}" "${GRADIENT_MICRO_BATCH:-}" >> "$TEST_SHARED/work/claims"
if [ "${TEST_PAUSE_FAMILY:-}" = "$key" ]; then
  marker="$TEST_SHARED/work/pause-once-$key"
  if mkdir "$marker" 2>/dev/null; then
    : > "$TEST_SHARED/work/pause-ready"
    while [ ! -e "$TEST_SHARED/work/pause-release" ]; do /bin/sleep 0.05; done
  fi
fi
/bin/sleep 0.05
for drift in $REGIME_DRIFTS; do
  run="$REGIME_ROOT/$REGIME_MODEL_TAG-s$REGIME_SEEDS-$REGIME_DATASETS-d$drift"
  mkdir -p "$run"; printf 'done\n' > "$run/DONE"
done
''',
    )
    executable(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        "  *query-compute-apps=pid*)\n"
        '    pid="${TEST_GPU_PID:-}"\n'
        '    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then\n'
        '      state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d " ")\n'
        '      [[ "$state" == Z* ]] || printf "%s\\n" "$pid"\n'
        "    fi ;;\n"
        "  *memory.used*) printf '0\\n0\\n0\\n0\\n' ;;\n"
        "  *) printf 'NVIDIA H100 80GB HBM3\\nNVIDIA H100 80GB HBM3\\nNVIDIA H100 80GB HBM3\\nNVIDIA H100 80GB HBM3\\n' ;;\n"
        "esac\n",
    )
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.com"], cwd=checkout, check=True
    )
    subprocess.run(["git", "config", "user.name", "fixture"], cwd=checkout, check=True)
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=checkout, check=True)
    return checkout, {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TEST_SHARED": str(shared),
        "HF_TOKEN": "must-not-reach-compute",
        "HUGGING_FACE_HUB_TOKEN": "must-not-reach-compute",
        "OM_PIPELINE_CACHE": str(tmp_path / "pipeline-cache"),
        "OM_RLZERO_PREFLIGHT_WAIT_SECONDS": "1",
        "OM_RLZERO_SIGNAL_TIMEOUT_SECONDS": "5",
        "OM_RLZERO_SMOKE_TIMEOUT_SECONDS": "5",
        "OM_RLZERO_PREFLIGHT_KILL_GRACE_SECONDS": "1",
        "OM_RLZERO_QUEUE_WAIT_SECONDS": "1",
        "OM_RLZERO_FAMILY_RETRY_SECONDS": "0",
        "OM_RLZERO_STATUS_PROBE_SECONDS": "0",
    }


def run_worker(checkout: Path, env: dict[str, str], local: Path) -> subprocess.Popen:
    return subprocess.Popen(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run"],
        cwd=checkout,
        env={**env, "OM_LOCAL_LOCK_DIR": str(local)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_three_workers_claim_every_family_exactly_once(tmp_path: Path) -> None:
    checkout, env = fixture_checkout(tmp_path)
    workers = [run_worker(checkout, env, tmp_path / f"local-{index}") for index in range(3)]
    outputs = []
    for worker in workers:
        output, _ = worker.communicate(timeout=30)
        outputs.append(output)
        assert worker.returncode == 0, output

    claims = (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()
    keys = [line.split("|")[1] for line in claims]
    assert len(keys) == 10
    assert len(set(keys)) == 10
    assert len({line.split("|")[0] for line in claims}) >= 2
    assert (Path(env["TEST_SHARED"]) / "work/results/olmo3-1025-7b-base-rlzero-grpo-v1/COMPLETE").is_file()


def test_h100_profile_uses_a_disjoint_root_and_runtime_contract(tmp_path: Path) -> None:
    checkout, env = fixture_checkout(tmp_path)
    shared_root = (
        Path(env["TEST_SHARED"])
        / "work/runs/olmo3-1025-7b-base-rlzero-grpo-h100-v2"
    )
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run", "h100"],
        cwd=checkout,
        env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "h100-local")},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    h100 = (
        Path(env["TEST_SHARED"])
        / "work/results/olmo3-1025-7b-base-rlzero-grpo-h100-v2/COMPLETE"
    )
    baseline = (
        Path(env["TEST_SHARED"])
        / "work/results/olmo3-1025-7b-base-rlzero-grpo-v1/COMPLETE"
    )
    assert h100.is_file()
    assert not baseline.exists()

    status = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "status", "h100", "verbose"],
        cwd=checkout,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    output = status.stdout + status.stderr
    assert status.returncode == 0, output
    assert "profile=h100" in output
    assert f"experiment_root={shared_root}" in output
    assert "runtime_contract generation_batch=8 gradient_micro_batch=4" in output
    assert "min_recovery_generation_batch=2" in output
    assert "math500/s0 complete" in output
    assert "d0 stage=complete" in output


def test_launcher_kills_stale_gpu_process_before_creating_fresh_contexts(
    tmp_path: Path,
) -> None:
    checkout, env = fixture_checkout(tmp_path)
    stale = subprocess.Popen(["sleep", "300"])
    try:
        result = subprocess.run(
            ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run", "h100"],
            cwd=checkout,
            env={
                **env,
                "OM_LOCAL_LOCK_DIR": str(tmp_path / "stale-gpu-local"),
                "TEST_GPU_PID": str(stale.pid),
            },
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        stale.wait(timeout=3)
        assert "TERM sent to 1 stale GPU compute processes" in result.stdout
        assert "all GPU compute contexts cleared" in result.stdout
        assert "fresh CUDA contexts passed on all four GPUs" in result.stdout
    finally:
        if stale.poll() is None:
            stale.kill()
            stale.wait()


def test_signal_qualification_has_a_hard_timeout(tmp_path: Path) -> None:
    checkout, env = fixture_checkout(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run", "h100"],
        cwd=checkout,
        env={
            **env,
            "OM_LOCAL_LOCK_DIR": str(tmp_path / "signal-timeout-local"),
            "OM_RLZERO_SIGNAL_TIMEOUT_SECONDS": "1",
            "TEST_HANG_SIGNAL": "1",
        },
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert "[preflight-timeout] signal-math500 exceeded 1s" in result.stdout


def test_four_gpu_smoke_has_a_hard_timeout(tmp_path: Path) -> None:
    checkout, env = fixture_checkout(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run", "h100"],
        cwd=checkout,
        env={
            **env,
            "OM_LOCAL_LOCK_DIR": str(tmp_path / "smoke-timeout-local"),
            "OM_RLZERO_SMOKE_TIMEOUT_SECONDS": "1",
            "TEST_HANG_SMOKE": "1",
        },
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert "[preflight-timeout] grpo-smoke-step1 exceeded 1s" in result.stdout


def test_status_shows_point_progress_and_untruncated_runtime_errors(
    tmp_path: Path,
) -> None:
    checkout, env = fixture_checkout(tmp_path)
    shared = Path(env["TEST_SHARED"])
    tag = "olmo3-1025-7b-base-rlzero-grpo-v1"
    root = shared / "work/runs" / tag
    family = root / "family-math500-s0"
    run = family / f"{tag}-s0-math500-d0"
    run25 = family / f"{tag}-s0-math500-d25"
    logs = run / "logs"
    logs.mkdir(parents=True)
    (run25 / "policy_step_25").mkdir(parents=True)
    (root / ".families").mkdir()
    (root / ".queue").mkdir()
    (root / "logs").mkdir()
    (root / ".queue/generation.git").write_text("abc123\n")
    (root / ".families/math500-s0.owner.json").write_text(
        '{"worker":"worker-status","host":"h100-test"}\n'
    )
    (root / "logs/worker-status.log").write_text(
        "[family-retry] math500/s0 rc=1; artifacts preserved\n"
    )
    (run / "rollouts_behavior_train.jsonl").write_text("{}\n{}\n")
    (run / "rollouts_fresh_train.shard0.partial").write_text("{}\n{}\n{}\n")
    (run / "rollout_recovery.jsonl").write_text(
        '{"configured_generation_batch":"8","failure_kind":"oom",'
        '"recovery_generation_batch":4,"status":"failed"}\n'
    )
    long_error = (
        "RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED "
        + "x" * 220
        + " END-OF-ERROR"
    )
    (logs / "regime-attempt-2.log").write_text("pipeline failed\n")
    (logs / "fresh-shard0.log").write_text(
        "[2026-09-02 12:00:00] rollout 46/100\n" + long_error + "\n"
    )
    (run25 / "policy_step_25/grpo_stats.jsonl").write_text(
        '{"grad_norm":0.42,"groups":4,"loss":0.0,"mean_ratio":1.0,'
        '"nonzero_advantage_groups":3,"reward_mean":0.25}\n'
    )

    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "status", "baseline", "verbose"],
        cwd=checkout,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "math500/s0 stale-owner" in output
    assert "verdict=DEAD" in output
    assert "d0 stage=fresh-rollout behavior_rows=2 fresh_rows=3" in output
    assert "recovery_batch=4 recovery_status=failed recovery_reason=oom" in output
    assert "d25 stage=initialized" in output
    assert "active_groups=3/4 loss=0.000e+00 grad_norm=4.200e-01" in output
    assert "learning_signal=update" in output
    assert "latest_log=" in output
    assert "rollout 46/100" in output
    assert "error_evidence_from_all_checked_logs:" in output
    assert long_error in output
    assert "worker_log=" in output


def test_failed_family_restarts_automatically_without_launcher_exit(tmp_path: Path) -> None:
    checkout, env = fixture_checkout(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run"],
        cwd=checkout,
        env={
            **env,
            "OM_LOCAL_LOCK_DIR": str(tmp_path / "retry-local"),
            "TEST_FAIL_FAMILY": "mbpp-s0",
            "TEST_FAIL_RC": "1",
            "OM_RLZERO_FAMILY_ATTEMPTS": "1",
        },
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[family-retry] mbpp/s0 rc=1" in result.stdout
    assert "allocation retained" in result.stdout
    claims = (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()
    assert len(claims) == 10
    assert len({line.split("|")[1] for line in claims}) == 10


def test_partial_suite_resumes_original_commit_after_git_pull(tmp_path: Path) -> None:
    checkout, env = fixture_checkout(tmp_path)
    first_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    failed = subprocess.Popen(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run"],
        cwd=checkout,
        env={
            **env,
            "OM_LOCAL_LOCK_DIR": str(tmp_path / "first-local"),
            "TEST_FAIL_FAMILY": "mbpp-s0",
            "TEST_FAIL_ALWAYS": "1",
            "TEST_FAIL_RC": "1",
            "OM_RLZERO_FAMILY_ATTEMPTS": "1",
            "OM_RLZERO_MAX_FAMILY_FAILURES": "100",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    failure_marker = Path(env["TEST_SHARED"]) / "work/fail-mbpp-s0"
    logs = Path(env["TEST_SHARED"]) / "work/runs/olmo3-1025-7b-base-rlzero-grpo-v1/logs"
    retry_recorded = False
    for _ in range(200):
        # The child writes failure_marker before the supervisor observes exit.
        # Wait for the actual retry transition before sending SIGTERM.
        retry_recorded = any(
            "[family-retry] mbpp/s0 rc=1" in path.read_text()
            for path in logs.glob("*.log")
        )
        if retry_recorded:
            break
        time.sleep(0.05)
    was_running = failed.poll() is None
    failed.terminate()
    failed_output, _ = failed.communicate(timeout=10)
    assert retry_recorded, failed_output
    assert failure_marker.exists()
    assert was_running
    assert "[family-retry] mbpp/s0 rc=1" in failed_output
    marker = Path(env["TEST_SHARED"]) / "work/runs/olmo3-1025-7b-base-rlzero-grpo-v1/.queue/generation.git"
    assert marker.read_text().strip() == first_commit

    (checkout / "note.txt").write_text("new supervisor\n")
    subprocess.run(["git", "add", "note.txt"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "new supervisor"], cwd=checkout, check=True)
    assert subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip() != first_commit

    # Reproduce an interrupted legacy cache whose path is not an independent
    # checkout. Repair must not add or prune shared Git worktree registrations.
    pipeline = Path(env["OM_PIPELINE_CACHE"]) / "clones" / first_commit
    shutil.rmtree(pipeline)
    pipeline.mkdir()
    (pipeline / "interrupted-cache").write_text("stale\n")
    worktrees_before = subprocess.check_output(
        ["git", "worktree", "list", "--porcelain"], cwd=checkout, text=True
    )

    resumed = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run"],
        cwd=checkout,
        env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "second-local")},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    claims = (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()
    assert len(claims) == 10
    assert {line.split("|")[2] for line in claims} == {first_commit}
    assert (pipeline / ".git").is_dir()
    worktrees_after = subprocess.check_output(
        ["git", "worktree", "list", "--porcelain"], cwd=checkout, text=True
    )
    assert worktrees_after == worktrees_before


def test_running_worker_is_unchanged_when_shared_checkout_commit_moves(
    tmp_path: Path,
) -> None:
    checkout, env = fixture_checkout(tmp_path)
    first_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    worker = run_worker(
        checkout,
        {**env, "TEST_PAUSE_FAMILY": "math500-s0"},
        tmp_path / "live-pull-local",
    )
    ready = Path(env["TEST_SHARED"]) / "work/pause-ready"
    for _ in range(200):
        if ready.exists():
            break
        time.sleep(0.05)
    assert ready.is_file()

    (checkout / "new-supervisor.txt").write_text("new\n")
    subprocess.run(["git", "add", "new-supervisor.txt"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "move shared checkout"], cwd=checkout, check=True)
    second_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    assert second_commit != first_commit
    (Path(env["TEST_SHARED"]) / "work/pause-release").touch()

    output, _ = worker.communicate(timeout=30)
    assert worker.returncode == 0, output
    claims = (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()
    assert len(claims) == 10
    assert {line.split("|")[2] for line in claims} == {first_commit}
    pipeline = Path(env["OM_PIPELINE_CACHE"]) / "clones" / first_commit
    assert (pipeline / ".git").is_dir()
    assert str(pipeline) not in subprocess.check_output(
        ["git", "worktree", "list", "--porcelain"], cwd=checkout, text=True
    )


def test_h100_worker_recovers_after_process_group_termination(tmp_path: Path) -> None:
    checkout, env = fixture_checkout(tmp_path)
    local = tmp_path / "restart-local"
    interrupted = subprocess.Popen(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run", "h100"],
        cwd=checkout,
        env={
            **env,
            "OM_LOCAL_LOCK_DIR": str(local),
            "TEST_INTERRUPT_FAMILY": "math500-s0",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    ready = Path(env["TEST_SHARED"]) / "work/interrupt-ready"
    for _ in range(200):
        if ready.exists():
            break
        time.sleep(0.05)
    assert ready.is_file()
    os.killpg(interrupted.pid, signal.SIGTERM)
    interrupted.communicate(timeout=10)
    assert interrupted.returncode != 0

    owner = (
        Path(env["TEST_SHARED"])
        / "work/runs/olmo3-1025-7b-base-rlzero-grpo-h100-v2/.families/math500-s0.owner.json"
    )
    # SIGKILL/job teardown may leave only the informational owner file. The
    # flock is the source of truth, so restart must replace and later remove it.
    assert owner.exists()

    resumed = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run", "h100"],
        cwd=checkout,
        env={
            **env,
            "OM_LOCAL_LOCK_DIR": str(local),
            "TEST_INTERRUPT_FAMILY": "math500-s0",
        },
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert not owner.exists()
    partial = (
        Path(env["TEST_SHARED"])
        / "work/runs/olmo3-1025-7b-base-rlzero-grpo-h100-v2/family-math500-s0/interrupted.partial"
    )
    assert partial.read_text().splitlines() == ["durable partial", "resumed"]
    claims = (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()
    assert len(claims) == 10
    assert len({line.split("|")[1] for line in claims}) == 10


def test_launcher_shares_the_node_primary_lock_with_other_experiments(
    tmp_path: Path,
) -> None:
    checkout, env = fixture_checkout(tmp_path)
    local = tmp_path / "locked-local"
    local.mkdir()
    stream = (local / "primary.lock").open("w")
    fcntl.flock(stream, fcntl.LOCK_EX)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run"],
        cwd=checkout,
        env={**env, "OM_LOCAL_LOCK_DIR": str(local)},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    fcntl.flock(stream, fcntl.LOCK_UN)
    stream.close()
    assert result.returncode != 0
    assert "another experiment already owns this physical node" in result.stdout
    assert not (Path(env["TEST_SHARED"]) / "work/claims").exists()


def test_supervisor_keepalive_covers_preflight_and_point_transitions() -> None:
    launcher = (ROOT / "scripts/run_olmo3_rlzero.sh").read_text()
    matrix = (ROOT / "scripts/run_matrix.sh").read_text()
    point = (ROOT / "scripts/run_point.sh").read_text()
    start = launcher.index('OM_GPU_KEEPALIVE_READY_FILE="$KEEPALIVE_READY"')
    signal = launcher.index("signal_qualify()")
    assert start < signal
    assert "export OM_EXTERNAL_GPU_KEEPALIVE=1" in launcher
    assert 'if [ "${OM_EXTERNAL_GPU_KEEPALIVE:-0}" = "1" ]' in point
    assert "git clone --quiet --no-hardlinks --no-checkout" in launcher
    assert 'remote remove origin' in launcher
    assert "git worktree" not in launcher
    assert "git worktree" not in matrix


def _run_config_digest(config: dict) -> str:
    import hashlib
    import json

    body = {k: v for k, v in config.items() if k != "digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_h100_passes_per_dataset_runtime_values_and_repairs_unfinished_points(tmp_path: Path) -> None:
    """math500 rollouts run at batch 32, mbpp at 16 with gradient micro-batch 1;
    a point recorded under the old values is rewritten before re-entry, a DONE
    point is not."""
    checkout, env = fixture_checkout(tmp_path)
    root = Path(env["TEST_SHARED"]) / "work/runs/olmo3-1025-7b-base-rlzero-grpo-h100-v2"
    tag = "olmo3-1025-7b-base-rlzero-grpo-h100-v2"
    family = root / "family-mbpp-s1"
    old = {"dataset": "mbpp", "seed": 1, "gen_batch": "8", "gradient_micro_batch": 4, "grpo_logprob_micro_batch": 4}
    done_run = family / f"{tag}-s1-mbpp-d0"
    live_run = family / f"{tag}-s1-mbpp-d25"
    for run, done in ((done_run, True), (live_run, False)):
        run.mkdir(parents=True)
        config = {**old, "drift": 0 if done else 25}
        config["digest"] = _run_config_digest(config)
        (run / "run_config.json").write_text(json.dumps(config, indent=1))
        if done:
            (run / "DONE").write_text("done\n")
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run", "h100"],
        cwd=checkout,
        env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "h100-local")},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    claims = (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()
    by_family = {line.split("|")[1]: line.split("|")[3:5] for line in claims}
    assert by_family["math500-s0"] == ["32", "4"]
    assert by_family["mbpp-s0"] == ["16", "1"]
    assert "[family] mbpp/s1 runtime: generation batch=16 gradient micro-batch=1" in result.stdout
    assert "[repair] " + live_run.name + ": gen_batch '8' -> '16'" in result.stdout
    assert "[repair] " + live_run.name + ": gradient_micro_batch 4 -> 1" in result.stdout
    assert done_run.name not in "".join(line for line in result.stdout.splitlines() if "[repair]" in line)
    live = json.loads((live_run / "run_config.json").read_text())
    assert live["gen_batch"] == "16" and live["gradient_micro_batch"] == 1
    assert live["digest"] == _run_config_digest(live)
    assert json.loads((done_run / "run_config.json").read_text())["gen_batch"] == "8"

    status = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "status", "h100", "verbose"],
        cwd=checkout,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    output = status.stdout + status.stderr
    assert status.returncode == 0, output
    assert "runtime_contract_errors=0" in output
    assert (
        "runtime_per_dataset math500:gen_batch=32,gradient_micro_batch=4 "
        "mbpp:gen_batch=16,gradient_micro_batch=1"
    ) in output


def test_run_accepts_a_dataset_filter_as_third_argument(tmp_path: Path) -> None:
    checkout, env = fixture_checkout(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run", "h100", "math500"],
        cwd=checkout,
        env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "filter-local")},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    claims = (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()
    families = {line.split("|")[1] for line in claims}
    assert families == {f"math500-s{seed}" for seed in range(5)}
    bad = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run", "h100", "gsm8k"],
        cwd=checkout,
        env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "filter-local-2")},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert bad.returncode == 2
    assert "unknown dataset filter: gsm8k" in bad.stdout + bad.stderr


def _order_fixture(tmp_path: Path):
    """mbpp/s3 has two finished points, math500/s2 was started, the rest untouched."""
    checkout, env = fixture_checkout(tmp_path)
    root = Path(env["TEST_SHARED"]) / "work/runs/olmo3-1025-7b-base-rlzero-grpo-h100-v2"
    tag = "olmo3-1025-7b-base-rlzero-grpo-h100-v2"
    family = root / "family-mbpp-s3"
    for drift in (0, 25):
        run = family / f"{tag}-s3-mbpp-d{drift}"
        run.mkdir(parents=True)
        config = {"dataset": "mbpp", "seed": 3, "drift": drift, "gen_batch": "16", "gradient_micro_batch": 1}
        config["digest"] = _run_config_digest(config)
        (run / "run_config.json").write_text(json.dumps(config))
        (run / "DONE").write_text("done\n")
    started = root / "family-math500-s2" / f"{tag}-s2-math500-d0"
    started.mkdir(parents=True)
    (started / "run_config.json").write_text(json.dumps({"dataset": "math500", "seed": 2, "gen_batch": "32", "gradient_micro_batch": 4}))
    return checkout, env


def _claims(env: dict[str, str]) -> list[str]:
    return [line.split("|")[1] for line in (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()]


def test_family_with_the_most_work_left_is_claimed_first(tmp_path: Path) -> None:
    """Seven workers, eight families, no node comes back once the GPU manager
    takes it: the family left waiting must be the shortest one. mbpp points cost
    about 1.35x math500 points and d400 about 1.7x a d0/d100 point, so an
    untouched mbpp family (4 points) goes first and the mbpp family with only
    d100+d400 left goes last."""
    checkout, env = _order_fixture(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run", "h100"],
        cwd=checkout,
        env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "order-local")},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    claims = _claims(env)
    assert claims[:4] == ["mbpp-s0", "mbpp-s1", "mbpp-s2", "mbpp-s4"], claims   # untouched mbpp, 4 points each
    assert claims[4] == "math500-s2", claims          # same work as other math500, but already started
    assert claims[5:9] == ["math500-s0", "math500-s1", "math500-s3", "math500-s4"], claims
    assert claims[9] == "mbpp-s3", claims             # only d100 + d400 left: least work, claimed last
    assert len(claims) == 10


def test_claim_order_progress_restores_the_2026_09_07_rule(tmp_path: Path) -> None:
    checkout, env = _order_fixture(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run", "h100"],
        cwd=checkout,
        env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "order-progress-local"), "OM_RLZERO_CLAIM_ORDER": "progress"},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    claims = _claims(env)
    assert claims[0] == "mbpp-s3"          # two points done
    assert claims[1] == "math500-s2"       # started, nothing done
    assert claims[2] == "math500-s0"       # untouched families in registered order
    assert len(claims) == 10


def test_cuda_runtime_faults_are_retried_without_tripping_the_loop_guard(tmp_path: Path) -> None:
    """Five CUDA runtime faults in a row exceed the loop guard (4) but are not
    the same failure repeating: the family is retried and finishes, and no
    .loop marker is written."""
    checkout, env = fixture_checkout(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run"],
        cwd=checkout,
        env={
            **env,
            "OM_LOCAL_LOCK_DIR": str(tmp_path / "cuda-local"),
            "TEST_CUDA_FAIL_FAMILY": "math500-s1",
            "TEST_CUDA_FAIL_TIMES": "5",
            "OM_RLZERO_FAMILY_ATTEMPTS": "1",
            "OM_RLZERO_STALE_PROCESS_TIMEOUT": "1",
            "OM_RLZERO_GPU_CLEANUP_TIMEOUT": "1",
        },
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[cuda-flaky] math500/s1: CUDA runtime fault #5" in result.stdout
    assert "[family-next] math500/s1 failed; this worker moves on to the next family now" in result.stdout
    assert "[family-loop]" not in result.stdout
    queue = Path(env["TEST_SHARED"]) / "work/runs/olmo3-1025-7b-base-rlzero-grpo-v1/.families"
    assert not (queue / "math500-s1.loop").exists()
    claims = [line.split("|")[1] for line in (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()]
    # five faults, each one handing the node to another family; it finishes last
    assert claims.index("math500-s1") > claims.index("mbpp-s1"), claims
    assert set(claims) == {f"{d}-s{s}" for s in range(5) for d in ("math500", "mbpp")}
    assert len(claims) == 10


def test_every_worker_clears_loop_markers_that_recorded_a_cuda_fault(tmp_path: Path) -> None:
    """No node has to be relaunched differently: a marker written for a CUDA
    runtime fault is cleared by whichever worker starts next; a marker for a
    real repeating failure (OOM) stays."""
    checkout, env = fixture_checkout(tmp_path)
    queue = Path(env["TEST_SHARED"]) / "work/runs/olmo3-1025-7b-base-rlzero-grpo-v1/.families"
    queue.mkdir(parents=True)
    (queue / "math500-s1.loop").write_text(
        "family=math500/s1 worker=w host=h consecutive_failures=4 last_rc=1\n"
        "last_error=RuntimeError: CUDA error: unspecified launch failure\nmarked_at_utc=2026-09-07T03:00:00Z\n"
    )
    (queue / "mbpp-s4.loop").write_text(
        "family=mbpp/s4 worker=w host=h consecutive_failures=4 last_rc=1\n"
        "last_error=torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 8.98 GiB\nmarked_at_utc=2026-09-07T03:00:00Z\n"
    )
    # a finished point re-entered under another generation batch (2026-09-08): fixed in run_matrix.sh
    (queue / "math500-s2.loop").write_text(
        "family=math500/s2 worker=w host=h consecutive_failures=4 last_rc=2\n"
        "last_error=[config-abort] existing artifacts use a different run config: ['gen_batch']\nmarked_at_utc=2026-09-08T00:00:00Z\n"
    )
    families = " ".join(f"{d}/s{s}" for s in range(5) for d in ("math500", "mbpp") if (d, s) != ("mbpp", 4))
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run"],
        cwd=checkout,
        env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "clear-local"), "OM_RLZERO_ONLY_FAMILIES": families},
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cleared loop marker math500-s1.loop: it recorded a CUDA runtime fault" in result.stdout
    assert "cleared loop marker math500-s2.loop: it recorded a re-entry config-abort" in result.stdout
    assert not (queue / "math500-s1.loop").exists()
    assert not (queue / "math500-s2.loop").exists()
    assert (queue / "mbpp-s4.loop").exists()
    claims = [line.split("|")[1] for line in (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()]
    assert "math500-s1" in claims and "math500-s2" in claims and "mbpp-s4" not in claims and len(claims) == 9


def test_repeated_cuda_faults_release_the_family_for_another_node(tmp_path: Path) -> None:
    checkout, env = fixture_checkout(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run"],
        cwd=checkout,
        env={
            **env,
            "OM_LOCAL_LOCK_DIR": str(tmp_path / "cuda-release-local"),
            "TEST_CUDA_FAIL_FAMILY": "math500-s1",
            "TEST_CUDA_FAIL_TIMES": "3",
            "OM_RLZERO_MAX_CUDA_FAILURES": "2",
            "OM_RLZERO_FAMILY_ATTEMPTS": "1",
            "OM_RLZERO_STALE_PROCESS_TIMEOUT": "1",
            "OM_RLZERO_GPU_CLEANUP_TIMEOUT": "1",
        },
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # the old wording claimed the family was "released for another node"; nothing
    # hands it to a node, this worker just moves on and anyone may take it
    assert "CUDA runtime faults in a row here" in result.stdout
    assert "this worker moves on" in result.stdout
    assert "[family-loop]" not in result.stdout
    claims = [line.split("|")[1] for line in (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()]
    # three faults, each one moving the worker to another family, then it finishes
    assert claims.index("math500-s1") > claims.index("mbpp-s1"), claims
    assert set(claims) == {f"{d}-s{s}" for s in range(5) for d in ("math500", "mbpp")}
    assert len(claims) == 10


def test_stale_cuda_line_does_not_exempt_a_real_repeating_failure(tmp_path: Path) -> None:
    """2026-09-08: five workers re-claimed the same family all night because an
    old CUDA line in a stage log classified every later failure as a CUDA
    fault. The attempt's own [point-failed] line decides now."""
    checkout, env = fixture_checkout(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run"],
        cwd=checkout,
        env={
            **env,
            "OM_LOCAL_LOCK_DIR": str(tmp_path / "stale-cuda-local"),
            "TEST_STALE_CUDA_FAIL_FAMILY": "math500-s1",
            "OM_RLZERO_MAX_FAMILY_FAILURES": "1",
            "OM_RLZERO_FAMILY_ATTEMPTS": "1",
            "OM_RLZERO_STALE_PROCESS_TIMEOUT": "1",
            "OM_RLZERO_GPU_CLEANUP_TIMEOUT": "1",
        },
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[cuda-flaky]" not in result.stdout
    assert "[family-loop] math500/s1 failed 1 times in a row (last: try 3/3 rc=2: [config-abort]" in result.stdout
    queue = Path(env["TEST_SHARED"]) / "work/runs/olmo3-1025-7b-base-rlzero-grpo-v1/.families"
    assert "last_error=try 3/3 rc=2: [config-abort]" in (queue / "math500-s1.loop").read_text()
    # Other primary families finish before admission of an independent experiment.
    claims = [line.split("|")[1] for line in (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()]
    assert "math500-s1" not in claims and len(claims) == 9
    assert "[queue] every family left is marked LOOPING: math500/s1" in result.stdout
    assert "[fixture-fallback] independent experiment admitted" in result.stdout


def test_worker_hands_off_when_only_looping_families_remain(tmp_path: Path) -> None:
    """A LOOPING family is skipped by every worker, so waiting for one is waiting
    for ever: the node held four GPUs all night printing '[queue] waiting for 1
    families' (2026-09-08)."""
    checkout, env = fixture_checkout(tmp_path)
    queue = Path(env["TEST_SHARED"]) / "work/runs/olmo3-1025-7b-base-rlzero-grpo-v1/.families"
    queue.mkdir(parents=True)
    (queue / "mbpp-s4.loop").write_text(
        "family=mbpp/s4 worker=w host=h consecutive_failures=4 last_rc=1\n"
        "last_error=torch.OutOfMemoryError: CUDA out of memory\nmarked_at_utc=2026-09-08T03:00:00Z\n"
    )
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run"],
        cwd=checkout,
        env={**env, "OM_LOCAL_LOCK_DIR": str(tmp_path / "looping-only-local")},
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[queue] every family left is marked LOOPING: mbpp/s4" in result.stdout
    assert "OM_RLZERO_CLEAR_LOOPS=1" in result.stdout
    claims = [line.split("|")[1] for line in (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()]
    assert "mbpp-s4" not in claims and len(claims) == 9
    assert (queue / "mbpp-s4.loop").is_file()
    assert "[fixture-fallback] independent experiment admitted" in result.stdout
    assert not list((queue.parent / ".workers").glob("*.json"))


def test_failed_family_is_retried_by_the_same_worker_before_moving_on(tmp_path: Path) -> None:
    checkout, env = fixture_checkout(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run"],
        cwd=checkout,
        env={
            **env,
            "OM_LOCAL_LOCK_DIR": str(tmp_path / "retry-same-local"),
            "TEST_FAIL_FAMILY": "math500-s2",
            "TEST_FAIL_RC": "1",
            "OM_RLZERO_FAMILY_ATTEMPTS": "1",
        },
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[family-next] math500/s2 failed; this worker moves on to the next family now" in result.stdout
    claims = [line.split("|")[1] for line in (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()]
    # the failed family does not hold the worker: every later family runs first,
    # and it is picked up again on a later pass (a failed attempt writes no claim)
    assert claims.index("math500-s2") > claims.index("mbpp-s2"), claims
    assert set(claims) == {f"{d}-s{s}" for s in range(5) for d in ("math500", "mbpp")}
    assert len(claims) == 10


def test_a_cuda_fault_that_reproduces_every_time_stops_being_exempt(tmp_path: Path) -> None:
    """A CUDA runtime fault is exempt from the loop guard because it is usually
    transient. With no lifetime bound, a fault that reproduces on every attempt
    cycled die -> wait -> die for ever and was never marked LOOPING."""
    checkout, env = fixture_checkout(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run"],
        cwd=checkout,
        env={
            **env,
            "OM_LOCAL_LOCK_DIR": str(tmp_path / "runtime-bound-local"),
            "TEST_CUDA_FAIL_FAMILY": "math500-s1",
            "TEST_CUDA_FAIL_TIMES": "99",
            "OM_RLZERO_MAX_RUNTIME_FAILURES": "1",
            "OM_RLZERO_FAMILY_ATTEMPTS": "1",
            "OM_RLZERO_STALE_PROCESS_TIMEOUT": "1",
            "OM_RLZERO_GPU_CLEANUP_TIMEOUT": "1",
        },
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert "every one a CUDA runtime fault" in result.stdout, result.stdout[-3000:]
    queue = Path(env["TEST_SHARED"]) / "work/runs/olmo3-1025-7b-base-rlzero-grpo-v1/.families"
    marker = (queue / "math500-s1.loop").read_text()
    assert "runtime_failures_total=" in marker
    # the other nine families still ran
    claims = [line.split("|")[1] for line in (Path(env["TEST_SHARED"]) / "work/claims").read_text().splitlines()]
    assert "math500-s1" not in claims and len(claims) == 9
