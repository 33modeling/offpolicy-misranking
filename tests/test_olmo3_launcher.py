"""Three-cluster assignment and commit-stable restart for the OLMo launcher."""

from __future__ import annotations

import fcntl
import os
import shutil
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
    shutil.copy2(ROOT / "configs/olmo3_rlzero.json", checkout / "configs")
    shutil.copy2(ROOT / "configs/olmo3_rlzero_h100.json", checkout / "configs")
    (checkout / "requirements.txt").write_text("fixture\n")
    (shared / "models/Olmo-3-1025-7B/config.json").write_text("{}\n")
    executable(checkout / "scripts/run_point.sh", "#!/bin/sh\nexit 0\n")
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
    while [ $# -gt 0 ]; do
      if [ "$1" = --output ]; then
        out=$2; mkdir -p "$out"
        for name in policy_train.json adapter_config.json adapter_model.safetensors optimizer.pt grpo_stats.jsonl; do printf 'pass\n' > "$out/$name"; done
        break
      fi
      shift
    done ;;
  -)
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
if [ "${TEST_FAIL_FAMILY:-}" = "$key" ]; then
  marker="$TEST_SHARED/work/fail-$key"
  mkdir -p "$(dirname "$marker")"
  if [ "${TEST_FAIL_ALWAYS:-0}" = 1 ] || [ ! -e "$marker" ]; then
    : > "$marker"
    exit 43
  fi
fi
git=$(git -C "$OM_PIPELINE_REPO" rev-parse HEAD)
printf '%s|%s|%s\n' "$WORKER_ID" "$key" "$git" >> "$TEST_SHARED/work/claims"
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
        "OM_RLZERO_QUEUE_WAIT_SECONDS": "1",
        "OM_RLZERO_FAMILY_RETRY_SECONDS": "0",
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


def test_failed_family_restarts_automatically_without_launcher_exit(tmp_path: Path) -> None:
    checkout, env = fixture_checkout(tmp_path)
    result = subprocess.run(
        ["/bin/bash", "scripts/run_olmo3_rlzero.sh", "run"],
        cwd=checkout,
        env={
            **env,
            "OM_LOCAL_LOCK_DIR": str(tmp_path / "retry-local"),
            "TEST_FAIL_FAMILY": "mbpp-s0",
        },
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[family-retry] mbpp/s0 rc=43" in result.stdout
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
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    failure_marker = Path(env["TEST_SHARED"]) / "work/fail-mbpp-s0"
    for _ in range(200):
        if failure_marker.exists():
            break
        time.sleep(0.05)
    assert failure_marker.exists()
    assert failed.poll() is None
    failed.terminate()
    failed_output, _ = failed.communicate(timeout=10)
    assert "[family-retry] mbpp/s0 rc=43" in failed_output
    marker = Path(env["TEST_SHARED"]) / "work/runs/olmo3-1025-7b-base-rlzero-grpo-v1/.queue/generation.git"
    assert marker.read_text().strip() == first_commit

    (checkout / "note.txt").write_text("new supervisor\n")
    subprocess.run(["git", "add", "note.txt"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "new supervisor"], cwd=checkout, check=True)
    assert subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip() != first_commit

    # Reproduce an interrupted node-local cache: Git still has a locked
    # registration while the path was replaced by a non-worktree directory.
    pipeline = Path(env["OM_PIPELINE_CACHE"]) / first_commit
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(pipeline), first_commit],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "worktree", "lock", str(pipeline)], cwd=checkout, check=True
    )
    shutil.rmtree(pipeline)
    pipeline.mkdir()
    (pipeline / "interrupted-cache").write_text("stale\n")

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
    point = (ROOT / "scripts/run_point.sh").read_text()
    start = launcher.index('OM_GPU_KEEPALIVE_READY_FILE="$KEEPALIVE_READY"')
    signal = launcher.index("signal_qualify()")
    assert start < signal
    assert "export OM_EXTERNAL_GPU_KEEPALIVE=1" in launcher
    assert 'if [ "${OM_EXTERNAL_GPU_KEEPALIVE:-0}" = "1" ]' in point
    assert 'worktree add -f -f --detach' in launcher
