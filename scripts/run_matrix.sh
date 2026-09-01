#!/usr/bin/env bash
# Regime discovery sweep. Run the same command on every independent cluster:
#
#   bash scripts/run_matrix.sh
#
# A shared flock queue assigns one seed x dataset family to each cluster.  A
# family creates one behavior pool and evaluates every drift on that exact pool.
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh
source scripts/_report_cache.sh

unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv python 없음: $PY"; exit 1; }
command -v flock >/dev/null || { echo "[abort] flock 없음"; exit 1; }
command -v setsid >/dev/null || { echo "[abort] setsid 없음"; exit 1; }

# Supervisors may be newer than a partially completed run. Generation always
# re-enters the immutable code snapshot recorded by that run.
SUPERVISOR_REPO="$PWD"
PIPELINE_EXPLICIT=0
if [ -n "${OM_PIPELINE_REPO+x}" ] || [ -n "${OM_PIPELINE_SCRIPT+x}" ]; then
  PIPELINE_EXPLICIT=1
fi
PIPELINE_REPO="${OM_PIPELINE_REPO:-$SUPERVISOR_REPO}"
PIPELINE_SCRIPT="${OM_PIPELINE_SCRIPT:-$PIPELINE_REPO/scripts/run_point.sh}"
[ -s "$PIPELINE_SCRIPT" ] || { echo "[abort] generation pipeline 없음: $PIPELINE_SCRIPT"; exit 1; }
PIPELINE_GIT=$(git -C "$PIPELINE_REPO" rev-parse HEAD 2>/dev/null) || {
  echo "[abort] generation pipeline is not a Git checkout: $PIPELINE_REPO"
  exit 1
}
PIPELINE_DIRTY=$(git -C "$PIPELINE_REPO" status --porcelain -- src scripts)
[ -z "$PIPELINE_DIRTY" ] || {
  echo "[abort] generation pipeline checkout is dirty: $PIPELINE_REPO"
  printf '%s\n' "$PIPELINE_DIRTY"
  exit 1
}

export MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/Qwen2.5-7B-Instruct}"
MODEL_TAG="${REGIME_MODEL_TAG:-$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')}"
ROOT="${REGIME_ROOT:-$OM_WORK/runs/regime-$MODEL_TAG}"
QUEUE="$ROOT/.queue"
RESULTS="${REGIME_RESULTS:-$OM_WORK/results/regime-$MODEL_TAG}"
QUARANTINE="${REGIME_QUARANTINE:-$OM_WORK/quarantine/regime-$MODEL_TAG}"
mkdir -p "$QUEUE" "$RESULTS"

# Bind an automatically created shared marker before any worker can initialize
# a run. This closes the empty-matrix race where different checkout revisions
# could otherwise claim different families concurrently.
GENERATION_MARKER="$QUEUE/generation.git"
GENERATION_CANDIDATE="${OM_GENERATION_GIT:-$PIPELINE_GIT}"
if [ -n "${OM_GENERATION_GIT:-}" ]; then
  if [ "$PIPELINE_GIT" != "$OM_GENERATION_GIT" ]; then
    echo "[abort] generation pipeline $PIPELINE_GIT conflicts with suite $OM_GENERATION_GIT"
    exit 1
  fi
  "$PY" src/regime_resume_commit.py "$ROOT" "$GENERATION_CANDIDATE" \
    --marker "$GENERATION_MARKER" --repair-to "$OM_GENERATION_GIT" \
    --quarantine-root "$QUARANTINE" || exit 1
fi
GENERATION_GIT=$("$PY" src/regime_resume_commit.py "$ROOT" "$GENERATION_CANDIDATE" \
  --marker "$GENERATION_MARKER") || exit 1
if [ -n "${OM_GENERATION_GIT:-}" ] && [ "$GENERATION_GIT" != "$OM_GENERATION_GIT" ]; then
  echo "[abort] matrix generation $GENERATION_GIT conflicts with suite $OM_GENERATION_GIT"
  exit 1
fi

# A new supervisor may resume a partial matrix, but generation must still use
# the exact source revision that initialized it. Materialize that revision in a
# node-local detached worktree so a pull between attempts remains restartable.
RESTORE_PIPELINE=0
if [ "$GENERATION_GIT" != "$PIPELINE_GIT" ]; then
  # A canonical launcher supplies OM_GENERATION_GIT and may deliberately run a
  # newer supervisor around an older immutable generation checkout.  Repair a
  # wrong/stale explicit checkout instead of asking the user to construct one.
  if [ "$PIPELINE_EXPLICIT" -eq 0 ] || [ -n "${OM_GENERATION_GIT:-}" ]; then
    RESTORE_PIPELINE=1
  fi
fi
if [ "$RESTORE_PIPELINE" -eq 1 ]; then
  PIPELINE_CACHE="${OM_PIPELINE_CACHE:-/tmp/offpolicy-misranking-$(id -u)/pipelines}"
  PIPELINE_REPO="$PIPELINE_CACHE/$GENERATION_GIT"
  mkdir -p "$PIPELINE_CACHE"
  (
    flock 7
    git -C "$SUPERVISOR_REPO" cat-file -e "$GENERATION_GIT^{commit}" 2>/dev/null || {
      echo "[abort] generation commit is unavailable locally: $GENERATION_GIT" >&2
      exit 1
    }
    recorded=$(git -C "$PIPELINE_REPO" rev-parse HEAD 2>/dev/null || true)
    dirty=invalid
    if [ "$recorded" = "$GENERATION_GIT" ]; then
      dirty=$(git -C "$PIPELINE_REPO" status --porcelain \
        -- src scripts 2>/dev/null || printf invalid)
    fi
    if [ "$recorded" != "$GENERATION_GIT" ] || [ -n "$dirty" ]; then
      if [ -e "$PIPELINE_REPO" ] || [ -L "$PIPELINE_REPO" ]; then
        stale="$PIPELINE_CACHE/.stale-$GENERATION_GIT-$(date +%s)-$$"
        mv -- "$PIPELINE_REPO" "$stale" || exit 1
        echo "[queue] invalid generation cache quarantined: $stale" >&2
      fi
      git -C "$SUPERVISOR_REPO" worktree prune --expire now || exit 1
      git -C "$SUPERVISOR_REPO" worktree add -f -f --detach "$PIPELINE_REPO" \
        "$GENERATION_GIT" >&2 || exit 1
    fi
    recorded=$(git -C "$PIPELINE_REPO" rev-parse HEAD 2>/dev/null) || {
      echo "[abort] invalid cached generation checkout: $PIPELINE_REPO" >&2
      exit 1
    }
    [ "$recorded" = "$GENERATION_GIT" ] || {
      echo "[abort] cached generation checkout is at $recorded, expected $GENERATION_GIT" >&2
      exit 1
    }
    dirty=$(git -C "$PIPELINE_REPO" status --porcelain -- src scripts)
    [ -z "$dirty" ] || {
      echo "[abort] cached generation checkout is dirty: $PIPELINE_REPO" >&2
      printf '%s\n' "$dirty" >&2
      exit 1
    }
  ) 7>"$PIPELINE_CACHE/.worktree.lock" || exit 1
  PIPELINE_SCRIPT="$PIPELINE_REPO/scripts/run_point.sh"
  [ -s "$PIPELINE_SCRIPT" ] || {
    echo "[abort] cached generation pipeline missing: $PIPELINE_SCRIPT"
    exit 1
  }
  PIPELINE_GIT="$GENERATION_GIT"
  echo "[queue] restored generation checkout=$PIPELINE_REPO"
fi
echo "[queue] generation_git=$GENERATION_GIT pipeline_git=$PIPELINE_GIT"

# Discovery uses three seeds.  Seeds 3-4 are a held-back confirmation extension.
SEEDS=(${REGIME_SEEDS:-0 1 2})
DATASETS=(${REGIME_DATASETS:-gsm8k math500})
DRIFTS=(${REGIME_DRIFTS:-0 25 100 400})
MAX_RETRIES="${REGIME_MAX_RETRIES:-3}"
CONTRACT="${REGIME_MATRIX:-}"
N_VAL_DEFAULT="${REGIME_N_VAL:-100}"
N_TRAIN_BY_DATASET="${REGIME_N_TRAIN_BY_DATASET:-}"
BEHAVIOR_K_DEFAULT="${REGIME_BEHAVIOR_K:-8}"
FRESH_K_DEFAULT="${REGIME_FRESH_K:-32}"
VAL_K_DEFAULT="${REGIME_VAL_K:-8}"
MICRO_GROUP_DEFAULT="${REGIME_MICRO_GROUP:-4}"
MAX_NEW_TOKENS_DEFAULT="${REGIME_MAX_NEW_TOKENS:-512}"
PROJ_DIM_DEFAULT="${REGIME_PROJ_DIM:-4096}"
GRAD_LAYERS_DEFAULT="${REGIME_GRAD_LAYERS:-4}"
CLIP_CAP_DEFAULT="${REGIME_CLIP_CAP:-10}"
TOPK_FRAC_DEFAULT="${REGIME_TOPK_FRAC:-0.10}"
TEMPERATURE_DEFAULT="${REGIME_TEMPERATURE:-1.0}"
GRPO_WORLD_SIZE_DEFAULT="${GRPO_WORLD_SIZE:-4}"
GRPO_GROUP_SIZE_DEFAULT="${GRPO_GROUP_SIZE:-8}"
GRPO_CLIP_EPSILON_DEFAULT="${GRPO_CLIP_EPSILON:-0.2}"
GRPO_LEARNING_RATE_DEFAULT="${GRPO_LEARNING_RATE:-1e-5}"
GRPO_EPOCHS_PER_BATCH_DEFAULT="${GRPO_EPOCHS_PER_BATCH:-2}"
GRPO_MAX_GRAD_NORM_DEFAULT="${GRPO_MAX_GRAD_NORM:-1.0}"
GRPO_ADVANTAGE_EPSILON_DEFAULT="${GRPO_ADVANTAGE_EPSILON:-1e-4}"
GRPO_LORA_RANK_DEFAULT="${GRPO_LORA_RANK:-16}"
GRPO_LORA_ALPHA_DEFAULT="${GRPO_LORA_ALPHA:-32}"
RLVR_METHOD_DEFAULT="${RLVR_METHOD:-grpo}"
export RLVR_METHOD="$RLVR_METHOD_DEFAULT"
WATCH_INTERVAL_SECONDS="${REGIME_WATCH_INTERVAL_SECONDS:-15}"
STALL_SECONDS="${REGIME_STALL_SECONDS:-$(( ${OM_STALL_MINUTES:-5} * 60 ))}"
WATCH_KILL_GRACE_SECONDS="${REGIME_WATCH_KILL_GRACE_SECONDS:-5}"
WATCH_GPU_SAMPLES="${REGIME_WATCH_GPU_SAMPLES:-3}"

n_train_for_dataset() {
  local dataset=$1 item key value
  for item in $N_TRAIN_BY_DATASET; do
    key=${item%%=*}
    value=${item#*=}
    if [ "$key" = "$dataset" ]; then
      case "$value" in
        ''|*[!0-9]*|0) echo "[abort] invalid candidate count: $item" >&2; return 2 ;;
      esac
      printf '%s\n' "$value"
      return
    fi
  done
  if [ -n "$N_TRAIN_BY_DATASET" ]; then
    echo "[abort] candidate-count mapping is missing dataset: $dataset" >&2
    return 2
  fi
  if [ -n "${REGIME_N_TRAIN:-}" ]; then
    printf '%s\n' "$REGIME_N_TRAIN"
  elif [ "$dataset" = "math500" ]; then
    printf '400\n'
  else
    printf '512\n'
  fi
}

for value_name in WATCH_INTERVAL_SECONDS STALL_SECONDS WATCH_GPU_SAMPLES; do
  value=${!value_name}
  case "$value" in
    ''|*[!0-9]*|0) echo "[abort] invalid $value_name=$value"; exit 2 ;;
  esac
done
case "$WATCH_KILL_GRACE_SECONDS" in
  ''|*[!0-9]*) echo "[abort] invalid WATCH_KILL_GRACE_SECONDS=$WATCH_KILL_GRACE_SECONDS"; exit 2 ;;
esac

if [ -n "$CONTRACT" ] && [ ! -s "$CONTRACT" ]; then
  echo "[abort] regime matrix contract 없음: $CONTRACT"
  exit 1
fi

contract_run() {
  local command=$1 run=$2 dataset=$3 seed=$4 drift=$5 source=$6
  shift 6
  local args=("$PY" src/regime_contract.py "$command" --matrix "$CONTRACT"
    --run "$run" --dataset "$dataset" --seed "$seed" --drift "$drift")
  [ -z "$source" ] || args+=(--behavior-source "$source")
  "${args[@]}" "$@"
}

run_dir() {
  printf '%s/%s-s%s-%s-d%s\n' "$ROOT" "$MODEL_TAG" "$2" "$1" "$3"
}

run_complete() {
  local run=$1 dataset=$2 seed=$3 drift=$4 source=$5 artifact
  local previous_drift=0 expected_parent="" candidate
  for candidate in "${DRIFTS[@]}"; do
    [ "$candidate" = "$drift" ] && break
    previous_drift=$candidate
  done
  if [ "$previous_drift" -gt 0 ]; then
    expected_parent="$(run_dir "$dataset" "$seed" "$previous_drift")/policy_step_$previous_drift"
  fi
  for artifact in DONE run_config.json manifest.json score_protocol.json \
      oracle_protocol.json report.json scores_oracle.json scores_offpolicy.json \
      scores_splithalf.json divergence_stats.json oracle_micro_groups.pt val_groups.pt; do
    [ -s "$run/$artifact" ] || return 1
  done
  MODEL_PATH="$MODEL_PATH" DATASET="$dataset" SEED="$seed" DRIFT="$drift" \
    EXPECTED_N_TRAIN="$(n_train_for_dataset "$dataset")" \
    BEHAVIOR_SOURCE="$source" EXPECTED_GRPO_START="$previous_drift" \
    EXPECTED_GRPO_RESUME="$expected_parent" \
    "$PY" - "$run/run_config.json" <<'PYEOF' \
      >/dev/null 2>&1 || return 1
import json
import os
import sys
from pathlib import Path

config = json.load(open(sys.argv[1]))
dataset = os.environ["DATASET"]
drift = int(os.environ["DRIFT"])
n_train = int(os.environ["EXPECTED_N_TRAIN"])
expected = {
    "model_resolved": str(Path(os.environ["MODEL_PATH"]).resolve()),
    "dataset": dataset,
    "seed": int(os.environ["SEED"]),
    "drift": drift,
    "n_train": n_train,
    "n_val": int(os.environ.get("REGIME_N_VAL", "100")),
    "behavior_k": int(os.environ.get("REGIME_BEHAVIOR_K", "8")),
    "fresh_k": int(os.environ.get("REGIME_FRESH_K", "32")),
    "val_k": int(os.environ.get("REGIME_VAL_K", "8")),
    "micro_group": int(os.environ.get("REGIME_MICRO_GROUP", "4")),
    "max_new_tokens": int(os.environ.get("REGIME_MAX_NEW_TOKENS", "512")),
    "proj_dim": int(os.environ.get("REGIME_PROJ_DIM", "4096")),
    "grad_layers": int(os.environ.get("REGIME_GRAD_LAYERS", "4")),
    "clip_cap": float(os.environ.get("REGIME_CLIP_CAP", "10")),
    "temperature": float(os.environ.get("REGIME_TEMPERATURE", "1.0")),
    "topk_frac": float(os.environ.get("REGIME_TOPK_FRAC", "0.10")),
    "top_p": float(os.environ.get("OM_TOP_P", "1.0")),
    "thinking": os.environ.get("OM_THINKING", "off"),
    "prompt_format": os.environ.get("OM_PROMPT_FORMAT", "tokenizer_chat"),
    "attn": os.environ.get("OM_ATTN", "eager"),
    "lora_targets": os.environ.get("OM_LORA_TARGETS"),
    "skip_hybrid": os.environ.get("OM_SKIP_HYBRID", "1"),
    "training_objective": "base_control" if drift == 0 else os.environ.get("RLVR_METHOD", "grpo"),
    "policy_update": (
        "none"
        if drift == 0
        else (
            "reinforce_leave_one_out"
            if os.environ.get("RLVR_METHOD", "grpo") == "rloo"
            else "clipped_policy_gradient"
        )
    ),
    "reward_source": "none" if drift == 0 else "verifier",
    "supervised_loss": False,
    "positive_only_filter": False,
    "grpo_world_size": int(os.environ.get("GRPO_WORLD_SIZE", "4")),
    "grpo_group_size": int(os.environ.get("GRPO_GROUP_SIZE", "8")),
    "grpo_clip_epsilon": float(os.environ.get("GRPO_CLIP_EPSILON", "0.2")),
    "grpo_learning_rate": float(os.environ.get("GRPO_LEARNING_RATE", "1e-5")),
    "grpo_reference_kl_beta": 0.0,
    "grpo_epochs_per_batch": int(os.environ.get("GRPO_EPOCHS_PER_BATCH", "2")),
    "grpo_max_grad_norm": float(os.environ.get("GRPO_MAX_GRAD_NORM", "1.0")),
    "grpo_advantage_epsilon": float(os.environ.get("GRPO_ADVANTAGE_EPSILON", "1e-4")),
    "grpo_lora_rank": int(os.environ.get("GRPO_LORA_RANK", "16")),
    "grpo_lora_alpha": int(os.environ.get("GRPO_LORA_ALPHA", "32")),
    "grpo_start_step": int(os.environ["EXPECTED_GRPO_START"]),
    "grpo_resume_adapter": os.environ["EXPECTED_GRPO_RESUME"] or None,
    "behavior_source": os.environ["BEHAVIOR_SOURCE"] or None,
}
errors = [key for key, value in expected.items() if config.get(key) != value]
if errors:
    raise SystemExit("run config mismatch: " + ", ".join(errors))
PYEOF
  PYTHONPATH="$PIPELINE_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" - "$run" <<'PYEOF' >/dev/null 2>&1 || return 1
import json
import math
import sys
from pathlib import Path

from gate_rules import has_valid_analysis_protocol
from score_artifacts import load_complete_score_artifacts

run = Path(sys.argv[1])
if not has_valid_analysis_protocol(run):
    raise SystemExit("score/oracle protocol validation failed")
artifacts = load_complete_score_artifacts(run)
config = json.loads((run / "run_config.json").read_text())
expected_ids = set(range(int(config["n_train"])))
if set(artifacts.oracle) != expected_ids:
    raise SystemExit("score artifacts do not cover every candidate prompt")
halves = json.loads((run / "scores_splithalf.json").read_text())
for row in halves.values():
    if not isinstance(row, dict) or set(row) != {"r", "a", "b"}:
        raise SystemExit("scores_splithalf.json lacks exact R/A/B scores")
    if not all(math.isfinite(float(row[key])) for key in ("r", "a", "b")):
        raise SystemExit("scores_splithalf.json contains non-finite scores")
PYEOF
  if [ "$drift" -gt 0 ]; then
    for artifact in policy_train.json adapter_config.json adapter_model.safetensors \
        optimizer.pt grpo_stats.jsonl; do
      [ -s "$run/policy_step_$drift/$artifact" ] || return 1
    done
    if [ -z "$CONTRACT" ]; then
    MODEL_PATH="$MODEL_PATH" SEED="$seed" \
      REGIME_MAX_NEW_TOKENS="$MAX_NEW_TOKENS_DEFAULT" \
      GRPO_GROUP_SIZE="$GRPO_GROUP_SIZE_DEFAULT" \
      GRPO_CLIP_EPSILON="$GRPO_CLIP_EPSILON_DEFAULT" \
      GRPO_LEARNING_RATE="$GRPO_LEARNING_RATE_DEFAULT" \
      GRPO_EPOCHS_PER_BATCH="$GRPO_EPOCHS_PER_BATCH_DEFAULT" \
      GRPO_MAX_GRAD_NORM="$GRPO_MAX_GRAD_NORM_DEFAULT" \
      GRPO_ADVANTAGE_EPSILON="$GRPO_ADVANTAGE_EPSILON_DEFAULT" \
      GRPO_LORA_RANK="$GRPO_LORA_RANK_DEFAULT" \
      GRPO_LORA_ALPHA="$GRPO_LORA_ALPHA_DEFAULT" \
      GRPO_CHECKPOINT_EVERY="${GRPO_CHECKPOINT_EVERY:-5}" \
      PYTHONPATH="$SUPERVISOR_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY" - "$run/policy_step_$drift" "$drift" \
        "$GRPO_WORLD_SIZE_DEFAULT" "$RLVR_METHOD_DEFAULT" "$previous_drift" \
        "$expected_parent" <<'PYEOF' >/dev/null || return 1
import sys
from pathlib import Path
import os
from train_policy_grpo import GrpoConfig, validate_policy_lineage
config = GrpoConfig(
    group_size=int(os.environ.get("GRPO_GROUP_SIZE", "8")),
    clip_epsilon=float(os.environ.get("GRPO_CLIP_EPSILON", "0.2")),
    learning_rate=float(os.environ.get("GRPO_LEARNING_RATE", "1e-5")),
    epochs_per_batch=int(os.environ.get("GRPO_EPOCHS_PER_BATCH", "2")),
    max_grad_norm=float(os.environ.get("GRPO_MAX_GRAD_NORM", "1.0")),
    advantage_epsilon=float(os.environ.get("GRPO_ADVANTAGE_EPSILON", "1e-4")),
    lora_rank=int(os.environ.get("GRPO_LORA_RANK", "16")),
    lora_alpha=int(os.environ.get("GRPO_LORA_ALPHA", "32")),
    checkpoint_every=int(os.environ.get("GRPO_CHECKPOINT_EVERY", "5")),
)
validate_policy_lineage(
    Path(sys.argv[1]), target_steps=int(sys.argv[2]), world_size=int(sys.argv[3]),
    training_objective=sys.argv[4], expected_start_step=int(sys.argv[5]),
    expected_parent=Path(sys.argv[6]) if sys.argv[6] else None,
    expected_model=Path(os.environ["MODEL_PATH"]),
    expected_seed=int(os.environ["SEED"]),
    expected_max_new_tokens=int(os.environ.get("REGIME_MAX_NEW_TOKENS", "512")),
    expected_prompt_format=os.environ.get("OM_PROMPT_FORMAT", "tokenizer_chat"),
    expected_config=vars(config),
    expected_prompts=Path(sys.argv[1]).parent / "prompts.json",
)
PYEOF
    fi
  fi
  [ -z "$CONTRACT" ] || contract_run check-run "$run" "$dataset" "$seed" "$drift" "$source" \
    >/dev/null 2>&1
}

family_complete() {
  local dataset=$1 seed=$2 drift source
  source=$(run_dir "$dataset" "$seed" 0)
  for drift in "${DRIFTS[@]}"; do
    run_complete "$(run_dir "$dataset" "$seed" "$drift")" "$dataset" "$seed" "$drift" \
      "$([ "$drift" = 0 ] || printf '%s' "$source")" || return 1
  done
}

quarantine_prompt_target() {
  local run=$1 try=$2 destination suffix
  mkdir -p "$QUARANTINE" || return 1
  suffix="$(date +%Y%m%dT%H%M%S)-$$-$try"
  destination="$QUARANTINE/$(basename "$run")-prompt-mismatch-$suffix"
  mv -- "$run" "$destination" || return 1
  echo "[prompt-rebuild] quarantined $run -> $destination"
}

if [ "$GENERATION_GIT" != "$PIPELINE_GIT" ]; then
  incomplete=0
  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      family_complete "$dataset" "$seed" || incomplete=1
    done
  done
  if [ "$incomplete" -eq 1 ]; then
    echo "[abort] partial matrix requires generation commit $GENERATION_GIT"
    echo "[abort] use OM_PIPELINE_REPO=<clean checkout at $GENERATION_GIT>"
    exit 1
  fi
  echo "[queue] completed source matrix; current checkout is analysis-only"
fi

group_cpu_seconds() {
  ps -eo pgid=,cputimes= 2>/dev/null \
    | awk -v pgid="$1" '$1 == pgid { total += $2 } END { print total + 0 }'
}

gpu_peak_util() {
  local peak=0 util sample
  for sample in $(seq 1 "$WATCH_GPU_SAMPLES"); do
    util=$(timeout 10 nvidia-smi --query-gpu=utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk '$1 > peak { peak=$1 } END { print peak + 0 }')
    [ "${util:-0}" -le "$peak" ] || peak=$util
    [ "$sample" -eq "$WATCH_GPU_SAMPLES" ] || /bin/sleep 2
  done
  printf '%s\n' "$peak"
}

terminate_process_group() {
  local pgid=$1
  [ -n "$pgid" ] || return 0
  kill -TERM -- "-$pgid" 2>/dev/null || true
  /bin/sleep "$WATCH_KILL_GRACE_SECONDS"
  kill -KILL -- "-$pgid" 2>/dev/null || true
}

ACTIVE_PGID=""
ACTIVE_WATCHER=""
cleanup_active_pipeline() {
  [ -z "$ACTIVE_PGID" ] || terminate_process_group "$ACTIVE_PGID"
  [ -z "$ACTIVE_WATCHER" ] || kill "$ACTIVE_WATCHER" 2>/dev/null || true
  ACTIVE_PGID=""
  ACTIVE_WATCHER=""
}

run_pipeline_watchdog() {  # run_pipeline_watchdog <run> <attempt-log> <command...>
  local run=$1 attempt_log=$2 runner_pid watcher_pid rc
  shift 2
  local prev="" elapsed=0 cpu_mark cpu_now cpu_delta gpu_peak lf line sig
  local candidates=()

  mkdir -p "$run/logs" || return 1
  : > "$attempt_log" || return 1
  setsid "$@" >> "$attempt_log" 2>&1 &
  runner_pid=$!
  ACTIVE_PGID=$runner_pid

  (
    cpu_mark=$(group_cpu_seconds "$runner_pid")
    while kill -0 -- "-$runner_pid" 2>/dev/null; do
      /bin/sleep "$WATCH_INTERVAL_SECONDS"
      kill -0 -- "-$runner_pid" 2>/dev/null || break

      candidates=("$attempt_log")
      for lf in "$run"/logs/*.log; do
        [ -f "$lf" ] && candidates+=("$lf")
      done
      lf=$(ls -t -- "${candidates[@]}" 2>/dev/null | head -1)
      line=$(tail -n 1 "$lf" 2>/dev/null | cut -c1-160)
      sig=$(stat -c '%n:%y:%s' "$lf" 2>/dev/null || true)
      if [ -n "$sig" ] && [ "$sig" != "$prev" ]; then
        [ -n "$line" ] && echo "[regime-detail·$(basename "$lf" .log)] $line"
        prev=$sig
        elapsed=0
        cpu_mark=$(group_cpu_seconds "$runner_pid")
        continue
      fi

      elapsed=$((elapsed + WATCH_INTERVAL_SECONDS))
      [ "$elapsed" -lt "$STALL_SECONDS" ] || {
        cpu_now=$(group_cpu_seconds "$runner_pid")
        cpu_delta=$((cpu_now > cpu_mark ? cpu_now - cpu_mark : 0))
        gpu_peak=$(gpu_peak_util)
        if [ "$gpu_peak" -gt 0 ] || [ "$cpu_delta" -gt 2 ]; then
          message="[regime-watchdog] 로그 ${STALL_SECONDS}초 무변화지만 계산 활동 확인 (GPU ${gpu_peak}%, CPU +${cpu_delta}s) — 계속 실행"
          echo "$message"
          printf '%s\n' "$message" >> "$attempt_log"
          cpu_mark=$cpu_now
        else
          message="[regime-watchdog] 로그·GPU·CPU ${STALL_SECONDS}초 정지 — process group 종료 후 .partial 재개"
          echo "$message"
          printf '%s\n' "$message" >> "$attempt_log"
          terminate_process_group "$runner_pid"
          break
        fi
        elapsed=0
      }
    done
  ) &
  watcher_pid=$!
  ACTIVE_WATCHER=$watcher_pid

  wait "$runner_pid"
  rc=$?
  kill "$watcher_pid" 2>/dev/null || true
  wait "$watcher_pid" 2>/dev/null || true
  ACTIVE_PGID=""
  ACTIVE_WATCHER=""
  return "$rc"
}

run_point() {
  local dataset=$1 seed=$2 drift=$3 source=$4 resume_step=$5 resume_run=$6
  local run try n_train attempt_log rc prompt_root prompt_env
  run=$(run_dir "$dataset" "$seed" "$drift")
  n_train=$(n_train_for_dataset "$dataset") || return 43
  if [ -n "$CONTRACT" ]; then
    contract_run prepare-run "$run" "$dataset" "$seed" "$drift" "$source" \
      --quarantine-root "$QUARANTINE" || return 1
  fi
  run_complete "$run" "$dataset" "$seed" "$drift" "$source" && return 0

  # A partial target may have been initialized on a node that resolved a
  # different local dataset copy. Quarantine it before any retry; never mix its
  # prompt-dependent artifacts with the immutable d0 behavior family.
  if [ -n "$source" ] && [ -s "$source/prompts.json" ] \
      && [ -s "$source/run_config.json" ] && [ -s "$run/run_config.json" ]; then
    mkdir -p "$run/logs" || return 43
    "$PY" src/reuse_behavior.py --sync-prompts "$source" "$run" \
      >> "$run/logs/behavior-reuse.log" 2>&1
    rc=$?
    if [ "$rc" -eq 42 ]; then
      quarantine_prompt_target "$run" preflight || return 43
    elif [ "$rc" -ne 0 ]; then
      tail -20 "$run/logs/behavior-reuse.log" 2>/dev/null || true
      return 43
    fi
  fi

  # Legacy generation commits still run their own prep stage. Feed that old code a
  # source-derived local snapshot so it reconstructs byte-identical prompts
  # without changing the recorded generation revision or run configuration.
  prompt_root=""
  prompt_env=""
  case "$dataset" in
    gsm8k) prompt_env=GSM8K_DIR ;;
    math500) prompt_env=MATH500_DIR ;;
    mbpp) prompt_env=MBPP_DIR ;;
    kk) prompt_env=KK_DIR ;;
    arc-challenge) prompt_env=ARC_CHALLENGE_DIR ;;
    mmlu-pro-nonmath) prompt_env=MMLU_PRO_DIR ;;
  esac
  if [ -n "$source" ] && [ -n "$prompt_env" ]; then
    # experiment.py fixes the dataset split at seed 0 for every experiment
    # seed. Experiment seeds control rollout/training RNG, not pool membership.
    prompt_root=$("$PY" src/materialize_prompt_dataset.py "$source/prompts.json" \
      "$dataset" "$QUEUE/prompt-datasets" --seed 0) || return 43
  fi
  for try in $(seq 1 "$MAX_RETRIES"); do
    echo "[$(date '+%F %T')] $dataset/s$seed/d$drift try $try/$MAX_RETRIES -> $run"
    args=(env DATASET="$dataset" SEED="$seed" DRIFT="$drift" OUT_ROOT="$run"
      N_TRAIN="$n_train" N_VAL="$N_VAL_DEFAULT" BEHAVIOR_K="$BEHAVIOR_K_DEFAULT"
      FRESH_K="$FRESH_K_DEFAULT" VAL_K="$VAL_K_DEFAULT"
      MICRO_GROUP="$MICRO_GROUP_DEFAULT" PROJ_DIM="$PROJ_DIM_DEFAULT"
      GRAD_LAYERS="$GRAD_LAYERS_DEFAULT" CLIP_CAP="$CLIP_CAP_DEFAULT"
      TOPK_FRAC="$TOPK_FRAC_DEFAULT" MAX_NEW_TOKENS="$MAX_NEW_TOKENS_DEFAULT"
      TEMPERATURE="$TEMPERATURE_DEFAULT"
      GRPO_WORLD_SIZE="$GRPO_WORLD_SIZE_DEFAULT"
      GRPO_GROUP_SIZE="$GRPO_GROUP_SIZE_DEFAULT"
      GRPO_CLIP_EPSILON="$GRPO_CLIP_EPSILON_DEFAULT"
      GRPO_LEARNING_RATE="$GRPO_LEARNING_RATE_DEFAULT"
      GRPO_EPOCHS_PER_BATCH="$GRPO_EPOCHS_PER_BATCH_DEFAULT"
      GRPO_MAX_GRAD_NORM="$GRPO_MAX_GRAD_NORM_DEFAULT"
      GRPO_ADVANTAGE_EPSILON="$GRPO_ADVANTAGE_EPSILON_DEFAULT"
      GRPO_LORA_RANK="$GRPO_LORA_RANK_DEFAULT"
      GRPO_LORA_ALPHA="$GRPO_LORA_ALPHA_DEFAULT"
      OM_SKIP_HYBRID="${OM_SKIP_HYBRID:-1}" OM_RETRY_INDEX="$try")
    [ -z "$source" ] || args+=(OM_BEHAVIOR_SOURCE="$source")
    [ -z "$prompt_root" ] || args+=("$prompt_env=$prompt_root")
    if [ -n "$resume_run" ]; then
      args+=(OM_GRPO_START_STEP="$resume_step"
        OM_GRPO_RESUME_ADAPTER="$resume_run/policy_step_$resume_step"
        OM_GRPO_RESUME_OPTIMIZER="$resume_run/policy_step_$resume_step/optimizer.pt")
    fi
    attempt_log="$run/logs/regime-attempt-$try.log"
    if run_pipeline_watchdog "$run" "$attempt_log" \
        "${args[@]}" OM_REPO="$PIPELINE_REPO" \
        PYTHONPATH="$PIPELINE_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
        bash "$PIPELINE_SCRIPT"; then
      if [ -n "$CONTRACT" ]; then
        if ! contract_run check-run "$run" "$dataset" "$seed" "$drift" "$source" \
          --deep --mark; then
          echo "[contract-fail] $dataset/s$seed/d$drift 산출물 격리 후 재시도"
          contract_run prepare-run "$run" "$dataset" "$seed" "$drift" "$source" \
            --quarantine-root "$QUARANTINE" || return 1
          continue
        fi
      fi
      run_complete "$run" "$dataset" "$seed" "$drift" "$source" && return 0
    else
      rc=$?
      if [ "$rc" -ne 42 ] && [ "$rc" -ne 43 ] && grep -Eq \
          'prompts.json differs from the requested dataset/split|prompts.json: content hash differs' \
          "$attempt_log" 2>/dev/null; then
        rc=42
      fi
      if [ "$rc" -ne 42 ] && [ "$rc" -ne 43 ] && grep -Eq \
          'prompts\.(train|val).*qualified snapshot|prompts differ from the qualified matrix' \
          "$attempt_log" 2>/dev/null; then
        rc=43
      fi
      if [ "$rc" -eq 42 ] && [ -n "$source" ]; then
        if quarantine_prompt_target "$run" "$try"; then
          continue
        fi
        echo "[abort] prompt-mismatch quarantine failed: $run"
        return 43
      fi
      if [ "$rc" -eq 42 ] || [ "$rc" -eq 43 ]; then
        echo "[abort] permanent prompt/contract failure rc=$rc: $run"
        return 43
      fi
    fi
    bash scripts/diagnose_run_failure.sh "$run" "$attempt_log" 1 2>/dev/null || true
    sleep 20
  done
  return 1
}

run_family() {
  local dataset=$1 seed=$2 source drift rc previous_step=0 previous_run=""
  source=$(run_dir "$dataset" "$seed" 0)
  # d0 is the exact positive control: beta=pi with independent rollout noise.
  run_point "$dataset" "$seed" 0 "" "" ""
  rc=$?; [ "$rc" -eq 0 ] || return "$rc"
  for drift in "${DRIFTS[@]}"; do
    [ "$drift" = 0 ] && continue
    run_point "$dataset" "$seed" "$drift" "$source" "$previous_step" "$previous_run"
    rc=$?; [ "$rc" -eq 0 ] || return "$rc"
    previous_step=$drift
    previous_run=$(run_dir "$dataset" "$seed" "$drift")
  done
}

echo "== regime queue: model=$MODEL_TAG seeds=${SEEDS[*]} data=${DATASETS[*]} drift=${DRIFTS[*]}"
failures=0
while :; do
  remaining=0
  claimed=0
  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      family_complete "$dataset" "$seed" && continue
      remaining=$((remaining + 1))
      lock="$QUEUE/$dataset-s$seed.lock"
      (
        trap 'cleanup_active_pipeline; exit 130' INT TERM HUP
        flock -n 9 || exit 75
        family_complete "$dataset" "$seed" && exit 0
        run_family "$dataset" "$seed"
      ) 9>"$lock"
      rc=$?
      [ "$rc" -eq 75 ] && continue
      if [ "$rc" -eq 43 ]; then
        echo "[abort] permanent family contract failure: $dataset/s$seed"
        exit 43
      fi
      claimed=$((claimed + 1))
      if [ "$rc" -ne 0 ]; then
        echo "[family-fail] $dataset/s$seed"
        failures=$((failures + 1))
      fi
    done
  done
  [ "$remaining" -eq 0 ] && break
  if [ "$claimed" -eq 0 ]; then
    echo "[queue] 다른 클러스터의 ${remaining}개 family 완료 대기"
    sleep 60
  elif [ "$failures" -gt 0 ]; then
    echo "[abort] 이 worker에서 family 실패 ${failures}개; 같은 명령으로 artifact부터 재개"
    exit 1
  fi
done

# A suite-level supervisor may own aggregation across multiple isolated family
# roots. Each family has already passed run_complete above; avoid publishing an
# underpowered one-seed report and let the suite collect the full matrix once.
if [ "${REGIME_SKIP_COLLECTION:-0}" = 1 ]; then
  echo "== regime family complete; suite-level collection deferred"
  exit 0
fi

# Aggregate publication is serialized across workers. Machine outputs are
# JSON/CSV; the sole human-facing output is FINAL_REPORT.md.
if ! (
  flock 9
  runs=()
  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for drift in "${DRIFTS[@]}"; do
        run=$(run_dir "$dataset" "$seed" "$drift")
        source=$(run_dir "$dataset" "$seed" 0)
        point_source=""; [ "$drift" = 0 ] || point_source="$source"
        run_complete "$run" "$dataset" "$seed" "$drift" "$point_source" \
          || { echo "[collect-abort] incomplete: $run"; exit 1; }
        runs+=("$run")
      done
    done
  done
  if [ -n "$CONTRACT" ] && "$PY" src/regime_contract.py check-collection \
      --matrix "$CONTRACT" --results "$RESULTS" --runs "${runs[@]}" \
      >/dev/null 2>&1; then
    echo "[collect] matrix-bound analysis already current"
    exit 0
  fi

  # Discovery has no external matrix contract. Bind the analysis publication
  # to its exact completed inputs so every waiting cluster does not rerun the
  # same bootstrap regime map after acquiring collect.lock.
  analysis_code=(
    src/regime_map.py src/gate_rules.py src/score_artifacts.py
    src/select_rules.py src/first_interval.py src/run_provenance.py
  )
  analysis_inputs=()
  for run in "${runs[@]}"; do
    for artifact in DONE run_config.json manifest.json score_protocol.json \
        oracle_protocol.json report.json scores_oracle.json scores_offpolicy.json \
        scores_splithalf.json divergence_stats.json oracle_micro_groups.pt val_groups.pt; do
      analysis_inputs+=("$run/$artifact")
    done
  done
  [ -z "$CONTRACT" ] || analysis_inputs+=("$CONTRACT")
  analysis_key=$(report_cache_key "${analysis_code[@]}" -- "${analysis_inputs[@]}") \
    || exit 1
  analysis_key=$(report_cache_key_values "$analysis_key" \
    "first_bootstrap=${REGIME_FIRST_BOOTSTRAP:-10000}") || exit 1
  analysis_marker="$RESULTS/.regime_analysis.key"
  analysis_outputs=(
    "$RESULTS/REGIME.json" "$RESULTS/REGIME.csv"
    "$RESULTS/REGIME_SUMMARY.csv" "$RESULTS/FINAL_REPORT.md"
  )
  if report_cache_hit "$analysis_marker" "$analysis_key" "${analysis_outputs[@]}"; then
    echo "[collect] regime analysis already current; duplicate aggregation skipped"
    exit 0
  fi
  if [ -n "$CONTRACT" ]; then
    for seed in "${SEEDS[@]}"; do
      for dataset in "${DATASETS[@]}"; do
        for drift in "${DRIFTS[@]}"; do
          run=$(run_dir "$dataset" "$seed" "$drift")
          source=$(run_dir "$dataset" "$seed" 0)
          point_source=""; [ "$drift" = 0 ] || point_source="$source"
          contract_run check-run "$run" "$dataset" "$seed" "$drift" "$point_source" \
            --deep --mark || exit 1
        done
      done
    done
  fi
  analysis_args=(--output-dir "$RESULTS"
    --first-bootstrap "${REGIME_FIRST_BOOTSTRAP:-10000}")
  "$PY" src/regime_map.py "${runs[@]}" "${analysis_args[@]}" || exit 1
  for output in "${analysis_outputs[@]}"; do
    [ -s "$output" ] || { echo "[collect-abort] empty analysis output: $output"; exit 1; }
  done
  if [ -n "$CONTRACT" ]; then
    "$PY" src/regime_contract.py mark-collection --matrix "$CONTRACT" \
      --results "$RESULTS" --runs "${runs[@]}" || exit 1
  fi
  report_cache_write "$analysis_marker" "$analysis_key" \
    "${analysis_outputs[@]}" || exit 1
) 9>"$QUEUE/collect.lock"; then
  echo "[collect-abort] final validation or analysis failed"
  exit 1
fi

echo "== regime sweep complete: $RESULTS/FINAL_REPORT.md"
