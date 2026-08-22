#!/usr/bin/env bash
# Resume each interrupted v4 run from the Git commit in its own run_config.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
CURRENT_REPO=$PWD
CURRENT_PY=$PY

slot=${1:-}
case "$slot" in 1|2|3) ;; *) echo "usage: bash scripts/go_v4.sh <1|2|3>" >&2; exit 2;; esac

current=$(git rev-parse HEAD) || exit 1
plan=$(mktemp "$TMPDIR/v4-resume-plan.XXXXXX") || exit 1
trap 'rm -f "$plan"' EXIT
"$PY" src/v4_resume_commit.py plan "$OM_WORK/runs" "$slot" "$current" > "$plan" || exit 1

echo "== 이전 v4 프로세스 정리"
"$PY" src/cleanup_run_processes.py --run-prefix "$OM_WORK/runs/v4-" || exit 1
pkill -TERM -f -- "--run $OM_WORK/runs/v4-" 2>/dev/null || true
sleep 3
pkill -KILL -f -- "--run $OM_WORK/runs/v4-" 2>/dev/null || true

snapshot_root="$OM_WORK/code-snapshots"
mkdir -p "$snapshot_root" || exit 1

ensure_snapshot() {
  local target=$1 snapshot snapshot_git
  exec 8>"$snapshot_root/.v4-resume.lock"
  flock 8 || return 1
  if ! git cat-file -e "$target^{commit}" 2>/dev/null; then
    echo "== 기록된 commit 자동 fetch: ${target:0:12}"
    git fetch --no-tags origin "$target" || {
      flock -u 8
      echo "[resume-v4-abort] commit fetch 실패: $target" >&2
      return 1
    }
  fi
  if [ "$target" = "$current" ]; then
    SNAPSHOT=$CURRENT_REPO
    flock -u 8
    return 0
  fi
  snapshot="$snapshot_root/offpolicy-misranking-${target:0:12}"
  if [ -e "$snapshot/.git" ]; then
    snapshot_git=$(git -C "$snapshot" rev-parse HEAD 2>/dev/null || true)
    [ "$snapshot_git" = "$target" ] || {
      flock -u 8
      echo "[resume-v4-abort] snapshot commit mismatch: $snapshot" >&2
      return 1
    }
  elif [ -e "$snapshot" ]; then
    flock -u 8
    echo "[resume-v4-abort] snapshot path is not a worktree: $snapshot" >&2
    return 1
  else
    git worktree add --detach "$snapshot" "$target" || {
      flock -u 8
      return 1
    }
  fi
  SNAPSHOT=$snapshot
  flock -u 8
}

pending=0
while IFS=$'\t' read -r name model seed dataset target config_path; do
  [ -n "$name" ] || continue
  pending=$((pending + 1))
  ensure_snapshot "$target" || exit 1
  echo "== [$name] commit ${target:0:12}에서 재개"
  (
    cd "$SNAPSHOT" || exit 1
    unset OM_REPO PYTHONPATH OM_POOL_FILE OM_GEN_BATCH OM_LORA_TARGETS
    source scripts/setup_env.sh

    export BEHAVIOR_K=8 FRESH_K=32 VAL_K=8 MICRO_GROUP=4 HYBRID_PROMPTS=64
    export K_CELL=8 DRIFT=100 MAX_NEW_TOKENS=512 PROJ_DIM=4096 GRAD_LAYERS=4
    export CLIP_CAP=10.0 TEMPERATURE=1.0 TOPK_FRAC=0.10 RADIUS_MODE=gaussian
    export OM_TOP_P=1.0 OM_THINKING=off OM_ATTN=eager N_VAL=100
    if [ "$model" = 27b ]; then
      export MODEL_14B="$MODELS_DIR/Qwen3.8-27B-BF16"
      export N_TRAIN=512 OM_LORA_TARGETS=all-linear OM_GEN_BATCH=8
      export OM_SKIP_HYBRID=1 OM_STALL_MINUTES=20
    else
      export MODEL_14B="$MODELS_DIR/Qwen2.5-7B-Instruct"
      export N_TRAIN=512 OM_SKIP_HYBRID=0 OM_STALL_MINUTES=5
    fi
    [ "$dataset" = math500 ] && export N_TRAIN=400

    if [ "$config_path" != - ]; then
      config_env=$("$CURRENT_PY" "$CURRENT_REPO/src/v4_resume_commit.py" env "$config_path") \
        || exit 1
      eval "$config_env"
    fi

    export RUN_BASE="$OM_WORK/runs/v4-$model"
    export RESULTS_BASE="$OM_WORK/results/v4-$model"
    export RUN_LABEL="v4-$model-resume-cluster$slot"
    export RUN_BASE_SMOKE="$OM_WORK/runs/v4-$model-smoke-${target:0:12}-scluster$slot"
    export OM_SKIP_POSTPROCESS=1 OM_GPUS=0,1,2,3 OM_MAX_RETRIES=5
    SEEDS="$seed" DATASETS="$dataset" bash scripts/go_v2.sh
  ) || exit 1
done < "$plan"

if [ "$pending" -eq 0 ]; then
  echo "== cluster $slot: 배정된 run이 이미 전부 완료됨"
else
  echo "== cluster $slot: 미완료 run $pending개 재개 완료"
fi
