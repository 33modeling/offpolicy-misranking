#!/usr/bin/env bash
# Full regime discovery worker for one 4xH100 cluster.
# Run this same script on every cluster; the shared queue prevents duplicate work.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup_env.sh

# Preserve immutable generation provenance when this command is rerun after a
# pull. The wrapper selects the one commit already recorded by partial runs.
if [ "${OM_REGIME_RESUME_WRAPPED:-0}" != "1" ]; then
  exec bash scripts/resume_regime.sh
fi

[ -d "$GROUP_VOLUME" ] && { [ "$OM_WORK" = "$GROUP_VOLUME" ] || [[ "$OM_WORK" == "$GROUP_VOLUME/"* ]]; } || {
  echo "[abort] shared GROUP_VOLUME is required: GROUP_VOLUME=$GROUP_VOLUME OM_WORK=$OM_WORK"
  exit 1
}

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  DIRTY=$(git status --porcelain -- src scripts)
  [ -z "$DIRTY" ] || {
    echo "[abort] src/scripts worktree is dirty; commit and pull before launching"
    printf '%s\n' "$DIRTY"
    exit 1
  }
fi

PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv missing: $PY"; exit 1; }
command -v flock >/dev/null 2>&1 || { echo "[abort] flock command missing"; exit 1; }
export MODEL_14B="$MODELS_DIR/Qwen2.5-7B-Instruct"
[ -f "$MODEL_14B/config.json" ] || {
  echo "[abort] Qwen2.5-7B snapshot missing: $MODEL_14B"
  exit 1
}

mapfile -t GPU_NAMES < <(timeout 20 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)
GPU_COUNT=${#GPU_NAMES[@]}
H100_COUNT=$(printf '%s\n' "${GPU_NAMES[@]}" | grep -c 'H100' || true)
[ "$GPU_COUNT" -eq 4 ] && [ "$H100_COUNT" -eq 4 ] || {
  echo "[abort] four H100 GPUs required (detected: GPUs=$GPU_COUNT, H100=$H100_COUNT)"
  exit 1
}

bash scripts/check_data.sh gsm8k 512 100
bash scripts/check_data.sh math500 400 100

# Canonical discovery matrix. Clear inherited overrides that could silently
# turn this launcher into a reduced or incompatible run.
export REGIME_DATASETS="gsm8k math500"
export REGIME_SEEDS="0 1 2"
export REGIME_DRIFTS="0 25 100 400"
export REGIME_N_VAL=100
export REGIME_BEHAVIOR_K=8
export REGIME_FRESH_K=32
export REGIME_VAL_K=8
export REGIME_MICRO_GROUP=4
export REGIME_MAX_NEW_TOKENS=512
export REGIME_PROJ_DIM=4096
export REGIME_GRAD_LAYERS=4
export REGIME_CLIP_CAP=10
export REGIME_TOPK_FRAC=0.10
export REGIME_TEMPERATURE=1.0
export REGIME_FIRST_BOOTSTRAP=2000
export REGIME_MAX_RETRIES=3
export REGIME_MODEL_TAG="qwen2.5-7b-instruct"
export OM_ATTN=eager
export OM_SKIP_HYBRID=1
export OM_TOP_P=1.0
export OM_THINKING=off
export OM_SKIP_GPU_CHECK=0
export OM_ALLOW_DIRTY=0
export OM_STALL_MINUTES=5
export HYBRID_PROMPTS=24
export K_CELL=8
export RADIUS_MODE=gaussian
unset REGIME_N_TRAIN REGIME_ROOT REGIME_RESULTS REGIME_MATRIX REGIME_QUARANTINE
unset REGIME_FIRST_CALIBRATION OM_GPUS OM_BEHAVIOR_SOURCE OM_LORA_TARGETS OM_GEN_BATCH
unset OM_POOL_FILE OM_EOS_IDS

mkdir -p "$OM_WORK/console-logs"
LOG="$OM_WORK/console-logs/regime-discovery-$(hostname)-$(date +%F-%H%M%S).log"
LOCK_DIR="$OM_WORK/locks"
NODE_TAG=$(hostname 2>/dev/null || printf node)
NODE_TAG=$(printf '%s' "$NODE_TAG" | tr -cs 'a-zA-Z0-9._-' '-')
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/regime-discovery-$NODE_TAG.lock"
flock -n 9 || {
  echo "[abort] a discovery worker is already running on this node"
  exit 1
}

# A point already gets three attempts inside go_regime.sh. If all three hit a
# transient CUDA failure, restart the worker and let the shared queue plus
# durable .partial files resume the unfinished family.
MAX_WORKER_RESTARTS=12
WORKER_RESTARTS=0

wait_for_gpu_release() {
  local gpu_memory gpu_rows busy
  for _ in $(seq 1 30); do
    gpu_memory=$(timeout 20 nvidia-smi --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null) || {
        sleep 2
        continue
      }
    gpu_rows=$(printf '%s\n' "$gpu_memory" | awk 'NF {n++} END {print n+0}')
    [ "$gpu_rows" -eq 4 ] || {
      sleep 2
      continue
    }
    busy=$(printf '%s\n' "$gpu_memory" | awk '$1 > 2000 {n++} END {print n+0}')
    [ "$busy" -eq 0 ] && return 0
    sleep 2
  done
  echo "[abort] GPU memory did not clear after worker failure" | tee -a "$LOG"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv 2>/dev/null | tee -a "$LOG" || true
  return 1
}

echo "[discovery] Qwen2.5-7B / GSM8K+MATH-500 / seeds 0-2 / drift 0,25,100,400"
echo "[discovery] log: $LOG"

while :; do
  attempt=$((WORKER_RESTARTS + 1))
  echo "[$(date '+%F %T')] [supervisor] worker attempt $attempt/$((MAX_WORKER_RESTARTS + 1))" \
    | tee -a "$LOG"

  set +e
  bash scripts/go_regime.sh 2>&1 | tee -a "$LOG"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  rc=${pipeline_status[0]}
  tee_rc=${pipeline_status[1]}

  [ "$tee_rc" -eq 0 ] || {
    echo "[abort] console log write failed (rc=$tee_rc): $LOG"
    exit "$tee_rc"
  }
  [ "$rc" -ne 0 ] || break

  if [ "$WORKER_RESTARTS" -ge "$MAX_WORKER_RESTARTS" ]; then
    echo "[supervisor] restart limit reached ($MAX_WORKER_RESTARTS)" | tee -a "$LOG"
    break
  fi

  WORKER_RESTARTS=$((WORKER_RESTARTS + 1))
  echo "[supervisor] worker failed (rc=$rc); waiting for GPU release before restart $WORKER_RESTARTS/$MAX_WORKER_RESTARTS" \
    | tee -a "$LOG"
  wait_for_gpu_release || exit "$rc"
  sleep 15
done

if [ "$rc" -eq 0 ]; then
  echo "[discovery] complete (worker restarts=$WORKER_RESTARTS)"
else
  echo "[discovery] failed after $WORKER_RESTARTS worker restarts (rc=$rc): $LOG"
fi
exit "$rc"
