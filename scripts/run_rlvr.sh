#!/usr/bin/env bash
# Canonical RLVR launcher. Run this exact command on each of the three 4xH100 nodes.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup_env.sh

PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv missing: $PY"; exit 1; }
command -v flock >/dev/null 2>&1 || { echo "[abort] flock missing"; exit 1; }
[ -d "$GROUP_VOLUME" ] || { echo "[abort] shared GROUP_VOLUME missing: $GROUP_VOLUME"; exit 1; }
[[ "$OM_WORK" == "$GROUP_VOLUME" || "$OM_WORK" == "$GROUP_VOLUME/"* ]] || {
  echo "[abort] OM_WORK must be on the shared volume: $OM_WORK"
  exit 1
}

DIRTY=$(git status --porcelain -- src scripts requirements.txt)
[ -z "$DIRTY" ] || {
  echo "[abort] code is dirty; commit and pull the same revision on every node"
  printf '%s\n' "$DIRTY"
  exit 1
}

mapfile -t GPU_NAMES < <(
  timeout 20 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
)
GPU_COUNT=${#GPU_NAMES[@]}
H100_COUNT=$(printf '%s\n' "${GPU_NAMES[@]}" | grep -c 'H100' || true)
[ "$GPU_COUNT" -eq 4 ] && [ "$H100_COUNT" -eq 4 ] || {
  echo "[abort] exactly four H100 GPUs required (GPUs=$GPU_COUNT H100=$H100_COUNT)"
  exit 1
}

MODEL_7B="${MODEL_7B:-$MODELS_DIR/Qwen2.5-7B-Instruct}"
MODEL_27B="${MODEL_27B:-$MODELS_DIR/Qwen3.8-27B-BF16}"
for model in "$MODEL_7B" "$MODEL_27B"; do
  for name in config.json tokenizer_config.json; do
    [ -s "$model/$name" ] || { echo "[abort] model snapshot missing: $model/$name"; exit 1; }
  done
done

bash scripts/check_data.sh gsm8k 512 100
bash scripts/check_data.sh math500 400 100

NODE_TAG=$(hostname 2>/dev/null || printf node)
NODE_TAG=$(printf '%s' "$NODE_TAG" | tr -cs 'a-zA-Z0-9._-' '-')
mkdir -p "$OM_WORK/locks" "$OM_WORK/console-logs"
exec 9>"$OM_WORK/locks/rlvr-$NODE_TAG.lock"
flock -n 9 || { echo "[abort] an RLVR worker is already running on this node"; exit 1; }
LOG="$OM_WORK/console-logs/rlvr-$NODE_TAG-$(date +%F-%H%M%S).log"

# Clear inherited overrides that could silently change the registered matrix.
unset REGIME_ROOT REGIME_RESULTS REGIME_MATRIX REGIME_QUARANTINE
unset OM_GPUS OM_BEHAVIOR_SOURCE OM_GRPO_RESUME_ADAPTER OM_GRPO_RESUME_OPTIMIZER
unset OM_GRPO_START_STEP OM_POOL_FILE OM_EOS_IDS
export GRPO_WORLD_SIZE=4
export GRPO_GROUP_SIZE=8
export GRPO_CLIP_EPSILON=0.2
export GRPO_LEARNING_RATE=1e-5
export GRPO_EPOCHS_PER_BATCH=2
export GRPO_MAX_GRAD_NORM=1.0
export GRPO_ADVANTAGE_EPSILON=1e-4
export GRPO_LORA_RANK=16
export GRPO_LORA_ALPHA=32
export GRPO_CHECKPOINT_EVERY=5
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
export REGIME_FIRST_BOOTSTRAP=10000
export REGIME_MAX_RETRIES=3
export OM_TOP_P=1.0
export OM_THINKING=off
export OM_ATTN=eager
export OM_SKIP_HYBRID=1
export OM_SKIP_GPU_CHECK=0
export OM_ALLOW_DIRTY=0
export OM_ALLOW_ANALYSIS_UPGRADE=1
export OM_STALL_MINUTES=10
export HYBRID_PROMPTS=24
export K_CELL=8
export RADIUS_MODE=gaussian

wait_for_gpu_release() {
  local memory rows busy
  for _ in $(seq 1 30); do
    memory=$(timeout 20 nvidia-smi --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null) || { sleep 2; continue; }
    rows=$(printf '%s\n' "$memory" | awk 'NF {n++} END {print n+0}')
    busy=$(printf '%s\n' "$memory" | awk '$1 > 2000 {n++} END {print n+0}')
    [ "$rows" -eq 4 ] && [ "$busy" -eq 0 ] && return 0
    sleep 2
  done
  return 1
}

run_phase() {
  local name=$1 restarts=0 rc=0
  echo "[rlvr] phase=$name model=$MODEL_PATH seeds=$REGIME_SEEDS drifts=$REGIME_DRIFTS" \
    | tee -a "$LOG"
  while :; do
    set +e
    bash scripts/run_matrix.sh 2>&1 | tee -a "$LOG"
    statuses=("${PIPESTATUS[@]}")
    set -e
    rc=${statuses[0]}
    [ "${statuses[1]}" -eq 0 ] || exit "${statuses[1]}"
    [ "$rc" -ne 0 ] || break
    [ "$restarts" -lt 12 ] || break
    restarts=$((restarts + 1))
    echo "[rlvr] phase=$name worker failed rc=$rc; restart=$restarts/12" | tee -a "$LOG"
    wait_for_gpu_release || { echo "[abort] GPU memory did not clear"; return "$rc"; }
    sleep 15
  done
  [ "$rc" -eq 0 ] || {
    echo "[rlvr] phase=$name failed after $restarts restarts rc=$rc" | tee -a "$LOG"
    return "$rc"
  }
}

# The 27B policy is the primary experiment and runs first. Qwen3.8 must use
# verified FLA kernels; there is no slow fallback.
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/check_27b_fla.py | tee -a "$LOG"
done
export MODEL_PATH="$MODEL_27B"
export REGIME_MODEL_TAG="qwen3.8-27b-grpo-v1"
export REGIME_DATASETS="gsm8k math500"
export REGIME_SEEDS="0 1 2 3 4"
export REGIME_DRIFTS="0 25 100 400"
unset REGIME_N_TRAIN
export OM_LORA_TARGETS=all-linear
export OM_GEN_BATCH=8
run_phase 27b

# The 7B policy is a lower-scale replication with three preregistered seeds.
export MODEL_PATH="$MODEL_7B"
export REGIME_MODEL_TAG="qwen2.5-7b-grpo-v1"
export REGIME_DATASETS="gsm8k math500"
export REGIME_SEEDS="0 1 2"
export REGIME_DRIFTS="0 25 100 400"
unset REGIME_N_TRAIN OM_GEN_BATCH
export OM_LORA_TARGETS=q_proj,v_proj
run_phase 7b

bash scripts/harvest_results.sh | tee -a "$LOG"
echo "[rlvr] all matrices complete" | tee -a "$LOG"
echo "[rlvr] final bundle: $OM_WORK/readouts/rlvr-grpo" | tee -a "$LOG"
