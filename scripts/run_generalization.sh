#!/usr/bin/env bash
# Cross-model/domain RLVR generalization matrix. Run this exact command on each
# of the three independent 4xH100 nodes after pulling the same clean commit.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup_env.sh

PY="$VENV_DIR/bin/python"
CONFIG="${GENERALIZATION_CONFIG:-configs/domain_transfer.json}"
RUN_ID="${GENERALIZATION_RUN_ID:-generalization-grpo-v1}"
[ -x "$PY" ] || { echo "[abort] venv missing: $PY"; exit 1; }
[ -s "$CONFIG" ] || { echo "[abort] generalization config missing: $CONFIG"; exit 1; }
command -v flock >/dev/null 2>&1 || { echo "[abort] flock missing"; exit 1; }
[ -d "$GROUP_VOLUME" ] || { echo "[abort] shared GROUP_VOLUME missing: $GROUP_VOLUME"; exit 1; }
[[ "$OM_WORK" == "$GROUP_VOLUME" || "$OM_WORK" == "$GROUP_VOLUME/"* ]] || {
  echo "[abort] OM_WORK must be on the shared volume: $OM_WORK"
  exit 1
}
[[ "$RUN_ID" =~ ^[a-zA-Z0-9._-]+$ ]] || {
  echo "[abort] GENERALIZATION_RUN_ID must be one safe path component"
  exit 1
}

DIRTY=$(git status --porcelain -- src scripts configs requirements.txt)
[ -z "$DIRTY" ] || {
  echo "[abort] code is dirty; commit and pull the same revision on every node"
  printf '%s\n' "$DIRTY"
  exit 1
}
GIT=$(git rev-parse HEAD)

mapfile -t GPU_NAMES < <(
  timeout 20 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
)
GPU_COUNT=${#GPU_NAMES[@]}
H100_COUNT=$(printf '%s\n' "${GPU_NAMES[@]}" | grep -c 'H100' || true)
[ "$GPU_COUNT" -eq 4 ] && [ "$H100_COUNT" -eq 4 ] || {
  echo "[abort] exactly four H100 GPUs required (GPUs=$GPU_COUNT H100=$H100_COUNT)"
  exit 1
}

experiment_field() {
  "$PY" src/model_matrix.py --config "$CONFIG" experiment-field "$1"
}

grpo_field() {
  "$PY" src/model_matrix.py --config "$CONFIG" grpo-field "$1"
}

model_field() {
  "$PY" src/model_matrix.py --config "$CONFIG" --models-dir "$MODELS_DIR" \
    field "$1" "$2"
}

mapfile -t MODEL_KEYS < <(
  "$PY" src/model_matrix.py --config "$CONFIG" list-models
)
[ "${#MODEL_KEYS[@]}" -gt 0 ] || { echo "[abort] no configured models"; exit 1; }

DATASETS=$(experiment_field datasets)
SEEDS=$(experiment_field seeds)
DRIFTS=$(experiment_field drifts)
POLICY_METHOD=$(experiment_field policy_method)
export RLVR_METHOD="$POLICY_METHOD"
# Force the actual loader onto the exact directories qualified below. This
# prevents an older flat $OM_DATA copy from shadowing a pinned snapshot.
export GSM8K_DIR="$DATASETS_DIR/gsm8k"
export MATH500_DIR="$DATASETS_DIR/math500"
export MBPP_DIR="$DATASETS_DIR/mbpp"
export KK_DIR="$DATASETS_DIR/kk"
export ARC_CHALLENGE_DIR="$DATASETS_DIR/arc-challenge"
[ "$(grpo_field reference_kl_beta)" = "0.0" ] || {
  echo "[abort] current GRPO implementation requires reference_kl_beta=0.0"
  exit 1
}
[ "$(grpo_field world_size)" = "4" ] || {
  echo "[abort] this launcher requires grpo.world_size=4"
  exit 1
}

mkdir -p "$OM_WORK/locks" "$OM_WORK/console-logs" "$OM_WORK/contracts"
NODE_TAG=$(hostname 2>/dev/null || printf node)
NODE_TAG=$(printf '%s' "$NODE_TAG" | tr -cs 'a-zA-Z0-9._-' '-')
exec 9>"$OM_WORK/locks/$RUN_ID-$NODE_TAG.lock"
flock -n 9 || { echo "[abort] a $RUN_ID worker is already running on this node"; exit 1; }
LOG="$OM_WORK/console-logs/$RUN_ID-$NODE_TAG-$(date +%F-%H%M%S).log"

CONFIG_ID=$(sha256sum "$CONFIG" | cut -c1-16)
QUALIFICATION="$OM_WORK/contracts/$RUN_ID-datasets-$CONFIG_ID.json"
(
  flock 8
  "$PY" src/qualify_domain_data.py $DATASETS \
    --data-root "$DATASETS_DIR" \
    --n-train "$(experiment_field n_train)" \
    --n-val "$(experiment_field n_val)" \
    --seeds $SEEDS \
    --output "$QUALIFICATION"
) 8>"$OM_WORK/locks/$RUN_ID-dataset-qualification.lock" | tee -a "$LOG"

# Clear inherited overrides that could silently alter the registered matrix.
unset REGIME_ROOT REGIME_RESULTS REGIME_MATRIX REGIME_QUARANTINE
unset REGIME_DATASETS REGIME_SEEDS REGIME_DRIFTS REGIME_N_TRAIN REGIME_N_VAL
unset OM_GPUS OM_BEHAVIOR_SOURCE OM_GRPO_RESUME_ADAPTER OM_GRPO_RESUME_OPTIMIZER
unset OM_GRPO_START_STEP OM_POOL_FILE OM_EOS_IDS OM_GEN_BATCH

export REGIME_DATASETS="$DATASETS"
export REGIME_SEEDS="$SEEDS"
export REGIME_DRIFTS="$DRIFTS"
export REGIME_N_TRAIN="$(experiment_field n_train)"
export REGIME_N_VAL="$(experiment_field n_val)"
export REGIME_BEHAVIOR_K="$(experiment_field behavior_k)"
export REGIME_FRESH_K="$(experiment_field fresh_k)"
export REGIME_VAL_K="$(experiment_field val_k)"
export REGIME_MICRO_GROUP="$(experiment_field micro_group)"
export REGIME_MAX_NEW_TOKENS="$(experiment_field max_new_tokens)"
export REGIME_PROJ_DIM="$(experiment_field proj_dim)"
export REGIME_GRAD_LAYERS="$(experiment_field grad_layers)"
export REGIME_CLIP_CAP="$(experiment_field clip_cap)"
export REGIME_TOPK_FRAC="$(experiment_field topk_frac)"
export REGIME_TEMPERATURE="$(experiment_field temperature)"
export REGIME_FIRST_BOOTSTRAP="$(experiment_field first_bootstrap)"
export GRPO_WORLD_SIZE="$(grpo_field world_size)"
export GRPO_GROUP_SIZE="$(grpo_field group_size)"
export GRPO_CLIP_EPSILON="$(grpo_field clip_epsilon)"
export GRPO_LEARNING_RATE="$(grpo_field learning_rate)"
export GRPO_EPOCHS_PER_BATCH="$(grpo_field epochs_per_batch)"
export GRPO_MAX_GRAD_NORM="$(grpo_field max_grad_norm)"
export GRPO_ADVANTAGE_EPSILON="$(grpo_field advantage_epsilon)"
export GRPO_LORA_RANK="$(grpo_field lora_rank)"
export GRPO_LORA_ALPHA="$(grpo_field lora_alpha)"
export GRPO_CHECKPOINT_EVERY=5
export REGIME_MAX_RETRIES=3
export OM_TOP_P="$(experiment_field top_p)"
export OM_THINKING="$(experiment_field thinking)"
export OM_ATTN="$(experiment_field attn)"
export OM_SKIP_HYBRID="$(experiment_field skip_hybrid)"
export OM_SKIP_GPU_CHECK=0
export OM_ALLOW_DIRTY=0
export OM_ALLOW_ANALYSIS_UPGRADE=0
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
  echo "[generalization] method=$RLVR_METHOD model=$name datasets=$REGIME_DATASETS seeds=$REGIME_SEEDS drifts=$REGIME_DRIFTS" \
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
    echo "[generalization] model=$name failed rc=$rc; restart=$restarts/12" | tee -a "$LOG"
    wait_for_gpu_release || { echo "[abort] GPU memory did not clear"; return "$rc"; }
    sleep 15
  done
  [ "$rc" -eq 0 ] || return "$rc"
}

for model_key in "${MODEL_KEYS[@]}"; do
  MODEL_PATH=$(model_field "$model_key" path)
  OM_LORA_TARGETS=$(model_field "$model_key" lora_targets)
  export MODEL_PATH OM_LORA_TARGETS
  "$PY" src/model_matrix.py --config "$CONFIG" --models-dir "$MODELS_DIR" \
    check "$model_key" | tee -a "$LOG"

  SMOKE="$OM_WORK/contracts/$RUN_ID-smoke-$NODE_TAG-$model_key-$GIT.json"
  CUDA_VISIBLE_DEVICES=0 "$PY" src/transfer_smoke.py \
    --model "$MODEL_PATH" --lora-targets "$OM_LORA_TARGETS" --marker "$SMOKE" \
    | tee -a "$LOG"

  export REGIME_MODEL_TAG="$RUN_ID-$POLICY_METHOD-$model_key"
  export REGIME_ROOT="$OM_WORK/runs/$RUN_ID/$model_key"
  export REGIME_RESULTS="$OM_WORK/results/$RUN_ID/$model_key"
  export REGIME_QUARANTINE="$OM_WORK/quarantine/$RUN_ID/$model_key"
  export REGIME_MATRIX="$OM_WORK/contracts/$RUN_ID-$model_key-$CONFIG_ID.json"
  "$PY" src/regime_contract.py init \
    --matrix "$REGIME_MATRIX" --config "$CONFIG" --model-key "$model_key" \
    --model "$MODEL_PATH" --qualification "$QUALIFICATION" --git "$GIT" \
    | tee -a "$LOG"
  run_phase "$model_key"
done

echo "[generalization] all $POLICY_METHOD matrices complete" | tee -a "$LOG"
echo "[generalization] results: $OM_WORK/results/$RUN_ID/{${MODEL_KEYS[*]}}" | tee -a "$LOG"
