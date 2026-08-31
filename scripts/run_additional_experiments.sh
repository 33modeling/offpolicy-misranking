#!/usr/bin/env bash
# One queued entry point for every registered post-primary experiment.
# Run from a separate clean checkout on each of the three 4xH100 nodes.
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

MATRIX_CONFIGS=(
  configs/domain_transfer.json
  configs/generalization_dr_grpo.json
  configs/generalization_rloo.json
)
MATRIX_IDS=(
  generalization-grpo-v1
  method-dr-grpo-v1
  method-rloo-v1
)
MODE=${1:---run}
[ "$#" -le 1 ] || { echo "usage: $0 [--prepare|--run]"; exit 2; }
case "$MODE" in
  --prepare|--run) ;;
  *) echo "usage: $0 [--prepare|--run]"; exit 2 ;;
esac
for config in "${MATRIX_CONFIGS[@]}"; do
  [ -s "$config" ] || { echo "[abort] additional config missing: $config"; exit 1; }
done

clean_checkout() {
  local dirty
  dirty=$(git status --porcelain -- src scripts configs requirements.txt)
  [ -z "$dirty" ] || {
    echo "[abort] additional checkout is dirty"
    printf '%s\n' "$dirty"
    return 1
  }
}
clean_checkout
GIT=$(git rev-parse HEAD)

mkdir -p "$OM_WORK/locks" "$OM_WORK/console-logs" "$OM_WORK/contracts"

matrix_field() {
  "$PY" src/model_matrix.py --config "$1" experiment-field "$2"
}

grpo_field() {
  "$PY" src/model_matrix.py --config "$1" grpo-field "$2"
}

model_field() {
  "$PY" src/model_matrix.py --config "$1" --models-dir "$MODELS_DIR" \
    field "$2" "$3"
}

provision_registered_snapshots() {
  local config=${MATRIX_CONFIGS[0]} datasets
  local model_keys=()
  [ "${ADDITIONAL_SKIP_PROVISION:-0}" != "1" ] || return 0

  # Model and data destinations are shared. Only one node writes; the other
  # nodes subsequently hash-check the immutable snapshots.
  exec 7>"$OM_WORK/locks/additional-provision.lock"
  flock 7
  export OM_ONLINE=1
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  mapfile -t model_keys < <(
    "$PY" src/model_matrix.py --config "$config" list-models
  )
  datasets=$(matrix_field "$config" datasets)
  "$PY" src/model_matrix.py --config "$config" --models-dir "$MODELS_DIR" \
    download "${model_keys[@]}"
  bash scripts/fetch_datasets.sh $datasets
  flock -u 7
  exec 7>&-
}

# Explicit roots prevent old flat snapshots and the primary v1 protocol from
# shadowing any input or receiving any output from this suite.
export GSM8K_DIR="$DATASETS_DIR/gsm8k"
export MATH500_DIR="$DATASETS_DIR/math500"
export MBPP_DIR="$DATASETS_DIR/mbpp"
export KK_DIR="$DATASETS_DIR/kk"
export ARC_CHALLENGE_DIR="$DATASETS_DIR/arc-challenge"
export OM_MATH_VERIFIER=math_verify

if [ "$MODE" = "--prepare" ]; then
  config=${MATRIX_CONFIGS[0]}
  provision_registered_snapshots
  datasets=$(matrix_field "$config" datasets)
  seeds=$(matrix_field "$config" seeds)
  config_id=$(sha256sum "$config" | cut -c1-16)
  "$PY" src/qualify_domain_data.py $datasets \
    --data-root "$DATASETS_DIR" \
    --n-train "$(matrix_field "$config" n_train)" \
    --n-val "$(matrix_field "$config" n_val)" \
    --seeds $seeds \
    --output "$OM_WORK/contracts/additional-prepared-$config_id.json"
  echo "[additional] all pinned model/data snapshots prepared and qualified"
  exit 0
fi

HOST_TAG=$(hostname 2>/dev/null || printf node)
WORKER_SUFFIX=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || printf '%s-%s' "$$" "$(date +%s%N)")
WORKER_TAG="${ADDITIONAL_WORKER_ID:-${ADDITIONAL_NODE_TAG:-$HOST_TAG-$WORKER_SUFFIX}}"
WORKER_TAG=$(printf '%s' "$WORKER_TAG" | tr -cs 'a-zA-Z0-9._-' '-')
LOCAL_LOCK_DIR="${OM_LOCAL_LOCK_DIR:-/tmp/offpolicy-misranking-$(id -u)}"
local_lock_path=$(realpath -m "$LOCAL_LOCK_DIR")
shared_path=$(realpath -m "$GROUP_VOLUME")
[[ "$local_lock_path" != "$shared_path" && "$local_lock_path" != "$shared_path/"* ]] || {
  echo "[abort] OM_LOCAL_LOCK_DIR must be node-local, not on GROUP_VOLUME"
  exit 1
}
mkdir -p "$LOCAL_LOCK_DIR"
chmod 700 "$LOCAL_LOCK_DIR"

# Reject a duplicate additional-suite worker on this node, then wait on the
# exact lock held for the lifetime of revision 295dfea's primary launcher.
exec 9>"$LOCAL_LOCK_DIR/additional-suite.lock"
flock -n 9 || { echo "[abort] additional suite already queued on this physical node"; exit 1; }
exec 8>"$LOCAL_LOCK_DIR/primary.lock"
echo "[additional] worker=$WORKER_TAG queued behind local primary at git=$GIT"
flock 8
echo "[additional] local primary complete for worker=$WORKER_TAG"
clean_checkout

mapfile -t GPU_NAMES < <(
  timeout 20 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
)
GPU_COUNT=${#GPU_NAMES[@]}
H100_COUNT=$(printf '%s\n' "${GPU_NAMES[@]}" | grep -c 'H100' || true)
[ "$GPU_COUNT" -eq 4 ] && [ "$H100_COUNT" -eq 4 ] || {
  echo "[abort] exactly four H100 GPUs required (GPUs=$GPU_COUNT H100=$H100_COUNT)"
  exit 1
}

wait_for_gpu_release() {
  local memory rows busy
  for _ in $(seq 1 120); do
    memory=$(timeout 20 nvidia-smi --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null) || { sleep 5; continue; }
    rows=$(printf '%s\n' "$memory" | awk 'NF {n++} END {print n+0}')
    busy=$(printf '%s\n' "$memory" | awk '$1 > 2000 {n++} END {print n+0}')
    [ "$rows" -eq 4 ] && [ "$busy" -eq 0 ] && return 0
    sleep 5
  done
  return 1
}
wait_for_gpu_release || { echo "[abort] four GPUs did not become idle"; exit 1; }
provision_registered_snapshots

run_phase() {
  local log=$1 name=$2 restarts=0 rc=0
  echo "[additional] method=$RLVR_METHOD model=$name datasets=$REGIME_DATASETS seeds=$REGIME_SEEDS drifts=$REGIME_DRIFTS" \
    | tee -a "$log"
  while :; do
    set +e
    bash scripts/run_matrix.sh 2>&1 | tee -a "$log"
    statuses=("${PIPESTATUS[@]}")
    set -e
    rc=${statuses[0]}
    [ "${statuses[1]}" -eq 0 ] || exit "${statuses[1]}"
    [ "$rc" -ne 0 ] || break
    [ "$restarts" -lt 12 ] || break
    restarts=$((restarts + 1))
    echo "[additional] model=$name failed rc=$rc; restart=$restarts/12" | tee -a "$log"
    wait_for_gpu_release || { echo "[abort] GPU memory did not clear"; return "$rc"; }
    sleep 15
  done
  [ "$rc" -eq 0 ] || return "$rc"
}

run_registered_matrix() {
  local config=$1 run_id=$2 datasets seeds drifts method config_id qualification log
  local model_keys=()

  datasets=$(matrix_field "$config" datasets)
  seeds=$(matrix_field "$config" seeds)
  drifts=$(matrix_field "$config" drifts)
  method=$(matrix_field "$config" policy_method)
  [ "$(grpo_field "$config" reference_kl_beta)" = "0.0" ] || {
    echo "[abort] reference_kl_beta must be 0.0"
    return 1
  }
  [ "$(grpo_field "$config" world_size)" = "4" ] || {
    echo "[abort] additional matrices require world_size=4"
    return 1
  }
  mapfile -t model_keys < <(
    "$PY" src/model_matrix.py --config "$config" list-models
  )
  [ "${#model_keys[@]}" -gt 0 ] || { echo "[abort] no configured models"; return 1; }

  log="$OM_WORK/console-logs/$run_id-$WORKER_TAG-$(date +%F-%H%M%S).log"
  config_id=$(sha256sum "$config" | cut -c1-16)
  qualification="$OM_WORK/contracts/$run_id-datasets-$config_id.json"
  (
    flock 6
    "$PY" src/qualify_domain_data.py $datasets \
      --data-root "$DATASETS_DIR" \
      --n-train "$(matrix_field "$config" n_train)" \
      --n-val "$(matrix_field "$config" n_val)" \
      --seeds $seeds \
      --output "$qualification"
  ) 6>"$OM_WORK/locks/$run_id-dataset-qualification.lock" | tee -a "$log"

  unset REGIME_ROOT REGIME_RESULTS REGIME_MATRIX REGIME_QUARANTINE
  unset REGIME_DATASETS REGIME_SEEDS REGIME_DRIFTS REGIME_N_TRAIN REGIME_N_VAL
  unset OM_GPUS OM_BEHAVIOR_SOURCE OM_GRPO_RESUME_ADAPTER OM_GRPO_RESUME_OPTIMIZER
  unset OM_GRPO_START_STEP OM_POOL_FILE OM_EOS_IDS OM_GEN_BATCH

  export RLVR_METHOD="$method"
  export REGIME_DATASETS="$datasets"
  export REGIME_SEEDS="$seeds"
  export REGIME_DRIFTS="$drifts"
  export REGIME_N_TRAIN="$(matrix_field "$config" n_train)"
  export REGIME_N_VAL="$(matrix_field "$config" n_val)"
  export REGIME_BEHAVIOR_K="$(matrix_field "$config" behavior_k)"
  export REGIME_FRESH_K="$(matrix_field "$config" fresh_k)"
  export REGIME_VAL_K="$(matrix_field "$config" val_k)"
  export REGIME_MICRO_GROUP="$(matrix_field "$config" micro_group)"
  export REGIME_MAX_NEW_TOKENS="$(matrix_field "$config" max_new_tokens)"
  export REGIME_PROJ_DIM="$(matrix_field "$config" proj_dim)"
  export REGIME_GRAD_LAYERS="$(matrix_field "$config" grad_layers)"
  export REGIME_CLIP_CAP="$(matrix_field "$config" clip_cap)"
  export REGIME_TOPK_FRAC="$(matrix_field "$config" topk_frac)"
  export REGIME_TEMPERATURE="$(matrix_field "$config" temperature)"
  export REGIME_FIRST_BOOTSTRAP="$(matrix_field "$config" first_bootstrap)"
  export GRPO_WORLD_SIZE="$(grpo_field "$config" world_size)"
  export GRPO_GROUP_SIZE="$(grpo_field "$config" group_size)"
  export GRPO_CLIP_EPSILON="$(grpo_field "$config" clip_epsilon)"
  export GRPO_LEARNING_RATE="$(grpo_field "$config" learning_rate)"
  export GRPO_EPOCHS_PER_BATCH="$(grpo_field "$config" epochs_per_batch)"
  export GRPO_MAX_GRAD_NORM="$(grpo_field "$config" max_grad_norm)"
  export GRPO_ADVANTAGE_EPSILON="$(grpo_field "$config" advantage_epsilon)"
  export GRPO_LORA_RANK="$(grpo_field "$config" lora_rank)"
  export GRPO_LORA_ALPHA="$(grpo_field "$config" lora_alpha)"
  export GRPO_CHECKPOINT_EVERY=5 REGIME_MAX_RETRIES=3
  export OM_TOP_P="$(matrix_field "$config" top_p)"
  export OM_THINKING="$(matrix_field "$config" thinking)"
  export OM_ATTN="$(matrix_field "$config" attn)"
  export OM_SKIP_HYBRID="$(matrix_field "$config" skip_hybrid)"
  export OM_SKIP_GPU_CHECK=0 OM_ALLOW_DIRTY=0 OM_ALLOW_ANALYSIS_UPGRADE=0
  export OM_STALL_MINUTES=10 HYBRID_PROMPTS=24 K_CELL=8 RADIUS_MODE=gaussian

  for model_key in "${model_keys[@]}"; do
    MODEL_PATH=$(model_field "$config" "$model_key" path)
    OM_LORA_TARGETS=$(model_field "$config" "$model_key" lora_targets)
    export MODEL_PATH OM_LORA_TARGETS
    "$PY" src/model_matrix.py --config "$config" --models-dir "$MODELS_DIR" \
      check "$model_key" | tee -a "$log"
    wait_for_gpu_release || { echo "[abort] GPU memory did not clear"; return 1; }
    CUDA_VISIBLE_DEVICES=0 "$PY" src/transfer_smoke.py \
      --model "$MODEL_PATH" --lora-targets "$OM_LORA_TARGETS" \
      --marker "$OM_WORK/contracts/$run_id-smoke-$WORKER_TAG-$model_key-$GIT.json" \
      | tee -a "$log"

    export REGIME_MODEL_TAG="$run_id-$method-$model_key"
    export REGIME_ROOT="$OM_WORK/runs/$run_id/$model_key"
    export REGIME_RESULTS="$OM_WORK/results/$run_id/$model_key"
    export REGIME_QUARANTINE="$OM_WORK/quarantine/$run_id/$model_key"
    export REGIME_MATRIX="$OM_WORK/contracts/$run_id-$model_key-$config_id.json"
    "$PY" src/regime_contract.py init \
      --matrix "$REGIME_MATRIX" --config "$config" --model-key "$model_key" \
      --model "$MODEL_PATH" --qualification "$qualification" --git "$GIT" \
      | tee -a "$log"
    run_phase "$log" "$model_key"
  done
  echo "[additional] complete: method=$method results=$OM_WORK/results/$run_id" \
    | tee -a "$log"
}

for index in "${!MATRIX_CONFIGS[@]}"; do
  run_registered_matrix "${MATRIX_CONFIGS[$index]}" "${MATRIX_IDS[$index]}"
done
echo "[additional] all registered model/domain/method generalization matrices complete"
