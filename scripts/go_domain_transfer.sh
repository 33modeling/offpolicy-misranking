#!/usr/bin/env bash
# Run this same command on every independent cluster. The regime family locks
# divide model x dataset x seed work without duplicate generation.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh

PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv python 없음: $PY"; exit 1; }
CONFIG="${TRANSFER_CONFIG:-configs/domain_transfer.json}"
QUALIFICATION="$DATASETS_DIR/domain_dataset_qualification.json"
TRANSFER_ROOT_BASE="${TRANSFER_ROOT_BASE:-$OM_WORK/runs}"
TRANSFER_RESULTS_BASE="${TRANSFER_RESULTS_BASE:-$OM_WORK/results}"
config_field() {
  "$PY" src/model_matrix.py --config "$CONFIG" experiment-field "$1"
}

MODELS=(${TRANSFER_MODELS:-mistral7b olmo2-7b})
export REGIME_DATASETS="$(config_field datasets)"
export REGIME_SEEDS="$(config_field seeds)"
export REGIME_DRIFTS="$(config_field drifts)"
export REGIME_N_TRAIN="$(config_field n_train)"
export REGIME_N_VAL="$(config_field n_val)"
export REGIME_BEHAVIOR_K="$(config_field behavior_k)"
export REGIME_FRESH_K="$(config_field fresh_k)"
export REGIME_VAL_K="$(config_field val_k)"
export REGIME_MICRO_GROUP="$(config_field micro_group)"
export REGIME_MAX_NEW_TOKENS="$(config_field max_new_tokens)"
export REGIME_PROJ_DIM="$(config_field proj_dim)"
export REGIME_GRAD_LAYERS="$(config_field grad_layers)"
export REGIME_CLIP_CAP="$(config_field clip_cap)"
export REGIME_TOPK_FRAC="$(config_field topk_frac)"
export REGIME_TEMPERATURE="$(config_field temperature)"
export REGIME_FIRST_BOOTSTRAP="$(config_field first_bootstrap)"
export OM_TOP_P="$(config_field top_p)"
export OM_THINKING="$(config_field thinking)"
export OM_ATTN="$(config_field attn)"
export OM_SKIP_HYBRID="$(config_field skip_hybrid)"
unset REGIME_FIRST_CALIBRATION

# Snapshot hashes, deterministic split, and train/validation disjointness.
"$PY" src/qualify_domain_data.py ${REGIME_DATASETS} \
  --data-root "$DATASETS_DIR" --n-train "$REGIME_N_TRAIN" \
  --n-val "$REGIME_N_VAL" --seeds ${REGIME_SEEDS} --output "$QUALIFICATION"

for key in "${MODELS[@]}"; do
  "$PY" src/model_matrix.py --config "$CONFIG" --models-dir "$MODELS_DIR" check "$key"
  model=$($PY src/model_matrix.py --config "$CONFIG" --models-dir "$MODELS_DIR" field "$key" path)
  targets=$($PY src/model_matrix.py --config "$CONFIG" --models-dir "$MODELS_DIR" field "$key" lora_targets)
  export REGIME_ROOT="$TRANSFER_ROOT_BASE/regime-$key"
  export REGIME_RESULTS="$TRANSFER_RESULTS_BASE/regime-$key"
  export REGIME_MATRIX="$REGIME_ROOT/MATRIX.json"
  "$PY" src/regime_contract.py init --matrix "$REGIME_MATRIX" --config "$CONFIG" \
    --model-key "$key" --model "$model" --qualification "$QUALIFICATION" \
    --git "$(git rev-parse HEAD)"
  host_tag=$(hostname | tr -cs 'a-zA-Z0-9._-' '-')
  "$PY" src/transfer_smoke.py --model "$model" --lora-targets "$targets" \
    --marker "$REGIME_ROOT/.runtime-smoke-$host_tag.json"
  echo "== domain transfer: model=$key data=$REGIME_DATASETS seeds=$REGIME_SEEDS drift=$REGIME_DRIFTS"
  MODEL_14B="$model" REGIME_MODEL_TAG="$key" OM_LORA_TARGETS="$targets" \
    bash scripts/go_regime.sh
  unset REGIME_ROOT REGIME_RESULTS REGIME_MATRIX
done
