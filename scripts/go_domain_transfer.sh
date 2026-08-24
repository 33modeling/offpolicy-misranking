#!/usr/bin/env bash
# Run this same command on every independent cluster. The regime family locks
# divide model x dataset x seed work without duplicate generation.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh

PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv python 없음: $PY"; exit 1; }

MODELS=(${TRANSFER_MODELS:-mistral7b olmo2-7b})
export REGIME_DATASETS="${REGIME_DATASETS:-mbpp kk arc-challenge}"
export REGIME_SEEDS="${REGIME_SEEDS:-0 1 2}"
export REGIME_DRIFTS="${REGIME_DRIFTS:-0 25 100 400}"

# Snapshot hashes, deterministic split, and train/validation disjointness.
"$PY" src/qualify_domain_data.py ${REGIME_DATASETS} \
  --data-root "$DATASETS_DIR" --n-train 512 --n-val 100 --seeds ${REGIME_SEEDS}

for key in "${MODELS[@]}"; do
  "$PY" src/model_matrix.py --models-dir "$MODELS_DIR" check "$key"
  model=$($PY src/model_matrix.py --models-dir "$MODELS_DIR" field "$key" path)
  targets=$($PY src/model_matrix.py --models-dir "$MODELS_DIR" field "$key" lora_targets)
  echo "== domain transfer: model=$key data=$REGIME_DATASETS seeds=$REGIME_SEEDS drift=$REGIME_DRIFTS"
  MODEL_14B="$model" REGIME_MODEL_TAG="$key" OM_LORA_TARGETS="$targets" \
    bash scripts/go_regime.sh
done
