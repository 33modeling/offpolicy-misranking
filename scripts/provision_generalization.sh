#!/usr/bin/env bash
# Download and qualify every immutable model/data snapshot for a transfer matrix.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup_env.sh

PY="$VENV_DIR/bin/python"
CONFIG="${GENERALIZATION_CONFIG:-configs/domain_transfer.json}"
[ -x "$PY" ] || { echo "[abort] venv missing: $PY; run scripts/provision.sh first"; exit 1; }
[ -s "$CONFIG" ] || { echo "[abort] generalization config missing: $CONFIG"; exit 1; }

export OM_ONLINE=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

mapfile -t MODEL_KEYS < <(
  "$PY" src/model_matrix.py --config "$CONFIG" list-models
)
DATASETS=$("$PY" src/model_matrix.py --config "$CONFIG" experiment-field datasets)
SEEDS=$("$PY" src/model_matrix.py --config "$CONFIG" experiment-field seeds)

"$PY" src/model_matrix.py --config "$CONFIG" --models-dir "$MODELS_DIR" \
  download "${MODEL_KEYS[@]}"
bash scripts/fetch_datasets.sh $DATASETS

export GSM8K_DIR="$DATASETS_DIR/gsm8k"
export MATH500_DIR="$DATASETS_DIR/math500"
export MBPP_DIR="$DATASETS_DIR/mbpp"
export KK_DIR="$DATASETS_DIR/kk"
export ARC_CHALLENGE_DIR="$DATASETS_DIR/arc-challenge"
mkdir -p "$OM_WORK/contracts"
CONFIG_ID=$(sha256sum "$CONFIG" | cut -c1-16)
"$PY" src/qualify_domain_data.py $DATASETS \
  --data-root "$DATASETS_DIR" \
  --n-train "$("$PY" src/model_matrix.py --config "$CONFIG" experiment-field n_train)" \
  --n-val "$("$PY" src/model_matrix.py --config "$CONFIG" experiment-field n_val)" \
  --seeds $SEEDS \
  --output "$OM_WORK/contracts/generalization-provision-$CONFIG_ID.json"

echo "[provision] generalization snapshots qualified for $CONFIG"
