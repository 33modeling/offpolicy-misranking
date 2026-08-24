#!/usr/bin/env bash
# Download and qualify the fixed non-math snapshots before reserving GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=1
source scripts/setup_env.sh
CONFIG="${TRANSFER_CONFIG:-configs/domain_transfer.json}"
PY="$VENV_DIR/bin/python"
field() { "$PY" src/model_matrix.py --config "$CONFIG" experiment-field "$1"; }

bash scripts/fetch_datasets.sh mbpp kk arc-challenge
"$PY" src/qualify_domain_data.py --data-root "$DATASETS_DIR" \
  --n-train "$(field n_train)" --n-val "$(field n_val)" --seeds $(field seeds)

echo "[ready] MBPP + Knights & Knaves + ARC-Challenge"
