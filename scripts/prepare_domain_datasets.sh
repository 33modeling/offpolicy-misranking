#!/usr/bin/env bash
# Download and qualify the fixed non-math snapshots before reserving GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=1
source scripts/setup_env.sh

bash scripts/fetch_datasets.sh mbpp kk arc-challenge
"$VENV_DIR/bin/python" src/qualify_domain_data.py \
  --data-root "$DATASETS_DIR" --n-train 512 --n-val 100 --seeds 0 1 2

echo "[ready] MBPP + Knights & Knaves + ARC-Challenge"
