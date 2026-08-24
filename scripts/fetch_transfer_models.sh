#!/usr/bin/env bash
# Download immutable non-Qwen model snapshots onto the shared model volume.
set -euo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=1
source scripts/setup_env.sh

"$VENV_DIR/bin/python" src/model_matrix.py \
  --config "${TRANSFER_CONFIG:-configs/domain_transfer.json}" \
  --models-dir "$MODELS_DIR" download "$@"
