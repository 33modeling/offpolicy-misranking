#!/usr/bin/env bash
# Verify immutable primary dataset snapshots and real math reward runtime.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv missing: $PY"; exit 1; }
command -v flock >/dev/null 2>&1 || { echo "[abort] flock missing"; exit 1; }

export GSM8K_DIR="$DATASETS_DIR/gsm8k"
export MATH500_DIR="$DATASETS_DIR/math500"
mkdir -p "$OM_WORK/contracts" "$OM_WORK/locks"
(
  flock 8
  "$PY" src/qualify_domain_data.py gsm8k --data-root "$DATASETS_DIR" \
    --n-train 512 --n-val 100 --seeds 0 1 2 3 4 \
    --output "$OM_WORK/contracts/rlvr-primary-gsm8k.json"
  "$PY" src/qualify_domain_data.py math500 --data-root "$DATASETS_DIR" \
    --n-train 400 --n-val 100 --seeds 0 1 2 3 4 \
    --output "$OM_WORK/contracts/rlvr-primary-math500.json"
) 8>"$OM_WORK/locks/rlvr-primary-data-qualification.lock"
