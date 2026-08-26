#!/usr/bin/env bash
# Full regime discovery worker for one 4xH100 cluster.
# Run this same script on every cluster; the shared queue prevents duplicate work.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup_env.sh

PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv missing: $PY"; exit 1; }
[ -f "$MODEL_QWEN25_7B/config.json" ] || {
  echo "[abort] Qwen2.5-7B snapshot missing: $MODEL_QWEN25_7B"
  exit 1
}

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
H100_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c 'H100' || true)
[ "$GPU_COUNT" -ge 4 ] && [ "$H100_COUNT" -ge 4 ] || {
  echo "[abort] four H100 GPUs required (detected: GPUs=$GPU_COUNT, H100=$H100_COUNT)"
  exit 1
}

bash scripts/check_data.sh gsm8k 512 100
bash scripts/check_data.sh math500 400 100

# Canonical discovery matrix. Clear inherited overrides that could silently
# turn this launcher into a reduced or incompatible run.
export REGIME_DATASETS="gsm8k math500"
export REGIME_SEEDS="0 1 2"
export REGIME_DRIFTS="0 25 100 400"
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
unset REGIME_N_TRAIN

mkdir -p "$OM_WORK/console-logs"
LOG="$OM_WORK/console-logs/regime-discovery-$(hostname)-$(date +%F-%H%M%S).log"

echo "[discovery] Qwen2.5-7B / GSM8K+MATH-500 / seeds 0-2 / drift 0,25,100,400"
echo "[discovery] log: $LOG"

set +e
bash scripts/go_regime.sh 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

if [ "$rc" -eq 0 ]; then
  echo "[discovery] complete"
else
  echo "[discovery] failed (rc=$rc): $LOG"
fi
exit "$rc"
