#!/usr/bin/env bash
# Fast, final-compatible regime pilot for one 4xH100 node.
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

bash scripts/check_data.sh math500 400 100

export REGIME_DATASETS="math500"
export REGIME_SEEDS="0"
export REGIME_DRIFTS="0 400"
export REGIME_RESULTS="$OM_WORK/results/regime-pilot-math500-s0-d0-d400"

mkdir -p "$OM_WORK/console-logs"
LOG="$OM_WORK/console-logs/regime-pilot-$(hostname)-$(date +%F-%H%M%S).log"

echo "[pilot] MATH-500 / seed 0 / drift 0,400 / H100 x4"
echo "[pilot] log: $LOG"

set +e
bash scripts/go_regime.sh 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

if [ "$rc" -eq 0 ]; then
  echo "[pilot] complete: $REGIME_RESULTS/FINAL_REPORT.md"
else
  echo "[pilot] failed (rc=$rc): $LOG"
fi
exit "$rc"
