#!/usr/bin/env bash
# val 방향 심화 — fresh K 추가 수집 + val gradient 재계산 (GPU 1장, 7B ~1시간)
#   bash scripts/deepen_val.sh 7b [drift100] [추가K=24]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
source scripts/_find_root.sh "${1:-7b}"
DRIFT_DIR="${2:-drift100}"
ADDK="${3:-24}"
RUN="$OUT_ROOT/$DRIFT_DIR"
PY="$VENV_DIR/bin/python"
ADAPTER=$(ls -d "$RUN"/drift_* 2>/dev/null | head -1)
[ -n "$ADAPTER" ] || { echo "[abort] adapter 없음: $RUN"; exit 1; }
MODEL="${MODEL:-$MODELS_DIR/Qwen2.5-7B-Instruct}"
echo "== val 심화: $RUN (+K=$ADDK, adapter=$(basename "$ADAPTER"))"
CUDA_VISIBLE_DEVICES="${GPU:-0}" "$PY" src/experiment.py --stage val-deepen \
  --run "$RUN" --model "$MODEL" --adapter "$ADAPTER" --val-k "$ADDK"
CUDA_VISIBLE_DEVICES="${GPU:-0}" "$PY" src/experiment.py --stage val-grads \
  --run "$RUN" --model "$MODEL" --adapter "$ADAPTER"
echo "== 완료 — 재판정:  bash scripts/c2_sweep.sh 7b"
