#!/usr/bin/env bash
# P4-0 — K-스케일링 floor 곡선 판별 (GPU 0, 수 분). 로그는 group-volume에만.
#   bash scripts/kcurve.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
STAMP_DIR="$OM_WORK/readouts/$(date '+%Y-%m-%d_%H%M')-kcurve"
mkdir -p "$STAMP_DIR"
"$PY" src/kcurve_floor.py "$OM_WORK/runs" | tee "$STAMP_DIR/KCURVE.md"
echo
echo "== 저장: $STAMP_DIR/KCURVE.md"
