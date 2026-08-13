#!/usr/bin/env bash
# 수확 원스톱 — 한 번 실행하고 마지막 줄에 찍히는 폴더 하나만 전달하면 끝:
#   KCURVE(GPU 0, 수 분) + READOUT + REVERSAL(닻 포함) + 표 사본을 같은 폴더에.
#   bash scripts/harvest.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
STAMP_DIR="$OM_WORK/readouts/$(date '+%Y-%m-%d_%H%M')-harvest"
mkdir -p "$STAMP_DIR"

"$PY" src/kcurve_floor.py "$OM_WORK/runs" | tee "$STAMP_DIR/KCURVE.md" || true
"$PY" src/readout_summary.py "$OM_WORK/runs" | tee "$STAMP_DIR/READOUT.md"
"$PY" src/reversal_freq.py "$OM_WORK/runs" > "$STAMP_DIR/REVERSAL.md" 2>/dev/null || true
cp "$OM_WORK"/results/v2/TABLES.md "$OM_WORK"/results/v2/FRONTIER.md \
   results/TABLES.md "$STAMP_DIR/" 2>/dev/null || true
echo
echo "== 전달할 폴더 하나: $STAMP_DIR"
ls "$STAMP_DIR"
