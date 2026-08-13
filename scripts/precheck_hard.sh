#!/usr/bin/env bash
# P3-0 사전 검력 체크 — GPU 0, 수 분. go_hard 실행 여부를 여기서 먼저 판정.
#   bash scripts/precheck_hard.sh
# 보고서는 group-volume 날짜 폴더에만 저장한다.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
STAMP_DIR="$OM_WORK/readouts/$(date '+%Y-%m-%d_%H%M')-precheck-hard"
mkdir -p "$STAMP_DIR"
"$PY" src/precheck_hard.py "$OM_WORK/runs" | tee "$STAMP_DIR/PRECHECK.md"
echo
echo "== 저장: $STAMP_DIR/PRECHECK.md"
