#!/usr/bin/env bash
# B1 — K-스케일링 floor 곡선 전 조건 확장 (GPU 0, 수 분).
#   bash scripts/kcurve_all.sh
# 사전 등록 P4-0 판정은 불변, v1 게이트 포함 전 run을 확장 증거로 보고.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
STAMP_DIR="$OM_WORK/readouts/$(date '+%Y-%m-%d_%H%M')-kcurve-all"
mkdir -p "$STAMP_DIR"
PYTHONPATH=src "$PY" src/kcurve_all.py "$OM_WORK/runs" | tee "$STAMP_DIR/KCURVE_ALL.md"
echo
echo "== 저장: $STAMP_DIR/KCURVE_ALL.md"
