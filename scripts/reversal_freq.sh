#!/usr/bin/env bash
# 부호반전 빈도 재집계 (CPU, 기존 산출물만 — GPU·신규 롤아웃 불필요):
#   프롬프트 단위 반전율·경계 대역 피해자 비율·불일치 경보 조건부 반전율.
#   원고 primary table 후보 (리뷰어 질문 "existence vs prevalence" 대응).
#   bash scripts/reversal_freq.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
STAMP_DIR="$OM_WORK/readouts/$(date '+%Y-%m-%d_%H%M')-reversal"
mkdir -p "$STAMP_DIR"

"$PY" src/reversal_freq.py "$OM_WORK/runs" | tee "$STAMP_DIR/REVERSAL.md"
echo
echo "== 저장 완료: $STAMP_DIR/REVERSAL.md"
