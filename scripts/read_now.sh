#!/usr/bin/env bash
# 즉시 판독 — 사람이 읽는 보고서(READOUT.md) 생성:
#   ① 한눈 요약 표(수치+평문 판정) ② 자동 결론 ③ 용어 설명 ④ 상세(원시 출력)
# 그룹볼륨 날짜·시간 폴더에 보관하고 경로를 마지막 줄에 찍는다.
#   bash scripts/read_now.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
OUT="READOUT.md"

"$PY" src/readout_summary.py "$OM_WORK/runs" | tee "$OUT"

STAMP_DIR="$OM_WORK/readouts/$(date '+%Y-%m-%d_%H%M')"
mkdir -p "$STAMP_DIR"
cp "$OUT" "$STAMP_DIR/"
cp "$OM_WORK"/results/v2/TABLES.md "$OM_WORK"/results/v2/FRONTIER.md \
   results/TABLES.md "$STAMP_DIR/" 2>/dev/null || true
echo
echo "== 저장 완료: $STAMP_DIR/READOUT.md"
ls "$STAMP_DIR"
