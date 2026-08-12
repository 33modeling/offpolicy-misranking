#!/usr/bin/env bash
# 즉시 판독 원샷 — 완주(DONE)된 v2 run 전부에 judge 자동 판정 + 핵심 수치.
# 출력은 화면과 READOUT.txt 양쪽에 남는다 (사진은 화면 아무거나).
#   bash scripts/read_now.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3

{
  echo "===== 즉시 판독 $(date '+%F %T') ====="
  n=0
  for d in "$OM_WORK"/runs/v2-*; do
    [ -f "$d/DONE" ] || continue
    n=$((n + 1))
    echo
    echo "======== $(basename "$d") ========"
    "$PY" src/judge.py "$d" 2>&1 | tail -15
    echo "-- 통계 (precision·우연 p·부트스트랩 CI):"
    "$PY" src/stats_extra.py "$d" 2>&1 | sed -n '2,8p'
  done
  echo
  echo "===== 완주 ${n}개 판독 끝 — 이 출력 전체를 사진으로 전달 ====="
} | tee READOUT.txt
