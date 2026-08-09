#!/usr/bin/env bash
# 결과 보기 + 자동 판정 한 방:  bash scripts/result.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
OUT_ROOT="${OUT_ROOT:-$OM_WORK/runs/gate}"
PY="${VENV_DIR:+$VENV_DIR/bin/python}"; [ -x "${PY:-/none}" ] || PY=python3

FOUND=0
for r in "$OUT_ROOT"/drift*/report.md; do
  [ -f "$r" ] && { echo "──── $r"; cat "$r"; echo; FOUND=1; }
done
if [ "$FOUND" = 0 ]; then
  echo "!! report 없음 — 게이트 미완료. 자동 진단:"
  echo
  bash scripts/status.sh
  echo
  echo "-- main.log 실패/마지막 기록"
  grep -E "✘|실패|ERROR|abort" "$OUT_ROOT/logs/main.log" 2>/dev/null | tail -5
  tail -n 8 "$OUT_ROOT/logs/main.log" 2>/dev/null
  echo
  echo "-- 각 drift 로그 마지막 3줄"
  for lf in "$OUT_ROOT"/logs/drift*.log; do
    [ -f "$lf" ] && { echo "  ── $(basename "$lf")"; tail -n 3 "$lf" | sed 's/^/  /'; }
  done
  echo
  echo "→ 이 출력 전체를 분석 담당에게 전달할 것"
  exit 0
fi
"$PY" src/judge.py "$OUT_ROOT"
echo
echo "(선택된 문제 상세는:  bash scripts/selection.sh)"
