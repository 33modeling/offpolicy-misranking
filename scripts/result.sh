#!/usr/bin/env bash
# 결과 보기 + 자동 판정 한 방:  bash scripts/result.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
source scripts/_find_root.sh "${1:-}"
PY="${VENV_DIR:+$VENV_DIR/bin/python}"; [ -x "${PY:-/none}" ] || PY=python3


FOUND=0
for r in "$OUT_ROOT"/drift*/report.md "$OUT_ROOT"/report.md; do
  [ -f "$r" ] || continue
  run=$(dirname "$r")
  if [ ! -f "$run/score_protocol.json" ] || [ ! -f "$run/oracle_protocol.json" ]; then
    echo "[거부] 교정 protocol 없는 역사적 report: $r" >&2
    continue
  fi
  echo "──── $r"; cat "$r"; echo; FOUND=1
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
  echo
  LAST=$(tail -n 1 "$OUT_ROOT"/logs/drift*.log 2>/dev/null | tail -1 | cut -c1-60)
  ALIVE=$(pgrep -fc "src/experiment.py" 2>/dev/null || echo 0)
  NREP=$(ls "$OUT_ROOT"/drift*/report.json 2>/dev/null | wc -l)
  NORC=$(ls "$OUT_ROOT"/drift*/scores_oracle.json 2>/dev/null | wc -l)
  echo "한줄: 실행중=$ALIVE개, report=$NREP, oracle=$NORC, 마지막로그=[$LAST]"
  exit 2
fi
"$PY" src/judge.py "$OUT_ROOT"
echo
echo "(선택된 문제 상세는:  bash scripts/selection.sh)"
