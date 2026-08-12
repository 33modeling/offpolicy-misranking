#!/usr/bin/env bash
# 즉시 판독 원샷 — 완주(DONE)된 v2 run 전부에 judge 판정 + 핵심 통계를
# READOUT.md 로 저장하고, 자동으로 커밋·push까지 시도한다.
# push가 되면 사람이 옮길 것이 없다 — Claude가 GitHub에서 받아 판독한다.
#   bash scripts/read_now.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
OUT="READOUT.md"

{
  echo "# 즉시 판독 — $(date '+%F %T')"
  echo
  n=0
  for d in "$OM_WORK"/runs/v2-*; do
    [ -f "$d/DONE" ] || continue
    n=$((n + 1))
    echo "## $(basename "$d")"
    echo
    echo '```'
    "$PY" src/judge.py "$d" 2>&1 | tail -15
    echo '```'
    echo
    echo "통계 (precision · 우연 p · 부트스트랩 CI):"
    echo
    echo '```'
    "$PY" src/stats_extra.py "$d" 0.10 300 2>&1 | sed -n '1,9p'
    echo '```'
    echo
  done
  echo "---"
  echo "완주 ${n}개 판독 완료."
} | tee "$OUT"

echo "== 저장 완료: $(pwd)/$OUT"
