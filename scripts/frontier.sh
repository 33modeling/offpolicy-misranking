#!/usr/bin/env bash
# frontier 사후 분석 원샷:  bash scripts/frontier.sh [run경로...]
# 인자 없으면 v2 run 전부. 출력: $OM_WORK/results/v2/FRONTIER.md
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
if [ "$#" -gt 0 ]; then
  targets=("$@")
else
  # 완주(DONE) run만 — 부분 완주 상태에서 후처리 전체가 깨지는 것 방지
  targets=()
# 구버전 이중 접미사 run은 신형(단일 접미사) 완주가 있으면 제외 — 같은
# (dataset, seed)가 두 디렉터리로 이중 집계되는 것 방지
_legacy_dup() { case "$1" in
    *-dapo-math-dapo-math) echo "${1%-dapo-math}";;
    *-math500-math500)     echo "${1%-math500}";;
    *-mbpp-mbpp)           echo "${1%-mbpp}";;
    *-kk-kk)               echo "${1%-kk}";;
    *-apps-apps)           echo "${1%-apps}";;
    *) echo "";;
  esac; }
  for d in $(ls -d "$OM_WORK"/runs/v[0-9]*-s* 2>/dev/null | grep -v smoke); do
    dup=$(_legacy_dup "$d")
    [ -n "$dup" ] && [ -f "$dup/DONE" ] && { echo "  [skip] legacy 이중 접미사: $(basename "$d") (신형 완주 존재)"; continue; }
    [ -f "$d/DONE" ] && targets+=("$d")
  done
fi
[ "${#targets[@]}" -gt 0 ] || { echo "[abort] 대상 run 없음 (v<세대>-s*)"; exit 1; }
OM_RESULTS="${OM_RESULTS:-$OM_WORK/results/v2}" "$PY" src/frontier.py "${targets[@]}"
