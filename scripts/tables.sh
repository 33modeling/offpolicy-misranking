#!/usr/bin/env bash
# 결과 테이블 원샷:  bash scripts/tables.sh [7b|14b|7bm|14bm|<경로> ...]
# 인자 없으면 존재하는 표준 run 전부. 출력: results/TABLES.md + 화면.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3

resolve() {
  case "$1" in
    7b)   echo "$OM_WORK/runs/gate" ;;
    14b)  echo "$OM_WORK/runs/gate-14b" ;;
    7bm)  echo "$OM_WORK/runs/gate-7b-math500" ;;
    14bm) echo "$OM_WORK/runs/gate-14b-math500" ;;
    *)    echo "$1" ;;
  esac
}

targets=()
if [ "$#" -gt 0 ]; then
  for a in "$@"; do targets+=("$(resolve "$a")"); done
else
  for a in 7b 14b 7bm 14bm; do
    p="$(resolve "$a")"; [ -d "$p" ] && targets+=("$p")
  done
  # v2 계열 완주분도 기본 포함 (부분 완주 안전 — DONE만)
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
  for d in $(ls -d "$OM_WORK"/runs/v2-* 2>/dev/null | grep -v smoke); do
    dup=$(_legacy_dup "$d")
    [ -n "$dup" ] && [ -f "$dup/DONE" ] && { echo "  [skip] legacy 이중 접미사: $(basename "$d") (신형 완주 존재)"; continue; }
    [ -f "$d/DONE" ] && targets+=("$d")
  done
fi
[ "${#targets[@]}" -gt 0 ] || { echo "[abort] 대상 run 없음"; exit 1; }
"$PY" src/make_tables.py "${targets[@]}"
