#!/usr/bin/env bash
# 결과 테이블 원샷:  bash scripts/tables.sh [7b|14b|7bm|14bm|<경로> ...]
# 인자 없으면 존재하는 표준 run 전부. 출력: results/TABLES.md + 화면.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
# 사용률 기준 잡 킬 대응 — CPU 구간 동안 저강도 keepalive (GPU 있을 때만)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${NGPU:-0}" -gt 0 ]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU - 1))) "$PY" scripts/gpu_keepalive.py > /dev/null 2>&1 &
  trap "kill $! 2>/dev/null" EXIT
fi

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
fi
[ "${#targets[@]}" -gt 0 ] || { echo "[abort] 대상 run 없음"; exit 1; }
"$PY" src/make_tables.py "${targets[@]}"
