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
  targets=($(ls -d "$OM_WORK"/runs/v2-s* 2>/dev/null | grep -v smoke || true))
fi
[ "${#targets[@]}" -gt 0 ] || { echo "[abort] 대상 run 없음 (v2-s*)"; exit 1; }
OM_RESULTS="${OM_RESULTS:-$OM_WORK/results/v2}" "$PY" src/frontier.py "${targets[@]}"
