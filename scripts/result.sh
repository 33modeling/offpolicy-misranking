#!/usr/bin/env bash
# 결과 보기 + 자동 판정 한 방:  bash scripts/result.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
OUT_ROOT="${OUT_ROOT:-$OM_WORK/runs/gate}"
PY="${VENV_DIR:+$VENV_DIR/bin/python}"; [ -x "${PY:-/none}" ] || PY=python3

for r in "$OUT_ROOT"/drift*/report.md; do
  [ -f "$r" ] && { echo "──── $r"; cat "$r"; echo; }
done
"$PY" src/judge.py "$OUT_ROOT"
