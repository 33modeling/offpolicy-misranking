#!/usr/bin/env bash
# 방법별로 실제 어떤 문제가 뽑혔는지 + 겹침 행렬:  bash scripts/selection.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
PY="${VENV_DIR:+$VENV_DIR/bin/python}"; [ -x "${PY:-/none}" ] || PY=python3
source scripts/_find_root.sh
"$PY" src/show_selection.py "$OUT_ROOT" "$@"
