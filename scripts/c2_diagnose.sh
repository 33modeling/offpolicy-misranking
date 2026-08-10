#!/usr/bin/env bash
# 진단만 단독 실행:  bash scripts/c2_diagnose.sh 7b
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
source scripts/_find_root.sh "${1:-}"
PY="${VENV_DIR:+$VENV_DIR/bin/python}"; [ -x "${PY:-/none}" ] || PY=python3
"$PY" src/c2_diagnose.py "$OUT_ROOT"
