#!/usr/bin/env bash
# C2 재판정 스윕 (GPU 불필요, 수 분):  bash scripts/c2_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
source scripts/_find_root.sh
PY="${VENV_DIR:+$VENV_DIR/bin/python}"; [ -x "${PY:-/none}" ] || PY=python3
"$PY" src/c2_sweep.py "$OUT_ROOT"
