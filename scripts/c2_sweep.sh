#!/usr/bin/env bash
# C2 재판정 스윕 (GPU 불필요, 수 분):  bash scripts/c2_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
source scripts/_find_root.sh
PY="${VENV_DIR:+$VENV_DIR/bin/python}"; [ -x "${PY:-/none}" ] || PY=python3
"$PY" src/c2_sweep.py "$OUT_ROOT"
echo; echo "== 진단 (왜 실패했나 / 얼마나 더 관측해야 하나)"
"$PY" src/c2_diagnose.py "$OUT_ROOT"
