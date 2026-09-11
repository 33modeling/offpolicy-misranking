#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-cpu}
[ "$#" -eq 0 ] || shift
PY=${GATE_PYTHON:-python3}
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
case "$MODE" in
  cpu)
    exec "$PY" -m pytest -q tests/test_selection_gate.py tests/test_selection_gate_study.py tests/test_selection_gate_gpu.py "$@"
    ;;
  plan|features|fit|analyze|initialize|decide|cost|inspect|status)
    exec "$PY" src/selection_gate_study.py "$MODE" "$@"
    ;;
  *)
    printf '%s\n' 'usage: bash scripts/run_selection_gate.sh [cpu|plan|features|fit|analyze|initialize|decide|cost|inspect|status]'
    exit 2
    ;;
esac
