#!/bin/sh
# Local CPU-only verification (never touches the GPU): unit tests + end-to-end smoke.
#   sh v8/experiments/smoke_cpu.sh [out_dir]
# Uses the code repository at V8_CODE_ROOT (default /home/kms/dev/offpolicy-misranking).
# TORCH_PY (optional) = a CPU python with torch/transformers/peft for the tiny OLMo-3 path.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HOME/.cache/v8-smoke/out}
export CUDA_VISIBLE_DEVICES="" V8_FSYNC=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
PY=${PY:-python3}
TORCH_PY=${TORCH_PY:-$HOME/.venvs/proto-ml/bin/python}
cd "$HERE/tests"
nice -n 15 "$PY" -m unittest discover -s . -p 'test_*.py'
if [ -x "$TORCH_PY" ]; then nice -n 15 "$TORCH_PY" -m unittest test_tiny_torch; fi
cd "$HERE"
SMOKE_PY=$PY
if [ -x "$TORCH_PY" ]; then SMOKE_PY=$TORCH_PY; fi
nice -n 15 "$SMOKE_PY" -m switching.cli smoke --out "$OUT"
