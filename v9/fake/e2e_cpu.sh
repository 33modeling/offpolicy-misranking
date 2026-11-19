#!/bin/sh
# Local CPU end-to-end check of `sh v9/run.sh` on a fake $WORK (software check only, never paper data).
#   sh v9/fake/e2e_cpu.sh [dir]      TORCH_PY = a CPU python with torch/transformers/peft/numpy/scipy
# 1. build fake V8 outputs (V8 code, ToyBackend + tiny OLMo-3)   2. sh v9/run.sh (no arguments)
# 3. the same command again (nothing redone)   4. status
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$HERE/../.." && pwd)
DIR=${1:-${TMPDIR:-/tmp}/v9-e2e}
TORCH_PY=${TORCH_PY:-python3}
export GROUP_VOLUME="$DIR" OM_USER=u V8_CODE_ROOT="$REPO" V9_ALLOW_FAKE=1 V9_NPROC=1 CUDA_VISIBLE_DEVICES= \
       V8_FSYNC=0 V9_PYTHON="$TORCH_PY" OMP_NUM_THREADS=2
unset OM_WORK V8_ROOT V9_ROOT
if [ ! -f "$DIR/u/offpolicy-misranking/v9-fake-inputs/llama_math-locator.json" ]; then
    "$TORCH_PY" "$HERE/make_fake_work.py" "$DIR"
fi
sh "$REPO/v9/run.sh" || echo "[e2e] first run exit $?"
sh "$REPO/v9/run.sh" || echo "[e2e] second run exit $?"
sh "$REPO/v9/run.sh" status
ls -l "$DIR/u/offpolicy-misranking/v9/paper"
