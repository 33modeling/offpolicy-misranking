#!/bin/sh
# V9 (ICML 2027) experiments. Same command on every 4xH100 node, no arguments:
#   sh v9/run.sh            freeze-tau -> x1 -> x2 -> x5 -> x3 -> x4 -> x8 -> x9 -> x10 -> x11 -> x12 -> x13 -> x14 -> collect
#   sh v9/run.sh status     V8-style screen: time left, per stage done/running/waiting, what runs where, nodes (--watch)
#   sh v9/run.sh <stage>    one stage: freeze-tau | x1 | x2 | x3 | x4 | x5 | x8 | x9 | x10 | x11 | x12 | x13 | x14 | collect
#   sh v9/run.sh retry-failed   clear the failure count of stopped tasks (they also restart by themselves, 3 times)
# V8 results are read from $WORK/v8 (never written); V9 writes only to $WORK/v9.
# Tasks are leased per node, finished work is skipped, a stopped run resumes from its latest.pt.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export OM_WORK="$WORK"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Worker interpreter: the existing SRGC venv (same default as the V8 launchers); GPU children use the
# cohort's own interpreter (PAIR_PYTHON / SWITCH_PYTHON / QWEN_PYTHON / V8_PYTHON_<COHORT>).
PY=${V9_PYTHON:-${V8_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}}
if ! command -v "$PY" >/dev/null 2>&1; then PY=python3; fi
cd "$HERE"
exec "$PY" -m v9lib.orchestrate "$@"
