#!/bin/sh
# Same command on each idle four-H100 node; no changes to the original queue.
set -eu
cd "$(dirname "$0")/.."
DATASET=${1:-}
case "$DATASET" in math|mbpp|all) ;; *) echo "usage: sh scripts/run_srgc_research.sh math|mbpp|all n01..n08 [run|status|results|json]" >&2; exit 2 ;; esac
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
case "$DATASET" in mbpp) EXPLICIT=${SWITCH_PYTHON:-} ;; *) EXPLICIT=${PAIR_PYTHON:-} ;; esac
PY=${EXPLICIT:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
if ! command -v "$PY" >/dev/null 2>&1; then
    [ -z "$EXPLICIT" ] || { printf 'Python not found: %s\n' "$PY" >&2; exit 2; }
    PY=python3
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
exec "$PY" -m srgc_research.cli "$@"
