#!/bin/sh
# Run the same command on each allocated four-H100 node; queues lease tasks.
set -eu
cd "$(dirname "$0")/.."
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
PY=${QWEN_PYTHON:-$WORK/.venv-qwen35/bin/python}
command -v "$PY" >/dev/null 2>&1 || {
    echo "Qwen Python not found: $PY; set QWEN_PYTHON to a separate compatible environment" >&2; exit 2;
}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
exec "$PY" scripts/run_srgc_qwen35.py "$@"
