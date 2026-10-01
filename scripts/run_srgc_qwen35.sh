#!/bin/sh
# Run the same command on each allocated four-H100 node; queues lease tasks.
set -eu
cd "$(dirname "$0")/.."
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
case "${1:-}" in
    math|all) EXPLICIT=${PAIR_PYTHON:-} ;;
    mbpp) EXPLICIT=${SWITCH_PYTHON:-} ;;
    *) EXPLICIT= ;;
esac
# Match the OLMo launcher. An explicit Qwen override remains available for
# resuming an already recorded run, but a separate environment is not required.
EXPLICIT=${QWEN_PYTHON:-$EXPLICIT}
PY=${EXPLICIT:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
if ! command -v "$PY" >/dev/null 2>&1; then
    [ -z "$EXPLICIT" ] || { printf 'Python not found: %s\n' "$PY" >&2; exit 2; }
    PY=python3
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENBLAS_DEFAULT_NUM_THREADS=1 GOTO_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export OMP_THREAD_LIMIT=1 RAYON_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
exec "$PY" scripts/run_srgc_qwen35.py "$@"
