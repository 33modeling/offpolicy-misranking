#!/bin/sh
# Same command on each four-GPU node: local weights, shared queues, resume first.
set -eu
cd "$(dirname "$0")/.."
[ "$#" -gt 0 ] || set -- all
if [ "${1:-}" = status ] || [ "${1:-}" = results ]; then
    ACTION=$1
    shift
    set -- all "$ACTION" "$@"
fi
if [ "${2:-}" = results ]; then
    export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    exec python3 -m srgc_research.dispatch.model_results llama31 "$@"
fi
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
case "${1:-}" in
    math|all) EXPLICIT=${PAIR_PYTHON:-} ;;
    mbpp) EXPLICIT=${SWITCH_PYTHON:-} ;;
    *) EXPLICIT= ;;
esac
EXPLICIT=${LLAMA_PYTHON:-$EXPLICIT}
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
exec "$PY" -m srgc_research.dispatch.llama_run "$@"
