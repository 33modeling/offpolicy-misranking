#!/bin/sh
# Corrected Gemma text generation. Original v1 inputs/results are never reset.
set -eu
cd "$(dirname "$0")/.."
[ "$#" -gt 0 ] || set -- all
case "${1:-}" in
    status|results|stop|resume|doctor|prepare|preflight)
        action=$1
        shift
        set -- all "$action" "$@"
        ;;
esac
if [ "${2:-}" = results ]; then
    export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    exec python3 -m srgc_research.dispatch.model_results gemma4_text "$@"
fi
if [ "${2:-}" = status ]; then
    export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    exec python3 -m srgc_research.dispatch.model_status gemma4_text "$@"
fi
if [ "${2:-}" = stop ]; then
    export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    exec python3 -m srgc_research.dispatch.gemma_text_run "$@"
fi
work=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
explicit=${GEMMA_PYTHON:-${PAIR_PYTHON:-${SWITCH_PYTHON:-}}}
python=${explicit:-${VENV_DIR:-$work/.venv-cu126}/bin/python}
if ! command -v "$python" >/dev/null 2>&1; then
    [ -z "$explicit" ] || { printf 'Python not found: %s\n' "$python" >&2; exit 2; }
    python=python3
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENBLAS_DEFAULT_NUM_THREADS=1 GOTO_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export OMP_THREAD_LIMIT=1 RAYON_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
exec "$python" -m srgc_research.dispatch.gemma_text_run "$@"
