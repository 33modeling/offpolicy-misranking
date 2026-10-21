#!/bin/sh
# Same command on each four-GPU node: local weights, shared queues, resume first.
set -eu
cd "$(dirname "$0")/.."
if [ "${SRGC_GEMMA_LEGACY_PROTOCOL:-0}" = 1 ]; then
    export SRGC_GEMMA_TEXT_PROTOCOL=0
else
    export SRGC_GEMMA_TEXT_PROTOCOL=${SRGC_GEMMA_TEXT_PROTOCOL:-1}
fi
[ "$#" -gt 0 ] || set -- all
if [ "${1:-}" = status ] || [ "${1:-}" = results ] || [ "${1:-}" = audit ] || [ "${1:-}" = stop ] || [ "${1:-}" = resume ] || [ "${1:-}" = prepare ] || [ "${1:-}" = doctor ] || [ "${1:-}" = preflight ]; then
    ACTION=$1
    shift
    set -- all "$ACTION" "$@"
fi
if [ "${SRGC_GEMMA_TEXT_PROTOCOL:-0}" = 1 ]; then
    case "${2:-run}" in
        results|status)
            export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
            if [ "$2" = results ]; then
                exec python3 -m srgc_research.dispatch.model_results gemma4_text "$@"
            fi
            exec python3 -m srgc_research.dispatch.model_status gemma4_text "$@" ;;
        stop)
            export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
            exec python3 -m srgc_research.dispatch.gemma_text_run "$@" ;;
    esac
fi
if [ "${2:-}" = results ]; then
    export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
    exec python3 -m srgc_research.dispatch.model_results gemma4 "$@"
fi
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
case "${1:-}" in
    math|all) EXPLICIT=${PAIR_PYTHON:-} ;;
    mbpp) EXPLICIT=${SWITCH_PYTHON:-} ;;
    *) EXPLICIT= ;;
esac
EXPLICIT=${GEMMA_PYTHON:-$EXPLICIT}
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
if [ "${2:-}" = audit ]; then
    export CUDA_VISIBLE_DEVICES=""
    AUDIT_DATASET=$1
    shift 2
    exec "$PY" -m srgc_research.dispatch.gemma_initial_audit "$AUDIT_DATASET" "$@"
fi
# Training commands share one model queue; read-only/control actions stay model-specific.
if [ "${SRGC_MODEL_CHAIN:-1}" = 1 ] && [ "${SRGC_MODEL_CHAIN_CHILD:-0}" != 1 ]; then
    case "${1:-}:${2:-run}" in
        all:run|math:run|mbpp:run|all:resume|math:resume|mbpp:resume)
            exec "$PY" -m srgc_research.dispatch.model_chain --start gemma4 "$@" ;;
    esac
fi
if [ "${SRGC_GEMMA_TEXT_PROTOCOL:-0}" = 1 ]; then
    exec "$PY" -m srgc_research.dispatch.gemma_text_run "$@"
fi
exec "$PY" -m srgc_research.dispatch.gemma_run "$@"
