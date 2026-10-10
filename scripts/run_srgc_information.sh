#!/bin/sh
# Each node claims a different dataset/seed. Reports require an explicit action.
set -eu
cd "$(dirname "$0")/.."
DATASET=${1:-math}
ACTION=${2:-collect}
case "$DATASET" in math|mbpp|all) ;; *) echo "usage: sh scripts/run_srgc_information.sh [math|mbpp|all] [collect|status|report options]" >&2; exit 2 ;; esac
case "$ACTION" in collect|status|report) ;; *) echo "action must be collect, status or report" >&2; exit 2 ;; esac
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export OM_WORK="$WORK"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

if [ "$ACTION" = collect ]; then
    case "$DATASET" in mbpp) EXPLICIT=${SWITCH_PYTHON:-} ;; *) EXPLICIT=${PAIR_PYTHON:-} ;; esac
    PY=${EXPLICIT:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
    if ! command -v "$PY" >/dev/null 2>&1; then
        [ -z "$EXPLICIT" ] || { printf 'Python not found: %s\n' "$PY" >&2; exit 2; }
        PY=python3
    fi
else
    PY=python3
    export CUDA_VISIBLE_DEVICES=""
fi

if [ "$#" -le 1 ] || { [ "$#" -eq 2 ] && [ "$ACTION" = collect ]; }; then
    exec "$PY" -m srgc_research.dispatch.information_queue "$DATASET"
fi

if [ "$ACTION" = collect ]; then
    exec "$PY" -m srgc_research.dispatch.information_run "$@"
fi
exec "$PY" -m srgc_research.information_cli "$@"
