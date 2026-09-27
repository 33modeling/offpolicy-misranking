#!/bin/sh
set -eu
cd "$(dirname "$0")/.."

usage() {
    printf '%s\n' 'usage: sh scripts/run_srgc.sh math|mbpp [run|status|results|costs]'
}

DATASET=${1:-}
MODE=${2:-run}
case "$DATASET" in
    -h|--help) usage; exit 0 ;;
    math|mbpp) ;;
    *) usage >&2; exit 2 ;;
esac
[ "$#" -le 2 ] || { usage >&2; exit 2; }
case "$MODE" in run|status|results|costs) ;; *) usage >&2; exit 2 ;; esac

WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
case "$DATASET" in
    math) EXPLICIT=${PAIR_PYTHON:-} ;;
    mbpp) EXPLICIT=${SWITCH_PYTHON:-} ;;
esac
PY=${EXPLICIT:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
if ! command -v "$PY" >/dev/null 2>&1; then
    [ -z "$EXPLICIT" ] || { printf 'Python not found: %s\n' "$PY" >&2; exit 2; }
    PY=python3
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENBLAS_DEFAULT_NUM_THREADS=1 GOTO_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export OMP_THREAD_LIMIT=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

if [ "$MODE" = run ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0,1,2,3}
    exec "$PY" scripts/run_srgc_rebuttal.py worker --dataset "$DATASET" --fresh "${SRGC_RUN_NAME:-restart1}"
fi
export CUDA_VISIBLE_DEVICES=""
exec "$PY" scripts/run_srgc_rebuttal.py "$MODE" --dataset "$DATASET"
