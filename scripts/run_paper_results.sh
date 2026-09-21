#!/usr/bin/env bash
# CPU-only current paper data, with one compact TXT per experiment root.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" != results ]; then
  echo 'usage: bash scripts/run_paper_results.sh results rloo|pair|mbpp'; exit 2
fi
shift
KIND=${1:-}
[ "$#" -eq 0 ] || shift
export OM_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
case "$KIND" in
  rloo) exec bash scripts/run_rloo.sh results "$@" ;;
  pair)
    PY=${PAIR_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
    [ -x "$PY" ] || PY=python3
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    exec "$PY" scripts/selector_pair_results.py \
      --root "${PAIR_ROOT:-$OM_WORK/runs/selector-pair-v1}" "$@" ;;
  mbpp)
    PY=${SWITCH_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
    [ -x "$PY" ] || PY=python3
    export EXPERIMENTS_MBPP_SUITE=all
    source scripts/_mbpp_experiments.sh
    mbpp_queue_init
    ROOT_ARGS=()
    while IFS= read -r root; do ROOT_ARGS+=(--root "$root"); done < <(mbpp_observation_roots)
    REPAIR_ROOT=${MBPP_REPAIR_ROOT:-$OM_WORK/runs/selection-switch-mbpp-quality-repair-v1}
    if [ -e "$REPAIR_ROOT" ] || [ -L "$REPAIR_ROOT" ]; then
      ROOT_ARGS+=(--repair-root "$REPAIR_ROOT")
    fi
    exec "$PY" scripts/mbpp_paper_results.py "${ROOT_ARGS[@]}" "$@" ;;
  *) echo 'usage: bash scripts/run_paper_results.sh results rloo|pair|mbpp'; exit 2 ;;
esac
