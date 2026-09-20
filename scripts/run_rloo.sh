#!/usr/bin/env bash
# Direct RLOO GPU launcher. Existing experiments are never cleaned up or stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
[ "$#" -eq 0 ] || shift
export OM_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export RLOO_ROOT=${RLOO_ROOT:-$OM_WORK/runs/rloo-selector-v2}
PY=${RLOO_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
case "$MODE" in
  status)
    export CUDA_VISIBLE_DEVICES=""
    STATUS_ARGS=()
    STATUS_WATCH=
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --all|--json) STATUS_ARGS+=("$1"); shift ;;
        --watch)
          STATUS_WATCH=15; shift
          if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then STATUS_WATCH=$1; shift; fi
          [[ "$STATUS_WATCH" =~ ^[1-9][0-9]*$ ]] || { echo '[abort] watch interval must be a positive integer'; exit 2; }
          ;;
        *) echo 'usage: run_rloo.sh status [--all] [--watch [SECONDS]] [--json]'; exit 2 ;;
      esac
    done
    while :; do
      if [ -n "$STATUS_WATCH" ] && [ -t 1 ]; then printf '\033[2J\033[H'; fi
      rc=0
      "$PY" scripts/rloo_status.py --root "$RLOO_ROOT" "${STATUS_ARGS[@]}" || rc=$?
      [ -n "$STATUS_WATCH" ] || exit "$rc"
      sleep "$STATUS_WATCH"
    done ;;
  plan|prepare|report|check)
    export CUDA_VISIBLE_DEVICES=""
    exec "$PY" src/rloo_experiment.py "$MODE" --root "$RLOO_ROOT" "$@" ;;
  cpu)
    export CUDA_VISIBLE_DEVICES=""
    exec "$PY" -m pytest -q -p no:cacheprovider tests/test_rloo_experiment.py tests/test_rloo_status.py tests/test_grpo_policy.py "$@" ;;
  run) ;;
  *) echo 'usage: run_rloo.sh [run]|plan|prepare|status|report|check|cpu'; exit 2 ;;
esac
# Validate arguments and inputs before touching GPU admission or runtime setup.
if [ "$#" -eq 0 ]; then
  set -- --max-phase-seconds "${RLOO_MAX_PHASE_SECONDS:-86400}"
fi
[ "$#" -eq 2 ] && [ "$1" = --max-phase-seconds ] || {
  echo '[abort] optional run argument: --max-phase-seconds SECONDS'; exit 2;
}
"$PY" -c 'import math,sys; n=float(sys.argv[1]); sys.exit(0 if math.isfinite(n) and n>0 else 2)' "$2"
CUDA_VISIBLE_DEVICES="" "$PY" src/rloo_experiment.py ensure-prepared --root "$RLOO_ROOT"
CUDA_VISIBLE_DEVICES="" "$PY" src/rloo_experiment.py check --root "$RLOO_ROOT"
export OUT_ROOT="$RLOO_ROOT" E5_FORCE=0
source scripts/_e5_node.sh
e5_acquire_node
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t GPUS < <(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader)
  export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"
fi
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
[ "${#GPUS[@]}" -eq 4 ] || { echo '[abort] four allocated GPUs required'; exit 2; }
MEMORY=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
while read -r used; do
  [[ "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || exit 2
  [ "$used" -le 4000 ] || { echo '[busy] GPU occupied; existing jobs untouched'; exit 75; }
done <<< "$MEMORY"
export RLOO_GPU_TYPE
RLOO_GPU_TYPE=$(timeout 20 nvidia-smi --query-gpu=name --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES")
VERIFY=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps")
export PYTHONPATH="$VERIFY:$PYTHONPATH" OM_MATH_VERIFIER=math_verify OM_NODE_LOCK_HELD=1
source scripts/_selection_worker.sh
selection_run_worker "$PY" src/rloo_experiment.py "$MODE" --root "$RLOO_ROOT" "$@"
