#!/usr/bin/env bash
# A new, isolated G/D experiment. Never invokes the legacy selector/random fit.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
[ "$#" -eq 0 ] || shift
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export OM_WORK="$WORK"
PAIR_ROOT=${PAIR_ROOT:-$WORK/runs/selector-pair-v1}
PY=${PAIR_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Apply before the first Python import, including all four rollout/scoring
# workers. OMP/MKL alone do not bound pthread OpenBLAS or tokenizer Rayon pools.
export OPENBLAS_NUM_THREADS=1 OPENBLAS_DEFAULT_NUM_THREADS=1 GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export OMP_THREAD_LIMIT=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
case "$MODE" in
  cpu)
    export CUDA_VISIBLE_DEVICES=""
    exec "$PY" -m pytest -q -p no:cacheprovider tests/test_selector_pair.py tests/test_selector_pair_gpu.py "$@" ;;
  init|prepare|fit|report|status|check-code)
    export CUDA_VISIBLE_DEVICES=""
    exec "$PY" src/selector_pair_gpu.py "$MODE" --root "$PAIR_ROOT" "$@" ;;
  run|develop|freeze|test) ;;
  *) echo 'usage: run_selector_pair.sh init|prepare|run|develop|fit|freeze|test|report|status|check-code|cpu'; exit 2 ;;
esac
if [ "$#" -ne 0 ]; then
  echo '[abort] run uses the frozen preparation; new options require a new root'; exit 2
fi
if [ "$MODE" = run ] || [ "$MODE" = develop ]; then
  # No arguments needed: create the default setup and validate real inputs,
  # or resume the existing frozen request before admitting any GPU work.
  CUDA_VISIBLE_DEVICES="" "$PY" src/selector_pair_gpu.py ensure-prepared --root "$PAIR_ROOT"
else
  CUDA_VISIBLE_DEVICES="" "$PY" src/selector_pair_gpu.py check-code --root "$PAIR_ROOT"
fi
export OM_WORK="$WORK" OUT_ROOT="$PAIR_ROOT"
source scripts/_e5_node.sh
export E5_FORCE=0
e5_acquire_node
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t PAIR_GPUS < <(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader)
  export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${PAIR_GPUS[*]}")"
fi
IFS=, read -r -a PAIR_GPUS <<< "$CUDA_VISIBLE_DEVICES"
[ "${#PAIR_GPUS[@]}" -eq 4 ] || { echo '[abort] four allocated GPUs required'; exit 2; }
PAIR_MEMORY=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
while read -r used; do
  [[ "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || { echo '[abort] invalid GPU status'; exit 2; }
  [ "$used" -le 4000 ] || { echo '[busy] allocated GPU occupied; no existing job was stopped'; exit 75; }
done <<< "$PAIR_MEMORY"
PAIR_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$WORK/runtime-deps")
export PYTHONPATH="$PAIR_VERIFY_PATH:$PYTHONPATH" OM_MATH_VERIFIER=math_verify OM_NODE_LOCK_HELD=1
source scripts/_selection_worker.sh
selection_run_worker "$PY" src/selector_pair_gpu.py "$MODE" --root "$PAIR_ROOT"
