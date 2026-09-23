#!/usr/bin/env bash
# Additional checkpoint diagnosis only; never restarts or trains a Pair arm.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-results}
[ "$#" -eq 0 ] || shift
export OM_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
PAIR_ROOT=${PAIR_ROOT:-$OM_WORK/runs/selector-pair-v1}
PY=${PAIR_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
case "$MODE" in
  results|all-results) export CUDA_VISIBLE_DEVICES="" ;;
  measure|all-measure)
    export OUT_ROOT="$PAIR_ROOT/sr-gc-repeat"
    source scripts/_e5_node.sh
    export E5_FORCE=0
    e5_acquire_node
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
      mapfile -t DEVICES < <(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader)
      export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${DEVICES[*]}")"
    fi
    IFS=, read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
    [ "${#DEVICES[@]}" -eq 4 ] || { echo '[abort] four allocated GPUs required'; exit 2; }
    MEMORY=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
    while read -r used; do
      [[ "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || { echo '[abort] invalid GPU status'; exit 2; }
      [ "$used" -le 4000 ] || { echo '[busy] GPUs occupied; existing jobs unchanged'; exit 75; }
    done <<< "$MEMORY"
    VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps")
    export PYTHONPATH="$VERIFY_PATH:$PYTHONPATH" OM_MATH_VERIFIER=math_verify OM_NODE_LOCK_HELD=1
    ;;
  *) echo 'usage: run_selector_pair_srgc_repeat.sh results|measure|all-results|all-measure [--interval 25] [--out FILE.txt]'; exit 2 ;;
esac
# Keep the shell alive until the child exits so the allocation lease is retained.
if [[ "$MODE" == all-* ]]; then
  "$PY" scripts/selector_pair_srgc_all_d.py "${MODE#all-}" --root "$PAIR_ROOT" "$@"
else
  "$PY" scripts/selector_pair_srgc_repeat.py "$MODE" --root "$PAIR_ROOT" "$@"
fi
