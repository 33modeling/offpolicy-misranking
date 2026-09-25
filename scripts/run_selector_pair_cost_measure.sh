#!/usr/bin/env bash
# Fresh single-reference four-arm cost experiment; original Pair outputs are read-only.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-plan}
[ "$#" -eq 0 ] || shift
export OM_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
PAIR_ROOT=${PAIR_ROOT:-$OM_WORK/runs/selector-pair-v1}
ARGS=("$@")
CLI_SEED=
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --seed)
      i=$((i + 1))
      VALUE=${ARGS[i]:-}
      ;;
    --seed=*) VALUE=${ARGS[i]#--seed=} ;;
    *) continue ;;
  esac
  case "$VALUE" in
    3|4) ;;
    *) echo '[abort] --seed requires 3 or 4'; exit 2 ;;
  esac
  if [ -n "$CLI_SEED" ] && [ "$CLI_SEED" != "$VALUE" ]; then
    CLI_SEED=both
  else
    CLI_SEED=$VALUE
  fi
done
SEED=${PAIR_COST_MEASURE_SEED:-}
case "$SEED" in
  ""|3|4) ;;
  *) echo '[abort] PAIR_COST_MEASURE_SEED must be 3 or 4'; exit 2 ;;
esac
if [ -n "$SEED" ]; then
  [ -z "$CLI_SEED" ] || { echo '[abort] use --seed or PAIR_COST_MEASURE_SEED, not both'; exit 2; }
  set -- --seed "$SEED" "$@"
else
  SEED=$CLI_SEED
fi
[ "$SEED" != both ] || SEED=
OUTPUT=${PAIR_COST_MEASURE_ROOT:-$OM_WORK/runs/selector-pair-cost-measure${SEED:+-s$SEED}-v1}
PY=${PAIR_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
case "$MODE" in
  plan|status|results) export CUDA_VISIBLE_DEVICES="" ;;
  run)
    # Validate all sources and arguments before reserving a node or creating run files.
    "$PY" scripts/selector_pair_cost_measure.py plan --root "$PAIR_ROOT" --output "$OUTPUT" "$@" >/dev/null
    export OUT_ROOT="$OUTPUT"
    source scripts/_e5_node.sh
    export E5_FORCE=0
    SOURCE_PAIR_ROOT=$PAIR_ROOT
    PAIR_ROOT=$OUTPUT
    e5_recover_pair_gpu() { return 0; }
    e5_acquire_node
    PAIR_ROOT=$SOURCE_PAIR_ROOT
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
  *) echo 'usage: bash scripts/run_selector_pair_cost_measure.sh [plan|run|status|results] [--seed 3|4] [--replicates N]'; exit 2 ;;
esac
# Keep the shell and its node lease alive until all owned workers finish.
"$PY" scripts/selector_pair_cost_measure.py "$MODE" --root "$PAIR_ROOT" --output "$OUTPUT" "$@"
