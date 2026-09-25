#!/usr/bin/env bash
# Evaluate the Pair t=25 controls (Random, On-policy, SR) at the common step 275
# for seeds 3 and 4, so Figure 2's labels compare all four arms at one step.
#   bash scripts/run_selector_pair_step275_eval.sh            # this idle 4-GPU node evaluates unclaimed checkpoints
#   bash scripts/run_selector_pair_step275_eval.sh plan       # CPU: checkpoints that will be evaluated
#   bash scripts/run_selector_pair_step275_eval.sh status     # CPU: measured / missing
#   bash scripts/run_selector_pair_step275_eval.sh results    # CPU: ~/selector-pair-step275-results.txt
#   bash scripts/run_selector_pair_step275_eval.sh cost       # CPU: GPU time through 275 per arm, ~/selector-pair-step275-cost.txt
# Reads the Switch root; writes only under $OM_WORK/runs/selector-pair-step275-eval-v1.
# Cost reports use the separate ${PAIR_STEP275_ROOT}-cost sibling (PAIR_STEP275_COST_ROOT overrides it).
# Several nodes may run the same command; evaluations are leased, nothing is trained.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
[ "$#" -eq 0 ] || shift
export OM_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
PAIR_ROOT=${PAIR_ROOT:-$OM_WORK/runs/selector-pair-v1}
SWITCH_ROOT=${PAIR_SWITCH_ROOT:-$OM_WORK/runs/selector-pair-srgc-switch-v1}
OUTPUT=${PAIR_STEP275_ROOT:-$OM_WORK/runs/selector-pair-step275-eval-v1}
COST_OUTPUT=${PAIR_STEP275_COST_ROOT:-${OUTPUT}-cost}
PY=${PAIR_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
case "$MODE" in
  plan|status|results) export CUDA_VISIBLE_DEVICES="" ;;
  cost)
    export CUDA_VISIBLE_DEVICES=""
    exec "$PY" scripts/selector_pair_step275_cost.py --root "$PAIR_ROOT" --switch-root "$SWITCH_ROOT" \
      --eval-root "$OUTPUT" --output "$COST_OUTPUT" "$@" ;;
  run)
    export OUT_ROOT="$OUTPUT"
    source scripts/_e5_node.sh
    export E5_FORCE=0
    # Common node lease only; no Pair recovery, no source ledger changes.
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
  *) echo 'usage: bash scripts/run_selector_pair_step275_eval.sh [run|plan|status|results|cost] [--seed 3|4]'; exit 2 ;;
esac
"$PY" scripts/selector_pair_step275_eval.py "$MODE" --switch-root "$SWITCH_ROOT" --output "$OUTPUT" "$@"
