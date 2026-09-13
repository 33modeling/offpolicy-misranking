#!/usr/bin/env bash
# Positive control: a candidate pool where selection should matter. Half the
# candidates are MATH-500 prompts, half are off-task prompts (MBPP by default);
# the ranking validation set and the independent test set stay MATH. One new
# d0 point is built with the matched OLMo configuration of the existing MATH
# point (same seed), then the reduced E5 arms and the gate arm run on it.
#
#   bash scripts/run_mixed_pool.sh pool      # build the pool file (CPU)
#   bash scripts/run_mixed_pool.sh point     # build the d0 point on THIS idle 4xH100 node (about a day)
#   bash scripts/run_mixed_pool.sh e5        # random / difficulty / fresh / reused arms, MIX_STEPS updates (default 200)
#   bash scripts/run_mixed_pool.sh gate      # executed gate arm on the same point
#   bash scripts/run_mixed_pool.sh status    # progress (no GPU)
#   bash scripts/run_mixed_pool.sh results   # finished tables (no GPU); bundle with run_e5.sh export
#
# Knobs: MIX_OTHER (mbpp), MIX_MATH (200), MIX_N_OTHER (200), MIX_STEPS (200), MIX_SEED (0).
# Requires the completed d0 points of seed MIX_SEED for math500 and MIX_OTHER.
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-status}
case "$MODE" in pool|point|e5|gate|status|results) ;; *) echo "usage: bash scripts/run_mixed_pool.sh [pool|point|e5|gate|status|results]"; exit 2 ;; esac
trap '' HUP
trap 'echo "[mixed] interrupted"; exit 130' INT TERM
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
ROOT=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
OTHER=${MIX_OTHER:-mbpp}; N_MATH=${MIX_MATH:-200}; N_OTHER=${MIX_N_OTHER:-200}; STEPS=${MIX_STEPS:-200}; SEED=${MIX_SEED:-0}
NAME=math500mix
MATH_RUN="$ROOT/family-math500-s$SEED/$TAG-s$SEED-math500-d0"
OTHER_RUN="$ROOT/family-$OTHER-s$SEED/$TAG-s$SEED-$OTHER-d0"
POOL="$OM_WORK/inputs/mixed/pool-math500-$OTHER-s$SEED.jsonl"
POINT="$ROOT/family-$NAME-s$SEED/$TAG-s$SEED-$NAME-d0"
E5_ENV=(E5_DATASET="$NAME" E5_TEST_DATASET=math500 E5_STEPS="$STEPS" E5_SEEDS="$SEED")
echo "[mixed] pool=$POOL point=$POINT arms root=$OM_WORK/runs/e5-reduced/$NAME-d0 steps=$STEPS"
case "$MODE" in
  status)
    if [ -s "$POOL" ]; then echo "pool: ready ($POOL)"; else echo "pool: not built"; fi
    if [ -s "$POINT/DONE" ]; then echo "point: DONE ($POINT)"
    elif [ -s "$POINT/logs/main.log" ]; then echo "point: in progress; last progress line:"; grep -F '[progress]' "$POINT/logs/main.log" | tail -n 1
    else echo "point: not started"; fi
    env "${E5_ENV[@]}" bash scripts/run_e5.sh status d0 2>/dev/null | grep -v setup_env
    exit 0 ;;
  results)
    env "${E5_ENV[@]}" bash scripts/run_e5.sh results 2>/dev/null | grep -v setup_env
    exit 0 ;;
  pool)
    for run in "$MATH_RUN" "$OTHER_RUN"; do [ -s "$run/DONE" ] || { echo "[abort] source point not complete: $run"; exit 1; }; done
    "$PY" src/mixed_pool.py build --math-run "$MATH_RUN" --other-run "$OTHER_RUN" --out "$POOL" \
      --math "$N_MATH" --other "$N_OTHER" --val 100 --seed "$SEED" || exit 1
    exit 0 ;;
  point)
    [ -s "$POOL" ] || { echo "[abort] pool missing; run:  bash scripts/run_mixed_pool.sh pool"; exit 1; }
    [ -s "$MATH_RUN/run_config.json" ] || { echo "[abort] source point missing: $MATH_RUN"; exit 1; }
    if [ -s "$POINT/DONE" ]; then echo "[mixed] point already complete: $POINT"; exit 0; fi
    # one node builds the point; others skip it (lease on the shared filesystem)
    mkdir -p "$(dirname "$POINT")"
    exec 6>"$POINT.lease"
    if ! flock -n 6; then echo "[busy] the mixed point is being built on another node"; exit 0; fi
    export OUT_ROOT="$POINT"   # process marker for cleanup; the point runner uses the same variable
    source scripts/_e5_node.sh || exit 1
    e5_cleanup_previous "$POINT" || exit 1
    e5_acquire_node || exit "$?"
    # matched configuration of the existing MATH point; only the pool and the output change
    while IFS= read -r line; do [ -n "$line" ] && export "${line?}"; done < <("$PY" src/mixed_pool.py env --run "$MATH_RUN") || exit 1
    export DATASET=math500 DRIFT=0 SEED="$SEED" OM_POOL_FILE="$POOL"
    unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    echo "[mixed] building the d0 point with the configuration of $MATH_RUN (model=$MODEL_PATH, N_TRAIN=$N_TRAIN, BEHAVIOR_K=$BEHAVIOR_K, FRESH_K=$FRESH_K)"
    bash scripts/run_point.sh 7>&- 8>&- 9>&-
    rc=$?
    [ -s "$POINT/DONE" ] && echo "[mixed] point complete: $POINT   next:  bash scripts/run_mixed_pool.sh e5"
    exit "$rc" ;;
  e5)
    [ -s "$POINT/DONE" ] || { echo "[abort] point not complete; run:  bash scripts/run_mixed_pool.sh point"; exit 1; }
    exec env "${E5_ENV[@]}" bash scripts/run_e5.sh d0 ;;
  gate)
    [ -s "$POINT/DONE" ] || { echo "[abort] point not complete; run:  bash scripts/run_mixed_pool.sh point"; exit 1; }
    exec env "${E5_ENV[@]}" bash scripts/run_e5.sh gate d0 ;;
esac
