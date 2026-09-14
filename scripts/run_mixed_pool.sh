#!/usr/bin/env bash
# Positive control: a candidate pool where selection should matter. Half the
# candidates are MATH-500 prompts, half are off-task prompts (MBPP by default);
# the ranking validation set and the independent test set stay MATH. One new
# d0 point is built with the matched OLMo configuration of the existing MATH
# point (same seed), then the reduced E5 arms and the gate arm run on it.
#
#   bash scripts/run_mixed_pool.sh pool      # build the pool file (CPU)
#   bash scripts/run_mixed_pool.sh point     # build the d0 point on THIS idle 4xH100 node (about a day);
#                                            # resumable: a partial point re-enters the commit that started
#                                            # it and transient shard crashes are retried (MIX_POINT_ATTEMPTS=3)
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
    elif [ -s "$POINT/logs/main.log" ]; then
      echo "point: in progress; last progress line:"; grep -F '[progress]' "$POINT/logs/main.log" | tail -n 1
      newest=$(find "$POINT" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -n 1 | cut -d. -f1)
      [ -n "$newest" ] && echo "point: last file write $(( ($(date +%s) - newest) / 60 )) min ago"
      if [ -f "$POINT.lease" ]; then
        if ( exec 6<"$POINT.lease"; flock -n 6 ) 2>/dev/null; then held="free from this node's view"; else held="HELD (visible from this node)"; fi
        echo "point: lease $held; note: $(cat "$POINT.lease" 2>/dev/null || echo none)"
        echo "       (a lease taken on another node may not be visible here; the note names the node that took it)"
      fi
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
    # A point started before 2026-09-14 recorded the pool as a prescreened pool; every launch of it
    # aborts at the qualification stage and its run config cannot be changed. It has to be rebuilt.
    if [ -s "$POINT/run_config.json" ] && "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pool") else 1)' "$POINT/run_config.json"; then
      echo "[abort] this point was initialized with the pool declared as a prescreened pool, so every"
      echo "        launch stops at [qualification-abort]. Its run config is immutable; move it aside"
      echo "        and this command rebuilds it from the start:"
      echo "          mv $POINT $POINT.pre-20260914"
      exit 1
    fi
    # one node builds the point; others skip it (lease on the shared filesystem)
    mkdir -p "$(dirname "$POINT")"
    source scripts/_lease.sh
    exec 6>>"$POINT.lease"
    if ! flock -n 6; then
      echo "[busy] the mixed point is claimed: $(head -n 1 "$POINT.lease" 2>/dev/null || echo 'no lease note')"
      newest=$(find "$POINT" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -n 1)
      if [ -n "$newest" ]; then
        age=$(( $(date +%s) - ${newest%.*} ))
        echo "[busy] its last file write was $((age / 60)) min ago"
        [ "$age" -gt 1800 ] && echo "[busy] nothing written for over 30 min: the holder looks dead; on that node run  bash scripts/run_queue.sh stop"
      else
        echo "[busy] the point directory has no file yet; the holder may be starting up or dead"
      fi
      exit 0
    fi
    lease_note "$POINT.lease"
    export OUT_ROOT="$POINT"   # process marker for cleanup; the point runner uses the same variable
    source scripts/_e5_node.sh || exit 1
    e5_cleanup_previous "$POINT" || exit 1
    e5_acquire_node || exit "$?"
    # matched configuration of the existing MATH point; only the pool and the output change
    while IFS= read -r line; do [ -n "$line" ] && export "${line?}"; done < <("$PY" src/mixed_pool.py env --run "$MATH_RUN") || exit 1
    # OM_PROMPT_POOL_FILE, not OM_POOL_FILE: the latter declares a prescreened pool and makes
    # run_point.sh requalify it against the main run, which a mixed pool cannot satisfy.
    export DATASET=math500 DRIFT=0 SEED="$SEED" OM_PROMPT_POOL_FILE="$POOL"
    unset OM_POOL_FILE
    unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    echo "[mixed] building the d0 point with the configuration of $MATH_RUN (model=$MODEL_PATH, N_TRAIN=$N_TRAIN, BEHAVIOR_K=$BEHAVIOR_K, FRESH_K=$FRESH_K)"
    # run_point.sh pins the commit that initialized the point (run_config.json) and
    # refuses every stage from another revision ([code-abort]); after `git pull` here
    # a partial point re-enters its own commit through a node-local checkout, as the
    # matrix supervisor does. A new point runs this checkout.
    RUNNER=$PWD/scripts/run_point.sh
    PIN=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("git") or "")' "$POINT/run_config.json" 2>/dev/null || true)
    HEAD=$(git rev-parse HEAD 2>/dev/null || true)
    if [ -n "$PIN" ] && [ "$PIN" != "$HEAD" ]; then
      source scripts/_pin_checkout.sh
      PIPELINE=$(pin_checkout "$PIN") || { echo "[abort] cannot re-enter the point's pinned commit $PIN"; exit 1; }
      RUNNER=$PIPELINE/scripts/run_point.sh
      echo "[mixed] re-entering the partial point under its pinned commit ${PIN:0:9} (this checkout is ${HEAD:0:9}): $PIPELINE"
    fi
    # Transient shard crashes (CUDA faults, killed workers) are retried with a rotated GPU
    # order; contract failures (rc 2/43, [code-abort]/[config-abort]) are not.
    attempts=${MIX_POINT_ATTEMPTS:-3}; rc=1; reason=""
    for ((attempt = 1; attempt <= attempts; attempt++)); do
      echo "[mixed] point attempt $attempt/$attempts ($(date -u +%Y-%m-%dT%H:%MZ))"
      OM_RETRY_INDEX=$attempt bash "$RUNNER" 7>&- 8>&- 9>&-
      rc=$?
      if [ -s "$POINT/DONE" ]; then echo "[mixed] point complete: $POINT   next:  bash scripts/run_mixed_pool.sh e5"; exit 0; fi
      reason=$(grep -aE '\[(code-abort|config-abort|permanent-contract|regime-contract-abort|stage-fail|abort)\]|Traceback|Error' \
        "$POINT/logs/main.log" 2>/dev/null | tail -n 1 | cut -c1-200)
      echo "[mixed] point attempt $attempt failed rc=$rc: ${reason:-no error line in $POINT/logs/main.log}"
      case "$rc" in 2|43) echo "[mixed] contract/config failure; not retrying"; exit "$rc" ;; esac
      case "$reason" in *code-abort*|*config-abort*|*permanent-contract*) echo "[mixed] not a transient failure; not retrying"; exit 43 ;; esac
      if [ "$attempt" -lt "$attempts" ]; then
        echo "[mixed] retrying in ${MIX_RETRY_SLEEP:-60}s with a rotated GPU order"; sleep "${MIX_RETRY_SLEEP:-60}"
      fi
    done
    echo "[mixed] point failed $attempts times (last: ${reason:-?}); finished shards are kept, rerun:  bash scripts/run_mixed_pool.sh point"
    exit "$rc" ;;
  e5)
    [ -s "$POINT/DONE" ] || { echo "[abort] point not complete; run:  bash scripts/run_mixed_pool.sh point"; exit 1; }
    exec env "${E5_ENV[@]}" bash scripts/run_e5.sh d0 ;;
  gate)
    [ -s "$POINT/DONE" ] || { echo "[abort] point not complete; run:  bash scripts/run_mixed_pool.sh point"; exit 1; }
    exec env "${E5_ENV[@]}" bash scripts/run_e5.sh gate d0 ;;
esac
