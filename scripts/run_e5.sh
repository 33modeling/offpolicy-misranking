#!/usr/bin/env bash
# Reduced E5 (2026-09-10): does the selected data actually train better?
#   MATH-500, checkpoint d400, seeds 0 1 2, arms random / fresh_r / g11,
#   100 further GRPO updates per arm, evaluated on 300 held-out MATH-train
#   problems x 8 responses (never the ranking validation prompts).
#
#   bash scripts/run_e5.sh          # run the d400 branch on THIS idle 4xH100 node
#   bash scripts/run_e5.sh d0       # run the d0 branch (arms start from the base model)
#   bash scripts/run_e5.sh status   # progress of every branch, seed and arm, no GPU
#   Any mode accepts d0 or d400 as an extra word, e.g.  bash scripts/run_e5.sh d0 stop
#   bash scripts/run_e5.sh plan     # dry run: contracts and commands only
#   bash scripts/run_e5.sh stop     # stop E5 on this node (nothing else)
#   bash scripts/run_e5.sh force    # run, first stopping a non-matrix process that holds this node's GPU lock
#
# Several idle nodes may run the same command: arms are leased per seed, a
# busy arm is skipped, and a node moves on to the next seed. Rerunning after a
# kill resumes from the newest checkpoint or completed evaluation shard.
# Prerequisite once, in an online shell:  bash scripts/fetch_math_train.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=run; DRIFT=${E5_DRIFT:-400}; DRIFT_GIVEN=${E5_DRIFT:+1}
for arg in "$@"; do
  case "$arg" in
    d0) DRIFT=0; DRIFT_GIVEN=1 ;;
    d400) DRIFT=400; DRIFT_GIVEN=1 ;;
    run|status|plan|stop) MODE=$arg ;;
    force) MODE=run; export E5_FORCE=1 ;;
    *) echo "usage: bash scripts/run_e5.sh [run|status|plan|stop|force] [d0|d400]"; exit 2 ;;
  esac
done
trap '' HUP
trap 'echo "[e5] interrupted; nothing else will be started"; exit 130' INT TERM
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
ROOT=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
DATASET=${E5_DATASET:-math500}
read -r -a SEEDS <<< "${E5_SEEDS:-0 1 2}"
STEPS=${E5_STEPS:-100}; EVAL_K=${E5_EVAL_K:-8}; COUNT=${E5_TEST_COUNT:-300}
SELECTORS=${E5_SELECTORS:-random passrate_beta fresh_r g11}
POOL="$DATASETS_DIR/math_train/math_train.jsonl"; POOL_MANIFEST="$DATASETS_DIR/math_train/dataset_manifest.json"
TEST="$OM_WORK/inputs/e5-reduced/test-$DATASET-d$DRIFT.json"
OUT_ROOT="$OM_WORK/runs/e5-reduced/$DATASET-d$DRIFT"
# Exported marker: every process of this pass carries OUT_ROOT in its environment,
# so a later launch on the same node can find and stop the whole earlier pass
# (including this loop), while the matrix launchers never match.
export OUT_ROOT
run_dir() { printf '%s/family-%s-s%s/%s-s%s-%s-d%s\n' "$ROOT" "$DATASET" "$1" "$TAG" "$1" "$DATASET" "$DRIFT"; }

echo "[e5] $DATASET d$DRIFT seeds=${SEEDS[*]} arms=$SELECTORS steps=$STEPS eval_k=$EVAL_K test=$COUNT  out=$OUT_ROOT"
if [ "$MODE" = stop ]; then
  source scripts/_e5_node.sh || exit 1
  e5_cleanup_previous "$OUT_ROOT" || exit 1
  echo "[e5] previous E5 processes stopped on $(hostname); checkpoints retained"
  exit 0
fi
if [ "$MODE" = status ]; then
  # Without d0/d400 the status covers every branch that exists.
  if [ -n "${DRIFT_GIVEN:-}" ]; then roots=("$OUT_ROOT"); else mapfile -t roots < <(ls -d "$OM_WORK/runs/e5-reduced/$DATASET-d"* 2>/dev/null); fi
  [ "${#roots[@]}" -gt 0 ] || { echo "no E5 branch prepared yet"; exit 0; }
  for root in "${roots[@]}"; do
    echo "== branch $(basename "$root")"
    for seed in "${SEEDS[@]}"; do
      out="$root/s$seed"
      if [ -s "$out/experiment.json" ]; then
        "$PY" src/downstream_status.py --out "$out"
        # arms added after preparation live in arms.json; show their state too
        [ -s "$out/arms.json" ] && "$PY" src/evidence_downstream.py status --out "$out" | grep -E "^  (passrate_beta|g00|g10|g01|g11|fresh_r|random) " | grep -vFf <("$PY" -c 'import json,sys; print("\n".join("  "+a+" " for a in json.load(open(sys.argv[1]))["selectors"]))' "$out/experiment.json") | sed 's/^/  [added]/'
        [ -s "$out/downstream_results.csv" ] && { echo "  results:"; sed 's/^/    /' "$out/downstream_results.csv"; }
      else
        echo "seed $seed: not prepared"
      fi
    done
  done
  exit 0
fi

# 0. an earlier E5 launch on this node (dropped session) is stopped and resumed by the launcher itself
# 1. source points must be complete
runs=()
for seed in "${SEEDS[@]}"; do
  run=$(run_dir "$seed")
  [ -s "$run/DONE" ] || { echo "[abort] source point is not complete: $run"; exit 1; }
  runs+=("$run")
done
# 2. held-out pool (fetched once in an online shell)
[ -s "$POOL" ] && [ -s "$POOL_MANIFEST" ] || {
  echo "[abort] held-out MATH-train pool missing: $POOL"
  echo "        run once in an online shell:  bash scripts/fetch_math_train.sh"
  exit 1
}
REVISION=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_revision"])' "$POOL_MANIFEST")
# 3. freeze the independent test set (idempotent; shared by all seeds and nodes)
"$PY" src/evidence_downstream.py prepare-test --candidates "$POOL" --runs "${runs[@]}" --out "$TEST" \
  --count "$COUNT" --dataset EleutherAI/hendrycks_math --revision "$REVISION" --split train || exit 1
if [ "$MODE" = run ]; then
  source scripts/_e5_node.sh || exit 1
  e5_cleanup_previous "$OUT_ROOT" || exit 1
  # Own this node for the entire seed pass, not separately for every child.
  if [ "${OM_NODE_LOCK_HELD:-0}" != 1 ]; then e5_acquire_node || exit "$?"; fi
fi
# 4. one seed after another; arms are leased inside the launcher
rc_all=0
for seed in "${SEEDS[@]}"; do
  run=$(run_dir "$seed"); out="$OUT_ROOT/s$seed"
  echo "== seed $seed: $run"
  if [ "$MODE" = plan ]; then
    DOWNSTREAM_SELECTORS="$SELECTORS" bash scripts/run_downstream_independent.sh "$run" "$out" \
      --eval-prompts "$TEST" --steps "$STEPS" --eval-k "$EVAL_K" --dry-run | head -20 || rc_all=1
    continue
  fi
  OM_NODE_LOCK_HELD=1 OM_E5_CONTROLLER_PID="$$" DOWNSTREAM_SELECTORS="$SELECTORS" \
    bash scripts/run_downstream_independent.sh "$run" "$out" \
    --eval-prompts "$TEST" --steps "$STEPS" --eval-k "$EVAL_K" 7>&- 8>&-
  rc=$?
  if [ "$rc" -eq 75 ]; then echo "[abort] E5 admission failed; see the actual owner or resource error above"; exit 75; fi
  if [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then echo "[e5] stopped by signal; nothing else will be started"; exit "$rc"; fi
  [ "$rc" -eq 0 ] || rc_all=1
done
echo "[e5] pass complete; check:  bash scripts/run_e5.sh status"
exit "$rc_all"
