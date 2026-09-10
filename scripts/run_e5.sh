#!/usr/bin/env bash
# Reduced E5 (2026-09-10): does the selected data actually train better?
#   MATH-500, checkpoint d400, seeds 0 1 2, arms random / fresh_r / g11,
#   100 further GRPO updates per arm, evaluated on 300 held-out MATH-train
#   problems x 8 responses (never the ranking validation prompts).
#
#   bash scripts/run_e5.sh          # run on THIS idle 4xH100 node (no OLMo/Qwen launcher here)
#   bash scripts/run_e5.sh status   # progress of every seed and arm, no GPU
#   bash scripts/run_e5.sh plan     # dry run: contracts and commands only
#
# Several idle nodes may run the same command: arms are leased per seed, a
# busy arm is skipped, and a node moves on to the next seed. Rerunning after a
# kill resumes from the newest checkpoint or completed evaluation shard.
# Prerequisite once, in an online shell:  bash scripts/fetch_math_train.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
case "$MODE" in run|status|plan) ;; *) echo "usage: bash scripts/run_e5.sh [run|status|plan]"; exit 2 ;; esac
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
ROOT=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
DATASET=${E5_DATASET:-math500}; DRIFT=${E5_DRIFT:-400}
read -r -a SEEDS <<< "${E5_SEEDS:-0 1 2}"
STEPS=${E5_STEPS:-100}; EVAL_K=${E5_EVAL_K:-8}; COUNT=${E5_TEST_COUNT:-300}
SELECTORS=${E5_SELECTORS:-random fresh_r g11}
POOL="$DATASETS_DIR/math_train/math_train.jsonl"; POOL_MANIFEST="$DATASETS_DIR/math_train/dataset_manifest.json"
TEST="$OM_WORK/inputs/e5-reduced/test-$DATASET-d$DRIFT.json"
OUT_ROOT="$OM_WORK/runs/e5-reduced/$DATASET-d$DRIFT"
run_dir() { printf '%s/family-%s-s%s/%s-s%s-%s-d%s\n' "$ROOT" "$DATASET" "$1" "$TAG" "$1" "$DATASET" "$DRIFT"; }

echo "[e5] $DATASET d$DRIFT seeds=${SEEDS[*]} arms=$SELECTORS steps=$STEPS eval_k=$EVAL_K test=$COUNT  out=$OUT_ROOT"
if [ "$MODE" = status ]; then
  for seed in "${SEEDS[@]}"; do
    out="$OUT_ROOT/s$seed"
    if [ -s "$out/experiment.json" ]; then
      "$PY" src/evidence_downstream.py status --out "$out"
      [ -s "$out/downstream_results.csv" ] && { echo "  results:"; sed 's/^/    /' "$out/downstream_results.csv"; }
    else
      echo "seed $seed: not prepared"
    fi
  done
  exit 0
fi

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
  DOWNSTREAM_SELECTORS="$SELECTORS" bash scripts/run_downstream_independent.sh "$run" "$out" \
    --eval-prompts "$TEST" --steps "$STEPS" --eval-k "$EVAL_K"
  rc=$?
  if [ "$rc" -eq 75 ]; then echo "[abort] this node's GPUs belong to another experiment (OLMo or Qwen launcher); use an idle node"; exit 75; fi
  [ "$rc" -eq 0 ] || rc_all=1
done
echo "[e5] pass complete; check:  bash scripts/run_e5.sh status"
exit "$rc_all"
