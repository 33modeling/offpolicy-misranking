#!/usr/bin/env bash
# New experiment (2026-09-23): SR-GC decisions frozen before training on unseen seeds.
#   Seeds 3 and 4 at MATH steps 0/100/400; arms random, passrate_beta (cached SR),
#   fresh_r (on-policy); 100 further GRPO updates; 300 held-out questions x 8.
#   SR-GC D is computed from the parent point's saved gradients and the exact
#   subsets this experiment trains, and frozen BEFORE any training starts.
#
#   bash scripts/run_srgc_newseeds.sh           # prepare + freeze (CPU), then train/evaluate on this idle 4xH100 node
#   bash scripts/run_srgc_newseeds.sh freeze    # prepare + freeze only, no GPU work
#   bash scripts/run_srgc_newseeds.sh status    # decisions and per-arm progress, no GPU
#   bash scripts/run_srgc_newseeds.sh results   # one TXT: ~/srgc-newseeds-results.txt
#
# Writes only under $OM_WORK/runs/srgc-newseeds-v1. Existing E5 roots, the
# frozen E5 question files and all code are read, never written. Several idle
# nodes may run the same command; arms are leased and a busy node is left alone.
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
case "$MODE" in
  run|freeze|status|results) ;;
  *) echo "usage: bash scripts/run_srgc_newseeds.sh [run|freeze|status|results]"; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo "[abort] no additional options; settings are fixed in this script"; exit 2; }
trap '' HUP
trap 'echo "[srgc-newseeds] interrupted; nothing else will be started"; exit 130' INT TERM
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv missing: $PY"; exit 1; }
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
MATRIX=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
ROOT=${SRGC_NEWSEEDS_ROOT:-$OM_WORK/runs/srgc-newseeds-v1}
SEEDS=(3 4)
DRIFTS=(0 100 400)
SELECTORS="random passrate_beta fresh_r"
STEPS=100; EVAL_K=8
run_dir() { printf '%s/family-math500-s%s/%s-s%s-math500-d%s\n' "$MATRIX" "$1" "$TAG" "$1" "$2"; }
out_dir() { printf '%s/math500-d%s/s%s\n' "$ROOT" "$2" "$1"; }
test_file() { printf '%s/inputs/test-math500-d%s.json\n' "$ROOT" "$1"; }
e5_test() { printf '%s/inputs/e5-reduced/test-math500-d%s.json\n' "$OM_WORK" "$1"; }
decision_file() { printf '%s/decisions/s%s-d%s.json\n' "$ROOT" "$1" "$2"; }

if [ "$MODE" = results ]; then
  for drift in "${DRIFTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out=$(out_dir "$seed" "$drift")
      [ -s "$out/experiment.json" ] && [ -d "$out/before/evaluation" ] && \
        "$PY" src/evidence_downstream.py summarize --out "$out" --allow-partial >/dev/null 2>&1 || true
    done
  done
  exec "$PY" scripts/srgc_newseeds.py results --root "$ROOT" --out "$HOME/srgc-newseeds-results.txt"
fi

if [ "$MODE" = status ]; then
  echo "[srgc-newseeds] root=$ROOT seeds=${SEEDS[*]} steps=${DRIFTS[*]} arms=$SELECTORS"
  for drift in "${DRIFTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out=$(out_dir "$seed" "$drift"); decision=$(decision_file "$seed" "$drift")
      echo "== s$seed d$drift"
      if [ -s "$decision" ]; then
        "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(f"  SR-GC frozen: D={d[\"d\"]:+.6g} selector={d[\"selector\"]} prospective={d[\"prospective\"]} at {d[\"frozen_at_utc\"]}")' "$decision"
      else
        echo "  SR-GC: not frozen yet"
      fi
      if [ -s "$out/experiment.json" ]; then "$PY" src/downstream_status.py --out "$out"; else echo "  not prepared"; fi
      [ -s "$out/downstream_results.csv" ] && echo "  RESULT ready: $out/downstream_results.csv"
    done
  done
  exit 0
fi

# 1. Every source point must be complete (read only).
for drift in "${DRIFTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run=$(run_dir "$seed" "$drift")
    [ -s "$run/DONE" ] || { echo "[abort] source point is not complete: $run"; exit 1; }
  done
done
# 2. Own copy of the frozen E5 question set per step (the E5 file is only read).
for drift in "${DRIFTS[@]}"; do
  "$PY" scripts/srgc_newseeds.py copy-test --source "$(e5_test "$drift")" --dest "$(test_file "$drift")" >/dev/null || exit 1
done
# 3. Prepare subsets and 4. freeze SR-GC for every state before any training.
for drift in "${DRIFTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run=$(run_dir "$seed" "$drift"); out=$(out_dir "$seed" "$drift")
    DOWNSTREAM_SELECTORS="$SELECTORS" bash scripts/run_downstream_independent.sh "$run" "$out" \
      --eval-prompts "$(test_file "$drift")" --steps "$STEPS" --eval-k "$EVAL_K" --prepare-only >/dev/null || exit 1
    "$PY" scripts/srgc_newseeds.py freeze --run "$run" --out "$out" --decision "$(decision_file "$seed" "$drift")" || exit 1
  done
done
echo "[srgc-newseeds] all ${#SEEDS[@]}x${#DRIFTS[@]} SR-GC decisions frozen under $ROOT/decisions"
[ "$MODE" = run ] || exit 0

# 5. Train and evaluate (GPU). Node admission never stops another experiment.
export OUT_ROOT="$ROOT" E5_FORCE=0
source scripts/_e5_node.sh || exit 1
e5_acquire_node || exit "$?"
states=()
for drift in "${DRIFTS[@]}"; do for seed in "${SEEDS[@]}"; do states+=("$seed:$drift"); done; done
# Different nodes start at different states; arms are leased, so nodes do not overlap.
offset=$(( $(hostname | cksum | cut -d' ' -f1) % ${#states[@]} ))
states=("${states[@]:offset}" "${states[@]:0:offset}")
rc_all=0
for state in "${states[@]}"; do
  seed=${state%%:*}; drift=${state##*:}
  run=$(run_dir "$seed" "$drift"); out=$(out_dir "$seed" "$drift")
  echo "== s$seed d$drift: $out"
  OM_NODE_LOCK_HELD=1 OM_E5_CONTROLLER_PID="$$" DOWNSTREAM_SELECTORS="$SELECTORS" \
    bash scripts/run_downstream_independent.sh "$run" "$out" \
    --eval-prompts "$(test_file "$drift")" --steps "$STEPS" --eval-k "$EVAL_K" 7>&- 8>&-
  rc=$?
  if [ "$rc" -eq 75 ]; then echo "[abort] GPU admission failed; existing jobs untouched"; exit 75; fi
  if [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then echo "[srgc-newseeds] stopped by signal"; exit "$rc"; fi
  [ "$rc" -eq 0 ] || rc_all=1
done
echo "[srgc-newseeds] pass complete; results: bash scripts/run_srgc_newseeds.sh results"
exit "$rc_all"
