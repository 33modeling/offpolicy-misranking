#!/usr/bin/env bash
# Offline random-fallback gate on the completed E5 points (CPU, no GPU):
#
#   bash scripts/run_gate_decision.sh            # both branches (d0, d400), seeds 0 1 2
#   bash scripts/run_gate_decision.sh d0         # one branch
#
# Applies the frozen rule config/gate_rule.json (pilot size, threshold,
# confidence, seed; committed before the d0 rewards were examined) to the
# stored half scores of every source point: the difficulty score -|p-1/2| on
# the two halves of the behavior responses, the fresh a/b validation-alignment
# scores, and the reuse estimators' halves when scores_stale_splithalf.json
# exists (bash scripts/run_stale_splithalf.sh). Where the seed's
# downstream_results.csv exists, the decision is mapped to the fixed arms'
# test rewards (forgone reward = reward of the selector minus reward of the
# chosen branch). Writes gate_decision.{json,csv} into each E5 seed directory
# and one bundle under $OM_WORK/exports/gate-decision-<UTC>.txt.
set -uo pipefail
cd "$(dirname "$0")/.."
BRANCHES="0 400"
for arg in "$@"; do
  case "$arg" in
    d0) BRANCHES=0 ;; d400) BRANCHES=400 ;;
    *) echo "usage: bash scripts/run_gate_decision.sh [d0|d400]"; exit 2 ;;
  esac
done
export OM_ONLINE=0 CUDA_VISIBLE_DEVICES=""
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
ROOT=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
RULE=${E5_GATE_RULE:-config/gate_rule.json}
[ -s "$RULE" ] || { echo "[abort] frozen rule missing: $RULE"; exit 1; }
read -r -a SEEDS <<< "${E5_SEEDS:-0 1 2}"
mkdir -p "$OM_WORK/exports"
target="$OM_WORK/exports/gate-decision-$(date -u +%Y%m%dT%H%M%SZ).txt"
rc=0
{
  echo "# gate decision export $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) code=$(git rev-parse --short HEAD 2>/dev/null)"
  echo "# rule: $RULE"; cat "$RULE"
  for drift in $BRANCHES; do
    for seed in "${SEEDS[@]}"; do
      run="$ROOT/family-math500-s$seed/$TAG-s$seed-math500-d$drift"
      e5="$OM_WORK/runs/e5-reduced/math500-d$drift/s$seed"
      echo; echo "### d$drift seed $seed"
      [ -s "$run/DONE" ] || { echo "source point not complete: $run"; continue; }
      if [ -d "$e5" ]; then
        "$PY" src/gate_decision.py decide --run "$run" --rule "$RULE" --e5 "$e5" || rc=1
      else
        "$PY" src/gate_decision.py decide --run "$run" --rule "$RULE" --out "$OM_WORK/runs/e5-reduced/gate-decision-d$drift-s$seed.json" || rc=1
      fi
    done
  done
  echo; echo "### pilot size table"
  "$PY" src/gate_decision.py table --r-min "$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["r_min"])' "$RULE")" \
    --confidence "$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["confidence"])' "$RULE")" || rc=1
  exit "$rc"
} 2>&1 | tee "$target"
statuses=("${PIPESTATUS[@]}")
echo "[gate] export written: $target"
for status in "${statuses[@]}"; do [ "$status" -eq 0 ] || exit "$status"; done
exit 0
