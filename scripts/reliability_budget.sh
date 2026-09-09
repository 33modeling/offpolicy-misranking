#!/usr/bin/env bash
# Size the reference budget from the stored d=0 artifacts (CPU only, read-only).
#
#   bash scripts/reliability_budget.sh                 # d0 of every family with stored micro-groups (both datasets)
#   bash scripts/reliability_budget.sh math500         # one dataset
#   bash scripts/reliability_budget.sh math500 25      # another drift point (descriptive only)
#   bash scripts/reliability_budget.sh baseline        # the v1 root instead of h100 v2
#
# Reads oracle_micro_groups.pt and val_groups.pt of each point (root copy, or
# the copy parked under pinned-scoring/<stamp>/ while a rescored point waits for
# its GPU re-evaluation), re-draws the split-half reference at every observable
# half size, and prints how many responses per prompt the gate floor >= 2k/n
# would need (src/reliability_budget.py). Nothing under the experiment root is
# written; the report goes to $OM_WORK/exports and to the terminal.
#
# This sizes the separate reference-budget run (scripts/run_reliability_budget.sh).
# It reuses the locked rollouts descriptively and defines no registered label.
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
PROFILE=h100
case "${1:-}" in baseline|h100) PROFILE=$1; shift ;; esac
case "$PROFILE" in
  baseline) TAG=olmo3-1025-7b-base-rlzero-grpo-v1 ;;
  h100)     TAG=olmo3-1025-7b-base-rlzero-grpo-h100-v2 ;;
esac
TAG="${OM_OLMO3_MODEL_TAG:-$TAG}"
ROOT="${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}"
DATASET_FILTER="${1:-}"
DRIFT="${2:-0}"
case "$DRIFT" in ''|*[!0-9]*) echo "usage: bash scripts/reliability_budget.sh [h100|baseline] [<dataset>] [<drift>]"; exit 2 ;; esac
[ -d "$ROOT" ] || { echo "[abort] no experiment root: $ROOT"; exit 1; }

runs=(); labels=()
for dir in "$ROOT"/family-*; do
  [ -d "$dir" ] || continue
  name=${dir##*/family-}; dataset=${name%-s*}; seed=${name##*-s}
  [ -z "$DATASET_FILTER" ] || [ "$dataset" = "$DATASET_FILTER" ] || continue
  point="$dir/$TAG-s$seed-$dataset-d$DRIFT"
  if [ -s "$point/oracle_micro_groups.pt" ] && [ -s "$point/val_groups.pt" ]; then
    runs+=("$point"); labels+=("$dataset/s$seed d$DRIFT")
  elif compgen -G "$point/pinned-scoring/*/oracle_micro_groups.pt" >/dev/null; then
    runs+=("$point"); labels+=("$dataset/s$seed d$DRIFT")
  fi
done
if [ "${#runs[@]}" -eq 0 ]; then
  echo "[abort] no point with stored micro-groups under $ROOT (dataset='${DATASET_FILTER:-any}', d$DRIFT)"
  exit 1
fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EXPORTS="$OM_WORK/exports"; mkdir -p "$EXPORTS" || { echo "[abort] cannot create $EXPORTS"; exit 1; }
OUT="$EXPORTS/reliability-budget-$TAG-${DATASET_FILTER:-all}-d$DRIFT-$STAMP.txt"
echo "[reliability-budget] ${#runs[@]} point(s) under $ROOT -> $OUT"
args=()
for label in "${labels[@]}"; do args+=(--label "$label"); done
"$PY" src/reliability_budget.py "${runs[@]}" "${args[@]}" --out "$OUT" \
  --reps "${RB_REPS:-40}" --pairs "${RB_PAIRS:-20}" --target "${RB_TARGET:-0.20}" || exit 1
echo
echo "[reliability-budget] report: $OUT"
echo "[reliability-budget] KEY lines:"
grep "^KEY " "$OUT"
