#!/usr/bin/env bash
# Signal-resolution control on the completed d=0 points: is a coarser ranking
# (pass rate / learnability / hardest-first) reproducible between the same A/B
# response halves where the gradient-alignment ranking is not? CPU only,
# read-only for the experiment; writes one text report under $OM_WORK/exports
# and prints it (the KEY table is phone-readable).
#
#   bash scripts/reference_controls.sh                 # h100: every d0 point with stored responses
#   bash scripts/reference_controls.sh math500 0       # one family's d0 point
#   bash scripts/reference_controls.sh baseline        # the older baseline root
#   bash scripts/reference_controls.sh /abs/point ...  # explicit point directories
#   RC_REPS=40 RC_PAIRS=20 ...                         # resample / tie-stream counts
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="${VENV_DIR:-}/bin/python"; [ -x "$PY" ] || PY=python3

PROFILE=h100
case "${1:-}" in baseline|h100) PROFILE=$1; shift ;; esac
case "$PROFILE" in
  baseline) TAG=olmo3-1025-7b-base-rlzero-grpo-v1 ;;
  h100)     TAG=olmo3-1025-7b-base-rlzero-grpo-h100-v2 ;;
esac
TAG="${OM_OLMO3_MODEL_TAG:-$TAG}"
ROOT="${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}"

points=(); labels=()
if [ "$#" -ge 1 ] && [ -d "$1" ]; then
  for p in "$@"; do points+=("$p"); labels+=("$(basename "$p")"); done
elif [ "$#" -ge 2 ]; then
  dataset=$1; seed=$2
  p="$ROOT/family-$dataset-s$seed/$TAG-s$seed-$dataset-d0"
  [ -d "$p" ] || { echo "[abort] no such point: $p"; exit 1; }
  points+=("$p"); labels+=("$dataset/s$seed")
else
  [ -d "$ROOT" ] || { echo "[abort] no experiment root: $ROOT"; exit 1; }
  for dir in "$ROOT"/family-*; do
    [ -d "$dir" ] || continue
    name=${dir##*/family-}; dataset=${name%-s*}; seed=${name##*-s}
    p="$dir/$TAG-s$seed-$dataset-d0"
    [ -f "$p/rollouts_fresh_train.jsonl" ] || { echo "[skip] $dataset/s$seed: no stored d0 responses yet"; continue; }
    points+=("$p"); labels+=("$dataset/s$seed")
  done
fi
[ "${#points[@]}" -gt 0 ] || { echo "[abort] no d0 point with rollouts_fresh_train.jsonl under $ROOT"; exit 1; }

EXPORTS="$OM_WORK/exports"; mkdir -p "$EXPORTS" || { echo "[abort] cannot create $EXPORTS"; exit 1; }
OUT="$EXPORTS/reference-controls-$TAG-$(date -u +%Y%m%dT%H%M%SZ).txt"
label_args=(); for l in "${labels[@]}"; do label_args+=(--label "$l"); done
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" src/reference_controls.py "${points[@]}" "${label_args[@]}" \
  --out "$OUT" --reps "${RC_REPS:-40}" --pairs "${RC_PAIRS:-20}"
rc=$?
[ "$rc" -eq 0 ] && echo "report   $OUT   (upload this file; the KEY lines above are enough to read it)"
exit "$rc"
