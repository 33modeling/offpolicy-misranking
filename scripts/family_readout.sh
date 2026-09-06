#!/usr/bin/env bash
# Provisional analysis of ONE finished OLMo-3 family (its four points), without
# waiting for the other nine. Prints the regime report to the terminal and
# writes it under $OM_WORK/readouts/. Nothing here touches running workers.
#   bash scripts/family_readout.sh h100 mbpp 0
#   bash scripts/family_readout.sh h100 mbpp 0 10000   # final-quality bootstrap (slow)
set -uo pipefail
cd "$(dirname "$0")/.."
PROFILE=${1:-h100}; DATASET=${2:?dataset (math500|mbpp)}; SEED=${3:?seed}; BOOT=${4:-1000}
case "$PROFILE" in
  baseline) MODEL_TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-v1} ;;
  h100)     MODEL_TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2} ;;
  *) echo "[abort] profile must be baseline or h100"; exit 2 ;;
esac
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
ROOT="${OM_OLMO3_ROOT:-$OM_WORK/runs/$MODEL_TAG}"
FAMILY="$ROOT/family-$DATASET-s$SEED"
runs=(); missing=()
for d in 0 25 100 400; do
  run="$FAMILY/$MODEL_TAG-s$SEED-$DATASET-d$d"
  if [ -s "$run/DONE" ] && [ -s "$run/report.json" ]; then runs+=("$run"); else missing+=("d$d"); fi
done
echo "family   $DATASET/s$SEED   points done: ${#runs[@]}/4${missing:+   missing: ${missing[*]}}"
[ "${#runs[@]}" -gt 0 ] || { echo "DECISION nothing to analyse yet"; exit 1; }
OUT="$OM_WORK/readouts/family-$DATASET-s$SEED-$(git rev-parse --short HEAD)-boot$BOOT"
mkdir -p "$OUT"
echo "output   $OUT"
echo "bootstrap $BOOT replicates (final requires 10000; smaller = provisional)"
echo
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" src/regime_map.py "${runs[@]}" \
  --output-dir "$OUT" --first-bootstrap "$BOOT" 2>&1 | tee "$OUT/readout.log"
rc=${PIPESTATUS[0]}
echo
if [ "$rc" -eq 0 ]; then
  echo "DECISION report written: $OUT/FINAL_REPORT.md  (also REGIME.csv, REGIME_SUMMARY.csv, REGIME.json)"
  echo "handover: paste the table above, or copy $OUT off the cluster (scp) - it is text only, a few hundred KB"
else
  echo "DECISION analysis failed rc=$rc; the error is above and in $OUT/readout.log"
fi
exit "$rc"
