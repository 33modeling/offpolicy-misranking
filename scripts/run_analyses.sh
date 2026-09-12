#!/usr/bin/env bash
# All CPU analyses in one command, then the export bundle:
#
#   bash scripts/run_analyses.sh
#
# Runs the offline gate decisions, the gain-law comparison (matrix points plus
# the synthetic calibration), the cost accounting, and finally
# `run_e5.sh export`, which bundles every finished result file into one text
# file under $OM_WORK/exports. No GPU; safe to run while GPU jobs are running.
set -uo pipefail
cd "$(dirname "$0")/.."
rc=0
for step in run_gate_decision run_gain_law run_cost_accounting; do
  echo; echo "===== $step"
  bash "scripts/$step.sh" || { echo "[analyses] $step failed (continuing)"; rc=1; }
done
echo; echo "===== export"
bash scripts/run_e5.sh export | tail -n 3 || rc=1
echo; echo "[analyses] exports are under: ${OM_WORK:-\$OM_WORK}/exports"
exit "$rc"
