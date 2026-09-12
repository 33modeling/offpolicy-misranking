#!/usr/bin/env bash
# Compute-cost accounting from what is already on disk (CPU, no GPU):
#
#   bash scripts/run_cost_accounting.sh
#
# E5 arms (training, pilot, evaluation, benchmark GPU-seconds), the
# reliability-logging overhead (rlog random arm against the benchmark random
# arm), and the matrix points' stage times with per-prompt scoring costs.
# Writes $OM_WORK/exports/cost-accounting-<UTC>.{txt,json,csv}.
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0 CUDA_VISIBLE_DEVICES=""
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
ROOT=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
mkdir -p "$OM_WORK/exports"
prefix="$OM_WORK/exports/cost-accounting-$(date -u +%Y%m%dT%H%M%SZ)"
{
  echo "# cost accounting $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) code=$(git rev-parse --short HEAD 2>/dev/null)"
  "$PY" src/cost_accounting.py --work "$OM_WORK" --matrix "$ROOT" --out "$prefix"
} 2>&1 | tee "$prefix.txt"
statuses=("${PIPESTATUS[@]}")
echo "[cost] export written: $prefix.txt (plus .json, .csv)"
for status in "${statuses[@]}"; do [ "$status" -eq 0 ] || exit "$status"; done
