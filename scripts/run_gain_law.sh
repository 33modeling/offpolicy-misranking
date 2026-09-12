#!/usr/bin/env bash
# Reliability law against the completed matrix points (CPU, no GPU):
#
#   bash scripts/run_gain_law.sh
#
# For every completed point (both datasets, all seeds and drifts) the cross-half
# selection gain of the fresh validation-alignment score, of the difficulty
# score and, where src/stale_splithalf.py has run, of the reuse estimators is
# compared with the Gaussian prediction rho_h * c_{k,n}. Writes the table and
# the plot data under $OM_WORK/exports/gain-law-<UTC>.{txt,csv,dat} and runs
# the synthetic calibration alongside (gain_law_simulation.py).
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0 CUDA_VISIBLE_DEVICES=""
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
ROOT=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
mkdir -p "$OM_WORK/exports"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
prefix="$OM_WORK/exports/gain-law-$stamp"
{
  rc=0
  echo "# gain law export $stamp host=$(hostname) code=$(git rev-parse --short HEAD 2>/dev/null) root=$ROOT"
  "$PY" src/gain_vs_reliability.py --root "$ROOT" --frac "${OM_TOPK_FRAC:-0.1}" --out "$prefix" || { echo "[gain-law] matrix analysis failed"; rc=1; }
  echo; echo "### synthetic calibration (n=400, k=40)"
  "$PY" src/gain_law_simulation.py --n 400 --frac 0.1 --reps "${GAIN_LAW_REPS:-300}" --out "$prefix-synthetic" || rc=1
  exit "$rc"
} 2>&1 | tee "$prefix.txt"
statuses=("${PIPESTATUS[@]}")
echo "[gain-law] export written: $prefix.txt (plus .csv, .dat, -synthetic.dat)"
for status in "${statuses[@]}"; do [ "$status" -eq 0 ] || exit "$status"; done
