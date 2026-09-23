#!/usr/bin/env bash
# Read saved projections only; no GPU reservation or measurement.
set -euo pipefail
cd "$(dirname "$0")/.."
export OM_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
PAIR_ROOT=${PAIR_ROOT:-$OM_WORK/runs/selector-pair-v1}
PY=${PAIR_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
"$PY" scripts/export_selector_pair_srgc_inventory.py --root "$PAIR_ROOT" \
  --out "${1:-$HOME/srgc-t25-inventory.txt}"
