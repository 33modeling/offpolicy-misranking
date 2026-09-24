#!/usr/bin/env bash
# CPU-only export of recorded Pair endpoint and first-target costs.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
PY=${PAIR_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
exec "$PY" scripts/selector_pair_adaptive_costs.py "$@"
