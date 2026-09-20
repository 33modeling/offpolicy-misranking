#!/usr/bin/env bash
# One read-only TXT containing MBPP and Pair diagnostics; no total upload cap.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
PY=${SWITCH_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
exec env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  "$PY" scripts/mbpp_pair_diagnostic.py "$@"
