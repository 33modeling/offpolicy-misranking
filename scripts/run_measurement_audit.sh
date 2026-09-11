#!/usr/bin/env bash
# No environment setup, model load, GPU admission, or changes to source artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${OM_WORK:-$PWD/.work}
if [ -d "${GROUP_VOLUME:-/group-volume}" ]; then
  WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
fi
PY=${AUDIT_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
OUT="$WORK/exports/measurement-audit-$(date -u +%Y%m%dT%H%M%SZ)-$$"
exec "$PY" src/measurement_audit.py --out "$OUT" "$@"
