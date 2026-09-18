#!/usr/bin/env bash
# Inspect every discovered experiment root; no training, cleanup, or reset.
set -euo pipefail
cd "$(dirname "$0")/.."
AUDIT_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
AUDIT_PY=${SWITCH_PYTHON:-${VENV_DIR:-$AUDIT_WORK/.venv-cu126}/bin/python}
[ -x "$AUDIT_PY" ] || AUDIT_PY=python3
exec env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  "$AUDIT_PY" scripts/random_storage_audit.py --work "$AUDIT_WORK"
