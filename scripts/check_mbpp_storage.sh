#!/usr/bin/env bash
# Separate, read-only storage/deletion-evidence audit. Never starts training.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$#" -le 1 ] || { echo 'usage: bash scripts/check_mbpp_storage.sh [all|fresh|quality|difficulty]'; exit 2; }
export OM_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export EXPERIMENTS_MBPP_SUITE=${1:-all}
source scripts/_mbpp_experiments.sh
mbpp_queue_init
AUDIT_PY=${SWITCH_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
[ -x "$AUDIT_PY" ] || AUDIT_PY=python3
AUDIT_ROOTS=()
for root in "${MBPP_ROOTS[@]}"; do AUDIT_ROOTS+=(--root "$root"); done
if [ "$EXPERIMENTS_MBPP_SUITE" = quality ] || [ "$EXPERIMENTS_MBPP_SUITE" = difficulty ]; then
  AUDIT_ROOTS+=(--root "$SWITCH_MBPP_ROOT")
fi
exec env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  "$AUDIT_PY" scripts/mbpp_storage_audit.py --work "$OM_WORK" "${AUDIT_ROOTS[@]}"
