#!/usr/bin/env bash
# Experiment storage is read-only; saves a small TXT in home. Never starts training.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$#" -le 1 ] || { echo 'usage: bash scripts/check_mbpp_storage.sh [all|fresh|difficulty|long|quality]'; exit 2; }
export OM_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export EXPERIMENTS_MBPP_SUITE=${1:-all}
source scripts/_mbpp_experiments.sh
mbpp_queue_init
AUDIT_PY=${SWITCH_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
[ -x "$AUDIT_PY" ] || AUDIT_PY=python3
AUDIT_ROOTS=()
AUDIT_OPTIONS=()
[ "${MBPP_STORAGE_AUDIT_AUTOMATIC:-0}" != 1 ] || AUDIT_OPTIONS+=(--report-on-error --allow-branch-quarantine)
if [ "${MBPP_STORAGE_AUDIT_AUTOMATIC:-0}" = 1 ]; then
  for root in "${MBPP_ROOTS[@]}"; do AUDIT_ROOTS+=(--root "$root"); done
else
  while IFS= read -r root; do AUDIT_ROOTS+=(--root "$root"); done < <(mbpp_observation_roots)
fi
if [ "$EXPERIMENTS_MBPP_SUITE" = all ] || [ "$EXPERIMENTS_MBPP_SUITE" = quality ] || [ "$EXPERIMENTS_MBPP_SUITE" = difficulty ] || [ "$EXPERIMENTS_MBPP_SUITE" = long ]; then
  present=0
  for arg in "${AUDIT_ROOTS[@]}"; do [ "$arg" != "$SWITCH_MBPP_ROOT" ] || present=1; done
  [ "$present" = 1 ] || AUDIT_ROOTS+=(--root "$SWITCH_MBPP_ROOT")
fi
exec env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  "$AUDIT_PY" scripts/mbpp_storage_audit.py --work "$OM_WORK" "${AUDIT_ROOTS[@]}" "${AUDIT_OPTIONS[@]}"
