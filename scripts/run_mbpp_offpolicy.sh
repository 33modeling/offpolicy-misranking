#!/usr/bin/env bash
# Figure 2 follow-up: reuse MBPP policies/responses; never train or generate.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
case "$MODE" in
  run|status|results) ;;
  -h|--help)
    echo 'usage: bash scripts/run_mbpp_offpolicy.sh [run|status|results]'
    exit 0 ;;
  *) echo '[mbpp-offpolicy] expected run, status, or results' >&2; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo '[mbpp-offpolicy] no additional options required' >&2; exit 2; }
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY=${VENV_DIR}/bin/python
[ -x "$PY" ] || PY=python3
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
ROOT=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
export PYTHONDONTWRITEBYTECODE=1
if [ "$MODE" != run ]; then
  exec env CUDA_VISIBLE_DEVICES='' "$PY" src/mbpp_offpolicy_followup.py "$MODE" --root "$ROOT" --tag "$TAG"
fi
env CUDA_VISIBLE_DEVICES='' "$PY" src/mbpp_offpolicy_followup.py prepare --root "$ROOT" --tag "$TAG"
if env CUDA_VISIBLE_DEVICES='' "$PY" src/mbpp_offpolicy_followup.py status --root "$ROOT" --tag "$TAG"; then
  exec env CUDA_VISIBLE_DEVICES='' "$PY" src/mbpp_offpolicy_followup.py results --root "$ROOT" --tag "$TAG"
fi
# Fixed scope matches the MATH off-policy calibration: 3 seeds x 2 checkpoints.
export E5_SEEDS='0 1 2' STALE_CHECK_FULL=4
unset OM_ATTN OM_LORA_TARGETS OM_PROMPT_FORMAT
rc=0
for drift in d0 d400; do
  bash scripts/run_stale_splithalf.sh mbpp "$drift" || { rc=$?; break; }
done
env CUDA_VISIBLE_DEVICES='' "$PY" src/mbpp_offpolicy_followup.py results --root "$ROOT" --tag "$TAG" || rc=$?
exit "$rc"
