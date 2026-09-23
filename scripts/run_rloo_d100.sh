#!/usr/bin/env bash
# New experiment (2026-09-24): RLOO objective control at MATH d=100.
#   Same design as the d0/d400 RLOO control (seeds 0-2; random, cached success
#   rate, on-policy alignment; 100 RLOO updates from the GRPO d=100 parent;
#   300 held-out questions x 8), written to its own root.
#
#   bash scripts/run_rloo_d100.sh            # prepare (CPU) then train/evaluate on this idle 4-GPU node
#   bash scripts/run_rloo_d100.sh prepare    # prepare + subset check only, no GPU
#   bash scripts/run_rloo_d100.sh status     # per-arm progress, no GPU
#   bash scripts/run_rloo_d100.sh results    # one TXT: ~/rloo-d100-results.txt
#
# Several idle nodes may run the same command; arms are leased and a busy node
# is left alone. The original root (rloo-selector-v2) and src/ are never written.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
case "$MODE" in
  run|prepare|check|status|results|plan) ;;
  *) echo 'usage: bash scripts/run_rloo_d100.sh [run|prepare|check|status|results|plan]'; exit 2 ;;
esac
[ "$#" -le 1 ] || { echo '[abort] no additional options; settings are fixed in this script'; exit 2; }
export OM_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
# Deliberately not RLOO_ROOT: an exported original root must never be picked up here.
ROOT=${RLOO_D100_ROOT:-$OM_WORK/runs/rloo-selector-d100}
PY=${RLOO_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
D100=("$PY" scripts/rloo_d100.py)
case "$MODE" in
  plan|status|check|results)
    CUDA_VISIBLE_DEVICES="" exec "${D100[@]}" "$MODE" --root "$ROOT" ;;
  prepare)
    CUDA_VISIBLE_DEVICES="" exec "${D100[@]}" prepare --root "$ROOT" ;;
esac
# run: prepare and verify on CPU before GPU admission.
CUDA_VISIBLE_DEVICES="" "${D100[@]}" prepare --root "$ROOT"
CUDA_VISIBLE_DEVICES="" "${D100[@]}" check --root "$ROOT"
export RLOO_ROOT="$ROOT" OUT_ROOT="$ROOT" E5_FORCE=0
# An inherited Pair scope must never trigger Pair-only cleanup on RLOO entry.
unset PAIR_ROOT
source scripts/_e5_node.sh
e5_acquire_node
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t GPUS < <(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader)
  export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"
fi
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
[ "${#GPUS[@]}" -eq 4 ] || { echo '[abort] four allocated GPUs required'; exit 2; }
MEMORY=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
while read -r used; do
  [[ "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || exit 2
  [ "$used" -le 4000 ] || { echo '[busy] GPU occupied; existing jobs untouched'; exit 75; }
done <<< "$MEMORY"
export RLOO_GPU_TYPE
RLOO_GPU_TYPE=$(timeout 20 nvidia-smi --query-gpu=name --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES")
VERIFY=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps")
export PYTHONPATH="$VERIFY:$PYTHONPATH" OM_MATH_VERIFIER=math_verify OM_NODE_LOCK_HELD=1
# The shared worker only sets node identity for recognised launchers; do it here.
source scripts/_node_id.sh
source scripts/_selection_worker.sh
selection_run_worker "${D100[@]}" queue --root "$ROOT" --max-phase-seconds "${RLOO_MAX_PHASE_SECONDS:-86400}"
