#!/usr/bin/env bash
# Separate queue: never modifies or restarts the selected-prefix experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
[ "$#" -eq 0 ] || shift
case "$MODE" in prepare|run|retry|status|summarize|errors|recover-cost|cpu) ;;
  *) echo 'usage: bash scripts/run_mopps_comparison.sh [prepare|run|retry|status|summarize|errors|recover-cost|cpu]'; exit 2 ;;
esac
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export OM_WORK="$WORK"
export OUT_ROOT
OUT_ROOT=$(realpath -m "${MOPPS_ROOT:-$WORK/runs/mopps-comparison-v1}")
PARENT=$(realpath -m "${SWITCH_ROOT:-$WORK/runs/selection-switch-v1}")
case "$OUT_ROOT" in /|"$PWD"|"$WORK"|"$WORK/runs"|"$PARENT") echo '[abort] unsafe comparison root'; exit 2 ;; esac
PY=${MOPPS_PYTHON:-${SWITCH_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$MODE" = cpu ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" -m pytest -q tests/test_mopps.py tests/test_mopps_comparison_gpu.py tests/test_selection_worker_shutdown.py "$@"
fi
for arg in "$@"; do
  case "$arg" in --root|--root=*|--parent-root|--parent-root=*) echo '[abort] use MOPPS_ROOT and SWITCH_ROOT'; exit 2 ;; esac
done
case "$MODE" in
  run|retry|prepare|summarize)
    if [ "${SWITCH_RUNTIME_REPO:-}" != "$PWD" ]; then
      exec "$PY" scripts/selection_switch_runtime.py --repo "$PWD" \
        --cache "${SWITCH_RUNTIME_CACHE:-/tmp/offpolicy-misranking-$(id -u)/switch-runtimes}" \
        --kind mopps -- "$MODE" "$@"
    fi
    export OM_REPO="$PWD"
    ;;
esac
if [ "$MODE" = prepare ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" src/mopps_comparison_gpu.py prepare --root "$OUT_ROOT" --parent-root "$PARENT" "$@"
fi
if [ "$MODE" = status ] || [ "$MODE" = summarize ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" src/mopps_comparison_gpu.py "$MODE" --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = errors ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" scripts/selection_switch_errors.py --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = recover-cost ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" scripts/recover_selection_switch_cost.py --root "$OUT_ROOT" "$@"
fi
# Preparation checks disjoint roots before even creating launcher logs.
CUDA_VISIBLE_DEVICES="" "$PY" src/mopps_comparison_gpu.py prepare --root "$OUT_ROOT" --parent-root "$PARENT"
mkdir -p "$OUT_ROOT/logs"
HOST=$(hostname | tr -c 'a-zA-Z0-9._-' '_')
exec > >(tee -p -a "$OUT_ROOT/logs/launcher.$HOST.log") 2>&1
printf '[launcher-start] host=%s pid=%s mode=%s commit=%s utc=%s\n' "$HOST" "$$" "$MODE" "${SWITCH_RUNTIME_COMMIT:-unknown}" "$(date -u +%FT%TZ)"
trap 'rc=$?; printf "[launcher-exit] pid=%s mode=%s rc=%s utc=%s\n" "$$" "$MODE" "$rc" "$(date -u +%FT%TZ)"' EXIT
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
source scripts/_e5_node.sh
export E5_FORCE=0
e5_acquire_node
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t GPUS < <(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader)
  export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"
fi
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
[ "${#GPUS[@]}" -eq 4 ] || { echo '[abort] four allocated GPUs required'; exit 2; }
MEMORY=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
while read -r used; do
  [[ "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || { echo '[abort] invalid GPU memory status'; exit 2; }
  [ "$used" -le 4000 ] || { echo '[busy] GPU occupied; existing jobs were not stopped'; exit 75; }
done <<< "$MEMORY"
MATH_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps")
export PYTHONPATH="$MATH_VERIFY_PATH${PYTHONPATH:+:$PYTHONPATH}" OM_MATH_VERIFIER=math_verify OM_NODE_LOCK_HELD=1
trap '' HUP
source scripts/_selection_worker.sh
rc=0
selection_run_worker "$PY" src/mopps_comparison_gpu.py "$MODE" --root "$OUT_ROOT" "$@" || rc=$?
if [ "$rc" -ne 0 ]; then
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_errors.py --root "$OUT_ROOT" || true
fi
exit "$rc"
