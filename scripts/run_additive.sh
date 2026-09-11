#!/usr/bin/env bash
# New scoring only. Existing OLMo/Qwen/E5 launchers and scientific files are untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
[ "$#" -eq 0 ] || shift
case "$MODE" in run|prepare|status|check|plan|live|stop) ;;
  *) echo 'usage: bash scripts/run_additive.sh [run|prepare|status|check|plan|live|stop]'; exit 2 ;;
esac
for option in "$@"; do
  case "$option" in --root|--root=*|--out|--out=*)
    echo '[abort] use ADDITIVE_ROOT for the output root so ownership and worker paths stay identical'; exit 2 ;;
  esac
done
WORK=${OM_WORK:-$PWD/.work}
if [ -d "${GROUP_VOLUME:-/group-volume}" ]; then
  WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
fi
export OM_WORK="$WORK"
PY=${ADDITIVE_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
ROOT=${OM_OLMO3_ROOT:-$WORK/runs/$TAG}
export OUT_ROOT=${ADDITIVE_ROOT:-$WORK/runs/additive-correction-v1}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "$MODE" = plan ]; then
  printf '%s\n' '[additive] scoring-only extension: MATH d400; seeds 0 1 2' \
    '[methods] gadd and tay2_terminal; source component clipping cap retained' \
    '[cost] zero new response generation; two gradient estimators per prompt, not zero GPU cost' \
    '[resources] four allocated GPUs; one source point per node; nodes skip leased points' \
    '[resume] completed prompt scores and behavior log-probability caches retained' \
    '[scope] no GRPO training, no E5 restart, no modifications to registered artifacts' \
    "[source] $ROOT" "[output] $OUT_ROOT"
  exit 0
fi
if [ "$MODE" = live ]; then
  shopt -s nullglob
  LOGS=("$OUT_ROOT"/logs/launcher.*.log "$OUT_ROOT"/points/*/logs/rescoring-additive-*.log)
  [ "${#LOGS[@]}" -gt 0 ] || { echo "[no logs] $OUT_ROOT"; exit 0; }
  exec tail -n 15 -F "${LOGS[@]}"
fi
if [ "$MODE" = check ] || [ "$MODE" = status ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" src/additive_experiment.py "$MODE" --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = stop ]; then
  exec "$PY" src/cleanup_run_processes.py --run-prefix "$OUT_ROOT" \
    --command-pattern "$OUT_ROOT" --command-pattern 'scripts/run_additive.sh' \
    --require-environment "OUT_ROOT=$OUT_ROOT" --launcher-environment-from-child --timeout 15 --compact
fi
"$PY" src/additive_experiment.py prepare --root "$OUT_ROOT" --matrix "$ROOT" "$@"
[ "$MODE" != prepare ] || exit 0
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export E5_FORCE=0
source scripts/_e5_node.sh
# A repeated launch stops only this suite on this physical node.
"$PY" src/cleanup_run_processes.py --run-prefix "$OUT_ROOT" \
  --command-pattern "$OUT_ROOT" --command-pattern 'scripts/run_additive.sh' \
  --require-environment "OUT_ROOT=$OUT_ROOT" --launcher-environment-from-child --timeout 15 --compact
"$PY" src/cleanup_run_processes.py --run-prefix "$OUT_ROOT" \
  --command-pattern "$OUT_ROOT" --timeout 15 --compact
mkdir -p "$OUT_ROOT/logs"
HOST=$(hostname | tr -c 'a-zA-Z0-9._-' '_')
exec > >(tee -a "$OUT_ROOT/logs/launcher.$HOST.log") 2>&1
e5_acquire_node
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t GPUS < <(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader)
  export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"
fi
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
[ "${#GPUS[@]}" -eq 4 ] || { echo '[abort] exactly four allocated GPUs required'; exit 2; }
for attempt in {1..12}; do
  MEMORY=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
  BUSY=0
  while read -r used; do
    [[ "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || { echo '[abort] unreadable GPU memory'; exit 2; }
    [ "$used" -le 4000 ] || BUSY=1
  done <<< "$MEMORY"
  [ "$BUSY" -eq 1 ] || break
  sleep 5
done
[ "$BUSY" -eq 0 ] || { echo '[busy] allocated GPUs remain occupied; no unrelated jobs were killed'; exit 75; }
export OM_NODE_LOCK_HELD=1
trap '' HUP
"$PY" src/additive_experiment.py run --root "$OUT_ROOT" 7>&- 8>&- &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 130' INT
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 143' TERM
rc=0
wait "$CHILD" || rc=$?
"$PY" src/additive_experiment.py status --root "$OUT_ROOT" || rc=1
exit "$rc"
