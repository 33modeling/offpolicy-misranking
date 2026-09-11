#!/usr/bin/env bash
# Same repository and GRPO engine; separate immutable experiment outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
[ "$#" -eq 0 ] || shift
case "$MODE" in run|score|train|prepare|status|check|plan|live|stop) ;;
  *) echo 'usage: bash scripts/run_low_order.sh [run|score|train|prepare|status|check|plan|live|stop]'; exit 2 ;;
esac
for option in "$@"; do
  case "$option" in --root|--root=*|--out|--out=*)
    echo '[abort] set LOW_ORDER_ROOT to change the output root'; exit 2 ;;
  esac
done
WORK=${OM_WORK:-$PWD/.work}
if [ -d "${GROUP_VOLUME:-/group-volume}" ]; then
  WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
fi
export OM_WORK="$WORK"
PY=${LOW_ORDER_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
ROOT=${OM_OLMO3_ROOT:-$WORK/runs/$TAG}
export OUT_ROOT
OUT_ROOT=$(realpath -m "${LOW_ORDER_ROOT:-$WORK/runs/low-order-reuse-v1}")
case "$OUT_ROOT" in /|"$PWD"|"$WORK"|"$WORK/runs") echo '[abort] unsafe suite root'; exit 2 ;; esac
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "$MODE" = plan ]; then
  printf '%s\n' '[low-order] default: OLMo MATH d100, seeds 0..4, K=8, G=8' \
    '[arms] random pair_u2 low_order; 100 continuation updates per arm' \
    '[evaluation] 300 independent questions x 8 responses; shared before-policy evaluation per seed' \
    '[run] validation direction -> checked scoring -> GRPO continuation -> independent test reward' \
    '[resources] one allocated four-GPU node; other nodes claim different points/arms' \
    '[cost] scoring, calibration, failed attempts and training consume GPU time; equal updates are NOT equal total cost' \
    '[safety] no source matrix, E5, Qwen, or manuscript edits; no downloads or GPU work in plan' \
    "[source] $ROOT" "[output] $OUT_ROOT"
  exit 0
fi
if [ "$MODE" = live ]; then
  shopt -s nullglob
  declare -A FOLLOWED=()
  TAILS=()
  stop_tails() { for pid in "${TAILS[@]}"; do kill "$pid" 2>/dev/null || true; done; }
  trap 'stop_tails; exit 130' INT
  trap 'stop_tails; exit 143' TERM
  trap stop_tails EXIT
  echo "[live] following node launchers and their worker progress under $OUT_ROOT"
  while true; do
    for path in "$OUT_ROOT"/logs/launcher.*.log; do
      [ -z "${FOLLOWED[$path]:-}" ] || continue
      FOLLOWED[$path]=1
      tail -n 5 -F "$path" & TAILS+=("$!")
    done
    sleep 2 & wait "$!"
  done
fi
if [ "$MODE" = check ] || [ "$MODE" = status ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" src/low_order_experiment.py "$MODE" --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = stop ]; then
  exec "$PY" src/cleanup_run_processes.py --run-prefix "$OUT_ROOT" \
    --command-pattern "$OUT_ROOT" --command-pattern 'scripts/run_low_order.sh' \
    --require-environment "OUT_ROOT=$OUT_ROOT" --launcher-environment-from-child --timeout 15 --compact
fi
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
if [ "$MODE" != train ] && { [ "$MODE" = prepare ] || [ ! -f "$OUT_ROOT/suite.json" ] || [ "$#" -gt 0 ]; }; then
  "$PY" src/low_order_experiment.py prepare --root "$OUT_ROOT" --matrix "$ROOT" \
    --pool "$DATASETS_DIR/math_train/math_train.jsonl" \
    --pool-manifest "$DATASETS_DIR/math_train/dataset_manifest.json" "$@"
fi
[ "$MODE" != prepare ] || exit 0
[ -f "$OUT_ROOT/suite.json" ] || { echo '[abort] prepare or run the suite first'; exit 2; }
if [ "$MODE" = train ] && [ "$#" -gt 0 ]; then
  echo '[abort] train resumes the frozen suite; specify settings during prepare'; exit 2
fi
export E5_FORCE=0
source scripts/_e5_node.sh
"$PY" src/cleanup_run_processes.py --run-prefix "$OUT_ROOT" \
  --command-pattern "$OUT_ROOT" --command-pattern 'scripts/run_low_order.sh' \
  --require-environment "OUT_ROOT=$OUT_ROOT" --launcher-environment-from-child --timeout 15 --compact
"$PY" src/cleanup_run_processes.py --run-prefix "$OUT_ROOT" \
  --command-pattern "$OUT_ROOT" --timeout 15 --compact
mkdir -p "$OUT_ROOT/logs"
HOST=$(hostname | tr -c 'a-zA-Z0-9._-' '_')
exec > >(tee -p -a "$OUT_ROOT/logs/launcher.$HOST.log") 2>&1
e5_acquire_node
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t GPUS < <(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader)
  export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"
fi
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
[ "${#GPUS[@]}" -eq 4 ] || { echo '[abort] exactly four allocated GPUs required'; exit 2; }
BUSY=1
for attempt in {1..12}; do
  MEMORY=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
  [ -n "$MEMORY" ] || { echo '[abort] unreadable GPU memory'; exit 2; }
  BUSY=0
  while read -r used; do
    [[ "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || { echo '[abort] unreadable GPU memory'; exit 2; }
    [ "$used" -le 4000 ] || BUSY=1
  done <<< "$MEMORY"
  [ "$BUSY" -eq 1 ] || break
  sleep 5
done
[ "$BUSY" -eq 0 ] || { echo '[busy] allocated GPUs occupied; unrelated jobs were not killed'; exit 75; }
if [ "$MODE" != score ]; then
  MATH_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps")
  export PYTHONPATH="$MATH_VERIFY_PATH${PYTHONPATH:+:$PYTHONPATH}" OM_MATH_VERIFIER=math_verify
fi
export OM_NODE_LOCK_HELD=1
trap '' HUP
"$PY" src/low_order_experiment.py "$MODE" --root "$OUT_ROOT" 7>&- 8>&- &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 130' INT
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 143' TERM
rc=0
wait "$CHILD" || rc=$?
exit "$rc"
