#!/usr/bin/env bash
# Selected-prefix experiment. Never stops or rewrites E5, Qwen or net-gain jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
[ "$#" -eq 0 ] || shift
case "$MODE" in run|smoke|prepare|status|fit|summarize|export|why|live|cpu|recover-cost|errors|check-code) ;;
  *) echo 'usage: bash scripts/run_selection_switch.sh [run|smoke|status|export|why|live|cpu|prepare|fit|summarize|recover-cost|errors|check-code]'; exit 2 ;;
esac
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export OM_WORK="$WORK"
export OUT_ROOT
OUT_ROOT=$(realpath -m "${SWITCH_ROOT:-$WORK/runs/selection-switch-v1}")
case "$OUT_ROOT" in /|"$PWD"|"$WORK"|"$WORK/runs") echo '[abort] unsafe experiment root'; exit 2 ;; esac
PY=${SWITCH_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$MODE" = cpu ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "${SWITCH_CPU_PYTHON:-$PY}" -m pytest -q tests/test_selection_switch.py tests/test_selection_switch_gpu.py \
    tests/test_net_gate_memory_math.py tests/test_logit_chunking.py tests/test_selection_switch_cost.py \
    tests/test_selection_switch_errors.py tests/test_selection_switch_status.py "$@"
fi
for arg in "$@"; do
  case "$arg" in --root|--root=*) echo '[abort] use SWITCH_ROOT for the output directory'; exit 2 ;; esac
done
if [ "$MODE" = status ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" scripts/selection_switch_status.py --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = check-code ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" src/selection_switch_gpu.py check-code --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = errors ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" scripts/selection_switch_errors.py --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = recover-cost ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" scripts/recover_selection_switch_cost.py --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = fit ] || [ "$MODE" = summarize ]; then
  export CUDA_VISIBLE_DEVICES=""
  "$PY" src/selection_switch_gpu.py "$MODE" --root "$OUT_ROOT" "$@"
  if [ "$MODE" = summarize ]; then
    "$PY" src/selection_switch_plot.py --root "$OUT_ROOT"
  fi
  exit 0
fi
if [ "$MODE" = export ] || [ "$MODE" = why ]; then
  [ -d "$OUT_ROOT" ] || { echo "[abort] no logs/results: $OUT_ROOT"; exit 2; }
  REPORT_DIR="$WORK/reports/selection-switch"
  mkdir -p "$REPORT_DIR"
  TARGET=$(mktemp "$REPORT_DIR/switch-$MODE-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.txt")
  (
    printf 'SELECTION SWITCH EXPERIMENT\nUTC: %s\nROOT: %s\nCOMMIT: ' "$(date -u +%FT%TZ)" "$OUT_ROOT"
    git rev-parse HEAD
    CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_status.py --root "$OUT_ROOT"
    if [ "$MODE" = export ] && [ -f "$OUT_ROOT/switch.json" ]; then
      CUDA_VISIBLE_DEVICES="" "$PY" src/selection_switch_gpu.py summarize --root "$OUT_ROOT"
    fi
    while IFS= read -r -d '' path; do
      printf '\n===== %s =====\n' "${path#"$OUT_ROOT"/}"
      cat "$path"
      printf '\n'
    done < <(find "$OUT_ROOT" -type f \( -name 'switch.json' -o -name 'model.json' \
      -o -name '*report.json' -o -name 'failure.json' -o -name 'progress.json' \
      -o -name 'decision.json' -o -name 'decisions-frozen.json' -o -name 'initial.json' \
      -o -name 'measurement.json' -o -name 'execution.json' -o -name 'result.json' \
      -o -name 'cost.jsonl' -o -name 'budget_stop.json' -o -name 'fit-cost.json' \
      -o -name 'kv-cache-runtime.json' -o -name 'cost-runtime.json' -o -name 'prefix-resume-runtime.json' \
      -o -name 'worker-logs-runtime.json' -o -name 'code-compat-runtime.json' \
      -o -path '*/cost-events/*.json' -o -path '*/pending-costs/*.json' \) -print0 | sort -z)
    while IFS= read -r -d '' path; do
      printf '\n===== LOG: %s (last 100 lines) =====\n' "${path#"$OUT_ROOT"/}"
      tail -n 100 "$path"
    done < <(find "$OUT_ROOT" -type f -name '*.log' -print0 | sort -z)
  ) > "$TARGET" 2>&1 || { printf '[export incomplete; see errors] %s\n' "$TARGET"; exit 1; }
  printf '[saved] %s\n' "$TARGET"
  exit 0
fi
if [ "$MODE" = live ]; then
  shopt -s nullglob
  LOGS=("$OUT_ROOT"/logs/launcher.*.log)
  [ "${#LOGS[@]}" -gt 0 ] || { echo "[no launcher logs] $OUT_ROOT/logs"; exit 1; }
  exec tail -n 20 -F "${LOGS[@]}"
fi
mkdir -p "$OUT_ROOT/logs"
HOST=$(hostname | tr -c 'a-zA-Z0-9._-' '_')
exec > >(tee -p -a "$OUT_ROOT/logs/launcher.$HOST.log") 2>&1
echo "[logs] $OUT_ROOT/logs/launcher.$HOST.log"
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
MATRIX=${OM_OLMO3_ROOT:-$WORK/runs/${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}}
if [ "$MODE" = prepare ] || [ ! -f "$OUT_ROOT/switch.json" ]; then
  "$PY" src/selection_switch_gpu.py prepare --root "$OUT_ROOT" --matrix "$MATRIX" \
    --gpu-type "${GATE_GPU_TYPE:-NVIDIA H100 80GB HBM3}" \
    --pool "$DATASETS_DIR/math_train/math_train.jsonl" \
    --pool-manifest "$DATASETS_DIR/math_train/dataset_manifest.json" "$@"
elif [ "$#" -gt 0 ]; then
  echo '[abort] experiment already frozen; run takes no new preparation options'; exit 2
fi
[ "$MODE" != prepare ] || exit 0
CUDA_VISIBLE_DEVICES="" "$PY" src/selection_switch_gpu.py check-code --root "$OUT_ROOT"
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
  [ "$used" -le 4000 ] || { echo '[busy] allocated GPU is occupied; other experiments were not stopped'; exit 75; }
done <<< "$MEMORY"
MATH_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps")
export PYTHONPATH="$MATH_VERIFY_PATH${PYTHONPATH:+:$PYTHONPATH}" OM_MATH_VERIFIER=math_verify OM_NODE_LOCK_HELD=1
trap '' HUP
"$PY" src/selection_switch_gpu.py "$MODE" --root "$OUT_ROOT" 7>&- 8>&- &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 130' INT
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 143' TERM
rc=0
wait "$CHILD" || rc=$?
if [ "$rc" -ne 0 ]; then
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_errors.py --root "$OUT_ROOT" || true
fi
exit "$rc"
