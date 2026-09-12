#!/usr/bin/env bash
# Fixed-policy g11 gate. Same command on up to four idle four-GPU nodes.
# Existing E5/Qwen/OLMo data and the historical gate_passrate arm are preserved.
#   bash scripts/run_fixed_gate.sh          # d400 + d0, seeds 0 1 2
#   bash scripts/run_fixed_gate.sh status   # CPU, all points
#   bash scripts/run_fixed_gate.sh live     # follow this suite's logs
#   bash scripts/run_fixed_gate.sh export   # CPU results and cost bundle
#   bash scripts/run_fixed_gate.sh cpu      # isolated tests, no GPU
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=run; DRIFTS=(400 0)
for arg in "$@"; do
  case "$arg" in
    run|status|results|plan|export|live|stop|cpu) MODE=$arg ;;
    d0) DRIFTS=(0) ;; d400) DRIFTS=(400) ;;
    *) echo 'usage: bash scripts/run_fixed_gate.sh [run|status|results|plan|export|live|stop|cpu] [d0|d400]'; exit 2 ;;
  esac
done
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "$MODE" = cpu ]; then
  PY=${FIXED_GATE_CPU_PYTHON:-$PWD/.work/.venv-cu126/bin/python}
  [ -x "$PY" ] || PY=python3
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" -m pytest -q tests/test_fixed_gate.py tests/test_review_regressions.py \
    tests/test_gate_arm.py tests/test_gate_decision.py tests/test_benchmark_eval.py \
    tests/test_gain_law.py tests/test_stale_splithalf.py tests/test_evidence_downstream.py \
    tests/test_selection_gate_gpu.py tests/test_cleanup_run_processes.py
fi
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] missing environment: $PY"; exit 1; }
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
MATRIX=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
export OUT_ROOT
OUT_ROOT=$(realpath -m "${FIXED_GATE_ROOT:-$OM_WORK/runs/fixed-checkpoint-gate-v1}")
read -r -a SEEDS <<< "${E5_SEEDS:-0 1 2}"
RULE=${FIXED_GATE_RULE:-config/fixed_gate_rule.json}
ARGS=(--root "$OUT_ROOT" --work "$OM_WORK" --matrix "$MATRIX" --rule "$RULE" --seeds "${SEEDS[@]}" --drifts "${DRIFTS[@]}")
if [ "$MODE" = status ] || [ "$MODE" = results ] || [ "$MODE" = plan ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" src/fixed_gate.py "$MODE" "${ARGS[@]}"
fi
if [ "$MODE" = export ]; then
  export CUDA_VISIBLE_DEVICES=""
  mkdir -p "$OM_WORK/exports"
  target="$OM_WORK/exports/fixed-gate-$(date -u +%Y%m%dT%H%M%SZ).txt"
  rc=0
  "$PY" src/fixed_gate.py results "${ARGS[@]}" || rc=$?
  {
    printf '# fixed-checkpoint gate export code=%s UTC=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
    for f in "$OUT_ROOT/results.json" "$OUT_ROOT"/d*/s*/contract.json "$OUT_ROOT"/d*/s*/diagnostic.json \
             "$OUT_ROOT"/d*/s*/decision.json "$OUT_ROOT"/d*/s*/cost.jsonl \
             "$OUT_ROOT"/d*/s*/progress.json "$OUT_ROOT"/d*/s*/baseline-failure.json; do
      [ -s "$f" ] || continue
      printf '\n### %s\n' "$f"
      cat "$f"
    done
    for f in "$OUT_ROOT"/d*/s*/*.log; do
      [ -s "$f" ] || continue
      printf '\n### Last 80 lines: %s\n' "$f"
      tail -n 80 "$f"
    done
  } > "$target"
  echo "[export] $target"
  exit "$rc"
fi
if [ "$MODE" = live ]; then
  mkdir -p "$OUT_ROOT/logs"
  shopt -s nullglob
  declare -A FOLLOWED=()
  TAILS=()
  stop_tails() { for pid in "${TAILS[@]}"; do kill "$pid" 2>/dev/null || true; done; }
  trap 'stop_tails; exit 130' INT
  trap 'stop_tails; exit 143' TERM
  trap stop_tails EXIT
  echo "[live] all nodes: $OUT_ROOT/logs/launcher-*.log"
  while true; do
    for path in "$OUT_ROOT"/logs/launcher-*.log; do
      [ -z "${FOLLOWED[$path]:-}" ] || continue
      FOLLOWED[$path]=1
      tail -n 20 -F "$path" & TAILS+=("$!")
    done
    sleep 2 & wait "$!"
  done
fi
if [ "$MODE" = stop ]; then
  exec "$PY" src/cleanup_run_processes.py --run-prefix "$OUT_ROOT" --command-pattern "$OUT_ROOT" \
    --require-environment "OUT_ROOT=$OUT_ROOT" --launcher-environment-from-child --timeout 15 --compact
fi
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
source scripts/_e5_node.sh
# Only this suite's previous local controller; unrelated experiments do not match.
e5_cleanup_previous "$OUT_ROOT"
E5_FORCE=0 e5_acquire_node
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
else mapfile -t GPUS < <(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader); fi
[ "${#GPUS[@]}" -eq 4 ] || { echo '[abort] exactly four allocated GPUs required'; exit 2; }
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"
MEMORY=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
while read -r used; do
  [[ "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || { echo '[abort] invalid GPU memory status'; exit 2; }
  [ "$used" -le 4000 ] || { echo '[busy] allocated GPUs occupied; unrelated jobs were not stopped'; exit 75; }
done <<< "$MEMORY"
export FIXED_GATE_GPU_TYPE
FIXED_GATE_GPU_TYPE=$(timeout 20 nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPUS[0]}")
MATH_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps")
export PYTHONPATH="$MATH_VERIFY_PATH:$PYTHONPATH" OM_MATH_VERIFIER=math_verify OM_NODE_LOCK_HELD=1
mkdir -p "$OUT_ROOT/logs"
exec > >(tee -p -a "$OUT_ROOT/logs/launcher-$(hostname).log" 7>&- 8>&-) 2>&1
trap '' HUP
"$PY" src/fixed_gate.py run "${ARGS[@]}" 7>&- 8>&- &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 130' INT
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 143' TERM
rc=0
wait "$CHILD" || rc=$?
echo '[fixed-gate] pass finished; bash scripts/run_fixed_gate.sh status'
exit "$rc"
