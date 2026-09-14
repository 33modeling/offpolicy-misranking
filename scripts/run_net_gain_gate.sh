#!/usr/bin/env bash
# V3 only. Existing E5, Qwen and gate launchers are not modified.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
[ "$#" -eq 0 ] || shift
case "$MODE" in run|prepare|status|summarize|live|stop|plan|cpu|fit|analyze|import-legacy|export) ;;
  *) echo 'usage: bash scripts/run_net_gain_gate.sh [run|prepare|status|summarize|live|stop|plan|cpu|fit|analyze|import-legacy|export]'; exit 2 ;;
esac
for option in "$@"; do
  case "$option" in --root|--root=*) echo '[abort] use NET_GATE_ROOT to set the output root'; exit 2 ;; esac
done
WORK=${OM_WORK:-$PWD/.work}
if [ -d "${GROUP_VOLUME:-/group-volume}" ]; then
  WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
fi
export OM_WORK="$WORK"
PY=${NET_GATE_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
MATRIX=${OM_OLMO3_ROOT:-$WORK/runs/$TAG}
export OUT_ROOT
OUT_ROOT=$(realpath -m "${NET_GATE_ROOT:-$WORK/runs/net-gain-gate-v3}")
case "$OUT_ROOT" in /|"$PWD"|"$WORK"|"$WORK/runs") echo '[abort] unsafe suite root'; exit 2 ;; esac
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [ "$MODE" = cpu ]; then
  export CUDA_VISIBLE_DEVICES=""
  CPU_PY=${NET_GATE_CPU_PYTHON:-${NET_GATE_PYTHON:-python3}}
  exec "$CPU_PY" -m pytest -q tests/test_net_gain_gate.py tests/test_net_gain_gate_gpu.py "$@"
fi
if [ "$MODE" = plan ]; then
  printf '%s\n' '[v3] one whole-pool cache scan plus existing checkpoint statistics; no per-epoch gate or bootstrap' \
    '[target] independent test reward: selected shortened continuation minus full-budget random' \
    '[study] default OLMo MATH d100, seeds 0/1/2; three matched-parent continuations per point' \
    '[test] --mode test --model MODEL --seeds 3 4; only if those trajectories were never used to fit MODEL' \
    '[test arms] always random, always selected, actual gate; diagnosis and scoring charged inside each cap' \
    '[selector] default low_order; --selector difficulty or pair_u2; NOT the old g11/fresh_r gradient scores' \
    '[scope] positive-drift OLMo MATH only; four GPUs per admitted node; no mixed pool' \
    '[horizon] one specified compute budget, not an estimated optimal stopping step' \
    '[cost] evaluation, model development and historical shared inputs reported separately' \
    "[source] $MATRIX" "[output] $OUT_ROOT"
  exit 0
fi
if [ "$MODE" = fit ] || [ "$MODE" = analyze ] || [ "$MODE" = import-legacy ]; then
  export CUDA_VISIBLE_DEVICES=""
  PY=${NET_GATE_CPU_PYTHON:-${NET_GATE_PYTHON:-python3}}
  if [ "$MODE" = fit ] && [ "$#" -eq 0 ]; then
    set -- --study "$OUT_ROOT/study.json" --out "$OUT_ROOT/model.json"
  fi
  exec "$PY" src/net_gain_gate.py "$MODE" "$@"
fi
if [ "$MODE" = status ]; then
  exec bash scripts/status_net_gain_gate.sh "$@"
fi
if [ "$MODE" = summarize ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" src/net_gain_gate_gpu.py "$MODE" --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = export ]; then
  [ -f "$OUT_ROOT/net_protocol.json" ] || { echo '[abort] no v3 suite at output root'; exit 2; }
  TARGET=${NET_GATE_EXPORT:-$HOME/net-gate-results-$(date -u +%Y%m%dT%H%M%SZ).txt}
  [ ! -e "$TARGET" ] || { echo '[abort] export already exists'; exit 2; }
  (
    set -o noclobber
    {
      printf 'V3 NET-GAIN GATE EXPORT\nUTC: %s\nROOT: %s\nCOMMIT: ' "$(date -u +%FT%TZ)" "$OUT_ROOT"
      git rev-parse HEAD
      CUDA_VISIBLE_DEVICES="" "$PY" src/net_gain_gate_gpu.py status --root "$OUT_ROOT"
      while IFS= read -r -d '' path; do
        printf '\n===== %s =====\n' "${path#"$OUT_ROOT"/}"
        cat "$path"
        printf '\n'
      done < <(find "$OUT_ROOT" -type f \( -name 'net_protocol.json' -o -name 'suite.json' \
          -o -name 'study.json' -o -name 'model.json' -o -name 'net_results.json' \
          -o -name 'cost.jsonl' -o -name 'initial.json' -o -name 'measurement.json' \
          -o -name 'decision.json' -o -name 'execution.json' -o -name 'result.json' \
          -o -name 'budget_stop.json' -o -name 'failure.json' -o -name 'measurement-failure.json' \) -print0 | sort -z)
      while IFS= read -r -d '' path; do
        printf '\n===== LOG TAIL: %s =====\n' "${path#"$OUT_ROOT"/}"
        tail -n 40 "$path"
      done < <(find "$OUT_ROOT" -type f -name '*.log' -print0 | sort -z)
    } > "$TARGET"
  )
  printf '[export] %s\n' "$TARGET"
  exit 0
fi
if [ "$MODE" = live ]; then
  shopt -s nullglob
  declare -A FOLLOWED=()
  TAILS=()
  cleanup_tails() { for pid in "${TAILS[@]}"; do kill "$pid" 2>/dev/null || true; done; }
  trap 'cleanup_tails; exit 130' INT
  trap 'cleanup_tails; exit 143' TERM
  trap cleanup_tails EXIT
  echo "[live] $OUT_ROOT/logs/launcher.*.log"
  while true; do
    for path in "$OUT_ROOT"/logs/launcher.*.log; do
      [ -z "${FOLLOWED[$path]:-}" ] || continue
      FOLLOWED[$path]=1
      tail -n 8 -F "$path" & TAILS+=("$!")
    done
    sleep 2 & wait "$!"
  done
fi
if [ "$MODE" = stop ]; then
  exec "$PY" src/cleanup_run_processes.py --run-prefix "$OUT_ROOT" \
    --command-pattern "$OUT_ROOT" --command-pattern 'scripts/run_net_gain_gate.sh' \
    --require-environment "OUT_ROOT=$OUT_ROOT" --launcher-environment-from-child --timeout 15 --compact
fi
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
if [ "$MODE" = prepare ] || [ ! -f "$OUT_ROOT/suite.json" ] || [ ! -f "$OUT_ROOT/net_protocol.json" ] || [ "$#" -gt 0 ]; then
  GPU_TYPE=${GATE_GPU_TYPE:-}
  if [ -z "$GPU_TYPE" ] && command -v nvidia-smi >/dev/null; then
    GPU_TYPE=$(timeout 20 nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" | sort -u)
  fi
  "$PY" src/net_gain_gate_gpu.py prepare --root "$OUT_ROOT" --matrix "$MATRIX" \
    --gpu-type "${GPU_TYPE:-NVIDIA H100 80GB HBM3}" \
    --pool "$DATASETS_DIR/math_train/math_train.jsonl" \
    --pool-manifest "$DATASETS_DIR/math_train/dataset_manifest.json" "$@"
fi
[ "$MODE" != prepare ] || exit 0
export E5_FORCE=0
source scripts/_e5_node.sh
"$PY" src/cleanup_run_processes.py --run-prefix "$OUT_ROOT" \
  --command-pattern "$OUT_ROOT" --command-pattern 'scripts/run_net_gain_gate.sh' \
  --require-environment "OUT_ROOT=$OUT_ROOT" --launcher-environment-from-child --timeout 15 --compact
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
MEMORY=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
[ -n "$MEMORY" ] || { echo '[abort] GPU memory status unavailable'; exit 2; }
while read -r used; do
  [[ "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || { echo '[abort] invalid GPU memory status'; exit 2; }
  [ "$used" -le 4000 ] || { echo '[busy] allocated GPUs occupied; unrelated jobs were not killed'; exit 75; }
done <<< "$MEMORY"
MATH_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps")
export PYTHONPATH="$MATH_VERIFY_PATH${PYTHONPATH:+:$PYTHONPATH}" OM_MATH_VERIFIER=math_verify OM_NODE_LOCK_HELD=1
trap '' HUP
"$PY" src/net_gain_gate_gpu.py run --root "$OUT_ROOT" 7>&- 8>&- &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 130' INT
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 143' TERM
rc=0
wait "$CHILD" || rc=$?
exit "$rc"
