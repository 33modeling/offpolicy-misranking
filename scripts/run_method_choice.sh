#!/usr/bin/env bash
# Separate prospective suite. Never changes the reduced E5 root or selectors.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
case "$MODE" in run|prepare|status|summarize|plan) ;; *) echo "usage: bash scripts/run_method_choice.sh [run|prepare|status|summarize|plan]"; exit 2 ;; esac
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
ROOT=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
OUT_ROOT=${METHOD_CHOICE_ROOT:-$OM_WORK/runs/method-choice-v1}
export OUT_ROOT
if [ "$MODE" = plan ]; then
  printf '%s\n' '[method-choice] MATH d100; seeds 0..4; 200 updates; 500 independent test questions x 32 responses' \
    '[arms] g00 g10 g01 g11 fresh_r passrate_beta random full_pool' \
    '[budget] 40 trained arms x 200 updates = 8000 updates; 45 policy evaluations including baselines' \
    '[resources] run: one allocated 4-GPU node per worker; multiple nodes lease different arms' \
    '[cost] phase GPU-seconds recorded; historical scoring/generation costs not assumed zero' \
    "[input] $ROOT" "[output] $OUT_ROOT" '[plan] no inputs created and no GPU work launched'
  exit 0
fi
if [ "$MODE" = status ] || [ "$MODE" = summarize ]; then
  exec "$PY" src/method_choice.py "$MODE" --root "$OUT_ROOT"
fi
"$PY" src/method_choice.py prepare --root "$OUT_ROOT" --matrix "$ROOT" \
  --pool "$DATASETS_DIR/math_train/math_train.jsonl" --pool-manifest "$DATASETS_DIR/math_train/dataset_manifest.json"
[ "$MODE" != prepare ] || exit 0
trap '' HUP
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export OM_SKIP_HYBRID=1
source scripts/_e5_node.sh
# Include this controller's command name: its late OUT_ROOT export may only
# be visible through descendants, while the shell itself retains node admission.
"$PY" src/cleanup_run_processes.py --run-prefix "$OUT_ROOT" \
  --command-pattern "$OUT_ROOT" --command-pattern 'scripts/run_method_choice.sh' \
  --require-environment "OUT_ROOT=$OUT_ROOT" --launcher-environment-from-child --timeout 15 --compact
"$PY" src/cleanup_run_processes.py --run-prefix "$OUT_ROOT" \
  --command-pattern "$OUT_ROOT" --timeout 15 --compact
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
[ "$BUSY" -eq 0 ] || { echo '[busy] GPU memory did not drain; checkpoints retained'; exit 75; }
MATH_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps")
export PYTHONPATH="$MATH_VERIFY_PATH${PYTHONPATH:+:$PYTHONPATH}" OM_MATH_VERIFIER=math_verify
export OM_NODE_LOCK_HELD=1
# Keep admission descriptors in the controller, not model/compile descendants.
"$PY" src/method_choice.py work --root "$OUT_ROOT" 7>&- 8>&- &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 130' INT
trap 'kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; exit 143' TERM
rc=0
wait "$CHILD" || rc=$?
"$PY" src/method_choice.py summarize --root "$OUT_ROOT" || rc=1
exit "$rc"
