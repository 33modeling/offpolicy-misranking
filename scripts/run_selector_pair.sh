#!/usr/bin/env bash
# A new, isolated G/D experiment. Never invokes the legacy selector/random fit.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
[ "$#" -eq 0 ] || shift
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export OM_WORK="$WORK"
PAIR_ROOT=${PAIR_ROOT:-$WORK/runs/selector-pair-v1}
PY=${PAIR_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Apply before the first Python import, including all four rollout/scoring
# workers. OMP/MKL alone do not bound pthread OpenBLAS or tokenizer Rayon pools.
export OPENBLAS_NUM_THREADS=1 OPENBLAS_DEFAULT_NUM_THREADS=1 GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export OMP_THREAD_LIMIT=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
case "$MODE" in
  results)
    export CUDA_VISIBLE_DEVICES=""
    exec "$PY" scripts/selector_pair_results.py --root "$PAIR_ROOT" "$@" ;;
  status)
    export CUDA_VISIBLE_DEVICES=""
    STATUS_ARGS=()
    STATUS_WATCH=
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --all|--json) STATUS_ARGS+=("$1"); shift ;;
        --watch)
          STATUS_WATCH=15; shift
          if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then STATUS_WATCH=$1; shift; fi
          [[ "$STATUS_WATCH" =~ ^[1-9][0-9]*$ ]] || { echo '[abort] watch interval must be a positive integer'; exit 2; }
          ;;
        *) echo 'usage: bash scripts/run_selector_pair.sh status [--all] [--watch [SECONDS]] [--json]'; exit 2 ;;
      esac
    done
    while :; do
      if [ -n "$STATUS_WATCH" ] && [ -t 1 ]; then printf '\033[2J\033[H'; fi
      rc=0
      "$PY" scripts/selector_pair_status.py --root "$PAIR_ROOT" "${STATUS_ARGS[@]}" || rc=$?
      [ -n "$STATUS_WATCH" ] || exit "$rc"
      sleep "$STATUS_WATCH"
    done ;;
  cpu)
    export CUDA_VISIBLE_DEVICES=""
    exec "$PY" -m pytest -q -p no:cacheprovider tests/test_selector_pair.py tests/test_selector_pair_gpu.py tests/test_selector_pair_operations.py tests/test_selector_pair_busy.py tests/test_selector_pair_lock_migration.py tests/test_selector_pair_queue.py tests/test_selector_pair_branch_queue.py tests/test_selector_pair_queue_migration.py tests/test_selector_pair_queue_barrier.py tests/test_selector_pair_wait.py tests/test_selector_pair_wait_migration.py tests/test_selector_pair_status.py "$@" ;;
  init|prepare|fit|report|check-code)
    export CUDA_VISIBLE_DEVICES=""
    exec "$PY" src/selector_pair_gpu.py "$MODE" --root "$PAIR_ROOT" "$@" ;;
  run|develop|freeze|test) ;;
  *) echo 'usage: run_selector_pair.sh init|prepare|run|develop|fit|freeze|test|report|results|status|check-code|cpu'; exit 2 ;;
esac
if [ "$#" -ne 0 ]; then
  echo '[abort] run uses the frozen preparation; new options require a new root'; exit 2
fi
pair_cpu_step() {
  local rc=0 retries=0
  while :; do
    rc=0
    CUDA_VISIBLE_DEVICES="" "$PY" src/selector_pair_gpu.py "$1" --root "$PAIR_ROOT" || rc=$?
    [ "$rc" -eq 75 ] || return "$rc"
    if [ "$retries" -ge 12 ]; then
      echo "[pair-wait-timeout] preparation lock still held: $PAIR_ROOT; stopped waiting without touching existing work" >&2
      return 76
    fi
    retries=$((retries+1))
    echo "[WAIT] pair preparation is in progress on another node; retry in 15s ($retries/12)"
    sleep 15
  done
}
PAIR_RESUME_TWO=0
if [ "$MODE" = run ]; then
  PAIR_PROBE_RC=0
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/selector_pair_resume_two.py probe --root "$PAIR_ROOT" || PAIR_PROBE_RC=$?
  case "$PAIR_PROBE_RC" in
    0) PAIR_RESUME_TWO=1 ;;
    3)
      # Ordinary prepared runs retain their reviewed launcher/trainer hashes.
      exec "$PY" scripts/selector_pair_resume_two.py frozen-run --root "$PAIR_ROOT" ;;
    *) exit "$PAIR_PROBE_RC" ;;
  esac
fi
if [ "$PAIR_RESUME_TWO" -eq 1 ]; then
  echo '[pair-resume] s1/t50 On-policy + s4/t100 Random: evaluate current saved finals; preserve the other 40 branches'
elif [ "$MODE" = run ] || [ "$MODE" = develop ]; then
  # No arguments needed: create the default setup and validate real inputs,
  # or resume the existing frozen request before admitting any GPU work.
  pair_cpu_step ensure-prepared
else
  pair_cpu_step check-code
fi
export OM_WORK="$WORK" OUT_ROOT="$PAIR_ROOT"
source scripts/_e5_node.sh
export E5_FORCE=0
if [ "$PAIR_RESUME_TWO" -eq 1 ]; then
  # This evaluation path must not repair or terminate source workers.
  e5_recover_pair_gpu() { return 0; }
  e5_cleanup_lock_helpers() { return 0; }
fi
e5_acquire_node
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t PAIR_GPUS < <(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader)
  export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${PAIR_GPUS[*]}")"
fi
IFS=, read -r -a PAIR_GPUS <<< "$CUDA_VISIBLE_DEVICES"
[ "${#PAIR_GPUS[@]}" -eq 4 ] || { echo '[abort] four allocated GPUs required'; exit 2; }
PAIR_MEMORY=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
while read -r used; do
  [[ "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || { echo '[abort] invalid GPU status'; exit 2; }
  [ "$used" -le 4000 ] || { echo '[busy] allocated GPU occupied; no existing job was stopped'; exit 75; }
done <<< "$PAIR_MEMORY"
PAIR_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$WORK/runtime-deps")
export PYTHONPATH="$PAIR_VERIFY_PATH:$PYTHONPATH" OM_MATH_VERIFIER=math_verify OM_NODE_LOCK_HELD=1
source scripts/_selection_worker.sh
PAIR_WORKER_RC=0
if [ "$PAIR_RESUME_TWO" -eq 1 ]; then
  selection_run_worker "$PY" scripts/selector_pair_resume_two.py run --root "$PAIR_ROOT" || PAIR_WORKER_RC=$?
else
  selection_run_worker "$PY" src/selector_pair_gpu.py "$MODE" --root "$PAIR_ROOT" || PAIR_WORKER_RC=$?
fi
exit "$PAIR_WORKER_RC"
