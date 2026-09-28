#!/bin/sh
set -eu
cd "$(dirname "$0")/.."

usage() {
    printf '%s\n' 'usage: sh scripts/run_srgc.sh math|mbpp|all [run|status|results|costs|backup|backup-watch]'
}

DATASET=${1:-}
MODE=${2:-run}
case "$DATASET" in
    -h|--help) usage; exit 0 ;;
    math|mbpp|all) ;;
    *) usage >&2; exit 2 ;;
esac
[ "$#" -le 2 ] || { usage >&2; exit 2; }
case "$MODE" in run|status|results|costs|backup|backup-watch) ;; *) usage >&2; exit 2 ;; esac

WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
case "$DATASET" in
    math|all) EXPLICIT=${PAIR_PYTHON:-} ;;
    mbpp) EXPLICIT=${SWITCH_PYTHON:-} ;;
esac
PY=${EXPLICIT:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
if ! command -v "$PY" >/dev/null 2>&1; then
    [ -z "$EXPLICIT" ] || { printf 'Python not found: %s\n' "$PY" >&2; exit 2; }
    PY=python3
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENBLAS_DEFAULT_NUM_THREADS=1 GOTO_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export OMP_THREAD_LIMIT=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

# Same startup rule as scripts/run_olmo3_rlzero.sh: before a worker starts, every GPU
# compute process owned by this user is terminated (TERM, then KILL after
# SRGC_GPU_CLEANUP_TIMEOUT seconds) and the GPUs must be below 2000 MiB used.
gpu_compute_pids() {
    timeout 20 nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | tr -d ' ' | grep -E '^[0-9]+$' || true
}

clear_gpu_memory() {
    uid=$(id -u); terminated=0
    for pid in $(gpu_compute_pids); do
        [ "$(stat -c %u "/proc/$pid" 2>/dev/null || true)" = "$uid" ] || continue
        kill -TERM "$pid" 2>/dev/null || true
        terminated=$((terminated + 1))
    done
    [ "$terminated" -eq 0 ] || printf '[startup-cleanup] TERM sent to %s GPU process(es) left on this node\n' "$terminated" >&2
    deadline=$(( $(date +%s) + ${SRGC_GPU_CLEANUP_TIMEOUT:-60} ))
    while :; do
        remaining=""
        for pid in $(gpu_compute_pids); do
            [ "$(stat -c %u "/proc/$pid" 2>/dev/null || true)" = "$uid" ] && remaining="$remaining $pid"
        done
        [ -n "$remaining" ] || break
        [ "$(date +%s)" -lt "$deadline" ] || break
        sleep 1
    done
    for pid in $remaining; do kill -KILL "$pid" 2>/dev/null || true; done
    [ -z "$remaining" ] || { printf '[startup-cleanup] KILL sent to:%s\n' "$remaining" >&2; sleep 2; }
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then set -- -i "$CUDA_VISIBLE_DEVICES"; else set --; fi
    memory=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits "$@" 2>/dev/null) || {
        printf '[abort] nvidia-smi could not report GPU memory\n' >&2; exit 1; }
    busy=$(printf '%s\n' "$memory" | awk '$1 > 2000 {n++} END {print n+0}')
    if [ "$busy" -ne 0 ]; then
        printf '[abort] GPU memory is still in use after cleanup (MiB per GPU):\n' >&2
        printf '%s\n' "$memory" >&2
        timeout 20 nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader >&2 || true
        exit 1
    fi
    printf '[startup-cleanup] GPU memory clear (MiB per GPU): %s\n' "$(printf '%s' "$memory" | tr '\n' ' ')" >&2
}

if [ "$MODE" = run ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0,1,2,3}
    if [ -n "${SRGC_SKIP_GPU_CLEANUP:-}" ]; then
        printf '[startup-cleanup] skipped (SRGC_SKIP_GPU_CLEANUP set)\n' >&2
    elif command -v nvidia-smi >/dev/null 2>&1; then
        clear_gpu_memory
    else
        printf '[startup-cleanup] nvidia-smi not found; skipping GPU cleanup\n' >&2
    fi
    # Failed tasks are retried automatically (two minutes apart) up to 50 attempts,
    # so a transient fault never leaves a task parked as attempts_exhausted.
    # SRGC_MAX_ATTEMPTS overrides the limit.
    if [ "$DATASET" = all ]; then
        # One worker per node serves both queues: MATH first, MBPP when MATH has nothing claimable.
        set -- worker --dataset math --with-dataset mbpp --retry-failed --max-attempts "${SRGC_MAX_ATTEMPTS:-50}" --retry-delay 120
        [ -z "${SRGC_RUN_NAME:-}" ] || { printf '%s\n' 'SRGC_RUN_NAME is not supported with all' >&2; exit 2; }
    else
        set -- worker --dataset "$DATASET" --retry-failed --max-attempts "${SRGC_MAX_ATTEMPTS:-50}" --retry-delay 120
    fi
    if [ -n "${SRGC_RUN_NAME:-}" ]; then
        exec "$PY" scripts/run_srgc_rebuttal.py "$@" --fresh "$SRGC_RUN_NAME"
    fi
    exec "$PY" scripts/run_srgc_rebuttal.py "$@"
fi
export CUDA_VISIBLE_DEVICES=""
if [ "$DATASET" = all ]; then
    rc=0
    for each in math mbpp; do
        "$PY" scripts/run_srgc_rebuttal.py "$MODE" --dataset "$each" || rc=$?
    done
    exit "$rc"
fi
exec "$PY" scripts/run_srgc_rebuttal.py "$MODE" --dataset "$DATASET"
