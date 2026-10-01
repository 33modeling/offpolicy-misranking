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
# Training children load the model with this attention kernel (recorded in each checkpoint);
# sdpa is several times faster than the frozen runner's eager default for 2048-token rollouts.
export SRGC_ATTENTION=${SRGC_ATTENTION:-sdpa}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENBLAS_DEFAULT_NUM_THREADS=1 GOTO_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export OMP_THREAD_LIMIT=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

if [ "$MODE" = run ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0,1,2,3}
    # Reduce allocator fragmentation across thousands of variable-length rollouts.
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    # Never kill an existing experiment when another worker is started.
    # Plan conflicts park before admission; GPU ownership is checked by the worker.
    # Failed tasks are retried automatically (two minutes apart) up to 50 attempts,
    # so a transient fault never leaves a task parked as attempts_exhausted.
    # SRGC_MAX_ATTEMPTS overrides the limit.
    if [ "$DATASET" = all ]; then
        # One worker per node serves both queues: MATH first, MBPP when MATH has nothing claimable.
        set -- worker --dataset math --with-dataset mbpp --retry-failed --max-attempts "${SRGC_MAX_ATTEMPTS:-50}" --retry-delay 120
    else
        set -- worker --dataset "$DATASET" --retry-failed --max-attempts "${SRGC_MAX_ATTEMPTS:-50}" --retry-delay 120
    fi
    [ -z "${SRGC_RUN_NAME:-}" ] || set -- "$@" --fresh "$SRGC_RUN_NAME"
    # The worker is supervised: when it dies for any reason other than a stop or an
    # interrupt (GPUs taken away by the operator, driver/NCCL hiccup, node hiccup), it is
    # restarted after SRGC_WORKER_RESTART_DELAY seconds, up to SRGC_WORKER_RESTARTS times.
    # A restarted worker resumes from the queue receipts, per-update checkpoints and the
    # rollout cache, so nothing finished is repeated.
    attempt=0
    while :; do
        "$PY" scripts/run_srgc_rebuttal.py "$@" && rc=0 || rc=$?   # set -e must not abort the supervisor
        [ "$rc" -ne 0 ] || exit 0
        case "$rc" in 130|143|2) exit "$rc" ;; esac    # Ctrl-C, SIGTERM, usage error: do not loop
        attempt=$((attempt + 1))
        [ "$attempt" -lt "${SRGC_WORKER_RESTARTS:-1000}" ] || exit "$rc"
        printf '[worker-restart] worker exited with %s; restarting in %ss (restart %s)\n' \
            "$rc" "${SRGC_WORKER_RESTART_DELAY:-90}" "$attempt" >&2
        sleep "${SRGC_WORKER_RESTART_DELAY:-90}"
    done
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
