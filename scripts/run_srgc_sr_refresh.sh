#!/bin/sh
# Extra arms forked from a seed's verified shared prefix (one seed, one arm, this node's 4 GPUs):
#   sh scripts/run_srgc_sr_refresh.sh math|mbpp <seed> [candidates|pool|sr_hold]        # SR success-rate refresh / cached-SR control
#   sh scripts/run_srgc_sr_refresh.sh math|mbpp <seed> switch_repeat|switch_fixed<N>    # repeated / fixed-schedule transitions
#   sh scripts/run_srgc_sr_refresh.sh math|mbpp <seed> direction_removed|direction_magnitude|direction_replaced
#   sh scripts/run_srgc_sr_refresh.sh math|mbpp <seed> replicate<k>-<random|sr|on_policy|switch|switch_fixed<N>>
#   sh scripts/run_srgc_sr_refresh.sh math|mbpp results [--json]           # per-seed rewards next to the recorded arms
# The seed's shared prefix must already be complete in the group-storage run root.
# Automatic planned repeats (same command on each empty node):
#   sh scripts/run_srgc_sr_refresh.sh all replicate
set -eu
cd "$(dirname "$0")/.."
DATASET=${1:-}; TARGET=${2:-}; SCOPE=${3:-candidates}
[ "$#" -le 3 ] || { echo "too many arguments" >&2; exit 2; }
case "$DATASET" in math|mbpp) ;; all) [ "$TARGET" = replicate ] || { echo "all supports replicate only" >&2; exit 2; } ;; *) echo "usage: sh scripts/run_srgc_sr_refresh.sh math|mbpp <seed>|results [arm], or all|math|mbpp replicate" >&2; exit 2 ;; esac
if [ "$TARGET" = replicate ]; then
    [ "$#" -eq 2 ] || { echo "usage: sh scripts/run_srgc_sr_refresh.sh all|math|mbpp replicate" >&2; exit 2; }
elif [ "$TARGET" = results ]; then
    case "${3:-}" in ''|--json) ;; *) echo "usage: math|mbpp results [--json]" >&2; exit 2 ;; esac
else
case "$TARGET" in ''|*[!0-9]*) echo "seed must be an integer" >&2; exit 2 ;; esac
case "$SCOPE" in
    candidates|pool) ARM_ARGS="--scope $SCOPE" ;;
    sr_hold|switch_repeat) ARM_ARGS="--arm $SCOPE" ;;
    switch_fixed[0-9]*) ARM_ARGS="--arm $SCOPE" ;;
    direction_removed|direction_magnitude|direction_replaced) ARM_ARGS="--arm $SCOPE" ;;
    replicate[0-9]*-*) ARM_ARGS="--arm $SCOPE" ;;
    *) echo "third argument must be candidates, pool, sr_hold, switch_repeat, switch_fixed<N> (e.g. switch_fixed200), direction_removed|direction_magnitude|direction_replaced, or replicate<k>-<random|sr|on_policy|switch|switch_fixed<N>> (e.g. replicate1-switch)" >&2; exit 2 ;;
esac
fi
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
case "$DATASET" in math|all) EXPLICIT=${PAIR_PYTHON:-} ;; mbpp) EXPLICIT=${SWITCH_PYTHON:-} ;; esac
PY=${EXPLICIT:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
if ! command -v "$PY" >/dev/null 2>&1; then
    [ -z "$EXPLICIT" ] || { printf 'Python not found: %s\n' "$PY" >&2; exit 2; }
    PY=python3
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Extra arms inherit their prefix's attention kernel; resumes keep their own
# saved kernel. An environment default must not change an existing experiment.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
if [ "$TARGET" = replicate ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0,1,2,3}
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    exec "$PY" scripts/srgc_replicate_worker.py --dataset "$DATASET"
fi
PLAN=$("$PY" - "$DATASET" "$TARGET" <<'PYEOF'
import os
import sys
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from srgc_shared_storage import route_plan
from srgc_pair_inputs import default_plan
from scripts.srgc_extra_plan import select_plan
source = default_plan(Path.cwd(), sys.argv[1], os.environ, writing=False)
target = sys.argv[2]
plan = route_plan(source, writing=False)
print(select_plan(plan, None if target == "results" else int(target)))
PYEOF
)
if [ "$TARGET" = results ]; then
    export CUDA_VISIBLE_DEVICES=""
    if [ "${3:-}" = --json ]; then
        exec "$PY" scripts/srgc_sr_refresh.py results --plan "$PLAN" --json
    fi
    exec "$PY" scripts/srgc_sr_refresh.py results --plan "$PLAN"
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Never share a node with a queue worker: an extra arm needs all four GPUs.
if pgrep -u "$(id -u)" -f 'run_srgc_rebuttal.py worker' >/dev/null 2>&1; then
    echo "[abort] a queue worker (run_srgc.sh ... run) is running on this node; stop it first or use another node" >&2
    exit 75
fi
# The Python launcher reaps orphan ranks before checking occupancy, acquires
# shared task/device leases, and runs the same NCCL admission as P0.
echo "[sr-refresh] dataset=$DATASET seed=$TARGET scope=$SCOPE plan=$PLAN" >&2
# Supervised like the queue worker: a crash (OOM, NCCL, pre-empted GPUs) is retried after a
# delay and resumes from the arm's checkpoint and rollout cache; Ctrl-C/SIGTERM stop it.
attempt=0
child=
stop_child() {
    trap '' INT TERM
    if [ -n "$child" ]; then
        kill -TERM "$child" 2>/dev/null || :
        wait "$child" 2>/dev/null || :
    fi
    exit "$1"
}
trap 'stop_child 130' INT
trap 'stop_child 143' TERM
while :; do
    "$PY" scripts/srgc_extra_worker.py --plan "$PLAN" --seed "$TARGET" $ARM_ARGS &
    child=$!
    wait "$child" && rc=0 || rc=$?
    child=
    [ "$rc" -ne 0 ] || exit 0
    case "$rc" in 130|143|2|75) exit "$rc" ;; esac
    attempt=$((attempt + 1))
    [ "$attempt" -lt "${SRGC_WORKER_RESTARTS:-50}" ] || exit "$rc"
    printf '[sr-refresh] exited with %s; retrying in %ss (retry %s)\n' "$rc" "${SRGC_WORKER_RESTART_DELAY:-120}" "$attempt" >&2
    sleep "${SRGC_WORKER_RESTART_DELAY:-120}" &
    child=$!
    wait "$child"
    child=
done
