#!/bin/sh
# SR with refreshed success rates (Limitations: "cache refresh ... unevaluated").
#   sh scripts/run_srgc_sr_refresh.sh math|mbpp <seed> [candidates|pool|switch_repeat]   # one seed on this node (4 GPUs)
#   sh scripts/run_srgc_sr_refresh.sh math|mbpp results                    # per-seed rewards next to the recorded arms
# The seed's shared prefix must already be complete in the group-storage run root.
set -eu
cd "$(dirname "$0")/.."
DATASET=${1:-}; TARGET=${2:-}; SCOPE=${3:-candidates}
case "$DATASET" in math|mbpp) ;; *) echo "usage: sh scripts/run_srgc_sr_refresh.sh math|mbpp <seed>|results [candidates|pool]" >&2; exit 2 ;; esac
case "$SCOPE" in
    candidates|pool) ARM_ARGS="--scope $SCOPE" ;;
    switch_repeat) ARM_ARGS="--arm switch_repeat" ;;
    switch_fixed[0-9]*) ARM_ARGS="--arm $SCOPE" ;;
    *) echo "third argument must be candidates, pool, switch_repeat or switch_fixed<N> (e.g. switch_fixed100)" >&2; exit 2 ;;
esac
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
case "$DATASET" in math) EXPLICIT=${PAIR_PYTHON:-} ;; mbpp) EXPLICIT=${SWITCH_PYTHON:-} ;; esac
PY=${EXPLICIT:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
if ! command -v "$PY" >/dev/null 2>&1; then
    [ -z "$EXPLICIT" ] || { printf 'Python not found: %s\n' "$PY" >&2; exit 2; }
    PY=python3
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PLAN=$("$PY" - "$DATASET" <<'PYEOF'
import os
import sys
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from srgc_shared_storage import route_plan
from srgc_pair_inputs import default_plan
source = default_plan(Path.cwd(), sys.argv[1], os.environ, writing=False)
print(route_plan(source, writing=False))
PYEOF
)
if [ "$TARGET" = results ]; then
    export CUDA_VISIBLE_DEVICES=""
    exec "$PY" scripts/srgc_sr_refresh.py results --plan "$PLAN"
fi
case "$TARGET" in ''|*[!0-9]*) echo "seed must be an integer" >&2; exit 2 ;; esac
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Never share a node with a queue worker: an extra arm needs all four GPUs.
if pgrep -u "$(id -u)" -f 'run_srgc_rebuttal.py worker' >/dev/null 2>&1; then
    echo "[abort] a queue worker (run_srgc.sh ... run) is running on this node; stop it first or use another node" >&2
    exit 75
fi
if command -v nvidia-smi >/dev/null 2>&1 && [ -z "${SRGC_SKIP_GPU_CLEANUP:-}" ]; then
    memory=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null) || {
        echo "[abort] nvidia-smi could not report GPU memory" >&2; exit 1; }
    busy=$(printf '%s\n' "$memory" | awk '$1 > 2000 {n++} END {print n+0}')
    if [ "$busy" -ne 0 ]; then
        echo "[abort] GPUs are in use on this node (MiB per GPU): $(printf '%s' "$memory" | tr '\n' ' ')" >&2
        timeout 20 nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader >&2 || true
        exit 75
    fi
fi
echo "[sr-refresh] dataset=$DATASET seed=$TARGET scope=$SCOPE plan=$PLAN" >&2
exec "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 --max_restarts=0 \
    scripts/srgc_sr_refresh.py run --plan "$PLAN" --seed "$TARGET" $ARM_ARGS
