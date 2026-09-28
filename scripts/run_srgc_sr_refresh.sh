#!/bin/sh
# SR with refreshed success rates (Limitations: "cache refresh ... unevaluated").
#   sh scripts/run_srgc_sr_refresh.sh math|mbpp <seed> [candidates|pool|switch_repeat]   # one seed on this node (4 GPUs)
#   sh scripts/run_srgc_sr_refresh.sh math|mbpp results                    # per-seed rewards next to the recorded arms
# The seed's shared prefix must already be complete in the group-storage run root.
set -eu
cd "$(dirname "$0")/.."
DATASET=${1:-}; TARGET=${2:-}; SCOPE=${3:-candidates}
case "$DATASET" in math|mbpp) ;; *) echo "usage: sh scripts/run_srgc_sr_refresh.sh math|mbpp <seed>|results [candidates|pool]" >&2; exit 2 ;; esac
case "$SCOPE" in candidates|pool|switch_repeat) ;; *) echo "third argument must be candidates, pool or switch_repeat" >&2; exit 2 ;; esac
if [ "$SCOPE" = switch_repeat ]; then ARM_ARGS="--arm switch_repeat"; else ARM_ARGS="--scope $SCOPE"; fi
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
case "$DATASET" in math) EXPLICIT=${PAIR_PYTHON:-} ;; mbpp) EXPLICIT=${SWITCH_PYTHON:-} ;; esac
PY=${EXPLICIT:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
command -v "$PY" >/dev/null 2>&1 || PY=python3
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
PLAN=$("$PY" - "$DATASET" <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from srgc_shared_storage import route_plan
name = "mbpp_seeds.json" if sys.argv[1] == "mbpp" else "additional_seeds.json"
print(route_plan(Path("srgc_rebuttal/experiments") / name, writing=False))
PYEOF
)
if [ "$TARGET" = results ]; then
    export CUDA_VISIBLE_DEVICES=""
    exec "$PY" scripts/srgc_sr_refresh.py results --plan "$PLAN"
fi
case "$TARGET" in ''|*[!0-9]*) echo "seed must be an integer" >&2; exit 2 ;; esac
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0,1,2,3}
if command -v nvidia-smi >/dev/null 2>&1 && [ -z "${SRGC_SKIP_GPU_CLEANUP:-}" ]; then
    uid=$(id -u)
    for pid in $(timeout 20 nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || true); do
        [ "$(stat -c %u "/proc/$pid" 2>/dev/null || true)" = "$uid" ] && kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 5
fi
echo "[sr-refresh] dataset=$DATASET seed=$TARGET scope=$SCOPE plan=$PLAN" >&2
exec "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 --max_restarts=0 \
    scripts/srgc_sr_refresh.py run --plan "$PLAN" --seed "$TARGET" $ARM_ARGS
