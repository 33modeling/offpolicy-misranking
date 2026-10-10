#!/bin/sh
# Default: collect MATH seed 7 at t0. Reports require an explicit report action.
set -eu
cd "$(dirname "$0")/.."
DATASET=${1:-math}
ACTION=${2:-collect}
case "$DATASET" in math|mbpp|all) ;; *) echo "usage: sh scripts/run_srgc_information.sh [math|mbpp|all] [collect|status|report options]" >&2; exit 2 ;; esac
case "$ACTION" in collect|status|report) ;; *) echo "action must be collect, status or report" >&2; exit 2 ;; esac
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
SOURCE_ROOT=${SRGC_STORAGE_ROOT:-$WORK/srgc-rebuttal}

default_plan() {
    case "$1" in
        math) printf '%s\n' "$SOURCE_ROOT/experiments/pair_seeds.json" ;;
        mbpp) printf '%s\n' "$SOURCE_ROOT/experiments/mbpp_pair_seeds.json" ;;
    esac
}

default_input() {
    python3 - "$1" <<'PY'
import sys
from pathlib import Path
from srgc_rebuttal.plan import input_path, load_plan
try:
    plan_path = Path(sys.argv[1])
    plan = load_plan(plan_path)
    if 7 not in plan["seeds"]:
        raise ValueError("default seed 7 is absent from the plan")
    source = input_path(plan_path, plan, 7)
    if not source.is_file():
        raise FileNotFoundError(f"input not found: {source}")
    print(source)
except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(2)
PY
}

if [ "$#" -le 1 ]; then
    if [ "$DATASET" = all ]; then
        # Resolve both inputs before starting either GPU measurement.
        for SRGC_DATASET in math mbpp; do
            default_input "$(default_plan "$SRGC_DATASET")" >/dev/null
        done
        for SRGC_DATASET in math mbpp; do
            sh scripts/run_srgc_information.sh "$SRGC_DATASET"
        done
        exit 0
    fi
    SRGC_PLAN=$(default_plan "$DATASET")
    SRGC_INPUT=$(default_input "$SRGC_PLAN")
    set -- "$DATASET" collect --plan "$SRGC_PLAN" --inputs "$SRGC_INPUT" \
        --seed 7 --stage 0 --attention sdpa \
        --output "$WORK/selection-information/$DATASET/seed-7/t0"
fi

if [ "$ACTION" = collect ]; then
    case "$DATASET" in mbpp) EXPLICIT=${SWITCH_PYTHON:-} ;; *) EXPLICIT=${PAIR_PYTHON:-} ;; esac
    PY=${EXPLICIT:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
    if ! command -v "$PY" >/dev/null 2>&1; then
        [ -z "$EXPLICIT" ] || { printf 'Python not found: %s\n' "$PY" >&2; exit 2; }
        PY=python3
    fi
else
    PY=python3
    export CUDA_VISIBLE_DEVICES=""
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
exec "$PY" -m srgc_research.information_cli "$@"
