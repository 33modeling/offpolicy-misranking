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
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

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

default_plan() {
    "$PY" - "$1" "$WORK" "$SOURCE_ROOT" <<'PY'
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "scripts"))
from scripts.srgc_extra_plan import select_plan
from scripts.srgc_pair_inputs import PLANS
from scripts.srgc_shared_storage import route_plan, storage_root
from srgc_rebuttal.plan import input_path, load_plan

def select_seed_plan(candidate):
    if 7 not in load_plan(candidate)["seeds"]:
        raise ValueError("default seed 7 is absent from the plan")
    return select_plan(candidate, 7)

try:
    dataset = sys.argv[1]
    os.environ["OM_WORK"] = sys.argv[2]
    os.environ["SRGC_STORAGE_ROOT"] = sys.argv[3]
    _, source_root = storage_root(os.environ)
    templates = Path.cwd() / "srgc_rebuttal/experiments"
    names = PLANS[dataset][:2]
    plan_path = None
    # The original worker follows active pointers, including fresh/... plans.
    # Read-only routing keeps the run, cached inputs and original runtime intact.
    for name in names:
        if (source_root / f".{Path(name).stem}-active.json").is_file():
            plan_path = select_seed_plan(route_plan(templates / name, writing=False))
            break
    if plan_path is None:
        for name in names:
            candidate = source_root / "experiments" / name
            if candidate.is_file():
                plan_path = select_seed_plan(candidate)
                break
    if plan_path is None:
        for name in names:
            candidate = templates / name
            if candidate.is_file() and input_path(candidate, load_plan(candidate), 7).is_file():
                plan_path = candidate
                break
    if plan_path is None:
        raise FileNotFoundError(f"no saved {dataset} plan with seed-7 input found under {source_root} or {templates}")
    print(plan_path)
except (OSError, ValueError, KeyError, TypeError) as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(2)
PY
}

default_input() {
    "$PY" - "$1" <<'PY'
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

exec "$PY" -m srgc_research.information_cli "$@"
