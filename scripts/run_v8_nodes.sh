#!/bin/sh
# No arguments: reuse the existing SRGC environment and dispatch both cohorts.
#   sh scripts/run_v8_nodes.sh            worker for this node (same command on every node)
#   sh scripts/run_v8_nodes.sh --status   queue state of both cohorts (read-only, JSON)
#   sh scripts/run_v8_nodes.sh status     status table like `run_srgc.sh <dataset> status` (read-only;
#                                         add --watch to refresh). Same as sh scripts/v8_status.sh
set -eu
# CDPATH would make `cd` print the directory and corrupt HERE.
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
# The supervisor/bootstrap need only the standard library: use the node's python3, or the existing
# SRGC venv (same default path as scripts/run_srgc.sh) when python3 is not installed.
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then PY=${VENV_DIR:-$WORK/.venv-cu126}/bin/python; fi
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "V8 startup: no python3 on PATH and no existing venv at ${VENV_DIR:-$WORK/.venv-cu126}" >&2
    exit 2
fi
# `status` as a word (the run_srgc.sh habit) used to be taken as a cohort name and failed with
# "configs/cohorts/status.json not found". It now shows the status table instead.
for arg in "$@"; do
    if [ "$arg" = status ]; then
        WATCH=
        for a in "$@"; do [ "$a" = --watch ] && WATCH=--watch; done
        exec "$PY" "$HERE/v8_status.py" $WATCH
    fi
done
if [ $# -gt 0 ]; then
    case "$1" in
        -*|all|olmo_math|qwen_mbpp) ;;
        *)
            CFG="$HERE/../v8/experiments/configs/cohorts/$1.json"
            if [ ! -f "$CFG" ]; then
                echo "V8 startup: unknown command or cohort '$1'." >&2
                echo "  worker (both cohorts):  sh scripts/run_v8_nodes.sh" >&2
                echo "  one cohort:             sh scripts/run_v8_nodes.sh olmo_math | qwen_mbpp" >&2
                echo "  status table:           sh scripts/run_v8_nodes.sh status   (or --status for JSON)" >&2
                exit 2
            fi ;;
    esac
fi
exec "$PY" "$HERE/v8_supervisor.py" "$@"
