#!/bin/sh
# No arguments: reuse SRGC, verify this node and retry failed tasks with saved checkpoints.
#   sh scripts/run_v8_nodes.sh            worker for this node (same command on every node)
#   sh scripts/run_v8_nodes.sh --status   queue state of all three cohorts (read-only, JSON)
#   sh scripts/run_v8_nodes.sh status     status table like `run_srgc.sh <dataset> status` (read-only;
#                                         add --watch to refresh). Same as sh scripts/v8_status.sh
#   sh scripts/run_v8_nodes.sh stop       stop this node's worker (same as sh scripts/v8_stop.sh)
#   cohort words: olmo_math|math, qwen_mbpp|mbpp, olmo_mbpp, all; mode words: run (default), status, stop
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
# Same words as scripts/run_srgc.sh; full cohort names also select olmo_mbpp.
# Words used to be taken as a cohort name ("status" -> configs/cohorts/status.json not found).
usage() {
    echo "usage: sh scripts/run_v8_nodes.sh [olmo_math|qwen_mbpp|olmo_mbpp|all] [run|status|stop] [--watch [SECONDS]] [--new-queue] [--status] [--retry-failed]" >&2
    echo "  sh scripts/run_v8_nodes.sh              worker for olmo_math, qwen_mbpp and olmo_mbpp (same command on every node)" >&2
    echo "  sh scripts/run_v8_nodes.sh status       status table like run_srgc.sh status (--watch to refresh)" >&2
    echo "  sh scripts/run_v8_nodes.sh stop         stop this node's V8 worker (checkpoints and queue kept)" >&2
    echo "  math = olmo_math, mbpp = qwen_mbpp" >&2
}
COHORT=; MODE=; WATCH=; VALUE_NEXT=; WATCH_NEXT=; VIEW=
n=$#
for arg in "$@"; do
    if [ -n "$VALUE_NEXT" ]; then VALUE_NEXT=; set -- "$@" "$arg"; continue; fi
    if [ -n "$WATCH_NEXT" ]; then
        WATCH_NEXT=
        # `--watch 30` (seconds) as in scripts/v8_status.sh; any other word is parsed normally below.
        case "$arg" in ''|*[!0-9.]*|.|*.*.*) ;; *) WATCH="--watch $arg"; continue ;; esac
    fi
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --gpus|--root|--python|--poll-seconds|--runner-root|--max-attempts|--retry-delay) VALUE_NEXT=1; set -- "$@" "$arg" ;;
        --watch) WATCH=--watch; WATCH_NEXT=1 ;;
        --watch=*) WATCH="--watch ${arg#--watch=}" ;;
        --all-nodes|--simple|--classic) VIEW="$VIEW $arg" ;;
        -*) set -- "$@" "$arg" ;;
        olmo_math|math) [ -z "$COHORT" ] && COHORT=olmo_math || { usage; exit 2; } ;;
        qwen_mbpp|mbpp) [ -z "$COHORT" ] && COHORT=qwen_mbpp || { usage; exit 2; } ;;
        olmo_mbpp) [ -z "$COHORT" ] && COHORT=olmo_mbpp || { usage; exit 2; } ;;
        all) [ -z "$COHORT" ] && COHORT=all || { usage; exit 2; } ;;
        run|status|stop) [ -z "$MODE" ] && MODE=$arg || { usage; exit 2; } ;;
        *)
            if [ -z "$COHORT" ] && [ -f "$HERE/../v8/experiments/configs/cohorts/$arg.json" ]; then COHORT=$arg
            else echo "V8 startup: unknown command or cohort '$arg'." >&2; usage; exit 2
            fi ;;
    esac
done
shift "$n"
case "${MODE:-run}" in
    status)
        if [ -n "$COHORT" ] && [ "$COHORT" != all ]; then exec "$PY" "$HERE/v8_status.py" --cohort "$COHORT" $WATCH $VIEW; fi
        exec "$PY" "$HERE/v8_status.py" $WATCH $VIEW ;;
    stop)
        if [ -n "$COHORT" ] || [ "$#" -gt 0 ]; then
            echo "V8 stop: stops every V8 process of this node (all cohorts, checkpoints and queue kept); the words/options '${COHORT:+$COHORT }$*' are not used by stop." >&2
        fi
        exec sh "$HERE/v8_stop.sh" ;;
esac
# run: options (in their original order, values and spaces intact), then the cohort (none = all three).
if [ -n "$COHORT" ] && [ "$COHORT" != all ]; then set -- "$@" "$COHORT"; fi
exec "$PY" "$HERE/v8_supervisor.py" "$@"
