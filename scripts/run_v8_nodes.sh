#!/bin/sh
# No arguments: reuse the existing SRGC environment and dispatch both cohorts.
#   sh scripts/run_v8_nodes.sh            worker for this node (same command on every node)
#   sh scripts/run_v8_nodes.sh --status   queue state of both cohorts (read-only)
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
exec "$PY" "$HERE/v8_supervisor.py" "$@"
