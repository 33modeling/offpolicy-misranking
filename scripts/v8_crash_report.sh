#!/bin/sh
# Read-only V8 native-crash report: sh scripts/v8_crash_report.sh -> screen summary + ~/v8-crash-report.txt
set -eu
# Code checkout stays code-only (no __pycache__ next to the scripts); CDPATH would corrupt HERE.
export PYTHONDONTWRITEBYTECODE=1
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then PY=${VENV_DIR:-$WORK/.venv-cu126}/bin/python; fi
exec "$PY" "$HERE/v8_crash_report.py" "$@"
