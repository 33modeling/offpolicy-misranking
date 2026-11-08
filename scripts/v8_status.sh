#!/bin/sh
# 읽기 전용 V8 현황: sh scripts/v8_status.sh  (계속 보기: sh scripts/v8_status.sh --watch)
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then PY=${VENV_DIR:-$WORK/.venv-cu126}/bin/python; fi
exec "$PY" "$HERE/v8_status.py" "$@"
