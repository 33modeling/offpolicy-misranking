#!/bin/sh
# V8 핵심 숫자만: sh scripts/v8_summary.sh  -> 화면 출력 + ~/v8-summary.txt (읽기 전용)
set -eu
export PYTHONDONTWRITEBYTECODE=1
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then PY=${VENV_DIR:-$WORK/.venv-cu126}/bin/python; fi
exec "$PY" "$HERE/v8_summary.py" "$@"
