#!/bin/sh
# V8 평가 1개 소요 시간과 남은 평가: sh scripts/v8_eval_time.sh  -> 화면 출력 + ~/v8-eval-time.txt (읽기 전용)
set -eu
export PYTHONDONTWRITEBYTECODE=1
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
PY=python3
if ! command -v "$PY" >/dev/null 2>&1; then PY=${VENV_DIR:-$WORK/.venv-cu126}/bin/python; fi
exec "$PY" "$HERE/v8_eval_time.py" "$@"
