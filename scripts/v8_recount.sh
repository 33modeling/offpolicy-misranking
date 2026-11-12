#!/bin/sh
# 학습 없이 V8 비용·예산 시점을 현재 코드로 다시 집계: sh scripts/v8_recount.sh -> 화면 + ~/v8-recount.txt
# 원래 결과(V8_ROOT)는 읽기만 하고, 새 집계는 ~/v8-recount/<cohort>/analysis 에 쓴다.
set -eu
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
PY=${V8_PYTHON:-${PAIR_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}}
command -v "$PY" >/dev/null 2>&1 || PY=python3
export V8_CODE_ROOT=${V8_CODE_ROOT:-$(CDPATH= cd -- "$HERE/.." && pwd)}
exec "$PY" "$HERE/v8_recount.py" "$@"
