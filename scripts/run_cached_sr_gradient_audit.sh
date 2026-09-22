#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${SR_AUDIT_PYTHON:-}"
if [[ -z "$PY" ]]; then
  WORK="${OM_WORK:-/group-volume/minsoo3.kim/offpolicy-misranking}"
  for candidate in "${VENV_DIR:-$WORK/.venv-cu126}/bin/python" "$ROOT/.venv-cu126/bin/python" "$ROOT/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then
      PY="$candidate"
      break
    fi
  done
fi
PY="${PY:-python3}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
exec "$PY" "$ROOT/src/cached_sr_gradient_audit.py" "$@"
