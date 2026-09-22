#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname -- "$SCRIPT_DIR")"
PY="${HISTORY_EXPORT_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for candidate in "${VENV_DIR:-$ROOT/.venv-cu126}/bin/python" "$ROOT/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then
      PY="$candidate"
      break
    fi
  done
fi
PY="${PY:-python3}"
"$PY" -c 'import sys; sys.exit("Python 3.9+ required; set HISTORY_EXPORT_PYTHON") if sys.version_info < (3, 9) else None'
exec "$PY" "$SCRIPT_DIR/export_continuous_selector_history.py" "$@"
