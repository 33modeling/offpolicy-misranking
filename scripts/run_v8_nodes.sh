#!/bin/sh
# No arguments: reuse the existing SRGC environment and dispatch both cohorts.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$HERE/v8_bootstrap.py" "$@"
