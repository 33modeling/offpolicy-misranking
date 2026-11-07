#!/bin/sh
# Run from the pinned paper checkout on EVERY allocated 4xH100 node.
# Example: sh "$EXPORT_CODE_ROOT/scripts/run_v8_nodes.sh" olmo_math
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
: "${V8_PYTHON:?set V8_PYTHON to the existing H100 environment python}"
exec "$V8_PYTHON" "$HERE/srgc_v8_nodes.py" "$@"
