#!/bin/sh
# Run from the pinned paper checkout on EVERY allocated 4xH100 node.
# One command on every node: automatically shares OLMo MATH and Qwen MBPP.
# Example: sh "$EXPORT_CODE_ROOT/scripts/run_v8_nodes.sh"
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
CONTROL_PYTHON=${V8_PYTHON:-${PAIR_PYTHON:-${SWITCH_PYTHON:-python3}}}
exec "$CONTROL_PYTHON" "$HERE/srgc_v8_nodes.py" "$@"
