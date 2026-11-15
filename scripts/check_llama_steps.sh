#!/bin/sh
# Refresh the Llama result file (no training) and print only the update counts.
set -eu
cd "$(dirname "$0")/.."
if ! scripts/run_srgc_llama31.sh results >/dev/null 2>&1; then
    echo "export failed; reading the existing result file"
fi
exec python3 scripts/check_llama_steps.py
