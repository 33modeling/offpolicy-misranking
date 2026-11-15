#!/bin/sh
# Refresh the Llama result file (no training) and save the update counts to exports/.
set -eu
cd "$(dirname "$0")/.."
scripts/run_srgc_llama31.sh results >/dev/null 2>&1 || :
exec python3 scripts/check_llama_steps.py
