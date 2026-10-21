#!/bin/sh
# Compatibility alias for the existing Gemma launcher.
set -eu
cd "$(dirname "$0")/.."
export SRGC_GEMMA_LEGACY_PROTOCOL=0 SRGC_GEMMA_TEXT_PROTOCOL=1 SRGC_MODEL_CHAIN=0
exec sh scripts/run_srgc_gemma4.sh "$@"
