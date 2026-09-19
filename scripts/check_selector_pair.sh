#!/usr/bin/env bash
# Read-only lock evidence, saved in one <=4 KiB TXT. No worker imports or setup.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -B scripts/selector_pair_diagnostic.py "$@"
