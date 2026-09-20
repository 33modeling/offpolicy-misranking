#!/usr/bin/env bash
# Read-only lock and cost evidence: one <=1 MiB TXT. No worker setup.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -B scripts/selector_pair_diagnostic.py --costs "$@"
