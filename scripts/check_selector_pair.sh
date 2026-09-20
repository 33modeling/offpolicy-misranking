#!/usr/bin/env bash
# Read-only evidence: one <=4 KiB TXT, or <=1 MiB with --costs. No worker setup.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 -B scripts/selector_pair_diagnostic.py "$@"
