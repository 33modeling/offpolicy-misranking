#!/usr/bin/env bash
# Keep the frozen GPU launcher unchanged; export current paper data on CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_paper_results.sh results pair "$@"
