#!/usr/bin/env bash
# Explicit 9B replication; never implicitly launch the retained 27B study.
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  prepare|check|run)
    [ "$#" -eq 1 ] || { echo "usage: $0 prepare|check|run"; exit 2; }
    exec bash scripts/run_additional_experiments.sh "--$1" qwen35
    ;;
  *) echo "usage: bash scripts/run_qwen35_9b.sh prepare|check|run"; exit 2 ;;
esac
