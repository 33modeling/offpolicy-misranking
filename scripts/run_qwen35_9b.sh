#!/usr/bin/env bash
# Explicit 9B replication; never implicitly launch the retained 27B study.
set -euo pipefail
cd "$(dirname "$0")/.."
# No argument = run. `run` already performs every check (snapshot seal, FLA,
# smoke) before the matrix, so there is no separate step to remember.
MODE=${1:-run}
case "$MODE" in
  prepare|check|run)
    [ "$#" -le 1 ] || { echo "usage: $0 [prepare|check|run]"; exit 2; }
    exec bash scripts/run_additional_experiments.sh "--$MODE" qwen35
    ;;
  *) echo "usage: bash scripts/run_qwen35_9b.sh [prepare|check|run]  (기본 run)"; exit 2 ;;
esac
