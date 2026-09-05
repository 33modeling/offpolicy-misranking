#!/usr/bin/env bash
# Pinned post-trained Qwen3.8 replication; no jobs start without an explicit mode.
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  prepare|check|run)
    [ "$#" -eq 1 ] || { echo "usage: $0 prepare|check|run"; exit 2; }
    exec bash scripts/run_additional_experiments.sh "--$1" qwen38
    ;;
  *) echo "usage: bash scripts/run_qwen38_27b.sh prepare|check|run"; exit 2 ;;
esac
