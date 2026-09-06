#!/usr/bin/env bash
# Explicit 9B replication; never implicitly launch the retained 27B study.
set -euo pipefail
cd "$(dirname "$0")/.."
# No argument = run. `run` already performs every check (snapshot seal, FLA,
# smoke) before the matrix, so there is no separate step to remember.
MODE=${1:-run}
# Never mutate the checkout from a launch/diagnostic command.
# Update explicitly in a separate idle checkout after reviewing local changes.
echo "[code] $(git rev-parse --short HEAD) (no automatic fetch/merge/reset)"
export OM_ALLOW_UNPINNED_SNAPSHOT=0 OM_TRUST_LOCAL_SNAPSHOT=0
case "$MODE" in
  doctor) exec bash scripts/doctor_qwen35.sh ;;
  status) exec bash scripts/status_qwen35.sh ;;
  prepare|check|run)
    [ "$#" -le 1 ] || { echo "usage: $0 [prepare|check|run|status|doctor]"; exit 2; }
    exec bash scripts/run_additional_experiments.sh "--$MODE" qwen35
    ;;
  *) echo "usage: bash scripts/run_qwen35_9b.sh [run|check|status|doctor|prepare]  (default run; status = one-screen progress; prepare = download, needs internet)"; exit 2 ;;
esac
