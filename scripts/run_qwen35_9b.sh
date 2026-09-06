#!/usr/bin/env bash
# Explicit 9B replication; never implicitly launch the retained 27B study.
set -euo pipefail
cd "$(dirname "$0")/.."
# No argument = run. `run` already performs every check (snapshot seal, FLA,
# smoke) before the matrix, so there is no separate step to remember.
MODE=${1:-run}
# Operator is often on a phone: make the bare command sufficient.
#  - pull the latest code first (ff-only, only if the checkout is clean; OLMo
#    workers run from node-local pinned clones, so this cannot disturb them)
#  - accept a snapshot uploaded from `main` (recorded as unverified provenance)
if [ -z "$(git status --porcelain -- src scripts configs requirements.txt)" ]; then
  git pull --ff-only -q 2>/dev/null || echo "[note] git pull failed; running current code $(git rev-parse --short HEAD)"
fi
export OM_ALLOW_UNPINNED_SNAPSHOT="${OM_ALLOW_UNPINNED_SNAPSHOT:-1}"
case "$MODE" in
  doctor) exec bash scripts/doctor_qwen35.sh ;;
  prepare|check|run)
    [ "$#" -le 1 ] || { echo "usage: $0 [prepare|check|run|doctor]"; exit 2; }
    exec bash scripts/run_additional_experiments.sh "--$MODE" qwen35
    ;;
  *) echo "usage: bash scripts/run_qwen35_9b.sh [run|check|doctor|prepare]  (default run; prepare = download, needs internet)"; exit 2 ;;
esac
