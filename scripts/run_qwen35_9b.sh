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
# Always run the latest pushed code. Local edits under src/scripts/configs would
# be lost by a reset, so in that case stop and say which files block the update.
DIRTY=$(git status --porcelain -- src scripts configs requirements.txt)
if [ -n "$DIRTY" ]; then
  echo "[abort] checkout has local changes; cannot update code:"; printf '%s\n' "$DIRTY"
  echo "ACTION: git stash  (or git checkout -- <file>)  then rerun"; exit 1
fi
if git fetch -q origin master 2>/dev/null; then
  if ! git merge -q --ff-only origin/master 2>/dev/null; then
    echo "[code] local branch diverged from origin/master; resetting to origin/master"
    git reset -q --hard origin/master
  fi
  echo "[code] $(git rev-parse --short HEAD) $(git log -1 --format=%s | cut -c1-60)"
else
  echo "[code] git fetch failed (offline?); running current code $(git rev-parse --short HEAD)"
fi
export OM_ALLOW_UNPINNED_SNAPSHOT="${OM_ALLOW_UNPINNED_SNAPSHOT:-1}"
# Uploaded weights are trusted: validate that they load, not that they match the
# Hub revision byte for byte. Set OM_TRUST_LOCAL_SNAPSHOT=0 for the strict check.
export OM_TRUST_LOCAL_SNAPSHOT="${OM_TRUST_LOCAL_SNAPSHOT:-1}"
case "$MODE" in
  doctor) exec bash scripts/doctor_qwen35.sh ;;
  prepare|check|run)
    [ "$#" -le 1 ] || { echo "usage: $0 [prepare|check|run|doctor]"; exit 2; }
    exec bash scripts/run_additional_experiments.sh "--$MODE" qwen35
    ;;
  *) echo "usage: bash scripts/run_qwen35_9b.sh [run|check|doctor|prepare]  (default run; prepare = download, needs internet)"; exit 2 ;;
esac
