#!/usr/bin/env bash
# Follow-up generalization matrices (docs/FOLLOWUP_GENERALIZATION_DESIGN.md).
#   bash scripts/run_followup.sh qwen35_2b|qwen35_4b|olmo3_domains [run|check|doctor|prepare]
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE=${1:-}
MODE=${2:-run}
case "$PROFILE" in
  qwen35_2b|qwen35_4b|olmo3_domains) ;;
  *) echo "usage: bash scripts/run_followup.sh qwen35_2b|qwen35_4b|olmo3_domains [run|check|doctor|prepare]"; exit 2 ;;
esac
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
export OM_TRUST_LOCAL_SNAPSHOT="${OM_TRUST_LOCAL_SNAPSHOT:-1}"
case "$MODE" in
  doctor) PROFILE_CONFIG=configs/${PROFILE/qwen35_/qwen35_}_grpo.json exec bash scripts/doctor_qwen35.sh ;;
  prepare|check|run) exec bash scripts/run_additional_experiments.sh "--$MODE" "$PROFILE" ;;
  *) echo "usage: bash scripts/run_followup.sh <profile> [run|check|doctor|prepare]"; exit 2 ;;
esac
