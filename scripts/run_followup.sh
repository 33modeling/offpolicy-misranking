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
if [ -z "$(git status --porcelain -- src scripts configs requirements.txt)" ]; then
  git pull --ff-only -q 2>/dev/null || echo "[note] git pull failed; running current code $(git rev-parse --short HEAD)"
fi
export OM_ALLOW_UNPINNED_SNAPSHOT="${OM_ALLOW_UNPINNED_SNAPSHOT:-1}"
export OM_TRUST_LOCAL_SNAPSHOT="${OM_TRUST_LOCAL_SNAPSHOT:-1}"
case "$MODE" in
  doctor) PROFILE_CONFIG=configs/${PROFILE/qwen35_/qwen35_}_grpo.json exec bash scripts/doctor_qwen35.sh ;;
  prepare|check|run) exec bash scripts/run_additional_experiments.sh "--$MODE" "$PROFILE" ;;
  *) echo "usage: bash scripts/run_followup.sh <profile> [run|check|doctor|prepare]"; exit 2 ;;
esac
