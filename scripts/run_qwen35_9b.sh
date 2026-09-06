#!/usr/bin/env bash
# Explicit 9B replication; never implicitly launch the retained 27B study.
set -euo pipefail
cd "$(dirname "$0")/.."
# No argument = run. `run` already performs every check (snapshot seal, FLA,
# smoke) before the matrix, so there is no separate step to remember.
MODE=${1:-run}
# Never mutate the checkout from a launch/diagnostic command.
# Update explicitly in a separate idle checkout after reviewing local changes.
# status is a read-only diagnostic; make sure it runs the latest pushed code.
# Workers run from node-local clones, so fast-forwarding the shared checkout
# never touches a running experiment. Never resets: local edits are reported.
self_update_for_status() {
  local before after dirty
  before=$(git rev-parse --short HEAD 2>/dev/null)
  if ! git fetch -q origin master 2>/dev/null; then
    echo "[code] $before (offline: could not fetch origin)"; return 0
  fi
  dirty=$(git status --porcelain -- src scripts configs 2>/dev/null)
  if [ -n "$dirty" ]; then
    echo "[code] $before but origin/master is $(git rev-parse --short origin/master); NOT updated: local edits block it:"
    printf '%s\n' "$dirty" | head -5
    echo "        to update: git stash && git pull --ff-only"
    return 0
  fi
  if git merge -q --ff-only origin/master 2>/dev/null; then
    after=$(git rev-parse --short HEAD 2>/dev/null)
    [ "$before" = "$after" ] && echo "[code] $after (up to date)" || echo "[code] updated $before -> $after"
  else
    echo "[code] $before but origin/master is $(git rev-parse --short origin/master); NOT updated: branch diverged"
    echo "        to update: git reset --hard origin/master   (shared checkout only; workers are unaffected)"
  fi
}
if [ "$MODE" = status ]; then
  self_update_for_status
  exec bash scripts/status_qwen35.sh
fi
echo "[code] $(git rev-parse --short HEAD) (no automatic fetch/merge/reset)"
export OM_ALLOW_UNPINNED_SNAPSHOT=0 OM_TRUST_LOCAL_SNAPSHOT=0
case "$MODE" in
  doctor) exec bash scripts/doctor_qwen35.sh ;;
  status) exec bash scripts/status_qwen35.sh ;;
  prepare|check|run)
    [ "$#" -le 1 ] || { echo "usage: $0 [prepare|check|run|status|doctor]"; exit 2; }
    exec bash scripts/run_additional_experiments.sh "--$MODE" qwen35
    ;;
  *) echo "usage: bash scripts/run_qwen35_9b.sh [run|check|status|doctor|prepare]  (default run; status = one screen; prepare = download, needs internet)"; exit 2 ;;
esac
