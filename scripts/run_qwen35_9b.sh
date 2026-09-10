#!/usr/bin/env bash
# Explicit 9B entrypoint: run on this idle node, independently of other nodes.
set -euo pipefail
cd "$(dirname "$0")/.."
# No argument = run. The node-local locks and GPU/model checks still apply;
# this operator-selected model does not wait for OLMo on other nodes.
MODE=${1:-run}
# Never mutate the checkout from a launch/diagnostic command.
# Update explicitly in a separate idle checkout after reviewing local changes.
# status is a read-only diagnostic; make sure it runs the latest pushed code.
# Workers run from node-local clones, so fast-forwarding the shared checkout
# never touches a running experiment. Never resets: local edits are reported.
self_update_for_status() {
  local before after dirty live
  before=$(git rev-parse --short HEAD 2>/dev/null)
  # A Qwen launcher whose generation commit equals this checkout's HEAD runs
  # run_point.sh and src/*.py straight from this shared checkout, stage by
  # stage. Replacing those files under it changes a running experiment, so the
  # update is skipped while any Qwen launcher is alive: on this node (pid) or on
  # another node (a session log without an [exit] line written in the last 20
  # minutes). Status still prints from the current code.
  live=$(pgrep -f 'run_additional_experiments[.]sh --run qwen35' 2>/dev/null | wc -l)  # [.] keeps this line from matching itself
  live=$(( ${live:-0} + $(find "${OM_WORK:-/nonexistent}/console-logs" -maxdepth 1 -name 'additional-qwen35-*.log' -mmin -20 2>/dev/null \
    | xargs -r grep -L '^\[exit\]' 2>/dev/null | wc -l) ))
  if [ "$live" -gt 0 ]; then
    if git fetch -q origin master 2>/dev/null && [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/master)" ]; then
      echo "[code] $before; origin/master is $(git rev-parse --short origin/master). Not updating: $live live Qwen launcher(s) read this checkout. Update after they exit, or run status from a second clone."
    fi
    return 0
  fi
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
  # `status`, `status verbose`; an extra profile word (`status h100`) is accepted
  # and ignored: the 9B matrix has one profile.
  self_update_for_status
  status_args=()
  for word in "${@:2}"; do [ "$word" != verbose ] || status_args+=(verbose); done
  exec bash scripts/status_qwen35.sh "${status_args[@]}"
fi
echo "[code] $(git rev-parse --short HEAD) (no automatic fetch/merge/reset)"
export OM_ALLOW_UNPINNED_SNAPSHOT=0 OM_TRUST_LOCAL_SNAPSHOT=0
case "$MODE" in
  doctor) exec bash scripts/doctor_qwen35.sh ;;
  status) exec bash scripts/status_qwen35.sh "${@:2}" ;;
  run|run-idle|restart-idle)
    [ "$#" -le 1 ] || { echo "usage: $0 [run|run-idle|restart-idle]"; exit 2; }
    export OM_QWEN_IDLE_NODE="$(hostname)"
    [ -n "$OM_QWEN_IDLE_NODE" ] || exit 1
    export OM_WAIT_PRIMARY=0
    export ADDITIONAL_GPU_WAIT_SECONDS="${ADDITIONAL_GPU_WAIT_SECONDS:-60}"
    # This is a separate model, never the primary's pinned generation/adapter.
    unset OM_PIPELINE_REPO OM_PIPELINE_SCRIPT OM_GENERATION_GIT
    unset REGIME_SKIP_COLLECTION REGIME_MATRIX MODEL_PATH OM_EXTERNAL_GPU_KEEPALIVE
    if [ "$MODE" = restart-idle ]; then
      source scripts/setup_env.sh
      echo "[cleanup] stopping this user's previous Qwen 9B processes on $(hostname), work=$OM_WORK; preserving all artifacts"
      "$VENV_DIR/bin/python" src/cleanup_run_processes.py \
        --run-prefix "$OM_WORK/runs/qwen35-9b-posttrained-math-code-grpo-v1" \
        --require-environment "OM_WORK=$OM_WORK" \
        --command-pattern 'scripts/run_additional_experiments.sh --run qwen35 ' \
        --timeout 15
    fi
    exec bash scripts/run_additional_experiments.sh --run qwen35
    ;;
  prepare|check)
    [ "$#" -le 1 ] || { echo "usage: $0 [prepare|check|run|status|doctor]"; exit 2; }
    exec bash scripts/run_additional_experiments.sh "--$MODE" qwen35
    ;;
  *) echo "usage: bash scripts/run_qwen35_9b.sh [run|run-idle|restart-idle|check|status|doctor|prepare]  (default: run 9B on this idle node; restart-idle first stops this node's previous 9B processes)"; exit 2 ;;
esac
