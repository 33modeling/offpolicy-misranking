#!/usr/bin/env bash
# Explicit 9B entrypoint: run on this idle node, independently of other nodes.
set -euo pipefail
cd "$(dirname "$0")/.."
# No argument = run. The node-local locks and GPU/model checks still apply;
# this operator-selected model does not wait for OLMo on other nodes.
MODE=${1:-run}
# Never mutate the checkout from a launch/diagnostic command.
# Update explicitly in a separate idle checkout after reviewing local changes.
# Status reports the installed revision without changing shared executable files.
if [ "$MODE" = status ]; then
  # `status`, `status verbose`; an extra profile word (`status h100`) is accepted
  # and ignored: the 9B matrix has one profile.
  echo "[code] $(git rev-parse --short HEAD 2>/dev/null || printf unknown) (read-only status; no automatic update)"
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
        --launcher-environment-from-child \
        --session-log-prefix "$OM_WORK/console-logs/additional-qwen35-run-" \
        --timeout 15
      LOCAL_LOCK_DIR="${OM_LOCAL_LOCK_DIR:-/tmp/offpolicy-misranking-$(id -u)}"
      if [ -d "$LOCAL_LOCK_DIR" ]; then
        for lock in additional-suite.lock primary.lock; do
          if ! flock -w 5 "$LOCAL_LOCK_DIR/$lock" true; then
            echo "[abort] Qwen cleanup finished but $lock still has another owner:"
            "$VENV_DIR/bin/python" src/cleanup_run_processes.py --list \
              --run-prefix "$OM_WORK/runs/qwen35-9b-posttrained-math-code-grpo-v1" \
              --open-file "$LOCAL_LOCK_DIR/$lock"
            exit 75
          fi
        done
      fi
      echo "[cleanup] previous Qwen processes exited; node locks are available"
    fi
    exec bash scripts/run_additional_experiments.sh --run qwen35
    ;;
  prepare|check)
    [ "$#" -le 1 ] || { echo "usage: $0 [prepare|check|run|status|doctor]"; exit 2; }
    exec bash scripts/run_additional_experiments.sh "--$MODE" qwen35
    ;;
  *) echo "usage: bash scripts/run_qwen35_9b.sh [run|run-idle|restart-idle|check|status|doctor|prepare]  (default: run 9B on this idle node; restart-idle first stops this node's previous 9B processes)"; exit 2 ;;
esac
