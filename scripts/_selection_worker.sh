#!/usr/bin/env bash
# Shared foreground launcher lifecycle for switch and MoPPS workers.
selection_stop_worker() {
  STOP_STATUS=$1
  trap '' INT TERM
  if [ -n "$CHILD" ]; then
    printf '[stopping] worker=%s; waiting for owned process groups and cost receipts\n' "$CHILD"
    kill -TERM "$CHILD" 2>/dev/null || true
    wait "$CHILD" || true
    [ -z "${SELECTION_LOG_PID:-}" ] || wait "$SELECTION_LOG_PID" || true
    exit "$STOP_STATUS"
  fi
}

selection_run_worker() {
  local tag= argument log_fd worker_repo
  local worker_argv=("$@")
  worker_repo=$(realpath -m "$(dirname "${BASH_SOURCE[0]}")/..")
  if [ "$#" -eq 5 ] && [ "$4" = --root ] \
      && [[ "${1##*/}" = python* ]] \
      && [ "$(realpath -m "$2")" = "$worker_repo/src/selector_pair_gpu.py" ]; then
    case "$3" in
      run|develop|test|freeze) worker_argv[1]="$worker_repo/scripts/queue_selector_pair_gpu.py" ;;
    esac
    if [ "$3" = run ] && [ -f "$worker_repo/scripts/selector_pair_resume_two.py" ]; then
      local pair_probe_rc=0
      CUDA_VISIBLE_DEVICES="" "$1" "$worker_repo/scripts/selector_pair_resume_two.py" probe --root "$5" || pair_probe_rc=$?
      case "$pair_probe_rc" in
        0) worker_argv[1]="$worker_repo/scripts/selector_pair_resume_two.py" ;;
        3) ;;
        *) return "$pair_probe_rc" ;;
      esac
    fi
  fi
  # Inspect the actual worker before inherited dataset settings: Pair can use
  # MBPP data without being the MBPP switch experiment.
  for argument in "${worker_argv[@]}"; do
    case "$argument" in
      */selector_pair_gpu.py|*/queue_selector_pair_gpu.py|*/selector_pair_resume_two.py) tag=pair; break ;;
      */queue_rloo.py|*/rloo_experiment.py) tag=rloo; break ;;
    esac
  done
  if [ -z "$tag" ] && { [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ] || [ "${SWITCH_DATASET:-}" = mbpp ]; }; then
    tag=mbpp
  fi
  # Direct Pair/RLOO launchers do not pass through run_experiments.sh. Set the
  # same GPU/container identity before their first meter or queue receipt.
  if [ "$tag" = pair ] || [ "$tag" = rloo ]; then
    source "$(dirname "${BASH_SOURCE[0]}")/_node_id.sh" || return $?
  fi
  CHILD=
  SELECTION_LOG_PID=
  STOP_STATUS=0
  trap 'selection_stop_worker 130' INT
  trap 'selection_stop_worker 143' TERM
  if [ -n "$tag" ]; then
    # Keep CHILD as the real worker, not a pipeline's formatter. Drain its final
    # output on both success and signal shutdown, without inheriting node locks.
    exec {log_fd}> >(trap '' INT TERM; exec 7>&- 8>&-; exec sed -u "/\\[$tag\\]$/!s/$/ [$tag]/")
    SELECTION_LOG_PID=$!
    "${worker_argv[@]}" 7>&- 8>&- >&"$log_fd" 2>&1 {log_fd}>&- &
    CHILD=$!
    exec {log_fd}>&-
  else
    "${worker_argv[@]}" 7>&- 8>&- &
    CHILD=$!
  fi
  if [ "$STOP_STATUS" -ne 0 ]; then
    selection_stop_worker "$STOP_STATUS"
  fi
  local rc=0
  wait "$CHILD" || rc=$?
  CHILD=
  [ -z "$SELECTION_LOG_PID" ] || wait "$SELECTION_LOG_PID" || true
  SELECTION_LOG_PID=
  trap - INT TERM
  return "$rc"
}

selection_hold_node() {
  local remaining=$1 interval
  while [ "$remaining" -gt 0 ]; do
    interval=$(( remaining < 15 ? remaining : 15 ))
    printf '[holding] node retained; next queue pass in %ss; no training active in this launcher\n' "$remaining"
    # Reuse stop handling and close inherited node-lock descriptors in sleep.
    selection_run_worker sleep "$interval" || return $?
    remaining=$((remaining-interval))
  done
}
