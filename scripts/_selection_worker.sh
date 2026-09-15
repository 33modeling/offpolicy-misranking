#!/usr/bin/env bash
# Shared foreground launcher lifecycle for switch and MoPPS workers.
selection_stop_worker() {
  STOP_STATUS=$1
  trap '' INT TERM
  if [ -n "$CHILD" ]; then
    printf '[stopping] worker=%s; waiting for owned process groups and cost receipts\n' "$CHILD"
    kill -TERM "$CHILD" 2>/dev/null || true
    wait "$CHILD" || true
    exit "$STOP_STATUS"
  fi
}

selection_run_worker() {
  CHILD=
  STOP_STATUS=0
  trap 'selection_stop_worker 130' INT
  trap 'selection_stop_worker 143' TERM
  "$@" 7>&- 8>&- &
  CHILD=$!
  if [ "$STOP_STATUS" -ne 0 ]; then
    selection_stop_worker "$STOP_STATUS"
  fi
  local rc=0
  wait "$CHILD" || rc=$?
  trap - INT TERM
  return "$rc"
}
