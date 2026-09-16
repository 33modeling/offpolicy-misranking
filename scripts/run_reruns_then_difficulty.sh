#!/usr/bin/env bash
# One-off (2026-09-17). On this node: reset the three invalid v1 branches once,
# run the v1 launcher so they are redone, and when all three have a result,
# stop the v1 launcher here and start the difficulty experiment on this node.
#
#   bash scripts/run_reruns_then_difficulty.sh      (on each node you give to the reruns)
#   log: $WORK/runs/experiments/logs/then.<host>.log
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
V1=${V1_ROOT:-$WORK/runs/selection-switch-v1}
BRANCHES=(states/s3-t100/points/view-100/random_reduced
          states/s4-t25/points/view-25/random_reduced
          states/s4-t25/points/view-25/gated)
HOST=$(hostname | tr -c 'a-zA-Z0-9._-' '_')
LOG_DIR="$WORK/runs/experiments/logs"
THEN_LOG="$LOG_DIR/then.$HOST.log"
RUN_RESET=${RUN_RESET:-bash scripts/run_selection_switch.sh reset-waived}
RUN_V1=${RUN_V1:-bash scripts/run_selection_switch.sh}
RUN_STOP=${RUN_STOP:-bash scripts/run_experiments.sh stop}
RUN_NEXT=${RUN_NEXT:-bash scripts/run_switch_difficulty.sh}
POLL=${THEN_POLL_SECONDS:-120}

results_done() {
  local n=0 b
  for b in "${BRANCHES[@]}"; do [ -f "$V1/$b/result.json" ] && n=$((n+1)); done
  echo "$n"
}
# An invalid original still in place: a result with no reset receipt (discards/).
# A rerun that already finished has both a result and discards/, and is kept.
invalid_left() {
  local n=0 b
  for b in "${BRANCHES[@]}"; do
    [ -f "$V1/$b/result.json" ] && [ ! -d "$V1/$b/discards" ] && n=$((n+1))
  done
  echo "$n"
}

if [ "${THEN_DETACHED:-0}" != 1 ]; then
  mkdir -p "$LOG_DIR"
  $RUN_RESET || true
  if [ "$(invalid_left)" -ne 0 ]; then
    echo "[then] abort: an invalid branch still has its result after reset-waived (a worker holds it?); nothing started"
    exit 2
  fi
  if [ "$(results_done)" -eq 3 ]; then
    echo "[then] all three reruns already have results; starting difficulty on this node now"
    $RUN_NEXT
    exit 0
  fi
  echo "[then] $(results_done)/3 reruns done; the rest run here first"
  THEN_DETACHED=1 setsid nohup bash "$0" >> "$THEN_LOG" 2>&1 < /dev/null &
  echo "[then] host=$HOST: watching the three v1 reruns; difficulty starts here when all three have results (log: $THEN_LOG)"
  $RUN_V1
  exit 0
fi

echo "[then] $(date -u +%FT%TZ) host=$HOST watcher started; waiting for 3 results under $V1"
while [ "$(results_done)" -lt 3 ]; do
  sleep "$POLL"
done
echo "[then] $(date -u +%FT%TZ) v1 reruns complete; stopping the v1 launcher here and starting difficulty"
$RUN_STOP || true
mkdir -p "$LOG_DIR"
CONSOLE_LOG="$LOG_DIR/console.$HOST.log"
PID_FILE="$LOG_DIR/launcher.$HOST.pid"
touch "$CONSOLE_LOG"
EXPERIMENTS_DETACHED=1 setsid nohup $RUN_NEXT run >> "$CONSOLE_LOG" 2>&1 < /dev/null &
echo "$!" > "$PID_FILE"
echo "[then] $(date -u +%FT%TZ) difficulty node launcher started (pid $!; console: $CONSOLE_LOG)"
