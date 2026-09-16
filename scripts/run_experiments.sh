#!/usr/bin/env bash
# One command per node for both experiments. Each pass closes stale costs,
# runs one selection-switch queue pass, then one MoPPS pass (which retries
# recorded failures first), and keeps the node between passes. The node is
# never assigned to one experiment: whatever has claimable work gets it.
#
#   bash scripts/run_experiments.sh          detach on this node and follow
#   bash scripts/run_experiments.sh stop     stop this node's launcher and workers
#   bash scripts/run_experiments.sh status   one screen for both experiments (also
#                                            what run_selection_switch.sh status and
#                                            run_mopps_comparison.sh status show)
#
# EXPERIMENTS_HOLD_SECONDS (default 300) is the pause between passes,
# EXPERIMENTS_AUTO_PULL=1 runs 'git pull --ff-only' before each pass.
set -euo pipefail
LAUNCHER_SELF=$(cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")
cd "$(dirname "$0")/.."
MODE=${1:-run}
[ "$#" -eq 0 ] || shift
case "$MODE" in run|stop|status) ;;
  *) echo 'usage: bash scripts/run_experiments.sh [run|stop|status]'; exit 2 ;;
esac
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export OM_WORK="$WORK"
SWITCH_ROOT=$(realpath -m "${SWITCH_ROOT:-$WORK/runs/selection-switch-v1}")
MOPPS_ROOT=$(realpath -m "${MOPPS_ROOT:-$WORK/runs/mopps-comparison-v1}")
export SWITCH_ROOT MOPPS_ROOT
PY=${SWITCH_PYTHON:-${MOPPS_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
LOG_DIR="$WORK/runs/experiments/logs"
HOST=$(hostname | tr -c 'a-zA-Z0-9._-' '_')
PID_FILE="$LOG_DIR/launcher.$HOST.pid"
CONSOLE_LOG="$LOG_DIR/console.$HOST.log"
launcher_pid_alive() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid=$(cat "$PID_FILE" 2>/dev/null) || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}
if [ "$MODE" = status ]; then
  # One screen for both experiments (switch first, MoPPS second, this node's
  # GPUs once). Accepts --all, --json and --watch [seconds]. Read-only.
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" scripts/experiments_status.py --switch-root "$SWITCH_ROOT" --mopps-root "$MOPPS_ROOT" "$@"
fi
if [ "$MODE" = stop ]; then
  if launcher_pid_alive; then
    pid=$(cat "$PID_FILE")
    echo "[stop] host=$HOST pid=$pid: sending TERM to the node launcher; inner launchers reap their ranks and close receipts"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 240); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && echo "[stop] pid=$pid still running after 240s; inspect $CONSOLE_LOG"
  else
    echo "[stop] no live node launcher on $HOST (pid file: $PID_FILE)"
  fi
  # Leftovers from either experiment (older launchers, ranks, keepalives).
  EXPERIMENTS_STOPPING=1 bash scripts/run_selection_switch.sh stop || true
  EXPERIMENTS_STOPPING=1 bash scripts/run_mopps_comparison.sh stop || true
  exit 0
fi
# --- run ---
if [ -t 1 ] && [ "${EXPERIMENTS_DETACHED:-0}" != 1 ]; then
  if launcher_pid_alive; then
    echo "[already running] host=$HOST pid=$(cat "$PID_FILE"); follow: tail -f $CONSOLE_LOG; stop: bash scripts/run_experiments.sh stop"
    exit 0
  fi
  mkdir -p "$LOG_DIR"
  touch "$CONSOLE_LOG"
  offset=$(stat -c %s "$CONSOLE_LOG")
  EXPERIMENTS_DETACHED=1 setsid nohup bash "$LAUNCHER_SELF" run >> "$CONSOLE_LOG" 2>&1 < /dev/null &
  pid=$!
  disown 2>/dev/null || true
  echo "$pid" > "$PID_FILE"
  echo "[detached] host=$HOST pid=$pid console=$CONSOLE_LOG"
  echo "[detached] Ctrl-C leaves the node working; stop with: bash scripts/run_experiments.sh stop"
  tail --pid="$pid" -c +"$((offset+1))" -F "$CONSOLE_LOG" 2>/dev/null || true
  exit 0
fi
mkdir -p "$LOG_DIR"
printf '[node-launcher-start] host=%s pid=%s utc=%s commit=%s\n' "$HOST" "$$" "$(date -u +%FT%TZ)" "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
KEEPALIVE_PID=
stop_keepalive() { [ -n "$KEEPALIVE_PID" ] && kill -TERM "$KEEPALIVE_PID" 2>/dev/null; KEEPALIVE_PID=; }
trap 'rc=$?; stop_keepalive; printf "[node-launcher-exit] pid=%s rc=%s utc=%s\n" "$$" "$rc" "$(date -u +%FT%TZ)"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
HOLD=${EXPERIMENTS_HOLD_SECONDS:-300}
# Keep the allocated GPUs visibly busy for this launcher's whole life: the
# cluster reclaims idle allocations, and an inner pass may spend minutes in
# admission or find nothing to claim. 727 MiB per GPU, well under the inner
# launchers' occupancy limit. EXPERIMENTS_KEEPALIVE=0 disables it.
if [ "${EXPERIMENTS_KEEPALIVE:-1}" != 0 ]; then
  "$PY" scripts/_gpu_keepalive.py > "$LOG_DIR/keepalive.$HOST.log" 2>&1 7>&- 8>&- &
  KEEPALIVE_PID=$!
  echo "[keepalive] pid=$KEEPALIVE_PID (log: $LOG_DIR/keepalive.$HOST.log)"
fi
[[ "$HOLD" =~ ^[0-9]+$ ]] || { echo '[abort] EXPERIMENTS_HOLD_SECONDS must be a whole number of seconds'; exit 2; }
# Inner launchers: foreground, single pass, no hold, no keepalive (this launcher holds the node).
inner() {
  env -u EXPERIMENTS_DETACHED SWITCH_FOREGROUND=1 SWITCH_HOLD_SECONDS=0 SWITCH_KEEPALIVE=0 bash "$@"
}
switch_complete() {
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_status.py --root "$SWITCH_ROOT" --json 2>/dev/null \
    | "$PY" -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("development_done")==18 and d.get("test_done")==30 else 1)'
}
mopps_complete() {
  [ "$(CUDA_VISIBLE_DEVICES="" "$PY" src/mopps_comparison_gpu.py status --root "$MOPPS_ROOT" 2>/dev/null | grep -c ' DONE ')" -ge 12 ]
}
recover_root() {
  [ -f "$1/switch.json" ] || [ -f "$1/mopps.json" ] || return 0
  # One summary line, then one line per open event that could not be closed and why.
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/recover_selection_switch_cost.py --root "$1" --stale --brief 2>&1 \
    | sed 's/^\[recovery blocked\]/[recover-cost] blocked:/' || true
}
pass=0
wait_seconds=$HOLD
blocked_passes=0
while :; do
  pass=$((pass+1))
  if [ "${EXPERIMENTS_AUTO_PULL:-0}" = 1 ]; then
    git pull -q --ff-only 2>&1 | tail -1 || true
  fi
  recover_root "$SWITCH_ROOT"
  recover_root "$MOPPS_ROOT"
  rc_switch=0
  if [ "${EXPERIMENTS_SKIP_SWITCH:-0}" != 1 ] && ! switch_complete; then
    echo "[pass $pass] selection switch"
    inner scripts/run_selection_switch.sh || rc_switch=$?
    case "$rc_switch" in 130|143) exit "$rc_switch" ;; esac
  fi
  rc_mopps=0
  if [ "${EXPERIMENTS_SKIP_MOPPS:-0}" != 1 ] && [ -f "$MOPPS_ROOT/mopps.json" ] && ! mopps_complete; then
    echo "[pass $pass] MoPPS comparison"
    inner scripts/run_mopps_comparison.sh || rc_mopps=$?
    case "$rc_mopps" in 130|143) exit "$rc_mopps" ;; esac
  fi
  if switch_complete && { [ ! -f "$MOPPS_ROOT/mopps.json" ] || mopps_complete; }; then
    echo '[done] both experiments are complete; releasing the node'
    exit 0
  fi
  if [ "$rc_switch" -eq 78 ] || [ "$rc_mopps" -eq 78 ]; then
    blocked_passes=$((blocked_passes+1))
    if [ "$blocked_passes" -ge 2 ]; then
      echo '[blocked] node admission failed on two passes; releasing this node (use another one)'
      exit 78
    fi
  else
    blocked_passes=0
  fi
  if [ "$rc_switch" -eq 0 ] && [ "$rc_mopps" -eq 0 ]; then wait_seconds=$HOLD; else wait_seconds=$(( wait_seconds*2 > 3600 ? 3600 : wait_seconds*2 )); fi
  echo "[hold] pass $pass ended (switch rc=$rc_switch, mopps rc=$rc_mopps); keeping this node's GPUs; next pass in ${wait_seconds}s (stop: bash scripts/run_experiments.sh stop)"
  remaining=$wait_seconds
  while [ "$remaining" -gt 0 ]; do
    step=$(( remaining < 15 ? remaining : 15 ))
    printf '[holding] node retained; next pass in %ss\n' "$remaining"
    sleep "$step" & wait $! || true
    remaining=$((remaining-step))
  done
done
