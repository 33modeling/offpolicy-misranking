#!/usr/bin/env bash
# Everything that remains, in priority order, on THIS node:
#
#   bash scripts/run_queue.sh          # start the queue on this GPU node, detached from the terminal
#   bash scripts/run_queue.sh status   # one screen: nodes first, then one line per step (no GPU)
#
# The queue runs in the background (setsid, output only to
# $OM_WORK/queue/<host>.log), so closing the terminal that started it does
# not stop it. Every step is leased (node lock, per-seed arms, per-point
# locks), so the same command on several nodes shares the work: a step that
# is finished or claimed elsewhere is skipped and the queue moves on.
# Rerunning resumes. Before starting, GPU processes of an earlier queue whose
# driver died are stopped (their leases are gone; they would collide with
# fresh work) - only when no live driver holds this node's lock.
# Order: mixed-pool positive control (pool, point, arms, gate) -> reuse
# split-half scores (d400, d0) -> public benchmarks (d0, d400) -> d100
# continuation -> CPU analyses and the export bundle.
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
case "$MODE" in run|status|worker) ;; *) echo "usage: bash scripts/run_queue.sh [status]"; exit 2 ;; esac
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
NOTES="$OM_WORK/queue"; mkdir -p "$NOTES"
HOST=$(hostname)
NOTE="$NOTES/$HOST.txt"; BEAT="$NOTES/$HOST.beat"; QLOG="$NOTES/$HOST.log"; WPID="$NOTES/$HOST.worker.pid"
# Every node that runs this command (run or status) gets a background watcher
# that reports the node every minute, so the status on any node lists all of
# them as busy, idle or gone (scripts/_node_watch.sh).
ensure_watch() {
  local pidfile="$NOTES/$HOST.watch.pid" pid
  pid=$(cat "$pidfile" 2>/dev/null)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then return 0; fi
  setsid nohup bash scripts/_node_watch.sh >/dev/null 2>&1 < /dev/null &
  disown 2>/dev/null || true
  echo "[queue] node watcher started on $HOST: this node now reports itself every minute"
}
ensure_watch
if [ "$MODE" = status ]; then
  "$PY" src/queue_status.py
  exit $?
fi
if [ "$MODE" = run ]; then
  pid=$(cat "$WPID" 2>/dev/null)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "[queue] already running on $HOST (pid $pid); progress:  bash scripts/run_queue.sh status"
    exit 0
  fi
  echo "===== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] queue started on $HOST code=$(git rev-parse --short HEAD 2>/dev/null)" >> "$QLOG"
  setsid nohup bash scripts/run_queue.sh worker >> "$QLOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
  echo "[queue] started in the background on $HOST; closing this terminal does not stop it"
  echo "        progress:  bash scripts/run_queue.sh status        log: $QLOG"
  exit 0
fi
# ---- worker (background)
trap '' HUP
echo $$ > "$WPID"
note() { printf 'host=%s pid=%s step=%s since=%s\n' "$HOST" "$$" "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$NOTE"; }
( while :; do date -u +%Y-%m-%dT%H:%M:%SZ > "$BEAT" 2>/dev/null; sleep 60; done ) &
BEAT_PID=$!
trap 'kill "$BEAT_PID" 2>/dev/null; note "stopped"; exit 130' INT TERM
note "starting"
# Orphans of an earlier queue on this node (driver dead, GPU children alive).
"$PY" src/queue_status.py --kill-orphans
QUEUE=("run_mixed_pool.sh pool" "run_mixed_pool.sh point" "run_mixed_pool.sh e5" "run_mixed_pool.sh gate" \
       "run_stale_splithalf.sh" "run_stale_splithalf.sh d0" "run_e5_bench.sh d0" "run_e5_bench.sh" \
       "run_e5.sh d100")
for job in "${QUEUE[@]}"; do
  echo; echo "===== [$(date -u +%H:%M)] $job"
  note "$job"
  bash scripts/$job; rc=$?
  echo "===== [$(date -u +%H:%M)] $job finished (rc=$rc)"
done
echo; echo "===== CPU analyses and export"
note "run_analyses.sh"
bash scripts/run_analyses.sh
note "done"
kill "$BEAT_PID" 2>/dev/null
echo "===== [$(date -u +%H:%M)] queue done on $HOST; check:  bash scripts/run_queue.sh status"
