#!/usr/bin/env bash
# Node watcher, started once per node by run_queue.sh (run or status) and left
# running in the background: every minute it records what this node is doing
# (queue processes by their OUT_ROOT marker, node lock) under
# $OM_WORK/queue/<host>.seen.json, so `run_queue.sh status` on any node can
# list every node as busy, idle or gone. Dies with the allocation; a newer
# watcher on the same host replaces it through the pid file.
cd "$(dirname "$0")/.."
trap '' HUP INT
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OM_WORK/queue"
PIDFILE="$OM_WORK/queue/$(hostname).watch.pid"
echo $$ > "$PIDFILE"
while :; do
  [ "$(cat "$PIDFILE" 2>/dev/null)" = "$$" ] || exit 0
  "$PY" src/queue_status.py --record >/dev/null 2>&1
  sleep 60
done
