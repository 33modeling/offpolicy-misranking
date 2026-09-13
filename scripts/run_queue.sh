#!/usr/bin/env bash
# Everything that remains, in priority order, on THIS node:
#
#   bash scripts/run_queue.sh          # run the queue (GPU node)
#   bash scripts/run_queue.sh status   # one-screen overview: one line per step (no GPU)
#
# Every step is leased (node lock, per-seed arms, per-point locks), so the
# same command on several nodes shares the work: a step that is finished or
# claimed elsewhere is skipped and the queue moves on. Rerunning resumes.
# Order: mixed-pool positive control (pool, point, arms, gate) -> reuse
# split-half scores (d400, d0) -> public benchmarks (d0, d400) -> d100
# continuation -> CPU analyses and the export bundle.
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
# Every node that runs this command (run or status) gets a background watcher
# that reports the node every minute, so the status on any node lists all of
# them as busy, idle or gone (scripts/_node_watch.sh).
ensure_watch() {
  local pidfile="$OM_WORK/queue/$(hostname).watch.pid" pid
  pid=$(cat "$pidfile" 2>/dev/null)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then return 0; fi
  mkdir -p "$OM_WORK/queue"
  setsid nohup bash scripts/_node_watch.sh >/dev/null 2>&1 < /dev/null &
  disown 2>/dev/null || true
  echo "[queue] node watcher started on $(hostname): this node now reports itself every minute"
}
ensure_watch
if [ "${1:-run}" = status ]; then
  # One screen: nodes first, then one line per queue step with a state word.
  PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
  PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" src/queue_status.py
  exit $?
fi
trap '' HUP
# Per-node note on the shared filesystem: which step this node is on, plus a
# heartbeat file touched every minute, so `run_queue.sh status` can list the
# nodes and tell a live queue from a killed one.
NOTES="$OM_WORK/queue"; mkdir -p "$NOTES"
NOTE="$NOTES/$(hostname).txt"; BEAT="$NOTES/$(hostname).beat"
note() { printf 'host=%s pid=%s step=%s since=%s\n' "$(hostname)" "$$" "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$NOTE"; }
( while :; do date -u +%Y-%m-%dT%H:%M:%SZ > "$BEAT" 2>/dev/null; sleep 60; done ) &
BEAT_PID=$!
trap 'kill "$BEAT_PID" 2>/dev/null; note "stopped"; exit 130' INT TERM
QUEUE=("run_mixed_pool.sh pool" "run_mixed_pool.sh point" "run_mixed_pool.sh e5" "run_mixed_pool.sh gate" \
       "run_stale_splithalf.sh" "run_stale_splithalf.sh d0" "run_e5_bench.sh d0" "run_e5_bench.sh" \
       "run_e5.sh d100")
for job in "${QUEUE[@]}"; do
  echo; echo "===== [$(date -u +%H:%M)] $job"
  note "$job"
  bash scripts/$job 2>&1 | grep -v setup_env
  echo "===== [$(date -u +%H:%M)] $job finished (rc=${PIPESTATUS[0]})"
done
echo; echo "===== CPU analyses and export"
note "run_analyses.sh"
bash scripts/run_analyses.sh 2>&1 | grep -v setup_env | tail -n 6
note "done"
kill "$BEAT_PID" 2>/dev/null
echo "[queue] done on $(hostname); check:  bash scripts/run_queue.sh status"
