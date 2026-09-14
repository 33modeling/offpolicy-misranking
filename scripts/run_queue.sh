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
case "$MODE" in run|status|worker|dump|stop) ;; *) echo "usage: bash scripts/run_queue.sh [status|dump|stop]"; exit 2 ;; esac
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
if [ "$MODE" = dump ]; then
  # One file with everything needed to see why a node is not working: the status, this node's
  # nvidia-smi, every node's queue log tail, and the mixed point's own log tail. Copy it like an export.
  mkdir -p "$OM_WORK/exports"
  target="$OM_WORK/exports/queue-dump-$(date -u +%Y%m%dT%H%M%SZ).txt"
  {
    echo "# queue dump $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$HOST code=$(git rev-parse --short HEAD 2>/dev/null)"
    echo; echo "### status"; "$PY" src/queue_status.py 2>&1
    echo; echo "### nvidia-smi on $HOST"; timeout 20 nvidia-smi 2>&1 | head -40
    echo; echo "### processes with a queue marker on $HOST"
    for pid in $(pgrep -u "$(id -u)" 2>/dev/null); do
      m=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep '^OUT_ROOT=' | head -1)
      [ -n "$m" ] && echo "pid=$pid $m $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-120)"
    done
    for f in "$NOTES"/*.log; do [ -s "$f" ] || continue; echo; echo "### queue log $(basename "$f") (last 60 lines)"; tail -n 60 "$f"; done
    TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
    ROOT=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
    POINT="$ROOT/family-math500mix-s${MIX_SEED:-0}/$TAG-s${MIX_SEED:-0}-math500mix-d0"
    echo; echo "### mixed point $POINT"
    ls -la "$POINT" 2>&1 | head -40
    echo; echo "### mixed point lease note"; cat "$POINT.lease" 2>/dev/null
    for f in "$POINT"/logs/*.log; do [ -s "$f" ] || continue; echo; echo "### point log $(basename "$f") (last 40 lines)"; tail -n 40 "$f"; done
    for out in "$OM_WORK"/runs/e5-reduced/math500mix-d0/s*; do
      [ -d "$out" ] || continue
      for f in "$out"/logs/*.log; do [ -s "$f" ] || continue; echo; echo "### mixed arms log $(basename "$f") (last 30 lines)"; tail -n 30 "$f"; done
    done
  } > "$target" 2>&1
  echo "[queue] dump written: $target"
  exit 0
fi
# A pid file on the shared filesystem outlives its allocation, and the number can be reused by an
# unrelated process. Only a live process whose command line is this worker counts as running.
worker_alive() {
  local pid=$1
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'run_queue.sh' || return 1
  return 0
}
if [ "$MODE" = stop ]; then
  # Stop this node's queue worker, then the step processes it started (they carry the OUT_ROOT
  # marker); leases are released with them. Finished shards and checkpoints stay; rerunning resumes.
  pid=$(cat "$WPID" 2>/dev/null)
  if worker_alive "$pid"; then
    echo "[queue] stopping worker pid $pid on $HOST"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
  else
    echo "[queue] no live worker on $HOST"
  fi
  "$PY" src/queue_status.py --kill-orphans
  printf 'host=%s pid=%s step=stopped since=%s\n' "$HOST" "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$NOTE"
  echo "[queue] stopped on $HOST; restart with:  bash scripts/run_queue.sh"
  exit 0
fi
if [ "$MODE" = run ]; then
  pid=$(cat "$WPID" 2>/dev/null)
  if worker_alive "$pid"; then
    echo "[queue] already running on $HOST (pid $pid); progress:  bash scripts/run_queue.sh status"
    echo "        stop it with:  bash scripts/run_queue.sh stop"
    exit 0
  fi
  if [ -n "$pid" ]; then echo "[queue] stale worker note on $HOST (pid $pid is not a queue worker); starting a new one"; fi
  # Byte offset before this run, so the terminal shows everything this run writes even when the
  # queue finishes before the follower attaches (every step can abort in seconds).
  offset=$(stat -c %s "$QLOG" 2>/dev/null || echo 0)
  echo "===== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] queue started on $HOST code=$(git rev-parse --short HEAD 2>/dev/null)" >> "$QLOG"
  setsid nohup bash scripts/run_queue.sh worker >> "$QLOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
  echo "[queue] started in the background on $HOST; closing this terminal does not stop it"
  echo "        progress:  bash scripts/run_queue.sh status        log: $QLOG"
  # Following the log neither feeds nor stops the worker: closing this terminal, or Ctrl-C here,
  # leaves it running. The follower stops by itself when the worker exits.
  follow=${OM_QUEUE_FOLLOW_SECONDS:-120}
  if [ "$follow" -gt 0 ]; then
    echo "        ---- this run's queue log (Ctrl-C here does not stop the queue) ----"
    timeout "$follow" tail -c "+$((offset + 1))" -f "$QLOG" 2>/dev/null &
    tailpid=$!
    ( sleep 3
      while :; do
        worker_alive "$(cat "$WPID" 2>/dev/null)" || break
        sleep 3
      done
      sleep 2
      kill "$tailpid" 2>/dev/null ) &
    watchpid=$!
    wait "$tailpid" 2>/dev/null
    kill "$watchpid" 2>/dev/null
    echo "        ---- end of this run's log; state:  bash scripts/run_queue.sh status ----"
  fi
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
# The mixed-pool positive control was dropped on 2026-09-14: it is not part of the manuscript and
# its point could not be built in time. Its scripts remain (bash scripts/run_mixed_pool.sh point)
# but the queue no longer stops on it. MIX_IN_QUEUE=1 puts its four steps back at the front.
QUEUE=("run_stale_splithalf.sh" "run_stale_splithalf.sh d0" "run_e5_bench.sh d0" "run_e5_bench.sh" \
       "run_e5.sh d100")
if [ "${MIX_IN_QUEUE:-0}" = 1 ]; then
  QUEUE=("run_mixed_pool.sh pool" "run_mixed_pool.sh point" "run_mixed_pool.sh e5" "run_mixed_pool.sh gate" "${QUEUE[@]}")
fi
failed=()
for job in "${QUEUE[@]}"; do
  echo; echo "===== [$(date -u +%H:%M)] $job"
  note "$job"
  bash scripts/$job; rc=$?
  echo "===== [$(date -u +%H:%M)] $job finished (rc=$rc)"
  [ "$rc" -eq 0 ] || failed+=("${job// /:}(rc=$rc)")
done
echo; echo "===== CPU analyses and export"
note "run_analyses.sh"
bash scripts/run_analyses.sh || failed+=("run_analyses.sh(rc=$?)")
if [ "${#failed[@]}" -gt 0 ]; then
  echo "===== FAILED STEPS: ${failed[*]}"
  echo "      rerun:  bash scripts/run_queue.sh   (finished work is skipped; a partial point re-enters its pinned commit)"
  note "done failed=$(IFS=,; echo "${failed[*]}")"
else
  note "done"
fi
kill "$BEAT_PID" 2>/dev/null
echo "===== [$(date -u +%H:%M)] queue done on $HOST; check:  bash scripts/run_queue.sh status"
