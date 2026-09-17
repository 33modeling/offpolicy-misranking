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
#   bash scripts/run_experiments.sh why      one report file for both experiments
#                                            (also what either launcher's why writes)
#
# EXPERIMENTS_HOLD_SECONDS (default 300) is the pause between passes,
# EXPERIMENTS_AUTO_PULL=0 stops the per-pass 'git pull --ff-only' and in-place restart on new code.
set -euo pipefail
LAUNCHER_SELF=$(cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")
cd "$(dirname "$0")/.."
MODE=${1:-run}
[ "$#" -eq 0 ] || shift
case "$MODE" in run|stop|status|progress|why) ;;
  *) echo 'usage: bash scripts/run_experiments.sh [run|stop|status|progress|why]'; exit 2 ;;
esac
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export OM_WORK="$WORK"
# OUT_ROOT is the per-experiment marker the inner launchers, workers and stop
# sweeps use to recognise "the run"; this launcher and its keepalive must not
# carry it (run_selection_switch.sh exports it before handing over to us).
unset OUT_ROOT
SWITCH_ROOT=$(realpath -m "${SWITCH_ROOT:-$WORK/runs/selection-switch-v1}")
MOPPS_ROOT=$(realpath -m "${MOPPS_ROOT:-$WORK/runs/mopps-comparison-v1}")
export SWITCH_ROOT MOPPS_ROOT
PY=${SWITCH_PYTHON:-${MOPPS_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
LOG_DIR="$WORK/runs/experiments/logs"
# Node identity (hostname plus GPU suffix); copies of this launcher without the helper use the hostname.
if [ -f scripts/_node_id.sh ]; then source scripts/_node_id.sh; fi
export EXPERIMENTS_NODE_ID=${EXPERIMENTS_NODE_ID:-$(hostname)}
HOST=$(printf '%s\n' "$EXPERIMENTS_NODE_ID" | tr -c 'a-zA-Z0-9._-' '_')
PID_FILE="$LOG_DIR/launcher.$HOST.pid"
CONSOLE_LOG="$LOG_DIR/console.$HOST.log"
launcher_pid_alive() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid=$(cat "$PID_FILE" 2>/dev/null) || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}
if [ "$MODE" = progress ]; then
  # One phone-width screen: per experiment root, branch counts, each running
  # branch with its updates so far and node, each failed branch with why. Read-only.
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" scripts/experiments_progress.py --work "$WORK" "$@"
fi
if [ "$MODE" = status ]; then
  # One screen for both experiments (switch first, MoPPS second, this node's
  # GPUs once). Accepts --all, --json and --watch [seconds]. Read-only.
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" scripts/experiments_status.py --switch-root "$SWITCH_ROOT" --mopps-root "$MOPPS_ROOT" "$@"
fi
if [ "$MODE" = why ]; then
  # One read-only report: the combined status screen, then each experiment's own
  # why report (records, logs, node launcher console) back to back. Prints the path.
  export CUDA_VISIBLE_DEVICES=""
  REPORT_DIR="$WORK/reports/experiments"
  mkdir -p "$REPORT_DIR"
  TARGET=$(mktemp "$REPORT_DIR/experiments-why-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.txt")
  rc=0
  (
    printf 'EXPERIMENTS WHY\nUTC: %s\nSWITCH ROOT: %s\nMOPPS ROOT: %s\nCOMMIT: ' "$(date -u +%FT%TZ)" "$SWITCH_ROOT" "$MOPPS_ROOT"
    git rev-parse HEAD 2>/dev/null || echo unknown
    "$PY" scripts/experiments_status.py --switch-root "$SWITCH_ROOT" --mopps-root "$MOPPS_ROOT" --all || echo '[status unavailable]'
    for launcher in run_selection_switch.sh run_mopps_comparison.sh; do
      printf '\n\n######## %s why ########\n' "$launcher"
      part=$(EXPERIMENTS_COMBINED=0 bash "scripts/$launcher" why 2>&1) || true
      # The inner report prints its path last, as "[saved] PATH" or "[... incomplete; see errors] PATH".
      inner_path=$(printf '%s\n' "$part" | tail -n 1 | sed -E 's/^\[[^]]*\] //')
      if [ -f "$inner_path" ]; then
        printf '%s\n' "$part" | grep -v '^\[saved\]' || true
        cat "$inner_path"
      else
        printf '%s\n' "$part"
      fi
    done
  ) > "$TARGET" 2>&1 || rc=$?
  [ "$rc" -eq 0 ] || printf '[report incomplete; see errors] %s\n' "$TARGET"
  printf '[saved] %s\n' "$TARGET"
  exit "$rc"
fi
# Inner launchers: foreground, single pass, no hold, no keepalive (this launcher holds the node).
# EXPERIMENTS_INNER replaces bash for the inner launchers in tests only.
inner() {
  env -u EXPERIMENTS_DETACHED -u OUT_ROOT SWITCH_FOREGROUND=1 SWITCH_HOLD_SECONDS=0 SWITCH_KEEPALIVE=0 "${EXPERIMENTS_INNER:-bash}" "$@"
}
# --- node cleanup: the node is ours; nothing of an earlier run may hold it ---
ROOT_PROCESS_PATTERN='run_selection_switch\.sh|run_mopps_comparison\.sh|selection_switch_runtime\.py|selection_switch_gpu\.py|mopps_comparison_gpu\.py|torch\.distributed\.run|train_[a-z_]*grpo\.py|_gpu_keepalive\.py|selection_nccl_preflight\.py|selection_switch_score\.py|light_selection_gate_gpu\.py'
MY_PGID=$(cut -d')' -f2 "/proc/$$/stat" | awk '{print $3}')
pgid_of() { cut -d')' -f2 "/proc/$1/stat" 2>/dev/null | awk '{print $3}'; }
# A process group is alive while it has a member that is not a zombie
# (kill -0 on the group also counts unreaped zombies).
group_alive() {
  local pid fields
  for pid in $(ls /proc | grep -E '^[0-9]+$'); do
    fields=$(cut -d')' -f2 "/proc/$pid/stat" 2>/dev/null) || continue
    [ "$(echo "$fields" | awk '{print $3}')" = "$1" ] || continue
    [ "$(echo "$fields" | awk '{print $1}')" = "Z" ] || return 0
  done
  return 1
}
cmdline_of() { { tr '\0' ' ' < "/proc/$1/cmdline"; } 2>/dev/null | cut -c1-90; }
# Process groups of our own leftover experiment processes (either root's
# marker, our command names) outside this launcher's group.
leftover_groups() {
  local pid marker pgid
  for pid in $(ls /proc | grep -E '^[0-9]+$'); do
    [ "$pid" != "$$" ] && [ -O "/proc/$pid" ] || continue
    marker=$({ tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null | grep -m1 '^OUT_ROOT=' | cut -d= -f2-) || true
    case "$marker" in "$WORK"/runs/*) ;; *) continue ;; esac
    cmdline_of "$pid" | grep -qE "$ROOT_PROCESS_PATTERN" || continue
    pgid=$(pgid_of "$pid")
    [ -n "$pgid" ] && [ "$pgid" != "$MY_PGID" ] || continue
    echo "[clean] leftover pid=$pid pgid=$pgid $(cmdline_of "$pid")" >&2
    echo "$pgid"
  done | sort -u
}
# Process groups of our own processes still holding memory on the visible GPUs.
gpu_holder_groups() {
  local pid mem pgid
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  timeout 20 nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | while IFS=, read -r pid mem; do
    pid=$(echo "$pid" | tr -d ' '); mem=$(echo "$mem" | tr -d ' ')
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    if [ ! -O "/proc/$pid" ]; then echo "[clean] gpu pid=$pid ${mem}MiB belongs to another user; cannot stop it" >&2; continue; fi
    pgid=$(pgid_of "$pid")
    [ -n "$pgid" ] && [ "$pgid" != "$MY_PGID" ] || continue
    echo "[clean] gpu pid=$pid pgid=$pgid ${mem}MiB $(cmdline_of "$pid")" >&2
    echo "$pgid"
  done | sort -u
}
gpu_memory_line() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo "no nvidia-smi"; return 0; }
  timeout 20 nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); printf "gpu%s %sMiB  ", $1, $2}'
}
gpus_free() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local used
  while read -r used; do
    used=$(echo "$used" | tr -d ' ')
    [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -le 4000 ] || return 1
  done < <(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
}
clean_node() {
  echo "[clean] host=$HOST: stopping leftover experiment processes and freeing the allocated GPUs"
  local groups g alive
  groups=$({ leftover_groups; gpu_holder_groups; } | sort -u | tr '\n' ' ')
  if [ -n "${groups// /}" ]; then
    for g in $groups; do kill -TERM -- "-$g" 2>/dev/null || true; done
    for _ in $(seq 1 120); do
      alive=0
      for g in $groups; do group_alive "$g" && alive=1; done
      [ "$alive" -eq 1 ] || break
      sleep 1
    done
    for g in $groups; do
      if group_alive "$g"; then
        echo "[clean] pgid=$g ignored TERM for 120s; killing it (its open cost events close as stale later)"
        kill -KILL -- "-$g" 2>/dev/null || true
      fi
    done
  else
    echo "[clean] no leftover experiment process on $HOST"
  fi
  # (queue_status --kill-orphans is not used here: it stops every process that
  # carries the marker, read-only status viewers included; the sweeps above
  # cover launchers, controllers, ranks, probes, scorers, keepalives and any
  # own process still holding GPU memory.)
  for _ in $(seq 1 30); do gpus_free && break; sleep 2; done
  echo "[clean] gpu memory now: $(gpu_memory_line)"
  gpus_free || echo "[clean] a GPU still holds more than 4000MiB; the pass will report the node as busy"
}
switch_complete() {
  root_complete "$SWITCH_ROOT"
}
mopps_complete() {
  [ "$(CUDA_VISIBLE_DEVICES="" "$PY" src/mopps_comparison_gpu.py status --root "$MOPPS_ROOT" 2>/dev/null | grep -c ' DONE ')" -ge 12 ]
}
# One phrase per inner launcher exit code, for the hold lines and the status view.
rc_reason() {
  case "$1" in
    0) echo "nothing left to claim" ;;
    1) echo "failed tasks, see [failed] lines above" ;;
    75) echo "node busy: lock held or GPUs occupied" ;;
    78) echo "admission failed: NCCL/CUDA probe" ;;
    79) echo "cooling down after a GPU fault; GPU work resumes when the record expires" ;;
    130|143) echo "interrupted" ;;
    skipped) echo "skipped: complete or not prepared" ;;
    *) echo "launcher error, see lines above" ;;
  esac
}
recover_root() {
  [ -f "$1/switch.json" ] || [ -f "$1/mopps.json" ] || return 0
  # One summary line, then one line per open event that could not be closed and why.
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/recover_selection_switch_cost.py --root "$1" --stale --min-age "$STALE_CLOSE" --brief 2>&1 \
    | sed 's/^\[recovery blocked\]/[recover-cost] blocked:/' || true
  # A branch whose attempt hung after a GPU fault until its allocation limit is
  # infrastructure loss, not selector cost: return the allocation, discard the
  # attempt, and let the queue rerun it (waivers/ keeps the receipt).
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/waive_stalled_attempts.py --root "$1" --apply 2>&1 \
    | grep -v 'no failed branch to waive$' | sed 's/^/[auto-waive] /' || true
  # Curve rows metered in a sealed branch ledger by the pre-curve-ledger runtime block
  # the curve retry ("cost ledger changed"); move them to curve/cost.jsonl.
  if [ -f "$1/switch.json" ]; then
    CUDA_VISIBLE_DEVICES="" "$PY" scripts/split_curve_ledger.py --root "$1" --apply 2>&1 \
      | grep -v 'no published branch carries curve rows in its ledger$' | sed 's/^/[auto-repair] /' || true
  fi
}
# Every experiment root on the shared volume: nodes come and go and run several
# experiments, so a start or a stop sweeps them all, not just the two of this launcher.
all_roots() {
  local root
  for root in "$WORK"/runs/*/; do
    root=${root%/}
    if [ -f "$root/switch.json" ] || [ -f "$root/mopps.json" ]; then echo "$root"; fi
  done
}
# Switch roots this node also works on when its own root has nothing claimable:
# the shared queue is one pool of nodes, so a node idles only when every prepared
# experiment is out of claimable work. Priority: the original v1 (its reruns feed
# the paper), then difficulty, hard, quality, long, then the rest by name.
# EXPERIMENTS_HELP_SIBLINGS=0 keeps a node on its own root only.
root_rank() {
  case "$(basename "$1")" in
    selection-switch-v1) echo 0 ;;
    *difficulty*) echo 1 ;;
    *hard*) echo 2 ;;
    *quality*) echo 3 ;;
    *long*) echo 4 ;;
    *) echo 5 ;;
  esac
}
sibling_roots() {
  local root
  for root in $(all_roots); do
    [ -f "$root/switch.json" ] || continue
    [ "$root" != "$SWITCH_ROOT" ] || continue
    printf '%s %s\n' "$(root_rank "$root")" "$root"
  done | sort -k1,1n -k2,2 | cut -d' ' -f2-
}
# With sibling help on, the node stays until the sibling roots are complete too.
siblings_complete() {
  local root
  [ "${EXPERIMENTS_HELP_SIBLINGS:-1}" != 0 ] || return 0
  for root in $(sibling_roots); do
    root_complete "$root" || return 1
  done
  return 0
}
# Any branch claimable right now in a root this node serves (own root first, then the
# siblings it helps): READY in the status snapshot, which reads receipts only.
claimable_work() {
  local root roots
  roots=$SWITCH_ROOT
  [ "${EXPERIMENTS_HELP_SIBLINGS:-1}" != 0 ] && roots="$roots $(sibling_roots | tr '\n' ' ')"
  for root in $roots; do
    [ -f "$root/switch.json" ] || continue
    if CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_status.py --root "$root" --json 2>/dev/null \
        | "$PY" -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if any(t.get("status")=="READY" for t in d.get("tasks",[])) else 1)'; then
      basename "$root"
      return 0
    fi
  done
  return 1
}
root_complete() {
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_status.py --root "$1" --json 2>/dev/null \
    | "$PY" -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("development_done")==18 and d.get("test_done")==30 else 1)'
}
sweep_all_roots() {
  local root
  for root in $(all_roots); do
    if [ -f "$root/switch.json" ]; then
      SWITCH_ROOT=$root EXPERIMENTS_STOPPING=1 bash scripts/run_selection_switch.sh stop 2>&1 | sed "s|^|[sweep $(basename "$root")] |" || true
    else
      MOPPS_ROOT=$root EXPERIMENTS_STOPPING=1 bash scripts/run_mopps_comparison.sh stop 2>&1 | sed "s|^|[sweep $(basename "$root")] |" || true
    fi
  done
}
# Open cost events of dead attempts block their branch's retry. Ones this host
# started are dead once the node is swept (closed at once); ones a killed node
# left behind are closed after EXPERIMENTS_STALE_CLOSE_SECONDS (default 180) of
# silence: the meter heartbeat is written every 15s, so three minutes without it
# means the attempt is gone.
STALE_CLOSE=${EXPERIMENTS_STALE_CLOSE_SECONDS:-180}
close_dead_events() {
  local root
  for root in $(all_roots); do
    CUDA_VISIBLE_DEVICES="" "$PY" scripts/recover_selection_switch_cost.py --root "$root" --stale --min-age 0 --this-host --brief 2>&1 \
      | grep -v ' 0 stale event(s) closed, 0 still open$' | sed "s|^|[sweep $(basename "$root")] |" || true
    CUDA_VISIBLE_DEVICES="" "$PY" scripts/recover_selection_switch_cost.py --root "$root" --stale --min-age "$STALE_CLOSE" --brief 2>&1 \
      | grep -v ' 0 stale event(s) closed, 0 still open$' | sed "s|^|[sweep $(basename "$root")] |" || true
  done
}
full_clean() {
  sweep_all_roots
  clean_node
  close_dead_events
}
stop_node() {
  if launcher_pid_alive; then
    pid=$(cat "$PID_FILE")
    echo "[stop] host=$HOST pid=$pid: sending TERM to the node launcher; inner launchers reap their ranks and close receipts"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 240); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && echo "[stop] pid=$pid still running after 240s; inspect $CONSOLE_LOG"
  else
    echo "[stop] no live node launcher on $HOST (pid file: $PID_FILE)"
  fi
  # Everything of every experiment on this node: launchers, ranks, keepalives, GPU
  # memory, and the cost events those attempts left open.
  full_clean
}
if [ "$MODE" = stop ]; then
  stop_node
  exit 0
fi
# --- run ---
# One command restarts a node: a launcher already running here is stopped first
# (its ranks reaped, receipts closed), the shared checkout is pulled, then the
# node starts fresh. EXPERIMENTS_PULL=0 skips the pull.
if [ "${EXPERIMENTS_DETACHED:-0}" != 1 ]; then
  if launcher_pid_alive; then
    echo "[restart] host=$HOST: a node launcher is already running (pid $(cat "$PID_FILE")); stopping it first"
    stop_node
  fi
  # An operator restart is a deliberate second chance for this node: clear the
  # watchdog's GPU-fault record and let the admission probe decide.
  fault_record="$WORK/runs/experiments/node-faults/$EXPERIMENTS_NODE_ID.json"
  if [ -f "$fault_record" ]; then
    rm -f "$fault_record" && echo "[fault-reset] host=$HOST: cleared the GPU-fault record ($fault_record); the admission probe decides again"
  fi
  if [ "${EXPERIMENTS_PULL:-1}" != 0 ]; then
    if git pull -q --ff-only 2>/dev/null; then
      echo "[pull] checkout at $(git rev-parse --short HEAD)"
    else
      echo "[pull] skipped (offline or diverged); checkout at $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    fi
  fi
fi
if [ -t 1 ] && [ "${EXPERIMENTS_DETACHED:-0}" != 1 ]; then
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
LOADED_REV=$(git rev-parse HEAD 2>/dev/null || true)
printf '[node-launcher-start] host=%s pid=%s utc=%s commit=%s\n' "$HOST" "$$" "$(date -u +%FT%TZ)" "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
KEEPALIVE_PID=
WATCHDOG_PID=
stop_keepalive() {
  [ -n "$KEEPALIVE_PID" ] && kill -TERM "$KEEPALIVE_PID" 2>/dev/null; KEEPALIVE_PID=
  [ -n "$WATCHDOG_PID" ] && kill -TERM "$WATCHDOG_PID" 2>/dev/null; WATCHDOG_PID=
}
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
# Stall watchdog: a training phase whose worker logs stop moving (a rank dead
# after a CUDA fault) is terminated after EXPERIMENTS_STALL_SECONDS (default
# 1500) instead of running to its allocation limit, and this host is recorded
# under runs/experiments/node-faults so no launcher does GPU work here again.
if [ "${EXPERIMENTS_WATCHDOG:-1}" != 0 ]; then
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/_stall_watchdog.py --host "$EXPERIMENTS_NODE_ID" --roots "$SWITCH_ROOT" "$MOPPS_ROOT" $(sibling_roots | tr '\n' ' ') \
    --faults-dir "$WORK/runs/experiments/node-faults" --stall-seconds "${EXPERIMENTS_STALL_SECONDS:-1500}" \
    > "$LOG_DIR/stall.$HOST.log" 2>&1 7>&- 8>&- &
  WATCHDOG_PID=$!
  echo "[watchdog] pid=$WATCHDOG_PID stops a phase whose logs are silent for ${EXPERIMENTS_STALL_SECONDS:-1500}s (log: $LOG_DIR/stall.$HOST.log)"
fi
[[ "$HOLD" =~ ^[0-9]+$ ]] || { echo '[abort] EXPERIMENTS_HOLD_SECONDS must be a whole number of seconds'; exit 2; }
pass=0
wait_seconds=$HOLD
blocked_passes=0
need_clean=1
while :; do
  pass=$((pass+1))
  # Between passes nothing of this node runs, so pull the shared checkout and, if
  # it moved, restart this launcher in place (same pid, same pid file) so fixes
  # reach every node without anyone typing a command. EXPERIMENTS_AUTO_PULL=0 disables it.
  if [ "${EXPERIMENTS_AUTO_PULL:-1}" != 0 ]; then
    git pull -q --ff-only >/dev/null 2>&1 || true
    after=$(git rev-parse HEAD 2>/dev/null || true)
    # Compare with the revision this launcher loaded, not with the checkout before
    # its own pull: a peer node may already have moved the shared checkout.
    if [ -n "$LOADED_REV" ] && [ -n "$after" ] && [ "$after" != "$LOADED_REV" ]; then
      echo "[pull] checkout moved ${LOADED_REV:0:7} -> ${after:0:7}; restarting this launcher with the new code"
      stop_keepalive
      exec env EXPERIMENTS_DETACHED=1 bash "$LAUNCHER_SELF" run
    fi
  fi
  # Before the first pass, and again whenever a pass found the node busy.
  if [ "$need_clean" -eq 1 ] && [ "${EXPERIMENTS_CLEAN:-1}" != 0 ]; then
    full_clean
  fi
  need_clean=0
  recover_root "$SWITCH_ROOT"
  recover_root "$MOPPS_ROOT"
  for root in $(sibling_roots); do recover_root "$root"; done
  rc_switch=0 why_switch=skipped
  if [ "${EXPERIMENTS_SKIP_SWITCH:-0}" != 1 ] && ! switch_complete; then
    echo "[pass $pass] selection switch"
    inner scripts/run_selection_switch.sh || rc_switch=$?
    case "$rc_switch" in 130|143) exit "$rc_switch" ;; esac
    why_switch=$rc_switch
    echo "[pass $pass] selection switch ended: rc=$rc_switch, $(rc_reason "$rc_switch")"
  fi
  # Own root busy-or-blocked means the node itself is unusable; otherwise, once the
  # own root has nothing claimable, take the sibling experiments' work in priority order.
  rc_own=$rc_switch
  helped=""
  if [ "${EXPERIMENTS_HELP_SIBLINGS:-1}" != 0 ] && [ "$rc_switch" -ne 75 ] && [ "$rc_switch" -ne 78 ] && [ "$rc_switch" -ne 79 ]; then
    for root in $(sibling_roots); do
      root_complete "$root" && continue
      name=$(basename "$root")
      echo "[pass $pass] sibling $name"
      rc_sib=0
      SWITCH_ROOT=$root SWITCH_ONLY_SEEDS= SWITCH_ONLY_ARMS= inner scripts/run_selection_switch.sh || rc_sib=$?
      case "$rc_sib" in 130|143) exit "$rc_sib" ;; esac
      echo "[pass $pass] sibling $name ended: rc=$rc_sib, $(rc_reason "$rc_sib")"
      helped="$helped | $name rc=$rc_sib $(rc_reason "$rc_sib")"
      if [ "$rc_sib" -eq 75 ]; then need_clean=1; break; fi
      if [ "$rc_sib" -eq 78 ] || [ "$rc_sib" -eq 79 ]; then break; fi
      [ "$rc_sib" -eq 0 ] && rc_switch=0
    done
  fi
  rc_mopps=0 why_mopps=skipped
  if [ "${EXPERIMENTS_SKIP_MOPPS:-0}" != 1 ] && [ -f "$MOPPS_ROOT/mopps.json" ] && ! mopps_complete; then
    echo "[pass $pass] MoPPS comparison"
    inner scripts/run_mopps_comparison.sh || rc_mopps=$?
    case "$rc_mopps" in 130|143) exit "$rc_mopps" ;; esac
    why_mopps=$rc_mopps
    echo "[pass $pass] MoPPS comparison ended: rc=$rc_mopps, $(rc_reason "$rc_mopps")"
  fi
  reason="switch rc=$rc_own $(rc_reason "$why_switch")$helped | mopps rc=$rc_mopps $(rc_reason "$why_mopps")"
  if [ "$rc_switch" -eq 75 ] || [ "$rc_mopps" -eq 75 ]; then need_clean=1; fi
  if switch_complete && { [ ! -f "$MOPPS_ROOT/mopps.json" ] || mopps_complete; } && siblings_complete; then
    echo '[done] every experiment this node can work on is complete; releasing the node'
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
  echo "[hold] pass $pass ended ($reason); keeping this node's GPUs; next pass in ${wait_seconds}s (stop: bash scripts/run_experiments.sh stop)"
  remaining=$wait_seconds
  poll=${EXPERIMENTS_HOLD_POLL_SECONDS:-60}
  since_poll=0
  while [ "$remaining" -gt 0 ]; do
    step=$(( remaining < 15 ? remaining : 15 ))
    [ "$step" -gt "$poll" ] && step=$poll
    printf '[holding] node retained (%s); next pass in %ss\n' "$reason" "$remaining"
    sleep "$step" & wait $! || true
    remaining=$((remaining-step))
    since_poll=$((since_poll+step))
    # A hold is not a timer: the moment a branch becomes claimable (a waiver, a fitted
    # gate, a stale event closed elsewhere), the node goes back to work. A cooling-down
    # or blocked node cannot take GPU work, so it waits the hold out.
    if [ "$since_poll" -ge "$poll" ] && [ "$remaining" -gt 0 ] && [ "$rc_own" -ne 79 ] && [ "$rc_own" -ne 78 ]; then
      since_poll=0
      if found=$(claimable_work); then
        echo "[hold] claimable work in $found; starting the next pass now"
        break
      fi
    fi
  done
done
