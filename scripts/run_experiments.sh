#!/usr/bin/env bash
# One command per node for both experiments. Each pass closes stale costs,
# runs one selection-switch queue pass, then one MoPPS pass (which retries
# recorded failures first), and keeps the node between passes. The node is
# never assigned to one experiment: whatever has claimable work gets it.
#
#   bash scripts/run_experiments.sh          start, or preserve the existing controller
#   bash scripts/run_experiments.sh restart  explicitly interrupt and reload this node
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
case "$MODE" in run|restart|stop|logs|status|progress|why|evidence) ;;
  *) echo 'usage: bash scripts/run_experiments.sh [run|restart|stop|logs|status|progress|why|evidence]'; exit 2 ;;
esac
case "$MODE" in run|restart|stop|logs)
  [ "$#" -eq 0 ] || { echo "[abort] $MODE takes no arguments; existing work untouched"; exit 2; } ;;
esac
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export OM_WORK="$WORK"
if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
  source scripts/_mbpp_experiments.sh
  mbpp_queue_init
  export SWITCH_ROOT="${MBPP_ROOTS[0]}" EXPERIMENTS_SKIP_MOPPS=1
fi
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
if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ] && [ -z "${MBPP_GUARD_PID:-}" ] \
    && [ -f scripts/mbpp_controller_identity.py ]; then
  case "$MODE" in run|restart|stop|logs)
    live_node=$("$PY" scripts/mbpp_controller_identity.py --logs "$LOG_DIR" \
      --work "$WORK" --suite "$EXPERIMENTS_MBPP_SUITE") || exit $?
    if [ -n "$live_node" ] && [ "$live_node" != "$EXPERIMENTS_NODE_ID" ]; then
      echo "[node] preserving live local MBPP controller identity: $live_node (current probe: $EXPERIMENTS_NODE_ID)"
      export EXPERIMENTS_NODE_ID="$live_node"
    fi
    ;;
  esac
fi
HOST=$(printf '%s\n' "$EXPERIMENTS_NODE_ID" | tr -c 'a-zA-Z0-9._-' '_')
PID_FILE="$LOG_DIR/launcher.$HOST.pid"
CONSOLE_LOG="$LOG_DIR/console.$HOST.log"
if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
  # A stale math launcher's PID file is not authority to stop it for MBPP.
  PID_FILE="$LOG_DIR/launcher.mbpp.$HOST.pid"
  CONSOLE_LOG="$LOG_DIR/console.mbpp.$HOST.log"
fi
launcher_pid_alive() {
  local pid candidate command
  if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
    # PID files outlive processes; a reused PID must never authorize a signal.
    # Accept the old shared filename only when it really belongs to MBPP here.
    for candidate in "$PID_FILE" "$LOG_DIR/launcher.$HOST.pid"; do
      pid=$(cat "$candidate" 2>/dev/null) || continue
      [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null || continue
      command=$({ tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null) || continue
      case "$command" in *run_experiments.sh*|*_mbpp_node_guard.py*) ;; *) continue ;; esac
      { tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null | grep -Fxq "OM_WORK=$WORK" || continue
      { tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null | grep -Fxq "EXPERIMENTS_NODE_ID=$EXPERIMENTS_NODE_ID" || continue
      { tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null | grep -Eq '^EXPERIMENTS_MBPP_SUITE=(all|fresh|quality|difficulty|long)$' || continue
      NODE_LAUNCHER_PID=$pid
      return 0
    done
    if [ -z "${MBPP_GUARD_PID:-}" ] && [ -f scripts/mbpp_controller_identity.py ]; then
      pid=$("$PY" scripts/mbpp_controller_identity.py --logs "$LOG_DIR" \
        --work "$WORK" --suite "$EXPERIMENTS_MBPP_SUITE" --field pid \
        --node "$EXPERIMENTS_NODE_ID") || exit $?
      if [[ "$pid" =~ ^[0-9]+$ ]]; then
        NODE_LAUNCHER_PID=$pid
        return 0
      fi
    fi
    return 1
  fi
  [ -f "$PID_FILE" ] || return 1
  pid=$(cat "$PID_FILE" 2>/dev/null) || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null || return 1
  command=$({ tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null) || return 1
  case "$command" in *run_experiments.sh*) ;; *) return 1 ;; esac
  { tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null | grep -Fxq "OM_WORK=$WORK" || return 1
  { tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null | grep -Fxq "EXPERIMENTS_NODE_ID=$EXPERIMENTS_NODE_ID" || return 1
  NODE_LAUNCHER_PID=$pid
}
if [ "$MODE" = progress ]; then
  # One phone-width screen: per experiment root, branch counts, each running
  # branch with its updates so far and node, each failed branch with why. Read-only.
  export CUDA_VISIBLE_DEVICES=""
  PROGRESS_ROOTS=()
  if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
    # MBPP progress must show the MBPP suites only: the generic screen lists every
    # prepared root under runs/, which buried the MBPP root under the math suites
    # and omitted MBPP siblings that were still waiting for their prefixes.
    while IFS= read -r root; do PROGRESS_ROOTS+=(--root "$root"); done < <(mbpp_observation_roots)
  fi
  exec "$PY" scripts/experiments_progress.py --work "$WORK" "${PROGRESS_ROOTS[@]}" "$@"
fi
if [ "$MODE" = status ]; then
  # One screen for both experiments (switch first, MoPPS second, this node's
  # GPUs once). Accepts --all, --json and --watch [seconds]. Read-only.
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" scripts/experiments_status.py --switch-root "$SWITCH_ROOT" --mopps-root "$MOPPS_ROOT" "$@"
fi
if [ "$MODE" = evidence ]; then
  # Every prepared switch root on this node, one compact file each, copied home.
  # Read-only and GPU-free; this is what the paper's audit imports.
  export CUDA_VISIBLE_DEVICES=""
  # Enumerated here rather than through all_roots, which is defined further down
  # with the run-mode helpers this read-only path never reaches. Every prepared root
  # goes into ONE file: only one file then has to leave this cluster.
  ROOT_ARGS=()
  for root in "$WORK"/runs/*/; do
    root=${root%/}
    [ -f "$root/switch.json" ] && ROOT_ARGS+=(--root "$root")
  done
  [ "${#ROOT_ARGS[@]}" -gt 0 ] || { echo "[abort] no prepared switch root under $WORK/runs"; exit 2; }
  REPORT_DIR="$WORK/reports/selection-switch"
  mkdir -p "$REPORT_DIR"
  TARGET="$REPORT_DIR/switch-evidence-$(date -u +%Y%m%dT%H%M%SZ).txt"
  PY_BIN=${SWITCH_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
  [ -x "$PY_BIN" ] || PY_BIN=python3
  if "$PY_BIN" scripts/switch_evidence_export.py "${ROOT_ARGS[@]}" --out "$TARGET"; then
    if [ -n "${HOME:-}" ] && [ -d "$HOME" ] && cp -f "$TARGET" "$HOME/" 2>/dev/null; then
      echo "[evidence] copied to $HOME/$(basename "$TARGET")"
    fi
    echo "[evidence] done: $TARGET"
    exit 0
  fi
  echo "[evidence failed] no file written; the error is above"
  exit 1
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
  # Returning after an unclaimable pass is what lets this node serve other
  # suites. HOLD=0 alone does not disable a worker's internal peer wait.
  env -u EXPERIMENTS_DETACHED -u OUT_ROOT SWITCH_FOREGROUND=1 SWITCH_HOLD_SECONDS=0 SWITCH_KEEPALIVE=0 \
    SWITCH_QUEUE_PASS=1 "${EXPERIMENTS_INNER:-bash}" "$@"
}
dispatch_evidence() {
  local root=$1 log_rc=0
  [ -f scripts/queue_dispatch_evidence.py ] || return 0
  # Informational only: one root, CPU metadata, no unbounded node/GPU scan.
  # Snapshot errors/timeouts must never change dispatch or its actual exit code.
  timeout -k 1 3 env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
    "$PY" scripts/queue_dispatch_evidence.py --root "$root" --checkout "${LOADED_REV:-unknown}" || log_rc=$?
  if [ "$log_rc" -ne 0 ]; then
    echo "[dispatch] root=$root metadata logging rc=$log_rc; continuing to worker validation"
  fi
  return 0
}
run_switch_root() {
  dispatch_evidence "$1"
  if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
    mbpp_queue_run "$1"
  else
    SWITCH_ROOT=$1 inner scripts/run_selection_switch.sh
  fi
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
    2) echo "configuration/runtime preflight failed; inspect errors above" ;;
    75) echo "node busy: lock held or GPUs occupied" ;;
    78) echo "admission failed: NCCL/CUDA probe" ;;
    79) echo "cooling down after a GPU fault; GPU work resumes when the record expires" ;;
    80) echo "only checkpoint-review branches remain; saved work preserved, not complete" ;;
    81) echo "repair contract validation blocked; original results preserved" ;;
    130|143) echo "interrupted" ;;
    skipped) echo "skipped: complete or not prepared" ;;
    node-unavailable) echo "skipped: node busy, failed admission or cooling down" ;;
    *) echo "launcher error, see lines above" ;;
  esac
}
recover_root() {
  [ -f "$1/switch.json" ] || [ -f "$1/mopps.json" ] || return 0
  # One summary line, then one line per open event that could not be closed and why.
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/recover_selection_switch_cost.py --root "$1" --stale --min-age "$STALE_CLOSE" --brief 2>&1 \
    | sed 's/^\[recovery blocked\]/[recover-cost] blocked:/' || true
  # A branch whose attempt hung after a GPU fault until its allocation limit is
  # infrastructure loss. Selection failures are excluded: saved rollout/shard
  # work must not be automatically moved away or reused with refunded costs.
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/waive_stalled_attempts.py --root "$1" --apply --automatic 2>&1 \
    | grep -v 'no failed branch to waive$' | sed 's/^/[auto-waive] /' || true
  # Curve rows metered in a sealed branch ledger by the pre-curve-ledger runtime block
  # the curve retry ("cost ledger changed"); move them to curve/cost.jsonl.
  if [ -f "$1/switch.json" ]; then
    CUDA_VISIBLE_DEVICES="" "$PY" scripts/split_curve_ledger.py --root "$1" --apply 2>&1 \
      | grep -v 'no published branch carries curve rows in its ledger$' | sed 's/^/[auto-repair] /' || true
  fi
}
# Prepared roots for lease-checked cost recovery, never process termination.
all_roots() {
  local root
  if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
    for root in "${MBPP_ROOTS[@]}"; do
      [ ! -f "$root/switch.json" ] || printf '%s\n' "$root"
    done
    return 0
  fi
  {
    for root in "$WORK"/runs/*/; do
      root=${root%/}
      if [ -f "$root/switch.json" ] || [ -f "$root/mopps.json" ]; then echo "$root"; fi
    done
    if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
      for root in "${MBPP_ROOTS[@]}"; do
        [ ! -f "$root/switch.json" ] || echo "$root"
      done
    fi
  } | sort -u
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
  if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
    # Planned roots prevent early exit and can become ready on another node.
    for root in "${MBPP_ROOTS[@]}"; do
      [ "$root" = "$SWITCH_ROOT" ] || printf '%s\n' "$root"
    done
    return 0
  fi
  while IFS= read -r root; do
    [ -f "$root/switch.json" ] || continue
    [ "$root" != "$SWITCH_ROOT" ] || continue
    printf '%s %s\n' "$(root_rank "$root")" "$root"
  done < <(all_roots) | sort -k1,1n -k2,2 | cut -d' ' -f2-
}
# With sibling help on, the node stays until the sibling roots are complete too.
siblings_complete() {
  local root
  [ "${EXPERIMENTS_HELP_SIBLINGS:-1}" != 0 ] || return 0
  while IFS= read -r root; do
    root_complete "$root" || return 1
  done < <(sibling_roots)
  return 0
}
# Any branch claimable right now in a root this node serves (own root first, then the
# siblings it helps): READY, retryable, or a validated pending MBPP gate fit.
claimable_work() {
  local root
  local roots=("$SWITCH_ROOT") siblings=()
  if [ "${EXPERIMENTS_HELP_SIBLINGS:-1}" != 0 ]; then
    mapfile -t siblings < <(sibling_roots)
    roots+=("${siblings[@]}")
  fi
  for root in "${roots[@]}"; do
    if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ] && mbpp_queue_preparable "$root"; then
      basename "$root"
      return 0
    fi
    [ -f "$root/switch.json" ] || continue
    if CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_status.py --root "$root" --json 2>/dev/null \
        | CUDA_VISIBLE_DEVICES="" "$PY" -c 'import json,sys
d=json.load(sys.stdin)
ready=any(t.get("status")=="READY" or t.get("retryable") is True for t in d.get("tasks",[]))
if not ready and d.get("protocol",{}).get("dataset")=="mbpp":
    sys.path.insert(0,"scripts")
    from mbpp_queue_readiness import gate_fit_claimable
    ready=gate_fit_claimable(d)
sys.exit(0 if ready else 1)'; then
      basename "$root"
      return 0
    fi
  done
  return 1
}
root_complete() {
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_status.py --root "$1" --json 2>/dev/null \
    | "$PY" -c 'import json,sys; d=json.load(sys.stdin); busy=any(t.get("status")=="RUNNING" or t.get("heartbeat_fresh") or t.get("owner_active") or t.get("task_lease_held") for t in d.get("tasks",[])); sys.exit(0 if d.get("development_done")==18 and d.get("test_done")==30 and not busy else 1)'
}
experiments_complete() {
  switch_complete && { [ "${EXPERIMENTS_SKIP_MOPPS:-0}" = 1 ] || [ ! -f "$MOPPS_ROOT/mopps.json" ] || mopps_complete; } && siblings_complete
}
# Recover abandoned cost events only after the recovery tool checks owner and
# meter leases. A timestamp or shared node/root marker does not prove death.
STALE_CLOSE=${EXPERIMENTS_STALE_CLOSE_SECONDS:-180}
close_dead_events() {
  local root
  while IFS= read -r root; do
    CUDA_VISIBLE_DEVICES="" "$PY" scripts/recover_selection_switch_cost.py --root "$root" --stale --min-age 0 --this-host --brief 2>&1 \
      | grep -v ' 0 stale event(s) closed, 0 still open$' | sed "s|^|[sweep $(basename "$root")] |" || true
    CUDA_VISIBLE_DEVICES="" "$PY" scripts/recover_selection_switch_cost.py --root "$root" --stale --min-age "$STALE_CLOSE" --brief 2>&1 \
      | grep -v ' 0 stale event(s) closed, 0 still open$' | sed "s|^|[sweep $(basename "$root")] |" || true
  done < <(all_roots)
}
full_clean() {
  if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
    # The guard reclaims proven children of a dead MBPP controller by token.
    # A busy lease is NOT evidence that every process on this node is stale.
    echo "[clean] host=$HOST: MBPP owner-scoped recovery; no node-wide process/GPU sweep"
    close_dead_events
    return 0
  fi
  # A new controller does not own existing processes merely because their
  # node ID, command or root matches. Busy GPU admission must leave them alone.
  echo "[clean] host=$HOST: lease-checked cost recovery; no node-wide process/GPU sweep"
  close_dead_events
}
stop_node() {
  if launcher_pid_alive; then
    pid=$NODE_LAUNCHER_PID
    if [ -n "${AUTO_RELOAD_PID:-}" ] && [ "$pid" != "$AUTO_RELOAD_PID" ]; then
      echo '[reload] another invocation already replaced the controller; leaving its replacement running'
      return 0
    fi
    echo "[stop] host=$HOST pid=$pid: 기존 실행 종료 요청 (TERM); 자식 GPU 프로세스·비용 기록 정리 대기"
    if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
      # The PID file names the guard, which reaps its token-bound controller/ranks.
      # Never signal an unverified process group based on a stale PID file.
      kill -TERM "$pid" 2>/dev/null || true
    else
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
    stop_started=$SECONDS
    stop_report=$SECONDS
    while launcher_pid_alive && [ "$NODE_LAUNCHER_PID" = "$pid" ] && [ $((SECONDS-stop_started)) -lt 240 ]; do
      if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ] && [ "$SECONDS" -ge "$stop_report" ]; then
        {
          echo "[stop] host=$HOST pid=$pid 종료 대기 $((SECONDS-stop_started))s / 240s; 아직 재시작하지 않았습니다"
          timeout -k 1 5 "$PY" scripts/_mbpp_node_guard.py \
            --lock "$LOG_DIR/mbpp-controller.$HOST.lock" --inspect-cleanup "$pid" || true
        } | tee -a "$LOG_DIR/cleanup.mbpp.$HOST.log"
        stop_report=$((SECONDS+5))
      fi
      sleep 1
    done
    if launcher_pid_alive && [ "$NODE_LAUNCHER_PID" = "$pid" ]; then
      echo "[stop] pid=$pid still running after 240s; new worker was not started. Cleanup log: $LOG_DIR/cleanup.mbpp.$HOST.log"
      return 75
    fi
    echo "[stop] host=$HOST pid=$pid: 기존 컨트롤러 종료 확인"
  else
    echo "[stop] no live node launcher on $HOST (pid file: $PID_FILE)"
  fi
  # Never expand an explicit controller stop into a sweep of other experiments.
  full_clean
}
if [ "$MODE" = logs ]; then
  if launcher_pid_alive; then
    echo "[already running] ${EXPERIMENTS_MBPP_SUITE:+MBPP }controller pid=$NODE_LAUNCHER_PID; existing work continues: $CONSOLE_LOG"
    if [ -t 1 ]; then
      echo '[logs] Ctrl-C closes this viewer; existing workers continue.'
      tail -n 50 -F --pid="$NODE_LAUNCHER_PID" "$CONSOLE_LOG" 2>/dev/null || true
    else
      tail -n 50 "$CONSOLE_LOG" 2>/dev/null || echo "[logs] MBPP console log not available: $CONSOLE_LOG"
    fi
  else
    echo "[logs] no live controller on $HOST; last saved output: $CONSOLE_LOG"
    tail -n 50 "$CONSOLE_LOG" 2>/dev/null || true
  fi
  exit 0
fi
# Generic cleanup spans all prepared roots. A dedicated MBPP controller on this
# allocation is not a leftover: preserve it even when the other entry point is
# used. Reuse its strict PID/work/node/suite verification before any mutation.
if [ -z "${EXPERIMENTS_MBPP_SUITE:-}" ] && \
    EXPERIMENTS_MBPP_SUITE=all PID_FILE="$LOG_DIR/launcher.mbpp.$HOST.pid" launcher_pid_alive; then
  echo "[already running] MBPP controller pid=$NODE_LAUNCHER_PID; existing work continues: $LOG_DIR/console.mbpp.$HOST.log"
  if [ "$MODE" = stop ] || [ "$MODE" = restart ]; then
    echo '[abort] use the MBPP launcher to explicitly stop or restart its controller'
    exit 75
  fi
  exit 0
fi
if [ "$MODE" = stop ]; then
  stop_node
  exit 0
fi
# Repeating run is observational while a controller owns this node. Check
# before pulling shared code: neither an update nor a missing runtime receipt
# authorizes interrupting an active training/evaluation phase.
if [ "$MODE" = run ] && [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ] && \
    [ "${EXPERIMENTS_DETACHED:-0}" != 1 ] && launcher_pid_alive; then
  echo "[already running] MBPP controller pid=$NODE_LAUNCHER_PID; existing work continues: $CONSOLE_LOG"
  if [ -t 1 ]; then
    echo '[logs] Ctrl-C closes this viewer; existing workers continue.'
    tail -n 50 -F --pid="$NODE_LAUNCHER_PID" "$CONSOLE_LOG" 2>/dev/null || true
  else
    tail -n 50 "$CONSOLE_LOG" 2>/dev/null || true
  fi
  exit 0
fi
# Pull only for an idle start or an explicitly requested restart.
# Re-enter the wrapper after a pull so its settings and storage audit are fresh.
if { [ "$MODE" = run ] || [ "$MODE" = restart ]; } && [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ] && [ "${EXPERIMENTS_DETACHED:-0}" != 1 ]; then
  if [ "${EXPERIMENTS_PULL:-1}" != 0 ] && [ "${EXPERIMENTS_START_PULL_DONE:-0}" != 1 ]; then
    before=$(git rev-parse HEAD 2>/dev/null || true)
    if timeout -k 5 20 env GIT_TERMINAL_PROMPT=0 git pull -q --ff-only 2>/dev/null; then
      echo "[pull] checkout at $(git rev-parse --short HEAD)"
    else
      echo '[pull] skipped (offline or diverged); checking the local code'
    fi
    after=$(git rev-parse HEAD 2>/dev/null || true)
    if [ -n "$after" ] && [ "$before" != "$after" ]; then
      exec env EXPERIMENTS_START_PULL_DONE=1 bash scripts/run_mbpp_experiments.sh "$MODE" "$EXPERIMENTS_MBPP_SUITE"
    fi
  fi
  unset EXPERIMENTS_START_PULL_DONE
fi
# A default queue may now select an existing repair directly. Validate its
# immutable contract before stopping a controller or recovering any cost event.
if { [ "$MODE" = run ] || [ "$MODE" = restart ]; } && \
    [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ] && [ -f "$SWITCH_ROOT/repair.json" ]; then
  if ! mbpp_queue_check "$EXPERIMENTS_MBPP_SUITE"; then
    echo '[blocked] MBPP repair contract validation failed before restart or recovery; existing controller and costs preserved [mbpp]' >&2
    exit 81
  fi
fi
if [ "$MODE" = restart ]; then
  stop_node
  if launcher_pid_alive && { [ -z "${AUTO_RELOAD_PID:-}" ] || [ "$NODE_LAUNCHER_PID" = "$AUTO_RELOAD_PID" ]; }; then
    echo '[abort] previous controller has not stopped; refusing a second controller'
    exit 75
  fi
  MODE=run
  unset EXPERIMENTS_DETACHED
fi
# --- run ---
# Run preserves its controller, including a concurrent replacement.
# EXPERIMENTS_PULL=0 skips the pull when starting an idle node.
if [ "${EXPERIMENTS_DETACHED:-0}" != 1 ]; then
  if launcher_pid_alive; then
    if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
      echo "[already running] MBPP controller pid=$NODE_LAUNCHER_PID; existing work continues: $CONSOLE_LOG"
      if [ -t 1 ]; then
        echo '[logs] Ctrl-C closes this viewer; existing workers continue.'
        tail -n 50 -F --pid="$NODE_LAUNCHER_PID" "$CONSOLE_LOG" 2>/dev/null || true
      else
        tail -n 50 "$CONSOLE_LOG" 2>/dev/null || echo "[logs] MBPP console log not available: $CONSOLE_LOG"
      fi
      exit 0
    fi
    echo "[already running] controller pid=$NODE_LAUNCHER_PID; existing work continues: $CONSOLE_LOG"
    exit 0
  fi
  # Keep the legacy reset for non-MBPP launchers. MBPP restart is a code reload,
  # not authority to erase repeated-fault protection or its diagnostic evidence.
  fault_record="$WORK/runs/experiments/node-faults/$EXPERIMENTS_NODE_ID.json"
  if [ -z "${EXPERIMENTS_MBPP_SUITE:-}" ] && [ -f "$fault_record" ]; then
    rm -f "$fault_record" && echo "[fault-reset] host=$HOST: cleared the GPU-fault record ($fault_record); the admission probe decides again"
  fi
  if [ -z "${EXPERIMENTS_MBPP_SUITE:-}" ] && [ "${EXPERIMENTS_PULL:-1}" != 0 ]; then
    if timeout -k 5 20 env GIT_TERMINAL_PROMPT=0 git pull -q --ff-only 2>/dev/null; then
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
  # Only the admitted MBPP controller publishes its guard PID. Two terminal
  # launches racing here must not replace the winner's PID with the loser's.
  if [ -z "${EXPERIMENTS_MBPP_SUITE:-}" ]; then echo "$pid" > "$PID_FILE"; fi
  echo "[detached] host=$HOST pid=$pid console=$CONSOLE_LOG"
  echo "[detached] Ctrl-C leaves the node working; stop with: bash scripts/run_experiments.sh stop"
  viewer_rc=0
  tail --pid="$pid" -c +"$((offset+1))" -F "$CONSOLE_LOG" 2>/dev/null || viewer_rc=$?
  # Closing the viewer never stops the detached controller. If the controller
  # itself exited, preserve its failure instead of reporting successful launch.
  [ "$viewer_rc" -eq 0 ] || exit "$viewer_rc"
  controller_rc=0
  wait "$pid" || controller_rc=$?
  printf '[detached-exit] host=%s pid=%s rc=%s console=%s\n' "$HOST" "$pid" "$controller_rc" "$CONSOLE_LOG"
  exit "$controller_rc"
fi
if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ] && [ "${MBPP_GUARD_PID:-}" != "$PPID" ]; then
  exec "$PY" scripts/_mbpp_node_guard.py \
    --lock "$LOG_DIR/mbpp-controller.$HOST.lock" -- bash "$LAUNCHER_SELF" run
fi
mkdir -p "$LOG_DIR"
LOADED_REV=$(git rev-parse HEAD 2>/dev/null || true)
if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
  LOADED_FINGERPRINT=$("$PY" scripts/_mbpp_node_guard.py --fingerprint)
  "$PY" scripts/_mbpp_node_guard.py --lock "$LOG_DIR/mbpp-controller.$HOST.lock" \
    --record-runtime "$MBPP_GUARD_PID" --loaded-fingerprint "$LOADED_FINGERPRINT"
  # Publish the PID only after its version binding exists; a simultaneous
  # invocation must not mistake a newly admitted controller for a legacy one.
  printf '%s\n' "$MBPP_GUARD_PID" > "$PID_FILE"
fi
printf '[node-launcher-start] host=%s pid=%s utc=%s commit=%s\n' "$HOST" "$$" "$(date -u +%FT%TZ)" "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
KEEPALIVE_PID=
WATCHDOG_PID=
# A shared host/root is not authority to stop another controller's workers.
OM_EXPERIMENT_CONTROLLER_TOKEN=$("$PY" -c 'import uuid; print(uuid.uuid4().hex)')
export OM_EXPERIMENT_CONTROLLER_TOKEN
stop_keepalive() {
  [ -z "$KEEPALIVE_PID" ] || kill -TERM "$KEEPALIVE_PID" 2>/dev/null || true
  [ -z "$WATCHDOG_PID" ] || kill -TERM "$WATCHDOG_PID" 2>/dev/null || true
  KEEPALIVE_PID= WATCHDOG_PID=
}
trap 'rc=$?; stop_keepalive; printf "[node-launcher-exit] pid=%s rc=%s utc=%s\n" "$$" "$rc" "$(date -u +%FT%TZ)"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
HOLD=${EXPERIMENTS_HOLD_SECONDS:-300}
POLL=${EXPERIMENTS_HOLD_POLL_SECONDS:-60}
[[ "$HOLD" =~ ^[0-9]+$ ]] || { echo '[abort] EXPERIMENTS_HOLD_SECONDS must be a whole number of seconds'; exit 2; }
[[ "$POLL" =~ ^[0-9]+$ ]] && [ "$POLL" -gt 0 ] || {
  echo '[abort] EXPERIMENTS_HOLD_POLL_SECONDS must be a positive whole number'; exit 2;
}
if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
  # An in-place reload does not re-enter the wrapper: inherited legacy 600s
  # settings must be bounded here too, regardless of the inner exit code.
  HOLD=$(( HOLD < 1 ? 1 : HOLD > 60 ? 60 : HOLD ))
  POLL=$(( POLL > 5 ? 5 : POLL ))
  echo "[hold-policy] MBPP idle=${HOLD}s poll=${POLL}s maximum=60s; GPU faults require re-admission"
fi
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
  mapfile -t WATCH_ROOTS < <(sibling_roots)
  if [ "${EXPERIMENTS_SKIP_MOPPS:-0}" != 1 ]; then WATCH_ROOTS+=("$MOPPS_ROOT"); fi
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/_stall_watchdog.py --host "$EXPERIMENTS_NODE_ID" --roots "$SWITCH_ROOT" "${WATCH_ROOTS[@]}" \
    --owner-token "$OM_EXPERIMENT_CONTROLLER_TOKEN" \
    --faults-dir "$WORK/runs/experiments/node-faults" --stall-seconds "${EXPERIMENTS_STALL_SECONDS:-1500}" \
    > "$LOG_DIR/stall.$HOST.log" 2>&1 7>&- 8>&- &
  WATCHDOG_PID=$!
  echo "[watchdog] pid=$WATCHDOG_PID stops a phase whose logs are silent for ${EXPERIMENTS_STALL_SECONDS:-1500}s (log: $LOG_DIR/stall.$HOST.log)"
fi
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
    timeout -k 5 20 env GIT_TERMINAL_PROMPT=0 git pull -q --ff-only >/dev/null 2>&1 || true
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
  if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
    # Auto-pull re-enters this controller, not the outer wrapper. Check saved
    # storage before recovery can move anything or a queue pass can retrain.
    if ! CUDA_VISIBLE_DEVICES='' MBPP_STORAGE_AUDIT_AUTOMATIC=1 bash scripts/check_mbpp_storage.sh "$EXPERIMENTS_MBPP_SUITE"; then
      echo '[storage-blocked] no recovery/reset/training pass started; preserve saved work and inspect storage'
      exit 2
    fi
  fi
  if [ "$need_clean" -eq 1 ] && [ "${EXPERIMENTS_CLEAN:-1}" != 0 ]; then
    full_clean
  fi
  need_clean=0
  recover_root "$SWITCH_ROOT"
  if [ "${EXPERIMENTS_SKIP_MOPPS:-0}" != 1 ]; then recover_root "$MOPPS_ROOT"; fi
  while IFS= read -r root; do recover_root "$root"; done < <(sibling_roots)
  rc_switch=0 why_switch=skipped
  if [ "${EXPERIMENTS_SKIP_SWITCH:-0}" != 1 ] && ! switch_complete; then
    echo "[pass $pass] selection switch"
    run_switch_root "$SWITCH_ROOT" || rc_switch=$?
    case "$rc_switch" in 130|143) exit "$rc_switch" ;; esac
    why_switch=$rc_switch
    echo "[pass $pass] selection switch ended: rc=$rc_switch, $(rc_reason "$rc_switch")"
  fi
  # Own root busy-or-blocked means the node itself is unusable; otherwise, once the
  # own root has nothing claimable, take the sibling experiments' work in priority order.
  if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ] && [ "$rc_switch" -eq 2 ]; then
    echo '[blocked] MBPP configuration/runtime preflight failed; no automatic hold; saved work preserved'
    exit 2
  fi
  rc_own=$rc_switch
  helped=""
  if [ "${EXPERIMENTS_HELP_SIBLINGS:-1}" != 0 ] && [ "$rc_switch" -ne 75 ] && [ "$rc_switch" -ne 78 ] && [ "$rc_switch" -ne 79 ] && [ "$rc_switch" -ne 81 ]; then
    while IFS= read -r root; do
      root_complete "$root" && continue
      name=$(basename "$root")
      echo "[pass $pass] sibling $name"
      rc_sib=0
      SWITCH_ONLY_SEEDS= SWITCH_ONLY_ARMS= run_switch_root "$root" || rc_sib=$?
      case "$rc_sib" in 130|143) exit "$rc_sib" ;; esac
      echo "[pass $pass] sibling $name ended: rc=$rc_sib, $(rc_reason "$rc_sib")"
      helped="$helped | $name rc=$rc_sib $(rc_reason "$rc_sib")"
      if [ "$rc_sib" -eq 75 ]; then rc_switch=$rc_sib; need_clean=1; break; fi
      if [ "$rc_sib" -eq 78 ] || [ "$rc_sib" -eq 79 ]; then rc_switch=$rc_sib; break; fi
      if [ "$rc_sib" -eq 81 ]; then rc_switch=$rc_sib; break; fi
      if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ] && [ "$rc_sib" -eq 2 ]; then rc_switch=$rc_sib; break; fi
      # A pending sibling returns 0 without doing work. Do not erase the
      # MBPP failure that needs retrying or reset its retry backoff.
      if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
        if [ "$rc_sib" -ne 0 ]; then rc_switch=$rc_sib; fi
      elif [ "$rc_sib" -eq 0 ]; then
        rc_switch=0
      fi
    done < <(sibling_roots)
  fi
  rc_mopps=0 why_mopps=skipped
  case "$rc_switch" in 75|78|79) why_mopps=node-unavailable ;; esac
  if [ "${EXPERIMENTS_SKIP_MOPPS:-0}" != 1 ] && [ "$rc_switch" -ne 75 ] && [ "$rc_switch" -ne 78 ] && [ "$rc_switch" -ne 79 ] && [ -f "$MOPPS_ROOT/mopps.json" ] && ! mopps_complete; then
    echo "[pass $pass] MoPPS comparison"
    dispatch_evidence "$MOPPS_ROOT"
    # Finish owned work, then yield peer/prerequisite waits to this shared queue.
    inner scripts/run_mopps_comparison.sh || rc_mopps=$?
    case "$rc_mopps" in 130|143) exit "$rc_mopps" ;; esac
    why_mopps=$rc_mopps
    echo "[pass $pass] MoPPS comparison ended: rc=$rc_mopps, $(rc_reason "$rc_mopps")"
  fi
  reason="switch rc=$rc_own $(rc_reason "$why_switch")$helped | mopps rc=$rc_mopps $(rc_reason "$why_mopps")"
  if [ "$rc_switch" -eq 75 ] || [ "$rc_mopps" -eq 75 ]; then need_clean=1; fi
  if experiments_complete; then
    echo '[done] every experiment this node can work on is complete; releasing the node'
    exit 0
  fi
  if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ] && [ "$rc_own" -eq 80 ] && siblings_complete; then
    echo '[WAIT] only MBPP checkpoint-review branches remain; incomplete results preserved; releasing this node without another GPU admission'
    exit 80
  fi
  if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
    case "$rc_switch" in
      2)
        echo '[blocked] MBPP configuration/runtime preflight failed; no automatic hold; saved work preserved'
        exit 2
        ;;
      81)
        echo '[blocked] MBPP repair validation failed; no holding/retry loop; original results preserved'
        exit 81
        ;;
      75|78)
        # Occupancy is not a stale-owner proof. NCCL preflight has already
        # exhausted its bounded, evidence-based probe fallbacks. Neither case
        # should retain this allocation in an endless holding/retry loop.
        echo "[blocked] MBPP node unavailable: $(rc_reason "$rc_switch"); no holding/retry loop; releasing owned processes"
        exit "$rc_switch"
        ;;
    esac
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
  if [ "$rc_switch" -eq 0 ] && [ "$rc_mopps" -eq 0 ]; then
    wait_seconds=$HOLD
  else
    max_wait=3600
    # Bound the scheduler retry, NOT permission to use a faulty GPU: receipt
    # TTL, repeat-fault refusal and the admission probe remain mandatory.
    if [ -n "${EXPERIMENTS_MBPP_SUITE:-}" ]; then
      max_wait=60
    fi
    wait_seconds=$(( wait_seconds*2 > max_wait ? max_wait : wait_seconds*2 ))
  fi
  if [ "$rc_switch" -eq 79 ] || [ "$rc_mopps" -eq 79 ]; then
    fault_wait=$(CUDA_VISIBLE_DEVICES="" "$PY" scripts/node_fault_state.py \
      "$WORK/runs/experiments/node-faults/$EXPERIMENTS_NODE_ID.json" --remaining) || fault_wait=$wait_seconds
    if [[ "$fault_wait" =~ ^[0-9]+$ ]]; then
      # Do not overshoot expiry by another exponential backoff interval.
      fault_wait=$(( fault_wait < 1 ? 1 : fault_wait ))
      wait_seconds=$(( wait_seconds < fault_wait ? wait_seconds : fault_wait ))
    fi
  fi
  echo "[hold] pass $pass ended ($reason); keeping this node's GPUs; next pass in ${wait_seconds}s (stop: bash scripts/run_experiments.sh stop)"
  remaining=$wait_seconds
  hold_deadline=$((SECONDS+wait_seconds))
  poll=$POLL
  since_poll=0
  while [ "$remaining" -gt 0 ]; do
    step=$(( remaining < 15 ? remaining : 15 ))
    [ "$step" -gt "$poll" ] && step=$poll
    printf '[holding] node retained (%s); next pass in %ss\n' "$reason" "$remaining"
    sleep "$step" & wait $! || true
    # Status I/O counts toward the deadline, not as extra unreported delay.
    remaining=$((hold_deadline-SECONDS))
    since_poll=$((since_poll+step))
    # A hold is not a timer: the moment a branch becomes claimable (a waiver, a fitted
    # gate, a stale event closed elsewhere), the node goes back to work. A cooling-down
    # or blocked node cannot take GPU work, so it waits the hold out.
    if [ "$since_poll" -ge "$poll" ] && [ "$remaining" -gt 0 ]; then
      since_poll=0
      if experiments_complete; then
        echo '[done] peers completed every experiment; releasing the node during hold'
        exit 0
      fi
      if [ "$rc_switch" -ne 79 ] && [ "$rc_switch" -ne 78 ] && [ "$rc_mopps" -ne 79 ] && [ "$rc_mopps" -ne 78 ] && found=$(claimable_work); then
        echo "[hold] claimable work in $found; starting the next pass now"
        break
      fi
    fi
    remaining=$((hold_deadline-SECONDS))
  done
done
