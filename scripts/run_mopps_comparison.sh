#!/usr/bin/env bash
# Separate queue: never modifies or restarts the selected-prefix experiment.
set -euo pipefail
LAUNCHER_SELF=$(cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")
cd "$(dirname "$0")/.."
MODE=${1:-run}
[ "$#" -eq 0 ] || shift
case "$MODE" in prepare|run|retry|stop|status|why|summarize|errors|recover-cost|cpu) ;;
  *) echo 'usage: bash scripts/run_mopps_comparison.sh [prepare|run|retry|stop|status|why|summarize|errors|recover-cost|cpu]'; exit 2 ;;
esac
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export OM_WORK="$WORK"
export OUT_ROOT
OUT_ROOT=$(realpath -m "${MOPPS_ROOT:-$WORK/runs/mopps-comparison-v1}")
PARENT=$(realpath -m "${SWITCH_ROOT:-$WORK/runs/selection-switch-v1}")
case "$OUT_ROOT" in /|"$PWD"|"$WORK"|"$WORK/runs"|"$PARENT") echo '[abort] unsafe comparison root'; exit 2 ;; esac
PY=${MOPPS_PYTHON:-${SWITCH_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$MODE" = cpu ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" -m pytest -q tests/test_mopps.py tests/test_mopps_comparison_gpu.py \
    tests/test_selection_worker_shutdown.py tests/test_selection_nccl_preflight.py "$@"
fi
for arg in "$@"; do
  case "$arg" in --root|--root=*|--parent-root|--parent-root=*) echo '[abort] use MOPPS_ROOT and SWITCH_ROOT'; exit 2 ;; esac
done

# Phone terminals drop: keep the GPU controller off the terminal. GPU modes
# re-launch themselves in their own session with a file console, then only
# follow that file; Ctrl-C ends the view, not the run. Tests and pipelines
# (no tty) keep the direct foreground behaviour; SWITCH_FOREGROUND=1 forces it.
CONSOLE_DIR="$OUT_ROOT/logs"
LAUNCH_HOST=$(hostname | tr -c 'a-zA-Z0-9._-' '_')
PID_FILE="$CONSOLE_DIR/launcher.$LAUNCH_HOST.pid"
CONSOLE_LOG="$CONSOLE_DIR/console.$LAUNCH_HOST.log"
launcher_pid_alive() {
  [ -f "$PID_FILE" ] || return 1
  local pid
  pid=$(cat "$PID_FILE" 2>/dev/null) || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}
# Read-only viewers (status --watch, why, errors) also carry OUT_ROOT; only
# launchers, controllers, ranks, probes, scorers and keepalives are "the run".
root_worker_cmdline() {
  { tr '\0' ' ' < "/proc/$1/cmdline"; } 2>/dev/null | grep -qE \
    'run_selection_switch\.sh|run_mopps_comparison\.sh|selection_switch_runtime\.py|selection_switch_gpu\.py|mopps_comparison_gpu\.py|torch\.distributed\.run|train_[a-z_]*grpo\.py|_gpu_keepalive\.py|selection_nccl_preflight\.py|selection_switch_score\.py|light_selection_gate_gpu\.py'
}
if [ "$MODE" = stop ]; then
  # Launchers started before detachment (foreground, tmux) have no pid file but
  # still export OUT_ROOT; so do their workers, ranks and keepalives. Find every
  # own process carrying this root, TERM its process group (the new code reaps
  # ranks and closes receipts on TERM), then sweep what is left.
  stop_root_processes() {
    local pid stat pgid found=0 groups=""
    for pid in $(ls /proc | grep -E '^[0-9]+$'); do
      [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ] || continue
      [ -O "/proc/$pid" ] || continue
      { tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null | grep -qx "OUT_ROOT=$OUT_ROOT" || continue
      root_worker_cmdline "$pid" || continue
      stat=$(cut -d')' -f2 "/proc/$pid/stat" 2>/dev/null) || continue
      pgid=$(echo "$stat" | awk '{print $3}')
      [ -n "$pgid" ] && [ "$pgid" != "$$" ] || continue
      found=1
      echo "[stop] found pid=$pid pgid=$pgid $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-90)"
      case " $groups " in *" $pgid "*) ;; *) groups="$groups $pgid" ;; esac
    done
    [ "$found" -eq 1 ] || return 1
    for pgid in $groups; do kill -TERM -- "-$pgid" 2>/dev/null || true; done
    for _ in $(seq 1 180); do
      sleep 1
      local alive=0
      for pgid in $groups; do kill -0 -- "-$pgid" 2>/dev/null && alive=1; done
      [ "$alive" -eq 1 ] || break
    done
    for pgid in $groups; do
      if kill -0 -- "-$pgid" 2>/dev/null; then echo "[stop] pgid=$pgid still alive after 180s; not killing harder (GPU ranks would be orphaned)"; fi
    done
    return 0
  }
  if ! launcher_pid_alive; then
    echo "[stop] no detached launcher pid file on $LAUNCH_HOST; looking for older launchers and workers of this root"
    if stop_root_processes; then
      echo "[stop] TERM sent to every process group of this root; sweeping orphaned GPU ranks whose driver is gone"
    else
      echo "[stop] no process of this root is running on $LAUNCH_HOST"
    fi
    CUDA_VISIBLE_DEVICES="" "$PY" src/queue_status.py --kill-orphans 2>/dev/null || true
    exit 0
  fi
  pid=$(cat "$PID_FILE")
  echo "[stop] host=$LAUNCH_HOST pid=$pid: sending TERM; the worker reaps its GPU ranks and closes cost receipts"
  # The detached launcher leads its own session: TERM to the group reaches the
  # controller and its worker together; both tolerate repeated stop signals.
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 180); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[stop] pid=$pid still running after 180s; not killing harder (GPU ranks would be orphaned). Retry stop or inspect: $CONSOLE_LOG"
    exit 1
  fi
  echo "[stop] launcher exited; last console lines:"
  tail -n 5 "$CONSOLE_LOG" 2>/dev/null || true
  stop_root_processes && echo "[stop] leftover processes of this root were also signalled"
  CUDA_VISIBLE_DEVICES="" "$PY" src/queue_status.py --kill-orphans 2>/dev/null || true
  exit 0
fi
case "$MODE" in run|retry)
  if [ -t 1 ] && [ "${SWITCH_DETACHED:-0}" != 1 ] && [ "${SWITCH_FOREGROUND:-0}" != 1 ]; then
    if launcher_pid_alive; then
      echo "[already running] host=$LAUNCH_HOST pid=$(cat "$PID_FILE"); follow: tail -f $CONSOLE_LOG; stop: bash scripts/$(basename "$0") stop"
      exit 0
    fi
    older=""
    for pid in $(ls /proc | grep -E '^[0-9]+$'); do
      [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ] || continue
      [ -O "/proc/$pid" ] || continue
      { tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null | grep -qx "OUT_ROOT=$OUT_ROOT" || continue
      root_worker_cmdline "$pid" || continue
      older="$older $pid"
    done
    if [ -n "$older" ]; then
      echo "[already running] host=$LAUNCH_HOST: older launcher/worker processes of this root without a pid file:$older"
      echo "[already running] stop them first with: bash scripts/$(basename "$0") stop"
      exit 0
    fi
    mkdir -p "$CONSOLE_DIR"
    touch "$CONSOLE_LOG"
    offset=$(stat -c %s "$CONSOLE_LOG")
    SWITCH_DETACHED=1 setsid nohup bash "$LAUNCHER_SELF" "$MODE" "$@" >> "$CONSOLE_LOG" 2>&1 < /dev/null &
    pid=$!
    disown 2>/dev/null || true
    echo "$pid" > "$PID_FILE"
    echo "[detached] host=$LAUNCH_HOST pid=$pid mode=$MODE console=$CONSOLE_LOG"
    echo "[detached] Ctrl-C leaves the run going; stop with: bash scripts/$(basename "$0") stop"
    tail --pid="$pid" -c +"$((offset+1))" -F "$CONSOLE_LOG" 2>/dev/null || true
    rc=$(grep -o 'launcher-exit\] pid='"$pid"' mode=[a-z]* rc=[0-9]*' "$CONSOLE_LOG" | tail -n 1 | grep -o 'rc=[0-9]*' | cut -d= -f2)
    exit "${rc:-0}"
  fi
  ;;
esac
case "$MODE" in
  run|retry|prepare|summarize)
    if [ "${SWITCH_RUNTIME_REPO:-}" != "$PWD" ]; then
      exec "$PY" scripts/selection_switch_runtime.py --repo "$PWD" \
        --cache "${SWITCH_RUNTIME_CACHE:-/tmp/offpolicy-misranking-$(id -u)/switch-runtimes}" \
        --kind mopps -- "$MODE" "$@"
    fi
    export OM_REPO="$PWD"
    ;;
esac
if [ "$MODE" = prepare ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" src/mopps_comparison_gpu.py prepare --root "$OUT_ROOT" --parent-root "$PARENT" "$@"
fi
if [ "$MODE" = status ]; then
  # Detailed read-only view in the switch status layout: nodes, current work,
  # node admission probes, parent prefixes, per-state grid with wait reasons.
  # 'status --brief' keeps the queue's own compact list.
  export CUDA_VISIBLE_DEVICES=""
  if [ "${1:-}" = --brief ]; then
    shift
    exec "$PY" src/mopps_comparison_gpu.py status --root "$OUT_ROOT" "$@"
  fi
  exec "$PY" scripts/mopps_comparison_status.py --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = summarize ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" src/mopps_comparison_gpu.py "$MODE" --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = why ]; then
  # One read-only report file for diagnosis: status, every recorded protocol,
  # failure, progress, decision, result, cost and node-admission record, then
  # the tail of every log under the comparison root. Prints the saved path.
  [ -d "$OUT_ROOT" ] || { echo "[abort] no logs/results: $OUT_ROOT"; exit 2; }
  REPORT_DIR="$WORK/reports/mopps-comparison"
  mkdir -p "$REPORT_DIR"
  TARGET=$(mktemp "$REPORT_DIR/mopps-why-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.txt")
  (
    printf 'MOPPS COMPARISON\nUTC: %s\nROOT: %s\nPARENT: %s\nCOMMIT: ' "$(date -u +%FT%TZ)" "$OUT_ROOT" "$PARENT"
    git rev-parse HEAD
    if [ -f "$OUT_ROOT/mopps.json" ]; then
      CUDA_VISIBLE_DEVICES="" "$PY" scripts/mopps_comparison_status.py --root "$OUT_ROOT" --all || echo '[status unavailable]'
      CUDA_VISIBLE_DEVICES="" "$PY" src/mopps_comparison_gpu.py status --root "$OUT_ROOT" || echo '[brief status unavailable]'
    else
      echo '[not prepared] mopps.json is absent'
    fi
    printf '\n===== parent switch status =====\n'
    CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_status.py --root "$PARENT" || echo '[parent status unavailable]'
    while IFS= read -r -d '' path; do
      printf '\n===== %s =====\n' "${path#"$OUT_ROOT"/}"
      cat "$path"
      printf '\n'
    done < <(find "$OUT_ROOT" -type f \( -name 'mopps.json' -o -name 'contract.json' -o -name 'import.done.json' \
      -o -name 'selector.json' -o -name 'failure.json' -o -name 'progress.json' -o -name 'result.json' \
      -o -name 'result.sha256.json' -o -name 'budget_stop.json' -o -name 'cost.jsonl' -o -name 'admission.json' \
      -o -name 'rank-*.json' -o -name '*-runtime.json' \
      -o -path '*/cost-events/*.json' -o -path '*/pending-costs/*.json' \) -print0 | sort -z)
    while IFS= read -r -d '' path; do
      printf '\n===== LOG: %s (last 100 lines) =====\n' "${path#"$OUT_ROOT"/}"
      tail -n 100 "$path"
    done < <(find "$OUT_ROOT" -type f -name '*.log' -print0 | sort -z)
  ) > "$TARGET" 2>&1 || { printf '[report incomplete; see errors] %s\n' "$TARGET"; exit 1; }
  printf '[saved] %s\n' "$TARGET"
  exit 0
fi
if [ "$MODE" = errors ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" scripts/selection_switch_errors.py --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = recover-cost ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" scripts/recover_selection_switch_cost.py --root "$OUT_ROOT" "$@"
fi
# Preparation checks disjoint roots before even creating launcher logs.
CUDA_VISIBLE_DEVICES="" "$PY" src/mopps_comparison_gpu.py prepare --root "$OUT_ROOT" --parent-root "$PARENT"
mkdir -p "$OUT_ROOT/logs"
HOST=$(hostname | tr -c 'a-zA-Z0-9._-' '_')
exec > >(tee -p -a "$OUT_ROOT/logs/launcher.$HOST.log") 2>&1
printf '[launcher-start] host=%s pid=%s mode=%s commit=%s utc=%s\n' "$HOST" "$$" "$MODE" "${SWITCH_RUNTIME_COMMIT:-unknown}" "$(date -u +%FT%TZ)"
trap 'rc=$?; printf "[launcher-exit] pid=%s mode=%s rc=%s utc=%s\n" "$$" "$MODE" "$rc" "$(date -u +%FT%TZ)"' EXIT
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
source scripts/_e5_node.sh
export E5_FORCE=0
e5_acquire_node
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t GPUS < <(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader)
  export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"
fi
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
[ "${#GPUS[@]}" -eq 4 ] || { echo '[abort] four allocated GPUs required'; exit 2; }
MEMORY=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
while read -r used; do
  [[ "$used" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || { echo '[abort] invalid GPU memory status'; exit 2; }
  [ "$used" -le 4000 ] || { echo '[busy] GPU occupied; existing jobs were not stopped'; exit 75; }
done <<< "$MEMORY"
# The allocation is reclaimed when its GPUs sit idle, which is what a launcher
# looks like while it waits for a prerequisite or holds between passes. Keep a
# tiny kernel running on every visible GPU for the launcher's lifetime
# (operator launches only; SWITCH_KEEPALIVE=0 disables it). Started after the
# occupancy check so it never counts as an existing job, killed on exit.
KEEPALIVE_PID=
stop_keepalive() { [ -n "$KEEPALIVE_PID" ] && kill -TERM "$KEEPALIVE_PID" 2>/dev/null; KEEPALIVE_PID=; }
if [ "${SWITCH_KEEPALIVE:-1}" != 0 ] && { [ "${SWITCH_DETACHED:-0}" = 1 ] || [ -t 1 ]; }; then
  "$PY" scripts/_gpu_keepalive.py > "$OUT_ROOT/logs/keepalive.$HOST.log" 2>&1 &
  KEEPALIVE_PID=$!
  echo "[keepalive] pid=$KEEPALIVE_PID keeps the allocated GPUs busy while this launcher waits or holds (log: logs/keepalive.$HOST.log)"
  trap 'rc=$?; stop_keepalive; printf "[launcher-exit] pid=%s mode=%s rc=%s utc=%s\n" "$$" "$MODE" "$rc" "$(date -u +%FT%TZ)"' EXIT
fi
MATH_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps")
export PYTHONPATH="$MATH_VERIFY_PATH${PYTHONPATH:+:$PYTHONPATH}" OM_MATH_VERIFIER=math_verify OM_NODE_LOCK_HELD=1
trap '' HUP
source scripts/_selection_worker.sh
rc=0
mopps_retry_failures() {
  # Retry each recorded branch failure once, in path order. Each item is still
  # the explicit single-branch retry of the Python entrypoint, behind the same
  # node admission check. Returns the last nonzero worker rc, or 0.
  local failure rel point arm seed step task_rc result=0
  mapfile -t FAILED < <(find "$OUT_ROOT/states" -mindepth 3 -maxdepth 3 -name failure.json 2>/dev/null | sort)
  [ "${#FAILED[@]}" -gt 0 ] || { echo '[retry] no recorded branch failures'; return 0; }
  for failure in "${FAILED[@]}"; do
    [ -f "$failure" ] || continue
    rel=${failure#"$OUT_ROOT/states/"}
    point=${rel%%/*}; arm=${rel#*/}; arm=${arm%%/*}
    seed=${point#s}; seed=${seed%%-t*}; step=${point##*-t}
    echo "[retry] s$seed/t$step/$arm ($failure)"
    task_rc=0
    # Other nodes may already own this branch. Try the next one without
    # spending the single-branch controller's default 600s waiting on its lock.
    selection_run_worker "$PY" scripts/selection_nccl_preflight.py --root "$OUT_ROOT" -- \
      "$PY" src/mopps_comparison_gpu.py retry --root "$OUT_ROOT" --seed "$seed" --step "$step" --arm "$arm" \
      --idle-timeout 0 || task_rc=$?
    [ "$task_rc" -eq 0 ] || result=$task_rc
    case "$task_rc" in
      78|130|137|143)
        echo "[retry stopped] worker rc=$task_rc; remaining branch failures were not retried"
        return "$task_rc" ;;
    esac
  done
  return "$result"
}
if [ "$MODE" = retry ] && [ "$#" -eq 0 ]; then
  mopps_retry_failures || rc=$?
else
  # Keep the node between passes (see run_selection_switch.sh): the allocation
  # ends with this process, and MoPPS work appears only as the switch experiment
  # publishes prefixes. SWITCH_HOLD_SECONDS=0 restores the single pass.
  # Holding applies to operator launches (detached or on a terminal); pipelines
  # and tests without a terminal keep the single pass unless they opt in.
  if [ "${SWITCH_DETACHED:-0}" = 1 ] || [ -t 1 ]; then HOLD_DEFAULT=600; else HOLD_DEFAULT=0; fi
  HOLD=${SWITCH_HOLD_SECONDS:-$HOLD_DEFAULT}
  [[ "$HOLD" =~ ^[0-9]+$ ]] || { echo '[abort] SWITCH_HOLD_SECONDS must be a whole number of seconds'; exit 2; }
  mopps_complete() {
    [ "$(CUDA_VISIBLE_DEVICES="" "$PY" src/mopps_comparison_gpu.py status --root "$OUT_ROOT" 2>/dev/null | grep -c ' DONE ')" -ge 12 ]
  }
  # The queue skips branches with a recorded failure, so a fresh node would only
  # wait for missing parent prefixes while the failed branches sit untouched.
  # Retry them once per launch first (this node has just passed admission);
  # MOPPS_AUTO_RETRY=0 keeps the old behaviour of leaving them to 'retry'.
  if [ "$MODE" = run ] && [ "$#" -eq 0 ] && [ "${MOPPS_AUTO_RETRY:-1}" != 0 ]; then
    mopps_retry_failures || rc=$?
    case "$rc" in 78|130|137|143) echo '[blocked] node admission or stop during the initial retry pass'; exit "$rc" ;; esac
    rc=0
  fi
  pass=0
  wait_seconds=$HOLD
  while :; do
    pass=$((pass+1))
    rc=0
    selection_run_worker "$PY" scripts/selection_nccl_preflight.py --root "$OUT_ROOT" -- \
      "$PY" src/mopps_comparison_gpu.py "$MODE" --root "$OUT_ROOT" "$@" || rc=$?
    case "$rc" in 78|130|137|143) break ;; esac
    if [ "$rc" -ne 0 ]; then
      CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_errors.py --root "$OUT_ROOT" || true
    fi
    [ "$MODE" = run ] && [ "$HOLD" -gt 0 ] || break
    if mopps_complete; then
      echo '[hold] all 12 comparison continuations are published; releasing the node'
      rc=0
      break
    fi
    if [ "$rc" -eq 0 ]; then wait_seconds=$HOLD; else wait_seconds=$(( wait_seconds*2 > 3600 ? 3600 : wait_seconds*2 )); fi
    echo "[hold] pass $pass ended rc=$rc; keeping this node's GPUs; next pass in ${wait_seconds}s (stop: bash scripts/run_mopps_comparison.sh stop)"
    selection_hold_node "$wait_seconds" || { rc=$?; break; }
  done
fi
if [ "$rc" -eq 78 ]; then
  echo '[blocked] node admission failed above; historical branch errors are not the cause of this launch'
elif [ "$rc" -ne 0 ] && [ "$MODE" = retry ]; then
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_errors.py --root "$OUT_ROOT" || true
  if [ "$rc" -eq 1 ]; then
    echo '[next] inspect the recorded failures; on a passing idle node use: bash scripts/run_mopps_comparison.sh retry'
  fi
fi
exit "$rc"
