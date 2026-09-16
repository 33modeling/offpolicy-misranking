#!/usr/bin/env bash
# Selected-prefix experiment. Never stops or rewrites E5, Qwen or net-gain jobs.
set -euo pipefail
LAUNCHER_SELF=$(cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")
cd "$(dirname "$0")/.."
MODE=${1:-run}
[ "$#" -eq 0 ] || shift
case "$MODE" in run|smoke|prepare|status|fit|summarize|export|why|live|cpu|recover-cost|errors|check-code|stop) ;;
  *) echo 'usage: bash scripts/run_selection_switch.sh [run|smoke|stop|status|export|why|live|cpu|prepare|fit|summarize|recover-cost|errors|check-code]'; exit 2 ;;
esac
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export OM_WORK="$WORK"
export OUT_ROOT
OUT_ROOT=$(realpath -m "${SWITCH_ROOT:-$WORK/runs/selection-switch-v1}")
case "$OUT_ROOT" in /|"$PWD"|"$WORK"|"$WORK/runs") echo '[abort] unsafe experiment root'; exit 2 ;; esac
PY=${SWITCH_PYTHON:-${VENV_DIR:-$WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$MODE" = cpu ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "${SWITCH_CPU_PYTHON:-$PY}" -m pytest -q tests/test_selection_switch.py tests/test_selection_switch_gpu.py \
    tests/test_net_gate_memory_math.py tests/test_logit_chunking.py tests/test_selection_switch_cost.py \
    tests/test_selection_switch_errors.py tests/test_selection_switch_status.py tests/test_selection_switch_runtime.py \
    tests/test_selection_worker_shutdown.py tests/test_selection_nccl_preflight.py "$@"
fi
for arg in "$@"; do
  case "$arg" in --root|--root=*) echo '[abort] use SWITCH_ROOT for the output directory'; exit 2 ;; esac
done
if [ "$MODE" = status ]; then
  export CUDA_VISIBLE_DEVICES=""
  # Both experiments on one screen; EXPERIMENTS_COMBINED=0 shows only this one.
  if [ "${EXPERIMENTS_COMBINED:-1}" != 0 ]; then
    exec bash scripts/run_experiments.sh status "$@"
  fi
  exec "$PY" scripts/selection_switch_status.py --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = check-code ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" src/selection_switch_gpu.py check-code --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = errors ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" scripts/selection_switch_errors.py --root "$OUT_ROOT" "$@"
fi
if [ "$MODE" = recover-cost ]; then
  export CUDA_VISIBLE_DEVICES=""
  exec "$PY" scripts/recover_selection_switch_cost.py --root "$OUT_ROOT" "$@"
fi

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
  NODE_PID_FILE="$WORK/runs/experiments/logs/launcher.$LAUNCH_HOST.pid"
  if [ -f "$NODE_PID_FILE" ] && [ "${EXPERIMENTS_STOPPING:-0}" != 1 ]; then
    node_pid=$(cat "$NODE_PID_FILE" 2>/dev/null || true)
    if [[ "$node_pid" =~ ^[0-9]+$ ]] && kill -0 "$node_pid" 2>/dev/null; then
      echo "[stop] node launcher (run_experiments.sh) pid=$node_pid is running; stopping it first"
      kill -TERM -- "-$node_pid" 2>/dev/null || kill -TERM "$node_pid" 2>/dev/null || true
      for _ in $(seq 1 240); do kill -0 "$node_pid" 2>/dev/null || break; sleep 1; done
    fi
  fi
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
      if kill -0 -- "-$pgid" 2>/dev/null; then
        # A keepalive holds no receipts or ranks; if it ignores TERM (stuck in CUDA) kill it outright.
        for pid in $(ls /proc | grep -E '^[0-9]+$'); do
          [ -O "/proc/$pid" ] || continue
          [ "$(cut -d')' -f2 "/proc/$pid/stat" 2>/dev/null | awk '{print $3}')" = "$pgid" ] || continue
          if { tr '\0' ' ' < "/proc/$pid/cmdline"; } 2>/dev/null | grep -q "_gpu_keepalive.py"; then
            echo "[stop] keepalive pid=$pid ignored TERM; killing it"; kill -KILL "$pid" 2>/dev/null || true
          fi
        done
        kill -0 -- "-$pgid" 2>/dev/null && echo "[stop] pgid=$pgid still alive after 180s; not killing harder (GPU ranks would be orphaned)"
      fi
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
case "$MODE" in run|smoke)
  if [ -t 1 ] && [ "${SWITCH_DETACHED:-0}" != 1 ] && [ "${SWITCH_FOREGROUND:-0}" != 1 ]; then
    # From a terminal, a plain 'run' hands the node to scripts/run_experiments.sh,
    # which closes stale costs, runs a switch pass and a MoPPS pass every cycle
    # and keeps the node in between. EXPERIMENTS_COMBINED=0 runs only this queue.
    if [ "$MODE" = run ] && [ "$#" -eq 0 ] && [ "${EXPERIMENTS_COMBINED:-1}" != 0 ]; then
      exec bash scripts/run_experiments.sh run
    fi
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
  run|smoke|prepare|fit|summarize)
    if [ "${SWITCH_RUNTIME_REPO:-}" != "$PWD" ]; then
      exec "$PY" scripts/selection_switch_runtime.py --repo "$PWD" \
        --cache "${SWITCH_RUNTIME_CACHE:-/tmp/offpolicy-misranking-$(id -u)/switch-runtimes}" -- "$MODE" "$@"
    fi
    export OM_REPO="$PWD"
    ;;
esac
if [ "$MODE" = fit ] || [ "$MODE" = summarize ]; then
  export CUDA_VISIBLE_DEVICES=""
  "$PY" src/selection_switch_gpu.py "$MODE" --root "$OUT_ROOT" "$@"
  if [ "$MODE" = summarize ]; then
    "$PY" src/selection_switch_plot.py --root "$OUT_ROOT"
  fi
  exit 0
fi
if [ "$MODE" = export ] || [ "$MODE" = why ]; then
  [ -d "$OUT_ROOT" ] || { echo "[abort] no logs/results: $OUT_ROOT"; exit 2; }
  REPORT_DIR="$WORK/reports/selection-switch"
  mkdir -p "$REPORT_DIR"
  TARGET=$(mktemp "$REPORT_DIR/switch-$MODE-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.txt")
  (
    printf 'SELECTION SWITCH EXPERIMENT\nUTC: %s\nROOT: %s\nCOMMIT: ' "$(date -u +%FT%TZ)" "$OUT_ROOT"
    git rev-parse HEAD
    CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_status.py --root "$OUT_ROOT"
    if [ "$MODE" = export ] && [ -f "$OUT_ROOT/switch.json" ]; then
      CUDA_VISIBLE_DEVICES="" "$PY" src/selection_switch_gpu.py summarize --root "$OUT_ROOT"
    fi
    while IFS= read -r -d '' path; do
      printf '\n===== %s =====\n' "${path#"$OUT_ROOT"/}"
      cat "$path"
      printf '\n'
    done < <(find "$OUT_ROOT" -type f \( -name 'switch.json' -o -name 'model.json' \
      -o -name '*report.json' -o -name 'failure.json' -o -name 'progress.json' \
      -o -name 'decision.json' -o -name 'decisions-frozen.json' -o -name 'gate-frozen.json' -o -name 'gate.json' -o -name 'initial.json' \
      -o -name 'measurement.json' -o -name 'execution.json' -o -name 'result.json' \
      -o -name 'cost.jsonl' -o -name 'budget_stop.json' -o -name 'fit-cost.json' \
      -o -name '*-runtime.json' -o -name 'admission.json' -o -name 'rank-*.json' \
      -o -path '*/cost-events/*.json' -o -path '*/pending-costs/*.json' \) -print0 | sort -z)
    while IFS= read -r -d '' path; do
      printf '\n===== LOG: %s (last 100 lines) =====\n' "${path#"$OUT_ROOT"/}"
      tail -n 100 "$path"
    done < <(find "$OUT_ROOT" -type f -name '*.log' -print0 | sort -z)
    # The node launcher (run_experiments.sh) logs live outside both roots.
    if [ -d "$WORK/runs/experiments/logs" ]; then
      while IFS= read -r -d '' path; do
        printf '\n===== NODE LAUNCHER LOG: %s (last 150 lines) =====\n' "${path#"$WORK"/}"
        grep -v '^\[holding\]' "$path" | tail -n 150
      done < <(find "$WORK/runs/experiments/logs" -type f -name '*.log' -print0 | sort -z)
    fi
  ) > "$TARGET" 2>&1 || { printf '[export incomplete; see errors] %s\n' "$TARGET"; exit 1; }
  printf '[saved] %s\n' "$TARGET"
  exit 0
fi
if [ "$MODE" = live ]; then
  shopt -s nullglob
  LOGS=("$OUT_ROOT"/logs/launcher.*.log)
  [ "${#LOGS[@]}" -gt 0 ] || { echo "[no launcher logs] $OUT_ROOT/logs"; exit 1; }
  exec tail -n 20 -F "${LOGS[@]}"
fi
mkdir -p "$OUT_ROOT/logs"
HOST=$(hostname | tr -c 'a-zA-Z0-9._-' '_')
exec > >(tee -p -a "$OUT_ROOT/logs/launcher.$HOST.log") 2>&1
printf '[launcher-start] host=%s pid=%s mode=%s commit=%s utc=%s\n' "$HOST" "$$" "$MODE" "${SWITCH_RUNTIME_COMMIT:-unknown}" "$(date -u +%FT%TZ)"
trap 'rc=$?; printf "[launcher-exit] pid=%s mode=%s rc=%s utc=%s\n" "$$" "$MODE" "$rc" "$(date -u +%FT%TZ)"' EXIT
echo "[logs] $OUT_ROOT/logs/launcher.$HOST.log"
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
MATRIX=${OM_OLMO3_ROOT:-$WORK/runs/${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}}
if [ "$MODE" = prepare ] || [ ! -f "$OUT_ROOT/switch.json" ]; then
  "$PY" src/selection_switch_gpu.py prepare --root "$OUT_ROOT" --matrix "$MATRIX" \
    --gpu-type "${GATE_GPU_TYPE:-NVIDIA H100 80GB HBM3}" \
    --pool "$DATASETS_DIR/math_train/math_train.jsonl" \
    --pool-manifest "$DATASETS_DIR/math_train/dataset_manifest.json" "$@"
elif [ "$#" -gt 0 ]; then
  echo '[abort] experiment already frozen; run takes no new preparation options'; exit 2
fi
[ "$MODE" != prepare ] || exit 0
CUDA_VISIBLE_DEVICES="" "$PY" src/selection_switch_gpu.py check-code --root "$OUT_ROOT"
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
  [ "$used" -le 4000 ] || { echo '[busy] allocated GPU is occupied; other experiments were not stopped'; exit 75; }
done <<< "$MEMORY"
# The allocation is reclaimed when its GPUs sit idle, which is what a launcher
# looks like while it waits for a prerequisite or holds between passes. Keep a
# tiny kernel running on every visible GPU for the launcher's lifetime
# (operator launches only; SWITCH_KEEPALIVE=0 disables it). Started after the
# occupancy check so it never counts as an existing job, killed on exit.
KEEPALIVE_PID=
stop_keepalive() { [ -n "$KEEPALIVE_PID" ] && kill -TERM "$KEEPALIVE_PID" 2>/dev/null; KEEPALIVE_PID=; }
if [ "${SWITCH_KEEPALIVE:-1}" != 0 ] && { [ "${SWITCH_DETACHED:-0}" = 1 ] || [ -t 1 ]; }; then
  "$PY" scripts/_gpu_keepalive.py > "$OUT_ROOT/logs/keepalive.$HOST.log" 2>&1 7>&- 8>&- &
  KEEPALIVE_PID=$!
  echo "[keepalive] pid=$KEEPALIVE_PID keeps the allocated GPUs busy while this launcher waits or holds (log: logs/keepalive.$HOST.log)"
  trap 'rc=$?; stop_keepalive; printf "[launcher-exit] pid=%s mode=%s rc=%s utc=%s\n" "$$" "$MODE" "$rc" "$(date -u +%FT%TZ)"' EXIT
fi
MATH_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps")
export PYTHONPATH="$MATH_VERIFY_PATH${PYTHONPATH:+:$PYTHONPATH}" OM_MATH_VERIFIER=math_verify OM_NODE_LOCK_HELD=1
trap '' HUP
source scripts/_selection_worker.sh
# The allocation lives only while this process does: an exit after "no
# claimable task" or a failed pass gives the GPUs back and the operator must
# re-request them and rerun the same command. In run mode keep the node and
# poll again instead; every pass re-admits the node and re-attempts failed
# tasks once. SWITCH_HOLD_SECONDS=0 restores the single pass.
# Holding applies to operator launches (detached or on a terminal); pipelines
# and tests without a terminal keep the single pass unless they opt in.
if [ "${SWITCH_DETACHED:-0}" = 1 ] || [ -t 1 ]; then HOLD_DEFAULT=600; else HOLD_DEFAULT=0; fi
HOLD=${SWITCH_HOLD_SECONDS:-$HOLD_DEFAULT}
[[ "$HOLD" =~ ^[0-9]+$ ]] || { echo '[abort] SWITCH_HOLD_SECONDS must be a whole number of seconds'; exit 2; }
switch_complete() {
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_status.py --root "$OUT_ROOT" --json 2>/dev/null \
    | "$PY" -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("development_done")==18 and d.get("test_done")==30 else 1)'
}
pass=0
wait_seconds=$HOLD
while :; do
  pass=$((pass+1))
  rc=0
  # Nodes are killed routinely here; an attempt that died without a receipt
  # otherwise blocks its branch until someone types recover-cost. Close
  # events whose owner has been silent for 15 minutes (operator decision,
  # see recover-cost --stale) before each pass. SWITCH_AUTO_RECOVER=0 disables.
  if [ "${SWITCH_AUTO_RECOVER:-1}" != 0 ]; then
    CUDA_VISIBLE_DEVICES="" "$PY" scripts/recover_selection_switch_cost.py --root "$OUT_ROOT" --stale --brief 2>&1 \
      | sed 's/^\[recovery blocked\]/[recover-cost] blocked:/' || true
  fi
  selection_run_worker "$PY" scripts/selection_nccl_preflight.py --root "$OUT_ROOT" -- \
    "$PY" src/selection_switch_gpu.py "$MODE" --root "$OUT_ROOT" || rc=$?
  case "$rc" in 78|130|137|143) break ;; esac
  if [ "$rc" -ne 0 ]; then
    CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_errors.py --root "$OUT_ROOT" || true
  fi
  [ "$MODE" = run ] && [ "$HOLD" -gt 0 ] || break
  if switch_complete; then
    echo '[hold] all 48 continuations are published; releasing the node'
    rc=0
    break
  fi
  if [ "$rc" -eq 0 ]; then wait_seconds=$HOLD; else wait_seconds=$(( wait_seconds*2 > 3600 ? 3600 : wait_seconds*2 )); fi
  echo "[hold] pass $pass ended rc=$rc; keeping this node's GPUs; next pass in ${wait_seconds}s (stop: bash scripts/run_selection_switch.sh stop)"
  selection_hold_node "$wait_seconds" || { rc=$?; break; }
done
if [ "$rc" -eq 78 ]; then
  echo '[blocked] node admission failed above; historical branch errors are not the cause of this launch'
fi
exit "$rc"
