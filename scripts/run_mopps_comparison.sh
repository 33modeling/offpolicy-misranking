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
if [ "$MODE" = stop ]; then
  if ! launcher_pid_alive; then
    echo "[stop] no live detached launcher on $LAUNCH_HOST (pid file: $PID_FILE)"
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
  exit 0
fi
case "$MODE" in run|retry)
  if [ -t 1 ] && [ "${SWITCH_DETACHED:-0}" != 1 ] && [ "${SWITCH_FOREGROUND:-0}" != 1 ]; then
    if launcher_pid_alive; then
      echo "[already running] host=$LAUNCH_HOST pid=$(cat "$PID_FILE"); follow: tail -f $CONSOLE_LOG; stop: bash scripts/$(basename "$0") stop"
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
if [ "$MODE" = status ] || [ "$MODE" = summarize ]; then
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
      CUDA_VISIBLE_DEVICES="" "$PY" src/mopps_comparison_gpu.py status --root "$OUT_ROOT" || echo '[status unavailable]'
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
MATH_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps")
export PYTHONPATH="$MATH_VERIFY_PATH${PYTHONPATH:+:$PYTHONPATH}" OM_MATH_VERIFIER=math_verify OM_NODE_LOCK_HELD=1
trap '' HUP
source scripts/_selection_worker.sh
rc=0
if [ "$MODE" = retry ] && [ "$#" -eq 0 ]; then
  # Phone-typeable form: retry each recorded branch failure once, in path order.
  # Each item is still the explicit single-branch retry of the Python entrypoint,
  # behind the same node admission check.
  mapfile -t FAILED < <(find "$OUT_ROOT/states" -mindepth 3 -maxdepth 3 -name failure.json 2>/dev/null | sort)
  [ "${#FAILED[@]}" -gt 0 ] || echo '[retry] no recorded branch failures'
  for failure in "${FAILED[@]}"; do
    rel=${failure#"$OUT_ROOT/states/"}
    point=${rel%%/*}; arm=${rel#*/}; arm=${arm%%/*}
    seed=${point#s}; seed=${seed%%-t*}; step=${point##*-t}
    echo "[retry] s$seed/t$step/$arm ($failure)"
    selection_run_worker "$PY" scripts/selection_nccl_preflight.py --root "$OUT_ROOT" -- \
      "$PY" src/mopps_comparison_gpu.py retry --root "$OUT_ROOT" --seed "$seed" --step "$step" --arm "$arm" || rc=$?
  done
else
  selection_run_worker "$PY" scripts/selection_nccl_preflight.py --root "$OUT_ROOT" -- \
    "$PY" src/mopps_comparison_gpu.py "$MODE" --root "$OUT_ROOT" "$@" || rc=$?
fi
if [ "$rc" -ne 0 ]; then
  CUDA_VISIBLE_DEVICES="" "$PY" scripts/selection_switch_errors.py --root "$OUT_ROOT" || true
fi
exit "$rc"
