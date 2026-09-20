#!/usr/bin/env bash
# Separate, authorized MBPP retries. Original publications and costs stay intact.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-run}
[ "$#" -eq 0 ] || shift
case "$MODE" in
  run|restart|stop|logs|status|results) ;;
  -h|--help)
    echo 'usage: bash scripts/run_mbpp_repair.sh [run|restart|stop|logs|status|results]'
    exit 0 ;;
  *) echo '[abort] unknown MBPP repair mode' >&2; exit 2 ;;
esac
if [ "$MODE" != status ] && [ "$#" -gt 0 ]; then
  echo '[abort] only status accepts viewer options' >&2
  exit 2
fi

export OM_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
OM_WORK=$(realpath -m "$OM_WORK")
SOURCE=$(realpath -m "${MBPP_REPAIR_SOURCE:-$OM_WORK/runs/selection-switch-mbpp-quality-v1}")
TARGET=$(realpath -m "${MBPP_REPAIR_ROOT:-$OM_WORK/runs/selection-switch-mbpp-quality-repair-v1}")
case "$TARGET" in
  "$SOURCE"|"$SOURCE/"*|/|"$OM_WORK"|"$OM_WORK/runs"|"$PWD")
    echo '[abort] repair output must be separate from original storage' >&2; exit 2 ;;
esac
case "$SOURCE" in
  "$TARGET/"*) echo '[abort] repair output must not contain original storage' >&2; exit 2 ;;
esac
PY=${SWITCH_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
[ -x "$PY" ] || PY=python3

if [ "$MODE" = results ]; then
  exec env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
    "$PY" scripts/mbpp_repair_results.py --root "$TARGET"
fi

case "$MODE" in
  run|restart|stop|logs)
    # The shared controller's identity is allocation-scoped, not root-scoped.
    # Never mistake an original quality controller for the separate repair run.
    owner=$("$PY" -B scripts/mbpp_controller_identity.py \
      --logs "$OM_WORK/runs/experiments/logs" --work "$OM_WORK" \
      --suite quality --field pid) || exit $?
    if [ -n "$owner" ]; then
      if ! [[ "$owner" =~ ^[1-9][0-9]*$ ]] || \
          ! { tr '\0' '\n' < "/proc/$owner/environ"; } 2>/dev/null \
            | grep -Fxq "SWITCH_ROOT=$TARGET"; then
        echo '[blocked] another MBPP root owns this allocation; original controller was not stopped. Use an idle allocation for repair.' >&2
        exit 75
      fi
    fi
    ;;
esac

if [ "$MODE" = run ] || [ "$MODE" = restart ]; then
  env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
    "$PY" scripts/mbpp_repair.py prepare --source "$SOURCE" --root "$TARGET"
fi

# Explicit quality keeps status/results scoped to this root, without legacy
# observation roots. The original fresh prefix root and frozen settings remain.
exec env -u OUT_ROOT -u SWITCH_ROOT -u SWITCH_RUNTIME_REPO \
  SWITCH_MBPP_QUALITY_ROOT="$TARGET" \
  bash scripts/run_mbpp_experiments.sh "$MODE" quality "$@"
