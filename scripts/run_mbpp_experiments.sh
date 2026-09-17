#!/usr/bin/env bash
# MBPP counterpart of the existing fresh / quality / difficulty switch suites.
# Run inside tmux on an allocated, idle four-GPU node. No training code is copied.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
SUITE=${2:-all}
usage() {
  echo 'usage: bash scripts/run_mbpp_experiments.sh [run|plan|check|status|results|why] [all|fresh|quality|difficulty]'
}
[ "$#" -le 2 ] || { usage; exit 2; }
case "$MODE" in run|plan|check|status|results|why) ;; -h|--help) usage; exit 0 ;; *) usage; exit 2 ;; esac
case "$SUITE" in all) SUITES=(fresh quality difficulty) ;; fresh|quality|difficulty) SUITES=("$SUITE") ;; *) usage; exit 2 ;; esac

export OM_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
FRESH_ROOT=${SWITCH_MBPP_ROOT:-$OM_WORK/runs/selection-switch-mbpp-v1}
QUALITY_ROOT=${SWITCH_MBPP_QUALITY_ROOT:-$OM_WORK/runs/selection-switch-mbpp-quality-v1}
DIFFICULTY_ROOT=${SWITCH_MBPP_DIFFICULTY_ROOT:-$OM_WORK/runs/selection-switch-mbpp-difficulty-v1}
FRESH_ROOT=$(realpath -m "$FRESH_ROOT")
QUALITY_ROOT=$(realpath -m "$QUALITY_ROOT")
DIFFICULTY_ROOT=$(realpath -m "$DIFFICULTY_ROOT")
for root in "$FRESH_ROOT" "$QUALITY_ROOT" "$DIFFICULTY_ROOT"; do
  case "$root" in /|"$PWD"|"$OM_WORK"|"$OM_WORK/runs") echo "[abort] unsafe root: $root"; exit 2 ;; esac
  for other in "$FRESH_ROOT" "$QUALITY_ROOT" "$DIFFICULTY_ROOT"; do
    [[ "$root" != "$other/"* ]] || { echo '[abort] suite roots must not contain each other'; exit 2; }
  done
done
[ "$FRESH_ROOT" != "$QUALITY_ROOT" ] && [ "$FRESH_ROOT" != "$DIFFICULTY_ROOT" ] && [ "$QUALITY_ROOT" != "$DIFFICULTY_ROOT" ] || {
  echo '[abort] fresh, quality and difficulty need separate output roots'; exit 2;
}
HOLD=${MBPP_HOLD_SECONDS:-600}
[[ "$HOLD" =~ ^[0-9]+$ ]] && [ "$HOLD" -gt 0 ] || { echo '[abort] MBPP_HOLD_SECONDS must be positive'; exit 2; }

settings() {
  PREFIX=; SELECTOR=fresh_r; ACCOUNTING=budget; GATE=final
  case "$1" in
    fresh) ROOT=$FRESH_ROOT ;;
    quality) ROOT=$QUALITY_ROOT; PREFIX=$FRESH_ROOT; ACCOUNTING=matched; GATE=convergence ;;
    difficulty) ROOT=$DIFFICULTY_ROOT; PREFIX=$FRESH_ROOT; SELECTOR=difficulty; GATE=convergence ;;
  esac
}
for suite in "${SUITES[@]}"; do
  settings "$suite"
  printf '[mbpp:%s] selector=%s accounting=%s gate=%s\n  root=%s\n' "$suite" "$SELECTOR" "$ACCOUNTING" "$GATE" "$ROOT"
  [ -z "$PREFIX" ] || printf '  shared prefixes and evaluation=%s\n' "$PREFIX"
done
if [ "$MODE" = plan ]; then
  echo '[plan] seeds 0..4; fresh-selected states at 25/50/100 updates; 18 development + 30 held-out continuations per suite'
  echo '[plan] MBPP execution rewards; final evaluation K=8; convergence curves: 3 checkpoints, K=4'
  echo '[plan] evaluation excludes every source train/validation prompt; available count checked before launch'
  echo "[plan] execution order: ${SUITES[*]}; each stage waits for its full queue to finish"
  echo '[plan] no files written or GPU work started; use check to validate local inputs'
  exit 0
fi

if [ "$MODE" = run ] || [ "$MODE" = check ]; then
  # Resolve the setup_env defaults without its directory/log maintenance: check
  # must remain read-only. The actual GPU launcher sources setup_env itself.
  GROUP_VOLUME=${GROUP_VOLUME:-/group-volume}
  if [ -d "$GROUP_VOLUME/${OM_USER:-minsoo3.kim}/datasets" ]; then
    DEFAULT_DATASETS="$GROUP_VOLUME/${OM_USER:-minsoo3.kim}/datasets"
  elif [ -d "$GROUP_VOLUME/datasets" ]; then
    DEFAULT_DATASETS="$GROUP_VOLUME/datasets"
  else
    DEFAULT_DATASETS="$OM_WORK/data"
  fi
  POOL_ROOT=${DATASETS_DIR:-$DEFAULT_DATASETS}
  PY=${SWITCH_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
  [ -x "$PY" ] || PY=python3
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/check_mbpp_experiments.py \
    --matrix "${OM_OLMO3_ROOT:-$OM_WORK/runs/${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}}" \
    --pool "$POOL_ROOT/mbpp/mbpp.jsonl" --manifest "$POOL_ROOT/mbpp/dataset_manifest.json" \
    --fresh-root "$FRESH_ROOT" --quality-root "$QUALITY_ROOT" --difficulty-root "$DIFFICULTY_ROOT" \
    --suite "$SUITE"
  [ "$MODE" != check ] || exit 0
fi

trap '' HUP
trap 'echo "[mbpp] interrupted; no next suite will start"; exit 130' INT
trap 'echo "[mbpp] terminated; no next suite will start"; exit 143' TERM
failed=0
for suite in "${SUITES[@]}"; do
  settings "$suite"
  echo "[mbpp:$suite] $MODE"
  rc=0
  # Foreground keeps the next suite behind completion, including on a terminal.
  # Do not inherit a math prefix, pilot filter, or another experiment's node lock.
  env -u SWITCH_PREFIX_SOURCE -u SWITCH_ONLY_SEEDS -u SWITCH_ONLY_ARMS \
    -u SWITCH_BUDGET_GPU_SECONDS -u OM_NODE_LOCK_HELD -u OUT_ROOT \
    -u SWITCH_RUNTIME_REPO -u SWITCH_DETACHED -u EXPERIMENTS_DETACHED \
    SWITCH_ROOT="$ROOT" SWITCH_PREFIX_SOURCE="$PREFIX" SWITCH_DATASET=mbpp \
    SWITCH_SELECTOR="$SELECTOR" SWITCH_ACCOUNTING="$ACCOUNTING" SWITCH_GATE="$GATE" \
    SWITCH_CURVE_POINTS=3 SWITCH_CURVE_K=4 \
    SWITCH_FOREGROUND=1 SWITCH_HOLD_SECONDS="$HOLD" \
    EXPERIMENTS_COMBINED=0 EXPERIMENTS_SKIP_MOPPS=1 E5_FORCE=0 \
    bash scripts/run_selection_switch.sh "$MODE" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[mbpp:$suite] failed rc=$rc; existing checkpoints and receipts retained"
    [ "$MODE" != run ] || exit "$rc"
    failed=1
  fi
done
exit "$failed"
