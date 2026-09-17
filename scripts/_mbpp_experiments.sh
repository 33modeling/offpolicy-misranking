# MBPP queue configuration. Lifecycle, recovery and GPU ownership stay in
# run_experiments.sh; this file only supplies roots and their prerequisites.
mbpp_queue_init() {
  case "${EXPERIMENTS_MBPP_SUITE:-all}" in
    all) MBPP_SUITES=(fresh quality difficulty) ;;
    fresh|quality|difficulty) MBPP_SUITES=("$EXPERIMENTS_MBPP_SUITE") ;;
    *) echo '[abort] unknown MBPP suite'; return 2 ;;
  esac
  export SWITCH_MBPP_ROOT SWITCH_MBPP_QUALITY_ROOT SWITCH_MBPP_DIFFICULTY_ROOT
  SWITCH_MBPP_ROOT=$(realpath -m "${SWITCH_MBPP_ROOT:-$OM_WORK/runs/selection-switch-mbpp-v1}")
  SWITCH_MBPP_QUALITY_ROOT=$(realpath -m "${SWITCH_MBPP_QUALITY_ROOT:-$OM_WORK/runs/selection-switch-mbpp-quality-v1}")
  SWITCH_MBPP_DIFFICULTY_ROOT=$(realpath -m "${SWITCH_MBPP_DIFFICULTY_ROOT:-$OM_WORK/runs/selection-switch-mbpp-difficulty-v1}")
  local root other work
  work=$(realpath -m "$OM_WORK")
  local roots=("$SWITCH_MBPP_ROOT" "$SWITCH_MBPP_QUALITY_ROOT" "$SWITCH_MBPP_DIFFICULTY_ROOT")
  for root in "${roots[@]}"; do
    case "$root" in /|"$PWD"|"$work"|"$work/runs") echo "[abort] unsafe root: $root"; return 2 ;; esac
    for other in "${roots[@]}"; do
      [[ "$root" != "$other/"* ]] || { echo '[abort] suite roots must not contain each other'; return 2; }
    done
  done
  [ "${roots[0]}" != "${roots[1]}" ] && [ "${roots[0]}" != "${roots[2]}" ] && [ "${roots[1]}" != "${roots[2]}" ] || {
    echo '[abort] fresh, quality and difficulty need separate output roots'; return 2;
  }
  MBPP_ROOTS=()
  local suite
  for suite in "${MBPP_SUITES[@]}"; do
    mbpp_queue_settings "$suite"
    MBPP_ROOTS+=("$MBPP_ROOT")
  done
}

mbpp_queue_settings() {
  MBPP_PREFIX=; MBPP_SELECTOR=fresh_r; MBPP_ACCOUNTING=budget; MBPP_GATE=final
  case "$1" in
    fresh|"$SWITCH_MBPP_ROOT") MBPP_SUITE=fresh; MBPP_ROOT=$SWITCH_MBPP_ROOT ;;
    quality|"$SWITCH_MBPP_QUALITY_ROOT")
      MBPP_SUITE=quality; MBPP_ROOT=$SWITCH_MBPP_QUALITY_ROOT
      MBPP_PREFIX=$SWITCH_MBPP_ROOT; MBPP_ACCOUNTING=matched; MBPP_GATE=convergence ;;
    difficulty|"$SWITCH_MBPP_DIFFICULTY_ROOT")
      MBPP_SUITE=difficulty; MBPP_ROOT=$SWITCH_MBPP_DIFFICULTY_ROOT
      MBPP_PREFIX=$SWITCH_MBPP_ROOT; MBPP_SELECTOR=difficulty; MBPP_GATE=convergence ;;
    *) echo "[abort] not a configured MBPP root: $1"; return 2 ;;
  esac
}

mbpp_prefixes_ready() {
  [ -f "$SWITCH_MBPP_ROOT/switch.json" ] || return 1
  local seed step
  for seed in 0 1 2 3 4; do
    for step in 25 50 100; do
      [ -f "$SWITCH_MBPP_ROOT/prefixes/seed-$seed/prefix-$step.json" ] || return 1
    done
  done
}

mbpp_queue_check() {
  local suite=$1 volume=${GROUP_VOLUME:-/group-volume} datasets python
  if [ -d "$volume/${OM_USER:-minsoo3.kim}/datasets" ]; then
    datasets="$volume/${OM_USER:-minsoo3.kim}/datasets"
  elif [ -d "$volume/datasets" ]; then
    datasets="$volume/datasets"
  else
    datasets="$OM_WORK/data"
  fi
  datasets=${DATASETS_DIR:-$datasets}
  python=${SWITCH_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
  [ -x "$python" ] || python=python3
  CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$python" scripts/check_mbpp_experiments.py \
    --matrix "${OM_OLMO3_ROOT:-$OM_WORK/runs/${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}}" \
    --pool "$datasets/mbpp/mbpp.jsonl" --manifest "$datasets/mbpp/dataset_manifest.json" \
    --fresh-root "$SWITCH_MBPP_ROOT" --quality-root "$SWITCH_MBPP_QUALITY_ROOT" \
    --difficulty-root "$SWITCH_MBPP_DIFFICULTY_ROOT" --suite "$suite"
}

mbpp_queue_preparable() {
  [ ! -f "$1/switch.json" ] || return 1
  mbpp_queue_settings "$1"
  if [ -n "$MBPP_PREFIX" ]; then mbpp_prefixes_ready; else return 1; fi
}

# Called once per root per node pass, never a private hold loop. Missing inputs
# are retried on a later pass; they do not prevent cleanup or release the node.
mbpp_queue_run() (
  mbpp_queue_settings "$1" || return
  if [ -n "$MBPP_PREFIX" ] && ! mbpp_prefixes_ready; then
    echo "[waiting] mbpp:$MBPP_SUITE: shared fresh prefixes are not ready; taking other queue work"
    return 0
  fi
  mbpp_queue_check "$MBPP_SUITE" || return 1
  unset SWITCH_ONLY_SEEDS SWITCH_ONLY_ARMS SWITCH_BUDGET_GPU_SECONDS OM_NODE_LOCK_HELD SWITCH_RUNTIME_REPO
  export SWITCH_ROOT="$MBPP_ROOT" SWITCH_PREFIX_SOURCE="$MBPP_PREFIX" SWITCH_DATASET=mbpp
  export SWITCH_SELECTOR="$MBPP_SELECTOR" SWITCH_ACCOUNTING="$MBPP_ACCOUNTING" SWITCH_GATE="$MBPP_GATE"
  export SWITCH_CURVE_POINTS=3 SWITCH_CURVE_K=4 EXPERIMENTS_SKIP_MOPPS=1 E5_FORCE=0 SWITCH_QUEUE_PASS=1
  inner scripts/run_selection_switch.sh
)
