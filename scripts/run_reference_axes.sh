#!/usr/bin/env bash
# Existing checkout/environment; new one-factor reference outputs only.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo 'usage: bash scripts/run_reference_axes.sh <math500|mbpp> <replicate 0..4> [--run|--plan|--check]'
  exit 2
fi
DATASET=$1; REPLICATE=$2; MODE=${3:---run}
case "$DATASET" in math500|mbpp) ;; *) echo '[abort] expected math500 or mbpp'; exit 2 ;; esac
case "$REPLICATE" in 0|1|2|3|4) ;; *) echo '[abort] replicate must be 0..4'; exit 2 ;; esac
case "$MODE" in --run|--plan|--check) ;; *) echo '[abort] unknown mode'; exit 2 ;; esac

# Do not inherit v2/Qwen checkout, source, Python, or output overrides.
# Defaults are the paths recorded in the transferred cluster logs.
export GROUP_VOLUME=${GROUP_VOLUME:-/group-volume} OM_USER=${OM_USER:-minsoo3.kim}
export OM_REPO="$PWD"
export OM_WORK="${REFERENCE_WORK:-$GROUP_VOLUME/$OM_USER/offpolicy-misranking}"
export VENV_DIR="${REFERENCE_VENV:-$OM_WORK/.venv-cu126}"
export MODELS_DIR="$GROUP_VOLUME/models"
export OM_OLMO3_MODEL_PATH="${REFERENCE_MODEL:-$MODELS_DIR/Olmo-3-1025-7B}"
export OM_OLMO3_MODEL_TAG=olmo3-1025-7b-base-rlzero-grpo-h100-v2
export OM_RLZERO_CONFIG="$PWD/configs/olmo3_rlzero_h100.json"
export OM_OLMO3_ROOT="${REFERENCE_SOURCE_ROOT:-$OM_WORK/runs/$OM_OLMO3_MODEL_TAG}"
export RB_RUNS_ROOT="$OM_WORK/runs/reference-axes"
export RB_EXPORTS_ROOT="$OM_WORK/exports/reference-axes"
export HF_HOME="$OM_WORK/cache/huggingface" PIP_CACHE_DIR="$OM_WORK/cache/pip"
export TMPDIR="$OM_WORK/tmp" OM_DATA="$OM_WORK/data" STORAGE_ROOT="$OM_WORK"
export PYTHONPATH="$PWD/src" PYTHONPYCACHEPREFIX="$OM_WORK/cache/pycache"
export OM_ONLINE=0 PYTHONNOUSERSITE=1
export OM_TOP_P=1.0 OM_THINKING=off
if [ "$DATASET" = math500 ]; then
  export OM_GEN_BATCH="${REFERENCE_GEN_BATCH:-32}" GRADIENT_MICRO_BATCH="${REFERENCE_GRAD_MICRO_BATCH:-2}"
else
  export OM_GEN_BATCH="${REFERENCE_GEN_BATCH:-16}" GRADIENT_MICRO_BATCH="${REFERENCE_GRAD_MICRO_BATCH:-1}"
fi
unset OUT_ROOT REGIME_ROOT OM_PROJECT_VERSION OM_NODE_LOCK_HELD PYTHONHOME
export RB_DRY=0
[ "$MODE" != --check ] || export RB_DRY=1
echo "[reference] checkout=$OM_REPO"
echo "[reference] existing_python=$VENV_DIR/bin/python"
echo "[reference] model=$OM_OLMO3_MODEL_PATH"
echo "[reference] read_only_source=$OM_OLMO3_ROOT"
echo "[reference] new_outputs=$RB_RUNS_ROOT"
if [ "$MODE" != --plan ]; then
  [ -x "$VENV_DIR/bin/python" ] || { echo '[abort] existing Python environment is missing'; exit 1; }
  [ -s "$OM_OLMO3_MODEL_PATH/config.json" ] || { echo '[abort] OLMo model snapshot is missing'; exit 1; }
  [ -d "$OM_OLMO3_ROOT" ] || { echo '[abort] primary source directory is missing'; exit 1; }
fi

ACTIVE=""
stop() {
  trap - INT TERM
  if [ -n "$ACTIVE" ]; then
    kill -TERM "$ACTIVE" 2>/dev/null || true
    wait "$ACTIVE" 2>/dev/null || true
  fi
  exit 143
}
trap stop INT TERM
failed=0
for budgets in '32 8' '64 8' '128 8' '32 16' '32 32'; do
  read -r fk vk <<< "$budgets"
  seed=$((100000 + REPLICATE * 10000 + fk * 40 + vk))
  echo "[condition] dataset=$DATASET replicate=$REPLICATE fresh_k=$fk val_k=$vk seed=$seed"
  [ "$MODE" != --plan ] || continue
  bash scripts/run_reliability_budget.sh "$DATASET" "$fk" "$vk" "$seed" &
  ACTIVE=$!
  wait "$ACTIVE"; rc=$?
  ACTIVE=""
  if [ "$rc" -ne 0 ]; then
    echo "[failed] fk=$fk vk=$vk rc=$rc; continuing with remaining conditions"
    failed=1
  fi
done
exit "$failed"
