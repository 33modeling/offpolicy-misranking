#!/usr/bin/env bash
# Extension E1 (2026-09-07): drift-resolution curve.
#
#   bash scripts/run_drift_curve.sh <config.json> <curve root> <curve results> [profile]
#
# The registered trainer keeps only the two most recent durable checkpoints of
# a chain, so intermediate policies cannot be read back from a finished
# registered family. The curve is therefore a separate chain family trained
# with the same launcher machinery (run_matrix.sh -> run_point.sh): one
# continuous adapter/optimizer lineage per dataset-seed through the fine grid
# below, with the behavior pool generated once at d=0 and reused by every
# point. No registered run directory is read or written; the registered
# labels are unaffected. Chained points are resumed from the previous point
# with the step-indexed RNG streams of the trainer, but bitwise identity with
# a registered point at the same cumulative step is not assumed or claimed.
#
# Defaults (override with the REGIME_* variables of run_matrix.sh):
#   DRIFT_CURVE_DRIFTS="0 5 10 25 50 100 200 400"
#   DRIFT_CURVE_SEEDS="0 1"
#   DRIFT_CURVE_DATASETS="math500"
set -uo pipefail
cd "$(dirname "$0")/.."
[ "$#" -ge 3 ] || { echo "usage: $0 <config.json> <curve root> <curve results> [profile]"; exit 2; }
CONFIG=$1; CURVE_ROOT=$2; CURVE_RESULTS=$3; PROFILE=${4:-}
[ -s "$CONFIG" ] || { echo "[abort] missing config: $CONFIG"; exit 1; }
export OM_ONLINE=0
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
MATRIX_TOOL="src/model_matrix.py"

field() { "$PY" "$MATRIX_TOOL" --config "$CONFIG" "$@"; }
MODEL_KEY="${DRIFT_CURVE_MODEL_KEY:-$(field list-models | head -1)}"
[ -n "$MODEL_KEY" ] || { echo "[abort] no model key in $CONFIG"; exit 1; }
model_field() { "$PY" "$MATRIX_TOOL" --config "$CONFIG" --models-dir "$MODELS_DIR" field "$MODEL_KEY" "$1"; }
export MODEL_PATH="${MODEL_PATH:-$(model_field path)}"
[ -f "$MODEL_PATH/config.json" ] || { echo "[abort] model snapshot missing: $MODEL_PATH"; exit 1; }
DATASETS="${DRIFT_CURVE_DATASETS:-math500}"
SEEDS="${DRIFT_CURVE_SEEDS:-0 1}"
DRIFTS="${DRIFT_CURVE_DRIFTS:-0 5 10 25 50 100 200 400}"
for d in $DRIFTS; do case "$d" in ''|*[!0-9]*) echo "[abort] drift must be an integer: $d"; exit 2 ;; esac; done
case "$DRIFTS" in 0\ *|0) ;; *) echo "[abort] the curve must start at drift 0 (behavior pool)"; exit 2 ;; esac
FORMAT=olmo_rlzero_math
case "$DATASETS" in *mbpp*) [ "$DATASETS" = mbpp ] || { echo "[abort] one dataset per invocation"; exit 2; }; FORMAT=olmo_rlzero_code ;; esac

# Same GRPO/experiment contract as the registered launcher, read from the config.
export GRPO_WORLD_SIZE=$(field grpo-field world_size) GRPO_GROUP_SIZE=$(field grpo-field group_size)
export GRPO_CLIP_EPSILON=$(field grpo-field clip_epsilon) GRPO_LEARNING_RATE=$(field grpo-field learning_rate)
export GRPO_EPOCHS_PER_BATCH=$(field grpo-field epochs_per_batch) GRPO_MAX_GRAD_NORM=$(field grpo-field max_grad_norm)
export GRPO_ADVANTAGE_EPSILON=$(field grpo-field advantage_epsilon) GRPO_LORA_RANK=$(field grpo-field lora_rank)
export GRPO_LORA_ALPHA=$(field grpo-field lora_alpha) GRPO_CHECKPOINT_EVERY=5 RLVR_METHOD=grpo
export REGIME_N_VAL=$(field experiment-field n_val) REGIME_BEHAVIOR_K=$(field experiment-field behavior_k)
export REGIME_FRESH_K=$(field experiment-field fresh_k) REGIME_VAL_K=$(field experiment-field val_k)
export REGIME_MICRO_GROUP=$(field experiment-field micro_group) REGIME_MAX_NEW_TOKENS=$(field experiment-field max_new_tokens)
export REGIME_PROJ_DIM=$(field experiment-field proj_dim) REGIME_GRAD_LAYERS=$(field experiment-field grad_layers)
export REGIME_CLIP_CAP=$(field experiment-field clip_cap) REGIME_TOPK_FRAC=$(field experiment-field topk_frac)
export REGIME_TEMPERATURE=$(field experiment-field temperature)
export REGIME_N_TRAIN_BY_DATASET="$(for ds in $DATASETS; do printf '%s=%s ' "$ds" "$(field dataset-n-train "$ds")"; done)"
export OM_ATTN=$(field experiment-field attn) OM_SKIP_HYBRID=1 OM_PROMPT_FORMAT="$FORMAT"
export OM_LORA_TARGETS="${OM_LORA_TARGETS:-$(model_field lora_targets)}"
for name in GRPO_WORLD_SIZE GRPO_GROUP_SIZE GRPO_EPOCHS_PER_BATCH REGIME_N_VAL REGIME_FRESH_K REGIME_MAX_NEW_TOKENS REGIME_N_TRAIN_BY_DATASET OM_LORA_TARGETS; do
  [ -n "${!name}" ] || { echo "[abort] $name resolved empty from $CONFIG"; exit 1; }
done
if [ -n "$PROFILE" ]; then
  export GRPO_LOGPROB_MICRO_BATCH=$(field runtime-field logprob_micro_batch)
  export GRPO_GRADIENT_CHECKPOINTING=$(field runtime-field gradient_checkpointing)
  export GRADIENT_MICRO_BATCH=$(field runtime-field gradient_micro_batch)
  export OM_GEN_BATCH=$(field runtime-field generation_batch)
fi
mkdir -p "$CURVE_ROOT" "$CURVE_RESULTS"
echo "[drift-curve] model=$MODEL_PATH datasets=$DATASETS seeds=$SEEDS drifts=$DRIFTS root=$CURVE_ROOT"
echo "[drift-curve] n_train=$REGIME_N_TRAIN_BY_DATASET n_val=$REGIME_N_VAL fresh_k=$REGIME_FRESH_K max_new_tokens=$REGIME_MAX_NEW_TOKENS world=$GRPO_WORLD_SIZE epochs=$GRPO_EPOCHS_PER_BATCH"
REGIME_ROOT="$CURVE_ROOT" REGIME_RESULTS="$CURVE_RESULTS" REGIME_MODEL_TAG="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')-curve" \
  REGIME_DATASETS="$DATASETS" REGIME_SEEDS="$SEEDS" REGIME_DRIFTS="$DRIFTS" REGIME_SKIP_COLLECTION=1 \
  bash scripts/run_matrix.sh
rc=$?
[ "$rc" -eq 0 ] || { echo "[drift-curve] run_matrix.sh failed rc=$rc"; exit "$rc"; }
echo "[drift-curve] complete. Aggregate with:"
echo "  $PY src/drift_curve.py --curve $CURVE_ROOT/*-d* --registered <registered runs> --output-dir <dir>"
