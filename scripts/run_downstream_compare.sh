#!/usr/bin/env bash
# Extension E5 (2026-09-07): matched downstream GRPO update per selector.
#
#   bash scripts/run_downstream_compare.sh <completed point run> <out root> [steps=50]
#
# From the point's policy_step_<d> adapter and optimizer, every selector's
# top-k prompt subset receives <steps> further registered GRPO updates with the
# same objective configuration (four ranks, K=8, one epoch, clip 0.2, lr 1e-5,
# LoRA q/v 16/32). The held-out reward is the mean verifier reward over the
# validation prompts with K_eval samples each, before (shared) and after each
# update. Results: <out root>/<family>-downstream/{eval-before.json,
# <selector>/policy, <selector>/eval-after.json, downstream_summary.csv}.
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
PY="$VENV_DIR/bin/python"

[ "$#" -ge 2 ] || { echo "usage: $0 <completed point run> <out root> [steps]"; exit 2; }
RUN=$1; OUT_ROOT=$2; STEPS=${3:-50}
[ -s "$RUN/DONE" ] || { echo "[abort] point is not complete: $RUN"; exit 1; }
cfg() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$RUN/run_config.json" "$1"; }
DRIFT=$(cfg drift)
[ "$DRIFT" -gt 0 ] || { echo "[abort] downstream comparison needs a positive-drift point (got d=$DRIFT)"; exit 1; }
POLICY="$RUN/policy_step_$DRIFT"
for artifact in adapter_config.json adapter_model.safetensors optimizer.pt policy_train.json; do
  [ -s "$POLICY/$artifact" ] || { echo "[abort] policy artifact missing: $POLICY/$artifact"; exit 1; }
done
MODEL_PATH="${MODEL_PATH:-$(cfg model)}"
[ -f "$MODEL_PATH/config.json" ] || { echo "[abort] model snapshot missing: $MODEL_PATH"; exit 1; }
export MODEL_PATH
export OM_ATTN="${OM_ATTN:-eager}"
export OM_PROMPT_FORMAT="${OM_PROMPT_FORMAT:-$(cfg prompt_format)}"
SEED=$(cfg seed); MAX_NEW_TOKENS=$(cfg max_new_tokens); TEMPERATURE=$(cfg temperature)
WORLD_SIZE="${GRPO_WORLD_SIZE:-$(cfg grpo_world_size)}"
EVAL_K="${DOWNSTREAM_EVAL_K:-8}"
family=$(basename "$RUN")
OUT="$OUT_ROOT/$family-downstream"
mkdir -p "$OUT/logs"
SELECTORS=(${DOWNSTREAM_SELECTORS:-fresh_r g11 g10 g01 g00 passrate_beta random})

"$PY" src/downstream_compare.py subsets --run "$RUN" --out "$OUT/subsets" \
  --frac "$(cfg topk_frac)" --seed "$SEED" | tee -a "$OUT/logs/main.log" || exit 1

if [ ! -s "$OUT/eval-before.json" ]; then
  echo "[eval] before: $POLICY" | tee -a "$OUT/logs/main.log"
  "$PY" src/downstream_compare.py evaluate --model "$MODEL_PATH" --adapter "$POLICY" \
    --prompts "$RUN/prompts.json" --k "$EVAL_K" --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" --seed "$((SEED + 7))" --out "$OUT/eval-before.json" \
    >> "$OUT/logs/eval-before.log" 2>&1 || { tail -5 "$OUT/logs/eval-before.log"; exit 1; }
fi

failures=0
for selector in "${SELECTORS[@]}"; do
  subset="$OUT/subsets/subset-$selector.json"
  [ -s "$subset" ] || { echo "[skip] no subset for $selector"; failures=$((failures + 1)); continue; }
  target="$OUT/$selector"; mkdir -p "$target"
  if [ -s "$target/eval-after.json" ]; then echo "[done] $selector"; continue; fi
  echo "[train] $selector: $STEPS GRPO updates from step $DRIFT" | tee -a "$OUT/logs/main.log"
  if ! "$PY" -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" src/train_policy_grpo.py \
      --model "$MODEL_PATH" --prompts "$subset" --output "$target/policy" \
      --target-steps "$((DRIFT + STEPS))" --start-step "$DRIFT" \
      --resume-adapter "$POLICY" --resume-optimizer "$POLICY/optimizer.pt" \
      --objective "${RLVR_METHOD:-grpo}" --expected-world-size "$WORLD_SIZE" \
      --group-size "${GRPO_GROUP_SIZE:-$(cfg grpo_group_size)}" \
      --clip-epsilon "${GRPO_CLIP_EPSILON:-$(cfg grpo_clip_epsilon)}" \
      --learning-rate "${GRPO_LEARNING_RATE:-$(cfg grpo_learning_rate)}" \
      --epochs-per-batch "${GRPO_EPOCHS_PER_BATCH:-$(cfg grpo_epochs_per_batch)}" \
      --max-grad-norm "${GRPO_MAX_GRAD_NORM:-$(cfg grpo_max_grad_norm)}" \
      --advantage-epsilon "${GRPO_ADVANTAGE_EPSILON:-$(cfg grpo_advantage_epsilon)}" \
      --lora-rank "${GRPO_LORA_RANK:-$(cfg grpo_lora_rank)}" \
      --lora-alpha "${GRPO_LORA_ALPHA:-$(cfg grpo_lora_alpha)}" \
      --checkpoint-every "${GRPO_CHECKPOINT_EVERY:-5}" \
      --logprob-micro-batch "${GRPO_LOGPROB_MICRO_BATCH:-$(cfg grpo_logprob_micro_batch)}" \
      $([ "${GRPO_GRADIENT_CHECKPOINTING:-$(cfg grpo_gradient_checkpointing)}" = 1 ] || echo --disable-gradient-checkpointing) \
      --max-new-tokens "$MAX_NEW_TOKENS" --seed "$SEED" >> "$target/grpo.log" 2>&1; then
    echo "[fail] $selector training; see $target/grpo.log" | tee -a "$OUT/logs/main.log"
    tail -8 "$target/grpo.log"; failures=$((failures + 1)); continue
  fi
  echo "[eval] after: $selector" | tee -a "$OUT/logs/main.log"
  "$PY" src/downstream_compare.py evaluate --model "$MODEL_PATH" --adapter "$target/policy" \
    --prompts "$RUN/prompts.json" --k "$EVAL_K" --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" --seed "$((SEED + 7))" --out "$target/eval-after.json" \
    >> "$target/eval-after.log" 2>&1 || { tail -5 "$target/eval-after.log"; failures=$((failures + 1)); continue; }
done
"$PY" src/downstream_compare.py summarize --results "$OUT" | tee -a "$OUT/logs/main.log"
[ "$failures" -eq 0 ] || { echo "[downstream] $failures selector(s) failed"; exit 1; }
echo "[downstream] complete: $OUT/downstream_summary.csv"
