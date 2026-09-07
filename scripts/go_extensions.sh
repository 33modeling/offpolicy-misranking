#!/usr/bin/env bash
# One-shot runner for the registered extensions E1-E6 (2026-09-07) on a
# 4xH100 node that has completed (or partially completed) the OLMo-3 matrix.
#
#   git pull --ff-only
#   bash scripts/go_extensions.sh [profile] [stage ...]
#
#   profile : baseline (default) | h100   -- selects the registered run root
#   stages  : any of  synthetic rescore analyze curve downstream  (default: all,
#             in that order). Every stage is idempotent and resumable.
#
# Run it in a foreground tmux window; everything is also appended to
# $OM_WORK/console-logs/extensions-<timestamp>.log. Results land under
# $OM_WORK/results/extensions-<model tag>/ as CSV/Markdown/dat files that the
# manuscript figures read directly.
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
PY="$VENV_DIR/bin/python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

PROFILE=${1:-baseline}; [ "$#" -ge 1 ] && shift
STAGES=("$@"); [ "${#STAGES[@]}" -gt 0 ] || STAGES=(synthetic rescore analyze curve downstream)
case "$PROFILE" in
  baseline) CONFIG=configs/olmo3_rlzero.json; MODEL_TAG="${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-v1}"; CURVE_PROFILE="" ;;
  h100)     CONFIG=configs/olmo3_rlzero_h100.json; MODEL_TAG="${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}"; CURVE_PROFILE=h100 ;;
  *) echo "[abort] unknown profile: $PROFILE"; exit 2 ;;
esac
ROOT="${OM_OLMO3_ROOT:-$OM_WORK/runs/$MODEL_TAG}"
OUT="$OM_WORK/results/extensions-$MODEL_TAG"
LOG="$OM_WORK/console-logs/extensions-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$OUT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[extensions] profile=$PROFILE root=$ROOT out=$OUT stages=${STAGES[*]}"

MODEL_KEY=olmo3-7b-base
model_field() { "$PY" src/model_matrix.py --config "$CONFIG" --models-dir "$MODELS_DIR" field "$MODEL_KEY" "$1"; }
export MODEL_PATH="${MODEL_PATH:-$(model_field path)}"
[ -f "$MODEL_PATH/config.json" ] || { echo "[abort] model snapshot missing: $MODEL_PATH"; exit 1; }
export OM_ATTN="${OM_ATTN:-eager}"
NGPU=$(timeout 20 nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}

completed_runs() {  # completed_runs [dataset] [drift]
  local run
  for run in "$ROOT"/family-*/"$MODEL_TAG"-s*-*-d*; do
    [ -s "$run/DONE" ] || continue
    [ -z "${1:-}" ] || [[ "$run" == *"-$1-d"* ]] || continue
    [ -z "${2:-}" ] || [[ "$run" == *"-d$2" ]] || continue
    printf '%s\n' "$run"
  done
}

want() { local s; for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

if want synthetic; then
  echo "== E4 synthetic pool study (CPU)"
  "$PY" src/synthetic_pool_study.py --output-dir "$OUT/synthetic" || exit 1
fi

if want rescore; then
  echo "== E6 re-scoring variants over completed runs ($NGPU GPUs)"
  mapfile -t runs < <(completed_runs)
  [ "${#runs[@]}" -gt 0 ] || echo "[rescore] no completed runs under $ROOT"
  i=0; pids=()
  for run in "${runs[@]}"; do
    if [ -s "$run/scores_offpolicy.variant-clip30.json" ] && [ -s "$run/scores_offpolicy.variant-bk2.json" ]; then
      echo "[rescore] done: $(basename "$run")"; continue
    fi
    gpu=$((i % NGPU)); i=$((i + 1))
    (
      export OM_PROMPT_FORMAT=olmo_rlzero_math; [[ "$run" == *-mbpp-d* ]] && export OM_PROMPT_FORMAT=olmo_rlzero_code
      CUDA_VISIBLE_DEVICES=$gpu "$PY" src/rescore_variants.py --run "$run" --model "$MODEL_PATH" \
        --variants bk2 bk4 clip3 clip30 --proj-dim "$("$PY" src/model_matrix.py --config "$CONFIG" experiment-field proj_dim)" \
        --grad-layers "$("$PY" src/model_matrix.py --config "$CONFIG" experiment-field grad_layers)" --micro-batch 2 \
        > "$run/logs/rescore-variants.log" 2>&1 && echo "[rescore] ok: $(basename "$run")" \
        || { echo "[rescore] FAIL: $(basename "$run") (see logs/rescore-variants.log)"; }
    ) &
    pids+=($!)
    if [ "${#pids[@]}" -ge "$NGPU" ]; then wait "${pids[@]}"; pids=(); fi
  done
  [ "${#pids[@]}" -eq 0 ] || wait "${pids[@]}"
fi

if want analyze; then
  echo "== E2/E3/E6 analyses over completed runs"
  mapfile -t runs < <(completed_runs)
  if [ "${#runs[@]}" -gt 0 ]; then
    "$PY" src/reversal_matrix.py "${runs[@]}" --output-dir "$OUT/reversal" || echo "[analyze] reversal_matrix failed"
    "$PY" src/margin_condition.py "${runs[@]}" --output-dir "$OUT/margin"; rc=$?
    [ "$rc" -ne 2 ] || echo "[analyze] MARGIN-CONDITION VIOLATION: see $OUT/margin/margin_condition_summary.json"
    "$PY" src/sensitivity_tables.py "${runs[@]}" --output-dir "$OUT/sensitivity" || echo "[analyze] sensitivity_tables failed"
    "$PY" src/diagnostics_vs_retention.py "${runs[@]}" --output-dir "$OUT/diagnostics" || echo "[analyze] diagnostics failed"
    mapfile -t curve_runs < <(ls -d "$OM_WORK/runs/$MODEL_TAG-curve"/*-d* 2>/dev/null)
    "$PY" src/drift_curve.py --registered "${runs[@]}" --curve "${curve_runs[@]}" --output-dir "$OUT/drift_curve" \
      || echo "[analyze] drift_curve failed"
  else
    echo "[analyze] no completed runs under $ROOT"
  fi
fi

if want curve; then
  echo "== E1 drift-resolution chain (MATH-500, seeds 0 1)"
  bash scripts/run_drift_curve.sh "$CONFIG" "$OM_WORK/runs/$MODEL_TAG-curve" "$OM_WORK/results/$MODEL_TAG-curve" $CURVE_PROFILE \
    || echo "[curve] run_drift_curve.sh failed; rerun this stage to resume"
fi

if want downstream; then
  echo "== E5 matched downstream update (MATH-500 d=100, seeds 0-2)"
  for seed in 0 1 2; do
    run="$(completed_runs math500 100 | grep -- "-s$seed-" | head -1)"
    [ -n "$run" ] || { echo "[downstream] seed $seed: no completed math500 d=100 run yet"; continue; }
    bash scripts/run_downstream_compare.sh "$run" "$OM_WORK/runs/$MODEL_TAG-downstream" 50 \
      || echo "[downstream] seed $seed failed; rerun this stage to resume"
  done
  mkdir -p "$OUT/downstream"
  for summary in "$OM_WORK/runs/$MODEL_TAG-downstream"/*-downstream/downstream_summary.csv; do
    [ -s "$summary" ] && cp "$summary" "$OUT/downstream/$(basename "$(dirname "$summary")").csv"
  done
fi

echo "[extensions] finished. Results: $OUT   Log: $LOG"
