#!/usr/bin/env bash
# Regime discovery sweep. Run the same command on every independent cluster:
#
#   bash scripts/go_regime.sh
#
# A shared flock queue assigns one seed x dataset family to each cluster.  A
# family creates one behavior pool and evaluates every drift on that exact pool.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
source scripts/_report_cache.sh

PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv python 없음: $PY"; exit 1; }
command -v flock >/dev/null || { echo "[abort] flock 없음"; exit 1; }

# Supervisors may be newer than a partially completed run. Generation always
# re-enters the immutable code snapshot recorded by that run.
PIPELINE_REPO="${OM_PIPELINE_REPO:-$PWD}"
PIPELINE_SCRIPT="${OM_PIPELINE_SCRIPT:-$PIPELINE_REPO/scripts/run_14b.sh}"
[ -s "$PIPELINE_SCRIPT" ] || { echo "[abort] generation pipeline 없음: $PIPELINE_SCRIPT"; exit 1; }

export MODEL_14B="${MODEL_14B:-$MODELS_DIR/Qwen2.5-7B-Instruct}"
MODEL_TAG="${REGIME_MODEL_TAG:-$(basename "$MODEL_14B" | tr '[:upper:]' '[:lower:]')}"
ROOT="${REGIME_ROOT:-$OM_WORK/runs/regime-$MODEL_TAG}"
QUEUE="$ROOT/.queue"
RESULTS="${REGIME_RESULTS:-$OM_WORK/results/regime-$MODEL_TAG}"
mkdir -p "$QUEUE" "$RESULTS"

# Discovery uses three seeds.  Seeds 3-4 are a held-back confirmation extension.
SEEDS=(${REGIME_SEEDS:-0 1 2})
DATASETS=(${REGIME_DATASETS:-gsm8k math500})
DRIFTS=(${REGIME_DRIFTS:-0 25 100 400})
MAX_RETRIES="${REGIME_MAX_RETRIES:-3}"
CONTRACT="${REGIME_MATRIX:-}"
QUARANTINE="${REGIME_QUARANTINE:-$OM_WORK/quarantine/regime-$MODEL_TAG}"
N_VAL_DEFAULT="${REGIME_N_VAL:-100}"
BEHAVIOR_K_DEFAULT="${REGIME_BEHAVIOR_K:-8}"
FRESH_K_DEFAULT="${REGIME_FRESH_K:-32}"
VAL_K_DEFAULT="${REGIME_VAL_K:-8}"
MICRO_GROUP_DEFAULT="${REGIME_MICRO_GROUP:-4}"
MAX_NEW_TOKENS_DEFAULT="${REGIME_MAX_NEW_TOKENS:-512}"
PROJ_DIM_DEFAULT="${REGIME_PROJ_DIM:-4096}"
GRAD_LAYERS_DEFAULT="${REGIME_GRAD_LAYERS:-4}"
CLIP_CAP_DEFAULT="${REGIME_CLIP_CAP:-10}"
TOPK_FRAC_DEFAULT="${REGIME_TOPK_FRAC:-0.10}"
TEMPERATURE_DEFAULT="${REGIME_TEMPERATURE:-1.0}"

if [ -n "$CONTRACT" ] && [ ! -s "$CONTRACT" ]; then
  echo "[abort] regime matrix contract 없음: $CONTRACT"
  exit 1
fi

contract_run() {
  local command=$1 run=$2 dataset=$3 seed=$4 drift=$5 source=$6
  shift 6
  local args=("$PY" src/regime_contract.py "$command" --matrix "$CONTRACT"
    --run "$run" --dataset "$dataset" --seed "$seed" --drift "$drift")
  [ -z "$source" ] || args+=(--behavior-source "$source")
  "${args[@]}" "$@"
}

run_dir() {
  printf '%s/%s-s%s-%s-d%s\n' "$ROOT" "$MODEL_TAG" "$2" "$1" "$3"
}

run_complete() {
  local run=$1 dataset=$2 seed=$3 drift=$4 source=$5 artifact
  for artifact in DONE run_config.json manifest.json score_protocol.json \
      oracle_protocol.json report.json scores_oracle.json scores_offpolicy.json \
      scores_splithalf.json divergence_stats.json oracle_micro_groups.pt val_groups.pt; do
    [ -s "$run/$artifact" ] || return 1
  done
  [ -z "$CONTRACT" ] || contract_run check-run "$run" "$dataset" "$seed" "$drift" "$source" \
    >/dev/null 2>&1
}

family_complete() {
  local dataset=$1 seed=$2 drift source
  source=$(run_dir "$dataset" "$seed" 0)
  for drift in "${DRIFTS[@]}"; do
    run_complete "$(run_dir "$dataset" "$seed" "$drift")" "$dataset" "$seed" "$drift" \
      "$([ "$drift" = 0 ] || printf '%s' "$source")" || return 1
  done
}

run_point() {
  local dataset=$1 seed=$2 drift=$3 source=$4 run try n_train
  run=$(run_dir "$dataset" "$seed" "$drift")
  n_train="${REGIME_N_TRAIN:-512}"
  [ -n "${REGIME_N_TRAIN:-}" ] || [ "$dataset" != "math500" ] || n_train=400
  if [ -n "$CONTRACT" ]; then
    contract_run prepare-run "$run" "$dataset" "$seed" "$drift" "$source" \
      --quarantine-root "$QUARANTINE" || return 1
  fi
  run_complete "$run" "$dataset" "$seed" "$drift" "$source" && return 0
  for try in $(seq 1 "$MAX_RETRIES"); do
    echo "[$(date '+%F %T')] $dataset/s$seed/d$drift try $try/$MAX_RETRIES -> $run"
    args=(env DATASET="$dataset" SEED="$seed" DRIFT="$drift" OUT_ROOT="$run"
      N_TRAIN="$n_train" N_VAL="$N_VAL_DEFAULT" BEHAVIOR_K="$BEHAVIOR_K_DEFAULT"
      FRESH_K="$FRESH_K_DEFAULT" VAL_K="$VAL_K_DEFAULT"
      MICRO_GROUP="$MICRO_GROUP_DEFAULT" PROJ_DIM="$PROJ_DIM_DEFAULT"
      GRAD_LAYERS="$GRAD_LAYERS_DEFAULT" CLIP_CAP="$CLIP_CAP_DEFAULT"
      TOPK_FRAC="$TOPK_FRAC_DEFAULT" MAX_NEW_TOKENS="$MAX_NEW_TOKENS_DEFAULT"
      TEMPERATURE="$TEMPERATURE_DEFAULT"
      OM_SKIP_HYBRID="${OM_SKIP_HYBRID:-1}" OM_RETRY_INDEX="$try")
    [ -z "$source" ] || args+=(OM_BEHAVIOR_SOURCE="$source")
    if "${args[@]}" OM_REPO="$PIPELINE_REPO" PYTHONPATH="$PIPELINE_REPO/src" \
        bash "$PIPELINE_SCRIPT"; then
      if [ -n "$CONTRACT" ]; then
        if ! contract_run check-run "$run" "$dataset" "$seed" "$drift" "$source" \
          --deep --mark; then
          echo "[contract-fail] $dataset/s$seed/d$drift 산출물 격리 후 재시도"
          contract_run prepare-run "$run" "$dataset" "$seed" "$drift" "$source" \
            --quarantine-root "$QUARANTINE" || return 1
          continue
        fi
      fi
      run_complete "$run" "$dataset" "$seed" "$drift" "$source" && return 0
    fi
    bash scripts/diagnose_run_failure.sh "$run" \
      "$run/logs/main.log" 1 2>/dev/null || true
    sleep 20
  done
  return 1
}

run_family() {
  local dataset=$1 seed=$2 source drift
  source=$(run_dir "$dataset" "$seed" 0)
  # d0 is the exact positive control: beta=pi with independent rollout noise.
  run_point "$dataset" "$seed" 0 "" || return 1
  for drift in "${DRIFTS[@]}"; do
    [ "$drift" = 0 ] && continue
    run_point "$dataset" "$seed" "$drift" "$source" || return 1
  done
}

echo "== regime queue: model=$MODEL_TAG seeds=${SEEDS[*]} data=${DATASETS[*]} drift=${DRIFTS[*]}"
failures=0
while :; do
  remaining=0
  claimed=0
  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      family_complete "$dataset" "$seed" && continue
      remaining=$((remaining + 1))
      lock="$QUEUE/$dataset-s$seed.lock"
      (
        flock -n 9 || exit 75
        family_complete "$dataset" "$seed" && exit 0
        run_family "$dataset" "$seed"
      ) 9>"$lock"
      rc=$?
      [ "$rc" -eq 75 ] && continue
      claimed=$((claimed + 1))
      if [ "$rc" -ne 0 ]; then
        echo "[family-fail] $dataset/s$seed"
        failures=$((failures + 1))
      fi
    done
  done
  [ "$remaining" -eq 0 ] && break
  if [ "$claimed" -eq 0 ]; then
    echo "[queue] 다른 클러스터의 ${remaining}개 family 완료 대기"
    sleep 60
  elif [ "$failures" -gt 0 ]; then
    echo "[abort] 이 worker에서 family 실패 ${failures}개; 같은 명령으로 artifact부터 재개"
    exit 1
  fi
done

# Aggregate publication is serialized across workers. Machine outputs are
# JSON/CSV; the sole human-facing output is FINAL_REPORT.md.
if ! (
  flock 9
  runs=()
  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for drift in "${DRIFTS[@]}"; do
        run=$(run_dir "$dataset" "$seed" "$drift")
        source=$(run_dir "$dataset" "$seed" 0)
        point_source=""; [ "$drift" = 0 ] || point_source="$source"
        run_complete "$run" "$dataset" "$seed" "$drift" "$point_source" \
          || { echo "[collect-abort] incomplete: $run"; exit 1; }
        runs+=("$run")
      done
    done
  done
  if [ -n "$CONTRACT" ] && "$PY" src/regime_contract.py check-collection \
      --matrix "$CONTRACT" --results "$RESULTS" --runs "${runs[@]}" \
      >/dev/null 2>&1; then
    echo "[collect] matrix-bound analysis already current"
    exit 0
  fi

  # Discovery has no external matrix contract. Bind the analysis publication
  # to its exact completed inputs so every waiting cluster does not rerun the
  # same 2,000-bootstrap regime map after acquiring collect.lock.
  analysis_code=(
    src/regime_map.py src/gate_rules.py src/score_artifacts.py
    src/select_rules.py src/first_interval.py
  )
  analysis_inputs=()
  for run in "${runs[@]}"; do
    for artifact in DONE run_config.json manifest.json score_protocol.json \
        oracle_protocol.json report.json scores_oracle.json scores_offpolicy.json \
        scores_splithalf.json divergence_stats.json oracle_micro_groups.pt val_groups.pt; do
      analysis_inputs+=("$run/$artifact")
    done
  done
  [ -z "$CONTRACT" ] || analysis_inputs+=("$CONTRACT")
  analysis_key=$(report_cache_key "${analysis_code[@]}" -- "${analysis_inputs[@]}") \
    || exit 1
  analysis_key=$(report_cache_key_values "$analysis_key" \
    "first_bootstrap=${REGIME_FIRST_BOOTSTRAP:-2000}" \
    "first_calibration=${REGIME_FIRST_CALIBRATION:-}") || exit 1
  analysis_marker="$RESULTS/.regime_analysis.key"
  analysis_outputs=(
    "$RESULTS/REGIME.json" "$RESULTS/REGIME.csv"
    "$RESULTS/REGIME_SUMMARY.csv" "$RESULTS/FINAL_REPORT.md"
  )
  if report_cache_hit "$analysis_marker" "$analysis_key" "${analysis_outputs[@]}"; then
    echo "[collect] regime analysis already current; duplicate aggregation skipped"
    exit 0
  fi
  if [ -n "$CONTRACT" ]; then
    for seed in "${SEEDS[@]}"; do
      for dataset in "${DATASETS[@]}"; do
        for drift in "${DRIFTS[@]}"; do
          run=$(run_dir "$dataset" "$seed" "$drift")
          source=$(run_dir "$dataset" "$seed" 0)
          point_source=""; [ "$drift" = 0 ] || point_source="$source"
          contract_run check-run "$run" "$dataset" "$seed" "$drift" "$point_source" \
            --deep --mark || exit 1
        done
      done
    done
  fi
  analysis_args=(--output-dir "$RESULTS"
    --first-bootstrap "${REGIME_FIRST_BOOTSTRAP:-2000}")
  if [ -n "${REGIME_FIRST_CALIBRATION:-}" ]; then
    analysis_args+=(--first-calibration "$REGIME_FIRST_CALIBRATION")
  fi
  "$PY" src/regime_map.py "${runs[@]}" "${analysis_args[@]}" || exit 1
  for output in "${analysis_outputs[@]}"; do
    [ -s "$output" ] || { echo "[collect-abort] empty analysis output: $output"; exit 1; }
  done
  if [ -n "$CONTRACT" ]; then
    "$PY" src/regime_contract.py mark-collection --matrix "$CONTRACT" \
      --results "$RESULTS" --runs "${runs[@]}" || exit 1
  fi
  report_cache_write "$analysis_marker" "$analysis_key" \
    "${analysis_outputs[@]}" || exit 1
) 9>"$QUEUE/collect.lock"; then
  echo "[collect-abort] final validation or analysis failed"
  exit 1
fi

echo "== regime sweep complete: $RESULTS/FINAL_REPORT.md"
