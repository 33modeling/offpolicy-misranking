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

PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv python 없음: $PY"; exit 1; }
command -v flock >/dev/null || { echo "[abort] flock 없음"; exit 1; }

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

run_dir() {
  printf '%s/%s-s%s-%s-d%s\n' "$ROOT" "$MODEL_TAG" "$2" "$1" "$3"
}

run_complete() {
  local run=$1 artifact
  for artifact in DONE run_config.json manifest.json score_protocol.json \
      oracle_protocol.json report.json scores_oracle.json scores_offpolicy.json \
      scores_splithalf.json oracle_micro_groups.pt val_groups.pt; do
    [ -s "$run/$artifact" ] || return 1
  done
}

family_complete() {
  local dataset=$1 seed=$2 drift
  for drift in "${DRIFTS[@]}"; do
    run_complete "$(run_dir "$dataset" "$seed" "$drift")" || return 1
  done
}

run_point() {
  local dataset=$1 seed=$2 drift=$3 source=$4 run try n_train
  run=$(run_dir "$dataset" "$seed" "$drift")
  n_train=512
  [ "$dataset" != "math500" ] || n_train=400
  run_complete "$run" && return 0
  for try in $(seq 1 "$MAX_RETRIES"); do
    echo "[$(date '+%F %T')] $dataset/s$seed/d$drift try $try/$MAX_RETRIES -> $run"
    args=(env DATASET="$dataset" SEED="$seed" DRIFT="$drift" OUT_ROOT="$run"
      N_TRAIN="$n_train" N_VAL=100 BEHAVIOR_K=8 FRESH_K=32 VAL_K=8
      MICRO_GROUP=4 PROJ_DIM=4096 GRAD_LAYERS=4 CLIP_CAP=10 TOPK_FRAC=0.10
      OM_SKIP_HYBRID=1 OM_RETRY_INDEX="$try")
    [ -z "$source" ] || args+=(OM_BEHAVIOR_SOURCE="$source")
    if "${args[@]}" bash scripts/run_14b.sh && run_complete "$run"; then
      return 0
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
(
  flock 9
  runs=()
  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for drift in "${DRIFTS[@]}"; do
        run=$(run_dir "$dataset" "$seed" "$drift")
        run_complete "$run" || { echo "[collect-abort] incomplete: $run"; exit 1; }
        runs+=("$run")
      done
    done
  done
  analysis_args=(--output-dir "$RESULTS"
    --first-bootstrap "${REGIME_FIRST_BOOTSTRAP:-2000}")
  if [ -n "${REGIME_FIRST_CALIBRATION:-}" ]; then
    analysis_args+=(--first-calibration "$REGIME_FIRST_CALIBRATION")
  fi
  "$PY" src/regime_map.py "${runs[@]}" "${analysis_args[@]}"
) 9>"$QUEUE/collect.lock"

echo "== regime sweep complete: $RESULTS/FINAL_REPORT.md"
