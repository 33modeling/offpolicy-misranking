#!/usr/bin/env bash
# Completed v4 runs only: validate 20 shared run directories, build model-separated
# TABLES/FRONTIER reports in staging, publish them, then run the final harvest.
# This script never starts or stops GPU work and never modifies a run directory.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh

required=(
  DONE run_config.json manifest.json score_protocol.json oracle_protocol.json
  report.json scores_oracle.json scores_offpolicy.json scores_splithalf.json
  oracle_micro_groups.pt val_groups.pt
)
missing_runs=0
incomplete_runs=0
missing_artifacts=0
targets_27b=()
targets_7b=()

echo "== v4 결과 20개 자동 수집 (GPU 사용 안 함)"
for model in 27b 7b; do
  for seed in 0 1 2 3 4; do
    for suffix in "" -math500; do
      run="$OM_WORK/runs/v4-$model-s$seed$suffix"
      if [ "$model" = 27b ]; then
        targets_27b+=("$run")
      else
        targets_7b+=("$run")
      fi
      if [ ! -d "$run" ]; then
        echo "[missing-run] $(basename "$run"): 실험 결과 디렉터리 없음" >&2
        missing_runs=$((missing_runs + 1))
        continue
      fi
      if [ ! -s "$run/DONE" ]; then
        echo "[incomplete-run] $(basename "$run"): DONE 없음 또는 비어 있음" >&2
        incomplete_runs=$((incomplete_runs + 1))
      fi
      for artifact in "${required[@]:1}"; do
        if [ ! -s "$run/$artifact" ]; then
          echo "[missing-artifact] $(basename "$run")/$artifact" >&2
          missing_artifacts=$((missing_artifacts + 1))
        fi
      done
      divergence=("$run"/divergence_stats*.json)
      if [ ! -s "${divergence[0]}" ]; then
        echo "[missing-artifact] $(basename "$run")/divergence_stats*.json" >&2
        missing_artifacts=$((missing_artifacts + 1))
      fi
    done
  done
done

if [ "$missing_runs" -gt 0 ] || [ "$incomplete_runs" -gt 0 ] \
   || [ "$missing_artifacts" -gt 0 ]; then
  echo "[collect-v4-abort] 결과 없는 run=$missing_runs, 미완료 run=$incomplete_runs, 후처리 산출물 누락=$missing_artifacts" >&2
  echo "[collect-v4-abort] GPU는 재실행하지 않았음" >&2
  exit 1
fi

results_root="$OM_WORK/results"
mkdir -p "$results_root" || exit 1
stage=$(mktemp -d "$results_root/.v4-collect.XXXXXX") || exit 1
cleanup() { rm -rf "$stage"; }
trap cleanup EXIT

build_model_reports() {
  local model=$1
  shift
  local out="$stage/v4-$model"
  mkdir -p "$out" || return 1
  OM_RESULTS="$out" bash scripts/tables.sh "$@" || return 1
  OM_RESULTS="$out" bash scripts/frontier.sh "$@" || return 1
  [ -s "$out/TABLES.md" ] || return 1
  [ -s "$out/FRONTIER.md" ] || return 1
  [ -s "$out/frontier.json" ] || return 1
}

build_model_reports 27b "${targets_27b[@]}" || {
  echo "[collect-v4-abort] 27B 표 생성 실패" >&2
  exit 1
}
build_model_reports 7b "${targets_7b[@]}" || {
  echo "[collect-v4-abort] 7B 표 생성 실패" >&2
  exit 1
}

publish_model_reports() {
  local model=$1 source="$stage/v4-$1" destination="$results_root/v4-$1"
  local file tmp
  mkdir -p "$destination" || return 1
  for file in TABLES.md FRONTIER.md frontier.json; do
    tmp="$destination/$file.tmp.$$"
    cp -- "$source/$file" "$tmp" || return 1
    mv "$tmp" "$destination/$file" || return 1
  done
}

publish_model_reports 27b || exit 1
publish_model_reports 7b || exit 1

echo "== 모델별 표 게시 완료; 최종 harvest 실행"
bash scripts/harvest.sh
