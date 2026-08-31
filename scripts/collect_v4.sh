#!/usr/bin/env bash
# Completed v4 runs only: validate 20 shared run directories, build model-separated
# TABLES/FRONTIER reports in staging, publish them, then run the final harvest.
# This script never starts or stops GPU work and never modifies a run directory.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
source scripts/_report_cache.sh

command -v flock >/dev/null 2>&1 || {
  echo "[collect-v4-abort] flock command missing" >&2
  exit 1
}
mkdir -p "$OM_WORK/locks" || exit 1
exec 7>"$OM_WORK/locks/collect-v4.lock"
flock 7 || exit 1

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
  local out="$stage/v4-$model" destination="$results_root/v4-$model"
  local run artifact file tmp key
  local data_files=()
  local code_files=(
    scripts/tables.sh scripts/frontier.sh src/make_tables.py src/frontier.py
    src/certagrad.py src/gate_rules.py src/select_rules.py
  )

  for run in "$@"; do
    for artifact in DONE run_config.json manifest.json score_protocol.json \
        oracle_protocol.json report.json scores_oracle.json scores_offpolicy.json \
        scores_splithalf.json oracle_micro_groups.pt val_groups.pt \
        rollouts_behavior_train.jsonl rollouts_fresh_train.jsonl; do
      data_files+=("$run/$artifact")
    done
    shopt -s nullglob
    for artifact in "$run"/divergence_stats*.json "$run"/scores_hybrid_*.json \
        "$run"/hybrid_protocol_*.json "$run"/downstream_*.json; do
      data_files+=("$artifact")
    done
    shopt -u nullglob
  done
  key=$(report_cache_key "${code_files[@]}" -- "${data_files[@]}") || return 1
  key=$(report_cache_key_values "$key" "model=$model" "report-schema=1") || return 1
  if report_cache_hit "$destination/.analysis.key" "$key" \
      "$destination/TABLES.md" "$destination/FRONTIER.md" \
      "$destination/frontier.json"; then
    echo "== v4-$model 분석 입력 변경 없음; TABLES/FRONTIER 재사용"
    return 0
  fi

  # Migrate a pre-cache report without recomputation only when every output is
  # nonempty and newer than every existing analyzer/input dependency.
  if [ ! -e "$destination/.analysis.key" ] \
     && [ -s "$destination/TABLES.md" ] \
     && [ -s "$destination/FRONTIER.md" ] \
     && [ -s "$destination/frontier.json" ]; then
    existing_is_fresh=1
    for artifact in "${code_files[@]}" "${data_files[@]}"; do
      [ -e "$artifact" ] || continue
      for file in TABLES.md FRONTIER.md frontier.json; do
        if [ "$artifact" -nt "$destination/$file" ]; then
          existing_is_fresh=0
          break 2
        fi
      done
    done
    if [ "$existing_is_fresh" -eq 1 ]; then
      report_cache_write "$destination/.analysis.key" "$key" \
        "$destination/TABLES.md" "$destination/FRONTIER.md" \
        "$destination/frontier.json" || return 1
      echo "== v4-$model 기존 TABLES/FRONTIER 입력 시각 검증 완료; 재계산 생략"
      return 0
    fi
  fi

  mkdir -p "$out" || return 1
  OM_RESULTS="$out" bash scripts/tables.sh "$@" || return 1
  OM_RESULTS="$out" bash scripts/frontier.sh "$@" || return 1
  [ -s "$out/TABLES.md" ] || return 1
  [ -s "$out/FRONTIER.md" ] || return 1
  [ -s "$out/frontier.json" ] || return 1

  mkdir -p "$destination" || return 1
  for file in TABLES.md FRONTIER.md frontier.json; do
    tmp="$destination/$file.tmp.$$"
    cp -- "$out/$file" "$tmp" || return 1
    mv "$tmp" "$destination/$file" || return 1
  done
  report_cache_write "$destination/.analysis.key" "$key" \
    "$destination/TABLES.md" "$destination/FRONTIER.md" \
    "$destination/frontier.json"
}

build_model_reports 27b "${targets_27b[@]}" || {
  echo "[collect-v4-abort] 27B 표 생성 실패" >&2
  exit 1
}
build_model_reports 7b "${targets_7b[@]}" || {
  echo "[collect-v4-abort] 7B 표 생성 실패" >&2
  exit 1
}

echo "== 모델별 표 게시 완료; 최종 harvest 실행"
bash scripts/harvest.sh
