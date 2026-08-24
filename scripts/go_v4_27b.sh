#!/usr/bin/env bash
# Current-code 27B-only rerun. With no arguments, independent four-H100
# clusters claim the 10 seed/dataset jobs from a shared lock queue.
set -uo pipefail
cd "$(dirname "$0")/.."

plan_only=0
mode=auto
if [ "${1:-}" = "--plan" ]; then
  plan_only=1
  mode=manual
  shift
fi
if [ "$#" -eq 2 ]; then
  mode=manual
elif [ "$#" -ne 0 ]; then
  echo "usage: bash scripts/go_v4_27b.sh [worker total-workers]" >&2
  exit 2
fi

jobs=(
  "0 gsm8k" "0 math500"
  "1 gsm8k" "1 math500"
  "2 gsm8k" "2 math500"
  "3 gsm8k" "3 math500"
  "4 gsm8k" "4 math500"
)
if [ "$mode" = manual ]; then
  worker=${1:-}
  workers=${2:-}
  if ! [[ "$worker" =~ ^[1-9][0-9]*$ && "$workers" =~ ^[1-9][0-9]*$ ]]; then
    echo "usage: bash scripts/go_v4_27b.sh [worker total-workers]" >&2
    exit 2
  fi
  worker=$((10#$worker))
  workers=$((10#$workers))
  if [ "$workers" -gt 10 ] || [ "$worker" -gt "$workers" ]; then
    echo "[abort] worker=$worker, total=$workers; require 1 <= worker <= total <= 10" >&2
    exit 2
  fi
  assigned=()
  for index in "${!jobs[@]}"; do
    if [ $((index % workers)) -eq $((worker - 1)) ]; then
      assigned+=("${jobs[$index]}")
    fi
  done
else
  assigned=("${jobs[@]}")
fi

if [ "$plan_only" -eq 1 ]; then
  printf '%s\n' "${assigned[@]}"
  exit 0
fi

source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv missing: $VENV_DIR" >&2; exit 1; }
command -v flock >/dev/null 2>&1 || { echo "[abort] flock command missing" >&2; exit 1; }
if [ -n "$(git status --porcelain -- src scripts)" ]; then
  echo "[abort] src/scripts worktree is dirty; commit/pull a clean snapshot first" >&2
  git status --short -- src scripts
  exit 1
fi

current=$(git rev-parse HEAD) || exit 1
code_tag=${current:0:12}
model="${MODEL_27B:-$MODELS_DIR/Qwen3.8-27B-BF16}"
[ -f "$model/config.json" ] || { echo "[abort] 27B model missing: $model" >&2; exit 1; }
model_hash=$(sha256sum "$model/config.json" | cut -d' ' -f1) || exit 1
ngpu=$(timeout 20 nvidia-smi -L 2>/dev/null | wc -l)
[ "${ngpu:-0}" -ge 4 ] || { echo "[abort] four H100s required; detected ${ngpu:-0}" >&2; exit 1; }

fla_ready() {
  "$PY" - <<'PYEOF'
try:
    from importlib.metadata import version
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    installed = version("fla-core")
    assert installed == "0.5.2", installed
    assert modeling.fused_recurrent_gated_delta_rule is not None
    assert modeling.chunk_gated_delta_rule is not None
except Exception as exc:
    print(f"[27b-runtime] FLA 0.5.2 not ready: {exc}")
    raise SystemExit(1)

print(f"[27b-runtime] FLA {installed} fused recurrent/chunk kernels ready")
PYEOF
}

if ! fla_ready; then
  echo "== Qwen3.8 FLA kernel 설치 (공유 venv에서 최초 한 번)"
  mkdir -p "$OM_WORK/locks"
  exec 7>"$OM_WORK/locks/install-fla-0.5.2.lock"
  flock 7 || exit 1
  if ! fla_ready; then
    "$VENV_DIR/bin/pip" install --timeout 60 --retries 2 \
      -c constraints/h100-cu126.txt \
      'flash-linear-attention[cuda]==0.5.2' || {
        echo "[abort] FLA 설치 실패; fallback recurrent 경로로 27B를 실행하지 않음" >&2
        exit 1
      }
  fi
  fla_ready || { echo "[abort] FLA 설치 후 kernel import 실패" >&2; exit 1; }
  flock -u 7
fi

if [ "$mode" = manual ]; then
  worker_tag="w${worker}of${workers}"
  echo "== v4 27B clean rerun manual worker $worker/$workers"
  printf '   assigned: %s\n' "${assigned[*]}"
else
  node_name=$(hostname 2>/dev/null || printf node)
  node_name=$(printf '%s' "$node_name" | tr -cs 'a-zA-Z0-9._-' '-')
  worker_tag="auto-${node_name:0:40}-$$"
  echo "== v4 27B automatic shared-queue worker: $worker_tag"
fi
echo "   commit=$current, gen_batch=4, retries=10"

echo "== 이전 로컬 v4 프로세스 정리"
"$PY" src/cleanup_run_processes.py --run-prefix "$OM_WORK/runs/v4-" || exit 1
pkill -TERM -f -- "--run $OM_WORK/runs/v4-" 2>/dev/null || true
sleep 3
pkill -KILL -f -- "--run $OM_WORK/runs/v4-" 2>/dev/null || true

gpu_memory=$(timeout 20 nvidia-smi --query-gpu=index,memory.used \
  --format=csv,noheader,nounits 2>/dev/null) || {
    echo "[abort] GPU memory query failed" >&2
    exit 1
  }
busy=$(printf '%s\n' "$gpu_memory" | awk -F', *' '$2 > 2000 { count++ } END { print count + 0 }')
if [ "$busy" -gt 0 ]; then
  echo "[abort] 정리 후에도 2GB 초과 GPU가 $busy개 남음" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  exit 1
fi

for gpu in 0 1 2 3; do
  echo "== GPU$gpu FLA kernel smoke"
  CUDA_VISIBLE_DEVICES="$gpu" TORCHINDUCTOR_COMPILE_THREADS=1 \
    timeout 300 "$PY" scripts/check_27b_fla.py || {
      echo "[abort] GPU$gpu FLA kernel smoke 실패; 이 클러스터에서 본실행하지 않음" >&2
      exit 1
    }
done

quarantine="$OM_WORK/quarantine/v4-27b-rerun"
smoke="$OM_WORK/runs/v4-27b-smoke-$code_tag-$worker_tag"
"$PY" src/prepare_run_path.py "$smoke" \
  --expected-git "$current" --quarantine-root "$quarantine" \
  --quarantine-unconfigured || exit 1

run_path() {
  local seed=$1 dataset=$2 path="$OM_WORK/runs/v4-27b-s$1"
  [ "$dataset" = gsm8k ] || path="$path-$dataset"
  printf '%s\n' "$path"
}

run_complete_27b() {
  local run=$1 artifact
  for artifact in DONE run_config.json manifest.json score_protocol.json \
      oracle_protocol.json report.json scores_oracle.json scores_offpolicy.json \
      scores_splithalf.json oracle_micro_groups.pt val_groups.pt; do
    [ -s "$run/$artifact" ] || return 1
  done
}

config_matches_27b() {
  local run=$1 seed=$2 dataset=$3
  "$PY" src/validate_v4_27b.py "$OM_WORK/runs" \
    --expected-git "$current" --expected-model-hash "$model_hash" \
    --single-run "$run" --seed "$seed" --dataset "$dataset" \
    >/dev/null 2>&1
}

run_reusable_27b() {
  local run=$1 seed=$2 dataset=$3
  run_complete_27b "$run" && config_matches_27b "$run" "$seed" "$dataset"
}

matrix_27b_complete() {
  local seed dataset
  for seed in 0 1 2 3 4; do
    for dataset in gsm8k math500; do
      run_reusable_27b "$(run_path "$seed" "$dataset")" "$seed" "$dataset" \
        || return 1
    done
  done
}

run_one_job() {
  local seed=$1 dataset=$2 run n_train
  run=$(run_path "$seed" "$dataset")
  if [ -s "$run/run_config.json" ] \
     && ! config_matches_27b "$run" "$seed" "$dataset"; then
    "$PY" src/prepare_run_path.py "$run" \
      --expected-git "$current" --quarantine-root "$quarantine" \
      --force-quarantine || return 1
  else
    "$PY" src/prepare_run_path.py "$run" \
      --expected-git "$current" --quarantine-root "$quarantine" \
      --quarantine-unconfigured || return 1
  fi
  if run_reusable_27b "$run" "$seed" "$dataset"; then
    echo "== 27B seed=$seed dataset=$dataset 이미 완료"
    return 0
  fi
  echo "== 27B seed=$seed dataset=$dataset"
  n_train=512; [ "$dataset" = math500 ] && n_train=400
  if MODEL_14B="$model" RUN_BASE="$OM_WORK/runs/v4-27b" \
     RESULTS_BASE="$OM_WORK/results/v4-27b" RUN_LABEL="v4-27b-rerun-$worker_tag" \
     RUN_BASE_SMOKE="$smoke" BEHAVIOR_K=8 FRESH_K=32 VAL_K=8 MICRO_GROUP=4 \
     HYBRID_PROMPTS=64 K_CELL=8 DRIFT=100 MAX_NEW_TOKENS=512 PROJ_DIM=4096 \
     GRAD_LAYERS=4 CLIP_CAP=10.0 TEMPERATURE=1.0 TOPK_FRAC=0.10 \
     RADIUS_MODE=gaussian OM_TOP_P=1.0 OM_THINKING=off OM_ATTN=eager \
     OM_LORA_TARGETS=all-linear OM_GEN_BATCH=4 OM_SKIP_HYBRID=1 \
     OM_STALL_MINUTES=20 OM_MAX_RETRIES=10 OM_SKIP_POSTPROCESS=1 OM_GPUS=0,1,2,3 \
     N_TRAIN="$n_train" N_VAL=100 SEEDS="$seed" DATASETS="$dataset" \
     bash scripts/go_v2.sh && run_reusable_27b "$run" "$seed" "$dataset"; then
    echo "== 27B seed=$seed dataset=$dataset 완료"
    return 0
  else
    echo "== 27B seed=$seed dataset=$dataset 미완료 — lock 해제 후 다른 worker가 재시도 가능"
    return 1
  fi
}

mkdir -p "$OM_WORK/locks"
failed=()
if [ "$mode" = manual ]; then
  for job in "${assigned[@]}"; do
    read -r seed dataset <<< "$job"
    exec 8>"$OM_WORK/locks/v4-27b-s$seed-$dataset.lock"
    flock 8 || exit 1
    run_one_job "$seed" "$dataset" || failed+=("s$seed/$dataset")
    flock -u 8
    exec 8>&-
  done
else
  declare -A attempted=()
  while ! matrix_27b_complete; do
    claimed=0
    pending_unattempted=0
    for job in "${jobs[@]}"; do
      read -r seed dataset <<< "$job"
      run=$(run_path "$seed" "$dataset")
      run_reusable_27b "$run" "$seed" "$dataset" && continue
      key="s$seed/$dataset"
      [ "${attempted[$key]:-0}" = 1 ] && continue
      pending_unattempted=1
      exec 8>"$OM_WORK/locks/v4-27b-s$seed-$dataset.lock"
      if ! flock -n 8; then
        exec 8>&-
        continue
      fi
      if run_reusable_27b "$run" "$seed" "$dataset"; then
        flock -u 8
        exec 8>&-
        continue
      fi
      attempted[$key]=1
      claimed=1
      echo "== 자동 선점: $key"
      run_one_job "$seed" "$dataset" || failed+=("$key")
      flock -u 8
      exec 8>&-
      break
    done
    matrix_27b_complete && break
    if [ "$claimed" -eq 0 ]; then
      if [ "$pending_unattempted" -eq 1 ]; then
        echo "== 미완료 작업은 다른 클러스터에서 실행 중 — 30초 대기"
        sleep 30
      else
        echo "[abort] 이 클러스터가 시도한 작업 중 미완료: ${failed[*]}" >&2
        exit 1
      fi
    fi
  done
fi

if [ "$mode" = manual ]; then
  if [ "${#failed[@]}" -gt 0 ] && ! matrix_27b_complete; then
    echo "[abort] 수동 배정 중 미완료: ${failed[*]}" >&2
    exit 1
  fi
  echo "== manual worker $worker/$workers 배정 완료"
else
  if [ "${#failed[@]}" -gt 0 ] && ! matrix_27b_complete; then
    echo "[abort] 미완료: ${failed[*]}" >&2
    exit 1
  fi
  echo "== 27B 10개 run 완료"
fi

matrix_complete() {
  local seed suffix run artifact
  matrix_27b_complete || return 1
  for seed in 0 1 2 3 4; do
    for suffix in "" -math500; do
      run="$OM_WORK/runs/v4-7b-s$seed$suffix"
      for artifact in DONE run_config.json manifest.json score_protocol.json \
          oracle_protocol.json report.json scores_oracle.json scores_offpolicy.json \
          scores_splithalf.json oracle_micro_groups.pt val_groups.pt; do
        [ -s "$run/$artifact" ] || return 1
      done
    done
  done
}

if matrix_complete; then
  exec 9>"$OM_WORK/runs/v4-finalize.lock"
  flock 9 || exit 1
  marker="$OM_WORK/results/v4/V4_COMPLETE"
  "$PY" src/validate_v4_27b.py "$OM_WORK/runs" \
    --expected-git "$current" --expected-model-hash "$model_hash" || exit 1
  if grep -Fxq "git=$current" "$marker" 2>/dev/null \
     && [ -s "$OM_WORK/results/v4-27b/TABLES.md" ] \
     && [ -s "$OM_WORK/results/v4-7b/TABLES.md" ]; then
    echo "== 20-run matrix 결과가 이미 수집됨"
  elif matrix_complete; then
    echo "== 20-run matrix 완료 — 최종 결과 자동 수집"
    bash scripts/collect_v4.sh || exit 1
    mkdir -p "$(dirname "$marker")"
    printf 'completed=%s\ngit=%s\n' "$(date -Is)" "$current" > "$marker.tmp"
    mv "$marker.tmp" "$marker"
  fi
  flock -u 9
else
  echo "== 전체 20-run matrix 미완료 — 마지막 자동 worker가 수집"
fi
