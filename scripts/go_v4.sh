#!/usr/bin/env bash
# v4 confirmatory rerun after the 2026-08-20 generation and validation-integrity fixes.
#
# This runs the corrected matrix without mixing models or historical artifacts:
#   - Qwen3.8-27B-BF16 seeds 0..4: current main-model confirmation
#   - Qwen2.5-7B-Instruct seeds 0..4: same-condition historical replication
#   - GSM8K and MATH500 for both; saturated 27B DAPO is not a valid test pool
#
# Cloud usage, after the currently running jobs finish and each checkout is updated:
# Run on three independent clusters (four H100s each):
#   cluster 1: 27B seeds 0,1; 7B seed 0
#   cluster 2: 27B seeds 2,3; 7B seed 1
#   cluster 3: 27B seed 4;    7B seeds 2,3,4
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || PY=python3

# A post-run git pull may contain analysis-only changes. Re-enter through the
# immutable generation snapshot recorded by existing run configs, so multi-day
# partial artifacts resume instead of being quarantined as a new experiment.
if [ "${OM_V4_RESUME_WRAPPED:-0}" != "1" ]; then
  exec bash scripts/resume_v4.sh "$@"
fi

if [ -n "$(git status --porcelain -- src scripts)" ]; then
  echo "[abort] src/scripts worktree is dirty; v4 requires a committed code snapshot"
  git status --short -- src scripts
  exit 1
fi

EXPECTED_V4_SEEDS="${EXPECTED_V4_SEEDS:-0 1 2 3 4}"
MODEL_7B="${MODEL_7B:-$MODELS_DIR/Qwen2.5-7B-Instruct}"
MODEL_27B="${MODEL_27B:-$MODELS_DIR/Qwen3.8-27B-BF16}"
RUN_BASE_7B="$OM_WORK/runs/v4-7b"
RUN_BASE_27B="$OM_WORK/runs/v4-27b"
RESULTS_BASE_7B="$OM_WORK/results/v4-7b"
RESULTS_BASE_27B="$OM_WORK/results/v4-27b"
V4_STATUS_ROOT="$OM_WORK/results/v4"

V4_WORKER_SLOT="${1:-}"
case "$V4_WORKER_SLOT" in
  1) SEEDS_27B="0 1"; SEEDS_7B="0" ;;
  2) SEEDS_27B="2 3"; SEEDS_7B="1" ;;
  3) SEEDS_27B="4";   SEEDS_7B="2 3 4" ;;
  *)
    echo "usage: bash scripts/go_v4.sh <cluster: 1|2|3>"
    exit 2
    ;;
esac

NGPU_V4=$(timeout 20 nvidia-smi -L 2>/dev/null | wc -l)
[ "${NGPU_V4:-0}" -ge 4 ] || {
  echo "[abort] v4 requires a 4-GPU node; detected ${NGPU_V4:-0}"
  exit 1
}

for model in "$MODEL_7B" "$MODEL_27B"; do
  [ -f "$model/config.json" ] || {
    echo "[abort] v4 model snapshot missing: $model"
    exit 1
  }
done
MODEL_HASH_7B=$(sha256sum "$MODEL_7B/config.json" | cut -d' ' -f1)
MODEL_HASH_27B=$(sha256sum "$MODEL_27B/config.json" | cut -d' ' -f1)
CURRENT_GIT=$(git rev-parse HEAD)
code_tag=${CURRENT_GIT:0:12}

echo "== 이전 v4 프로세스 및 GPU 점유 정리"
"$PY" src/cleanup_run_processes.py --run-prefix "$OM_WORK/runs/v4-" || exit 1
# Handle any child that raced with the process snapshot above.
pkill -TERM -f -- "--run $OM_WORK/runs/v4-" 2>/dev/null || true
sleep 3
pkill -KILL -f -- "--run $OM_WORK/runs/v4-" 2>/dev/null || true

gpu_clear=0
for _ in $(seq 1 30); do
  gpu_memory=$(timeout 20 nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null) || {
    sleep 2
    continue
  }
  gpu_rows=$(printf '%s\n' "$gpu_memory" | awk 'NF {n++} END {print n+0}')
  [ "$gpu_rows" -eq "$NGPU_V4" ] || {
    sleep 2
    continue
  }
  busy=$(printf '%s\n' "$gpu_memory" | awk '$1 > 2000 {n++} END {print n+0}')
  if [ "${busy:-1}" -eq 0 ]; then
    gpu_clear=1
    break
  fi
  sleep 2
done
if [ "$gpu_clear" -ne 1 ]; then
  echo "[abort] v4 프로세스 정리 후에도 GPU 점유가 남아 있음"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  exit 1
fi
echo "   GPU 메모리 해제 확인"

prepare_worker_paths() {
  local label=$1 run_base=$2 seeds=$3 seed run
  local quarantine="$OM_WORK/quarantine/v4"
  local smoke="$OM_WORK/runs/v4-$label-smoke-$code_tag-scluster$V4_WORKER_SLOT"
  "$PY" src/prepare_run_path.py "$smoke" \
    --expected-git "$CURRENT_GIT" --quarantine-root "$quarantine" || return 1
  for seed in $seeds; do
    for run in "$run_base-s$seed" "$run_base-s$seed-math500"; do
      "$PY" src/prepare_run_path.py "$run" \
        --expected-git "$CURRENT_GIT" --quarantine-root "$quarantine" || return 1
    done
  done
}

collect_targets() {
  local run_base=$1 expected_model_hash=$2 seed run artifact
  targets=()
  for seed in $EXPECTED_V4_SEEDS; do
    for run in "$run_base-s$seed" "$run_base-s$seed-math500"; do
      for artifact in DONE run_config.json manifest.json score_protocol.json oracle_protocol.json report.json; do
        [ -s "$run/$artifact" ] || {
          echo "[abort] incomplete v4 run: $run ($artifact missing or empty)"
          return 1
        }
      done
      targets+=("$run")
    done
  done
  "$PY" - "$expected_model_hash" "${targets[@]}" <<'PYEOF'
import json
import sys
from pathlib import Path

expected_model_hash = sys.argv[1]
runs = [Path(raw) for raw in sys.argv[2:]]
configs = []
for run in runs:
    path = run / "run_config.json"
    if not path.is_file():
        raise SystemExit(f"[abort] run config missing: {path}")
    config = json.loads(path.read_text())
    expected_seed = int(run.name.rsplit("-s", 1)[1].split("-", 1)[0])
    expected_dataset = "math500" if run.name.endswith("-math500") else "gsm8k"
    expected_n_train = 400 if expected_dataset == "math500" else 512
    if config.get("seed") != expected_seed or config.get("dataset") != expected_dataset:
        raise SystemExit(f"[abort] run path/config mismatch: {run}")
    if config.get("n_train") != expected_n_train or config.get("n_val") != 100:
        raise SystemExit(f"[abort] unexpected sample size in {run}: n_train={config.get('n_train')} n_val={config.get('n_val')}")
    if config.get("model_config_sha256") != expected_model_hash:
        raise SystemExit(f"[abort] unexpected model snapshot in {run}")
    configs.append(config)

same_keys = (
    "git", "git_diff_sha256", "git_status", "model_config_sha256",
    "tokenizer_config_sha256", "generation_config_sha256", "behavior_k",
    "fresh_k", "val_k", "micro_group", "hybrid_prompts", "k_cell",
    "drift", "max_new_tokens", "proj_dim", "grad_layers", "clip_cap",
    "temperature", "topk_frac", "radius_mode", "top_p", "thinking",
    "attn", "lora_targets", "skip_hybrid", "linear_attention_backend",
    "fla_core_version",
)
reference = configs[0]
for key in same_keys:
    values = {json.dumps(config.get(key), sort_keys=True) for config in configs}
    if len(values) != 1:
        raise SystemExit(f"[abort] mixed v4 provenance/config for {key}: {sorted(values)}")
if reference.get("git_status"):
    raise SystemExit("[abort] v4 run was initialized from a dirty src/scripts tree")
print(f"[v4 provenance] {len(runs)} runs, commit={reference['git'][:12]}, model={reference['model_config_sha256'][:12]}")
PYEOF
}

matrix_complete() {
  local run_base seed run artifact
  for run_base in "$RUN_BASE_27B" "$RUN_BASE_7B"; do
    for seed in $EXPECTED_V4_SEEDS; do
      for run in "$run_base-s$seed" "$run_base-s$seed-math500"; do
        for artifact in DONE run_config.json manifest.json score_protocol.json oracle_protocol.json report.json; do
          [ -s "$run/$artifact" ] || return 1
        done
      done
    done
  done
}

finalize_model() {
  local run_base=$1 results_base=$2 expected_model_hash=$3
  local run tag file base
  collect_targets "$run_base" "$expected_model_hash" || return 1
  mkdir -p "$results_base" || return 1
  for run in "${targets[@]}"; do
    tag=$(basename "$run")
    cp "$run/report.json" "$results_base/report-$tag.json" || return 1
    cp "$run/manifest.json" "$results_base/manifest-$tag.json" || return 1
    for file in "$run"/divergence_stats*.json; do
      [ -f "$file" ] || {
        echo "[abort] divergence stats missing: $run"
        return 1
      }
      base=$(basename "$file" .json)
      cp "$file" "$results_base/$base-$tag.json" || return 1
    done
    "$PY" src/judge.py "$run" \
      > "$results_base/judge-$tag.txt" 2>&1 || return 1
  done
  OM_RESULTS="$results_base" bash scripts/tables.sh "${targets[@]}" || return 1
  OM_RESULTS="$results_base" bash scripts/frontier.sh "${targets[@]}" || return 1
}

finalize_v4() {
  finalize_model "$RUN_BASE_27B" "$RESULTS_BASE_27B" "$MODEL_HASH_27B" || return 1
  finalize_model "$RUN_BASE_7B" "$RESULTS_BASE_7B" "$MODEL_HASH_7B" || return 1
  bash scripts/harvest.sh || return 1
  echo "== v4 aggregate complete: $RESULTS_BASE_27B, $RESULTS_BASE_7B"
}

finalize_v4_once() {
  local marker="$V4_STATUS_ROOT/V4_COMPLETE"
  mkdir -p "$V4_STATUS_ROOT" || return 1
  command -v flock >/dev/null 2>&1 || return 1
  exec 9>"$OM_WORK/runs/v4-finalize.lock"
  flock 9 || return 1
  if [ -s "$marker" ] \
     && [ -s "$RESULTS_BASE_27B/TABLES.md" ] && [ -s "$RESULTS_BASE_27B/FRONTIER.md" ] \
     && [ -s "$RESULTS_BASE_7B/TABLES.md" ] && [ -s "$RESULTS_BASE_7B/FRONTIER.md" ]; then
    echo "== v4 aggregate already complete"
    return 0
  fi
  finalize_v4 || return 1
  printf 'completed %s\n' "$(date -Is)" > "$marker.tmp"
  mv "$marker.tmp" "$marker"
}

worker_tag="cluster$V4_WORKER_SLOT"

echo "== v4 confirmatory rerun"
echo "   commit=$(git rev-parse HEAD)"
echo "   27B=$MODEL_27B -> $RUN_BASE_27B-s*"
echo "   7B=$MODEL_7B -> $RUN_BASE_7B-s*"
echo "   cluster=$V4_WORKER_SLOT, GPUs=0,1,2,3"
echo "   27B seeds=[$SEEDS_27B], 7B seeds=[$SEEDS_7B]"

run_model_worker() {
  local label=$1 model=$2 run_base=$3 results_base=$4 seeds=$5
  prepare_worker_paths "$label" "$run_base" "$seeds" || return 1
  (
    export MODEL_14B="$model" RUN_BASE="$run_base" RESULTS_BASE="$results_base"
    export RUN_LABEL="v4-$label-worker-s${worker_tag:-unknown}"
    # A smoke run is code-specific. Reusing one after git pull must not trip the
    # immutable run_config lock or mix artifacts from two implementations.
    export RUN_BASE_SMOKE="$OM_WORK/runs/v4-$label-smoke-$code_tag-s${worker_tag:-unknown}"
    export OM_SKIP_POSTPROCESS=1 OM_GPUS=0,1,2,3 OM_MAX_RETRIES=5
    if [ "$label" = "27b" ]; then
      # Four concurrent 27B snapshot loads can legitimately be silent for >5 min.
      export OM_LORA_TARGETS=all-linear OM_GEN_BATCH=8 OM_SKIP_HYBRID=1
      export OM_STALL_MINUTES=20
    else
      unset OM_LORA_TARGETS OM_GEN_BATCH
      export OM_SKIP_HYBRID=0 OM_STALL_MINUTES=5
    fi
    SEEDS="$seeds" DATASETS="gsm8k" N_TRAIN=512 N_VAL=100 \
      bash scripts/go_v2.sh || exit 1
    SEEDS="$seeds" DATASETS="math500" N_TRAIN=400 N_VAL=100 \
      bash scripts/go_v2.sh || exit 1
  )
}

# Current main model first; 7B follows as the same-condition replication axis.
run_model_worker 27b "$MODEL_27B" "$RUN_BASE_27B" "$RESULTS_BASE_27B" "$SEEDS_27B" || exit 1
run_model_worker 7b "$MODEL_7B" "$RUN_BASE_7B" "$RESULTS_BASE_7B" "$SEEDS_7B" || exit 1

# Verify only this worker's outputs. Global aggregation waits for all expected seeds.
verify_worker_outputs() {
  local run_base=$1 seeds=$2 seed run artifact
  for seed in $seeds; do
    for run in "$run_base-s$seed" "$run_base-s$seed-math500"; do
      for artifact in DONE run_config.json manifest.json score_protocol.json oracle_protocol.json report.json; do
        [ -s "$run/$artifact" ] || {
          echo "[abort] incomplete v4 run: $run ($artifact missing or empty)"
          exit 1
        }
      done
    done
  done
}
verify_worker_outputs "$RUN_BASE_27B" "$SEEDS_27B"
verify_worker_outputs "$RUN_BASE_7B" "$SEEDS_7B"

if matrix_complete; then
  finalize_v4_once || exit 1
else
  echo "== v4 worker complete: 27B seeds=[$SEEDS_27B], 7B seeds=[$SEEDS_7B]"
  echo "   cluster $V4_WORKER_SLOT 완료. 다른 클러스터 결과와 합친 뒤 최종 집계합니다."
fi
