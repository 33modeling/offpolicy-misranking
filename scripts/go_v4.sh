#!/usr/bin/env bash
# v4 confirmatory rerun after the 2026-08-20 generation and validation-integrity fixes.
#
# This runs the corrected matrix without mixing models or historical artifacts:
#   - Qwen3.8-27B-BF16 seeds 0..4: current main-model confirmation
#   - Qwen2.5-7B-Instruct seeds 0..4: same-condition historical replication
#   - GSM8K and MATH500 for both; saturated 27B DAPO is not a valid test pool
#
# Cloud usage, after the currently running jobs finish and the shared checkout is updated:
#   bash scripts/go_v4.sh
# Run the same command on three shared-storage nodes. Slots are claimed automatically:
#   slot 0 -> seeds 0,1; slot 1 -> seeds 2,3; slot 2 -> seed 4.
# The last completed worker aggregates automatically.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || PY=python3

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

command -v flock >/dev/null 2>&1 || {
  echo "[abort] flock is required for shared-cloud v4 workers"
  exit 1
}

if [ -n "${SEEDS_V4:-}" ]; then
  V4_WORKER_SLOT=manual
else
  claim_root="$OM_WORK/runs/v4-worker-claims"
  mkdir -p "$claim_root" || exit 1
  V4_WORKER_SLOT=
  for slot in 0 1 2; do
    exec {claim_fd}>"$claim_root/slot-$slot.lock"
    if flock -n "$claim_fd"; then
      V4_WORKER_SLOT=$slot
      V4_CLAIM_FD=$claim_fd
      break
    fi
    exec {claim_fd}>&-
  done
  [ -n "$V4_WORKER_SLOT" ] || {
    echo "[abort] v4 worker slots 0,1,2 are already occupied"
    exit 1
  }
  case "$V4_WORKER_SLOT" in
    0) SEEDS_V4="0 1" ;;
    1) SEEDS_V4="2 3" ;;
    2) SEEDS_V4="4" ;;
  esac
fi

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
    "attn", "lora_targets", "skip_hybrid",
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
  command -v flock >/dev/null 2>&1 || {
    echo "[abort] flock is required for shared-cloud v4 finalization"
    return 1
  }
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

worker_tag=$(printf '%s' "$SEEDS_V4" | tr -cs '0-9' '-' | sed 's/^-//; s/-$//')

echo "== v4 confirmatory rerun"
echo "   commit=$(git rev-parse HEAD)"
echo "   27B=$MODEL_27B -> $RUN_BASE_27B-s*"
echo "   7B=$MODEL_7B -> $RUN_BASE_7B-s*"
echo "   worker_slot=$V4_WORKER_SLOT, GPUs=0,1,2,3, seeds=[$SEEDS_V4]"

run_model_worker() {
  local label=$1 model=$2 run_base=$3 results_base=$4
  (
    export MODEL_14B="$model" RUN_BASE="$run_base" RESULTS_BASE="$results_base"
    export RUN_LABEL="v4-$label-worker-s${worker_tag:-unknown}"
    export RUN_BASE_SMOKE="$OM_WORK/runs/v4-$label-smoke-s${worker_tag:-unknown}"
    export OM_SKIP_POSTPROCESS=1 OM_GPUS=0,1,2,3
    if [ "$label" = "27b" ]; then
      export OM_LORA_TARGETS=all-linear OM_GEN_BATCH=8 OM_SKIP_HYBRID=1
    else
      unset OM_LORA_TARGETS OM_GEN_BATCH
      export OM_SKIP_HYBRID=0
    fi
    SEEDS="$SEEDS_V4" DATASETS="gsm8k" N_TRAIN=512 N_VAL=100 \
      bash scripts/go_v2.sh || exit 1
    SEEDS="$SEEDS_V4" DATASETS="math500" N_TRAIN=400 N_VAL=100 \
      bash scripts/go_v2.sh || exit 1
  )
}

# Current main model first; 7B follows as the same-condition replication axis.
run_model_worker 27b "$MODEL_27B" "$RUN_BASE_27B" "$RESULTS_BASE_27B" || exit 1
run_model_worker 7b "$MODEL_7B" "$RUN_BASE_7B" "$RESULTS_BASE_7B" || exit 1

# Verify only this worker's outputs. Global aggregation waits for all expected seeds.
for run_base in "$RUN_BASE_27B" "$RUN_BASE_7B"; do
  for seed in $SEEDS_V4; do
    for run in "$run_base-s$seed" "$run_base-s$seed-math500"; do
      for artifact in DONE run_config.json manifest.json score_protocol.json oracle_protocol.json report.json; do
        [ -s "$run/$artifact" ] || {
          echo "[abort] incomplete v4 run: $run ($artifact missing or empty)"
          exit 1
        }
      done
    done
  done
done

if matrix_complete; then
  finalize_v4_once || exit 1
else
  echo "== v4 worker complete: seeds=[$SEEDS_V4]"
  echo "   남은 seed worker가 끝나면 마지막 worker가 자동 집계합니다."
fi
