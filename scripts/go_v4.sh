#!/usr/bin/env bash
# v4 confirmatory rerun after the 2026-08-20 generation and validation-integrity fixes.
#
# This reruns the historical significance conditions without mixing artifacts:
#   - GSM8K seeds 0..4: same-dataset check of the old v2 significance
#   - MATH500 seeds 0..4: replication of the seed-dependent cell ordering
#
# Cloud usage, after the currently running jobs finish and the shared checkout is updated:
#   bash scripts/go_v4.sh
# Overrides:
#   SEEDS_V4="0" bash scripts/go_v4.sh
# Run one command per cloud worker, then aggregate once from any checkout:
#   OM_V4_FINALIZE_ONLY=1 bash scripts/go_v4.sh
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

export RUN_BASE="$OM_WORK/runs/v4"
export RESULTS_BASE="$OM_WORK/results/v4"
EXPECTED_V4_SEEDS="${EXPECTED_V4_SEEDS:-0 1 2 3 4}"
SEEDS_V4="${SEEDS_V4:-$EXPECTED_V4_SEEDS}"

collect_targets() {
  local seed run artifact
  targets=()
  for seed in $EXPECTED_V4_SEEDS; do
    for run in "$RUN_BASE-s$seed" "$RUN_BASE-s$seed-math500"; do
      for artifact in DONE run_config.json manifest.json score_protocol.json oracle_protocol.json report.json; do
        [ -s "$run/$artifact" ] || {
          echo "[abort] incomplete v4 run: $run ($artifact missing or empty)"
          return 1
        }
      done
      targets+=("$run")
    done
  done
  "$PY" - "${targets[@]}" <<'PYEOF'
import json
import sys
from pathlib import Path

runs = [Path(raw) for raw in sys.argv[1:]]
configs = []
for run in runs:
    path = run / "run_config.json"
    if not path.is_file():
        raise SystemExit(f"[abort] run config missing: {path}")
    config = json.loads(path.read_text())
    expected_seed = int(run.name.split("-s", 1)[1].split("-", 1)[0])
    expected_dataset = "math500" if run.name.endswith("-math500") else "gsm8k"
    expected_n_train = 400 if expected_dataset == "math500" else 512
    if config.get("seed") != expected_seed or config.get("dataset") != expected_dataset:
        raise SystemExit(f"[abort] run path/config mismatch: {run}")
    if config.get("n_train") != expected_n_train or config.get("n_val") != 100:
        raise SystemExit(f"[abort] unexpected sample size in {run}: n_train={config.get('n_train')} n_val={config.get('n_val')}")
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

finalize_v4() {
  local run tag file base
  collect_targets || return 1
  mkdir -p "$RESULTS_BASE" || return 1
  for run in "${targets[@]}"; do
    tag=$(basename "$run")
    cp "$run/report.json" "$RESULTS_BASE/report-$tag.json" || return 1
    cp "$run/manifest.json" "$RESULTS_BASE/manifest-$tag.json" || return 1
    for file in "$run"/divergence_stats*.json; do
      [ -f "$file" ] || {
        echo "[abort] divergence stats missing: $run"
        return 1
      }
      base=$(basename "$file" .json)
      cp "$file" "$RESULTS_BASE/$base-$tag.json" || return 1
    done
    "$PY" src/judge.py "$run" \
      > "$RESULTS_BASE/judge-$tag.txt" 2>&1 || return 1
  done
  OM_RESULTS="$RESULTS_BASE" bash scripts/tables.sh "${targets[@]}" || return 1
  OM_RESULTS="$RESULTS_BASE" bash scripts/frontier.sh "${targets[@]}" || return 1
  echo "== v4 aggregate complete: $RESULTS_BASE"
}

if [ "${OM_V4_FINALIZE_ONLY:-0}" = "1" ]; then
  finalize_v4
  exit $?
fi

worker_tag=$(printf '%s' "$SEEDS_V4" | tr -cs '0-9' '-' | sed 's/^-//; s/-$//')
export RUN_LABEL="v4-worker-s${worker_tag:-unknown}"
# Shared cloud workers must not race on the same smoke directory.
export RUN_BASE_SMOKE="${RUN_BASE_SMOKE:-$OM_WORK/runs/v4-smoke-s${worker_tag:-unknown}}"

echo "== v4 confirmatory rerun"
echo "   commit=$(git rev-parse HEAD)"
echo "   runs=$RUN_BASE"
echo "   seeds=[$SEEDS_V4]"

# Keep each dataset at its historical sample size so the corrected result is
# directly comparable to the exploratory snapshot.
OM_SKIP_POSTPROCESS=1 SEEDS="$SEEDS_V4" DATASETS="gsm8k" N_TRAIN=512 N_VAL=100 \
  bash scripts/go_v2.sh || exit 1
OM_SKIP_POSTPROCESS=1 SEEDS="$SEEDS_V4" DATASETS="math500" N_TRAIN=400 N_VAL=100 \
  bash scripts/go_v2.sh || exit 1

# Verify only this worker's outputs. Global aggregation waits for all expected seeds.
for seed in $SEEDS_V4; do
  for run in "$RUN_BASE-s$seed" "$RUN_BASE-s$seed-math500"; do
    for artifact in DONE run_config.json manifest.json score_protocol.json oracle_protocol.json report.json; do
      [ -s "$run/$artifact" ] || {
        echo "[abort] incomplete v4 run: $run ($artifact missing or empty)"
        exit 1
      }
    done
  done
done

if [ "$SEEDS_V4" = "$EXPECTED_V4_SEEDS" ]; then
  finalize_v4 || exit 1
else
  echo "== v4 worker complete: seeds=[$SEEDS_V4]"
  echo "   모든 worker 완료 후: OM_V4_FINALIZE_ONLY=1 bash scripts/go_v4.sh"
fi
