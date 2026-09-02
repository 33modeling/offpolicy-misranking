#!/usr/bin/env bash
# Resume one failed rollout stage with a smaller runtime generation batch.
set -uo pipefail

[ "$#" -eq 5 ] || {
  echo "usage: recover_rollout_stage.sh PIPELINE_REPO RUN STAGE BATCH GPU_CSV" >&2
  exit 2
}
PIPELINE_REPO=$1
RUN=$2
STAGE=$3
BATCH=$4
GPU_CSV=$5
PY=${OM_RECOVERY_PY:?OM_RECOVERY_PY is required}
INDEX=${OM_RECOVERY_INDEX:-1}

case "$STAGE" in
  rollout-behavior|rollout-fresh) ;;
  *) echo "[recovery-abort] unsupported stage=$STAGE" >&2; exit 2 ;;
esac
case "$BATCH" in
  ''|*[!0-9]*|0) echo "[recovery-abort] invalid batch=$BATCH" >&2; exit 2 ;;
esac
[ -x "$PY" ] || { echo "[recovery-abort] Python missing: $PY" >&2; exit 1; }
[ -s "$RUN/run_config.json" ] || {
  echo "[recovery-abort] run config missing: $RUN/run_config.json" >&2
  exit 1
}
[ -s "$PIPELINE_REPO/src/experiment.py" ] || {
  echo "[recovery-abort] pinned experiment code missing: $PIPELINE_REPO" >&2
  exit 1
}

assignments=$("$PY" - "$RUN/run_config.json" "$PIPELINE_REPO" <<'PYEOF'
import json
import shlex
import subprocess
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
repository = Path(sys.argv[2])
config = json.loads(config_path.read_text(encoding="utf-8"))
head = subprocess.check_output(
    ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
).strip()
if config.get("git") != head:
    raise SystemExit(
        f"[recovery-abort] run uses generation git {config.get('git')}, checkout is {head}"
    )

fields = {
    "MODEL_PATH": config["model_resolved"],
    "DATASET": config["dataset"],
    "DRIFT": config["drift"],
    "BEHAVIOR_K": config["behavior_k"],
    "FRESH_K": config["fresh_k"],
    "VAL_K": config["val_k"],
    "MICRO_GROUP": config["micro_group"],
    "HYBRID_PROMPTS": config.get("hybrid_prompts", 24),
    "MICRO_BATCH": config.get("gradient_micro_batch", 1),
    "N_TRAIN": config["n_train"],
    "N_VAL": config["n_val"],
    "SEED": config["seed"],
    "MAX_NEW_TOKENS": config["max_new_tokens"],
    "PROJ_DIM": config["proj_dim"],
    "GRAD_LAYERS": config["grad_layers"],
    "CLIP_CAP": config["clip_cap"],
    "TEMPERATURE": config["temperature"],
    "TOPK_FRAC": config["topk_frac"],
    "RADIUS_MODE": config.get("radius_mode", "gaussian"),
    "K_CELL": config.get("k_cell", 8),
    "OM_TOP_P": config.get("top_p", 1.0),
    "OM_THINKING": config.get("thinking", "off"),
    "OM_PROMPT_FORMAT": config.get("prompt_format", "tokenizer_chat"),
    "OM_ATTN": config.get("attn", "eager"),
    "CONFIGURED_GEN_BATCH": config.get("gen_batch"),
}
for key, value in fields.items():
    print(f"{key}={shlex.quote(str(value))}")
PYEOF
) || exit 1
eval "$assignments" || exit 1

if [ "$STAGE" = rollout-behavior ] && [ "$DRIFT" -ne 0 ]; then
  echo "[recovery-abort] behavior generation is only valid for the d0 source" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
NGPU=${#GPUS[@]}
[ "$NGPU" -gt 0 ] || { echo "[recovery-abort] empty GPU list" >&2; exit 2; }
for gpu in "${GPUS[@]}"; do
  case "$gpu" in
    ''|*[!0-9]*) echo "[recovery-abort] invalid GPU id=$gpu" >&2; exit 2 ;;
  esac
done

case "$STAGE" in
  rollout-behavior) RECOVERY_BASE=rollouts_behavior_train; RECOVERY_K=$BEHAVIOR_K ;;
  rollout-fresh) RECOVERY_BASE=rollouts_fresh_train; RECOVERY_K=$FRESH_K ;;
esac
record_recovery() {
  "$PY" - "$RUN" "$1" "$STAGE" "$RECOVERY_BASE" "$RECOVERY_K" \
    "$CONFIGURED_GEN_BATCH" "$BATCH" "$INDEX" "$GPU_CSV" <<'PYEOF'
import collections
import datetime
import json
import os
import sys
from pathlib import Path

run = Path(sys.argv[1])
status, stage, base = sys.argv[2:5]
k = int(sys.argv[5])
configured_batch, recovery_batch = sys.argv[6:8]
attempt = int(sys.argv[8])
gpus = sys.argv[9].split(",")
coverage = {}
for shard in range(len(gpus)):
    published = run / f"{base}.shard{shard}.jsonl"
    partial = run / f"{base}.shard{shard}.partial"
    path = published if published.is_file() else partial
    rows = collections.defaultdict(set)
    row_count = 0
    if path.is_file():
        for line in path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
                rows[int(row["prompt_idx"])].add(int(row["rollout_idx"]))
                row_count += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    coverage[str(shard)] = {
        "artifact": path.name if path.is_file() else None,
        "rows": row_count,
        "complete_prompt_ids": sorted(
            prompt for prompt, indices in rows.items() if indices == set(range(k))
        ),
    }
record = {
    "schema": "offpolicy-rollout-runtime-recovery/v1",
    "recorded_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "pid": os.getpid(),
    "status": status,
    "stage": stage,
    "attempt": attempt,
    "configured_generation_batch": configured_batch,
    "recovery_generation_batch": int(recovery_batch),
    "gpu_order": gpus,
    "coverage": coverage,
}
path = run / "rollout_recovery.jsonl"
with path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
PYEOF
}
record_recovery started || exit 1
record_recovery_on_exit() {
  local rc=$?
  trap - EXIT
  if [ "$rc" -eq 0 ]; then
    record_recovery completed || true
  else
    record_recovery failed || true
  fi
  exit "$rc"
}
trap record_recovery_on_exit EXIT

COMMON=(--run "$RUN" --model "$MODEL_PATH" --dataset "$DATASET"
  --behavior-k "$BEHAVIOR_K" --fresh-k "$FRESH_K" --val-k "$VAL_K"
  --micro-group "$MICRO_GROUP" --hybrid-prompts "$HYBRID_PROMPTS"
  --micro-batch "$MICRO_BATCH" --n-train "$N_TRAIN" --n-val "$N_VAL"
  --seed "$SEED" --max-new-tokens "$MAX_NEW_TOKENS" --proj-dim "$PROJ_DIM"
  --grad-layers "$GRAD_LAYERS" --clip-cap "$CLIP_CAP"
  --temperature "$TEMPERATURE" --topk-frac "$TOPK_FRAC"
  --radius-mode "$RADIUS_MODE" --k-cell "$K_CELL")
POLICY_ARGS=()
if [ "$STAGE" = rollout-fresh ] && [ "$DRIFT" -gt 0 ]; then
  ADAPTER="$RUN/policy_step_$DRIFT"
  [ -s "$ADAPTER/adapter_model.safetensors" ] || {
    echo "[recovery-abort] completed adapter missing: $ADAPTER" >&2
    exit 1
  }
  POLICY_ARGS=(--adapter "$ADAPTER")
fi

mkdir -p "$RUN/logs"
cd "$PIPELINE_REPO" || exit 1
pids=()
logs=()
for ((shard = 0; shard < NGPU; shard++)); do
  gpu=${GPUS[$(( (shard + INDEX - 1) % NGPU ))]}
  log="$RUN/logs/recovery-${STAGE}-attempt${INDEX}-shard${shard}.log"
  logs+=("$log")
  echo "[cuda-recovery] shard=$shard/$NGPU gpu=$gpu batch=$BATCH log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" OM_GEN_BATCH="$BATCH" OM_TOP_P="$OM_TOP_P" \
    OM_THINKING="$OM_THINKING" OM_PROMPT_FORMAT="$OM_PROMPT_FORMAT" \
    OM_ATTN="$OM_ATTN" PYTHONPATH="$PIPELINE_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" src/experiment.py --stage "$STAGE" "${COMMON[@]}" \
      "${POLICY_ARGS[@]}" --shard "$shard:$NGPU" > "$log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
  echo "[cuda-recovery] one or more shards failed" >&2
  for log in "${logs[@]}"; do
    echo "--- $log" >&2
    tail -30 "$log" >&2 || true
  done
  exit 1
fi
echo "[cuda-recovery] $STAGE shards complete; normal pipeline will validate and merge"
