#!/usr/bin/env bash
# New on-policy reference at a larger rollout budget (d=0, no training), to
# measure split-half reliability against the reference budget with new,
# source-disjoint rollout groups.
#
#   bash scripts/run_reliability_budget.sh math500             # 64 responses per candidate prompt, 32 per validation prompt
#   bash scripts/run_reliability_budget.sh mbpp 128 32         # <dataset> [fresh_k=64] [val_k=32] [seed=100]
#   RB_DRY=1 bash scripts/run_reliability_budget.sh math500    # print the plan, run nothing
#
# Runs on one node with the visible GPUs (4xH100 expected; CUDA_VISIBLE_DEVICES
# is honoured). It holds the node-local primary.lock that the registered
# launcher and the extension launchers hold (refuses a busy node), and one
# lease per run directory (refuses a second launcher for the same run on
# another node). TERM/INT stop this invocation's stage children. Writes only under
#   $OM_WORK/runs/reliability-budget-v1/<dataset>-fk<K>-vk<VK>-s<seed>/
# and $OM_WORK/exports. The registered matrix root is read once (prompts.json
# and run_config.json of a finished d0 point) and never written. Running the
# same command again resumes: finished shards and stages are skipped.
#
# Stages (the registered code path, src/experiment.py, in a new run directory):
#   1 rollout-fresh  K responses per candidate prompt and VK per validation prompt, new RNG domain
#   2 val-grads      one gradient per validation prompt                (last GPU)
#   3 oracle-grads   micro-group gradients, sharded over the other GPUs
#   4 merge-grads    oracle_micro_groups.pt
#   5 analysis       src/reliability_budget.py on the new run and on the registered d0 point
#                    -> $OM_WORK/exports/reliability-budget-run-<dataset>-...txt (KEY lines printed)
#
# Plan section 7 forbids building a reliability-versus-budget curve from the
# locked rollouts; this run generates new groups instead, which the reliability
# audit allows ("independently generated groups at multiple budgets"). It
# defines no registered label and touches no registered point.
set -uo pipefail
trap '' HUP
cd "$(dirname "$0")/.."
DATASET=${1:-}
FRESH_K=${2:-64}
VAL_K=${3:-32}
SEED=${4:-100}
usage() { echo "usage: bash scripts/run_reliability_budget.sh <math500|mbpp> [fresh_k=64] [val_k=32] [seed=100]"; }
case "$DATASET" in math500|mbpp) ;; *) usage; exit 2 ;; esac
for v in FRESH_K VAL_K SEED; do
  case "${!v}" in ''|*[!0-9]*) echo "[abort] $v must be a non-negative integer"; usage; exit 2 ;; esac
done
MICRO_GROUP=4
if [ $((FRESH_K % MICRO_GROUP)) -ne 0 ] || [ $((FRESH_K / MICRO_GROUP)) -lt 8 ] || [ $(((FRESH_K / MICRO_GROUP) % 4)) -ne 0 ]; then
  echo "[abort] fresh_k must be a multiple of 16 and at least 32 (micro-groups of $MICRO_GROUP, R/A/B partition)"; exit 2
fi
[ "$VAL_K" -ge 2 ] || { echo "[abort] val_k must be at least 2"; exit 2; }
DRY=${RB_DRY:-0}

export OM_ONLINE=0
source scripts/setup_env.sh
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv missing: $PY"; exit 1; }
CONFIG="${OM_RLZERO_CONFIG:-$PWD/configs/olmo3_rlzero_h100.json}"
[ -s "$CONFIG" ] || { echo "[abort] experiment config missing: $CONFIG"; exit 1; }
TAG="${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}"
SOURCE_ROOT="${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}"
field() { "$PY" src/model_matrix.py --config "$CONFIG" experiment-field "$1"; }
N_VAL=$(field n_val) || exit 1
MAX_NEW_TOKENS=$(field max_new_tokens) || exit 1
PROJ_DIM=$(field proj_dim) || exit 1
GRAD_LAYERS=$(field grad_layers) || exit 1
CLIP_CAP=$(field clip_cap) || exit 1
TOPK_FRAC=$(field topk_frac) || exit 1
ATTN=$(field attn) || exit 1
MODEL_PATH="${OM_OLMO3_MODEL_PATH:-$("$PY" src/model_matrix.py --config "$CONFIG" --models-dir "$MODELS_DIR" field olmo3-7b-base path 2>/dev/null)}"
if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
  if [ "$DRY" = 1 ]; then echo "[warn] OLMo-3 model path not found: '$MODEL_PATH'"; else echo "[abort] OLMo-3 model path not found: '$MODEL_PATH'"; exit 1; fi
fi
case "$DATASET" in
  math500) N_TRAIN=400; FORMAT=olmo_rlzero_math; GEN_BATCH=${OM_GEN_BATCH:-32}; GRAD_MICRO_BATCH=${GRADIENT_MICRO_BATCH:-2} ;;
  mbpp)    N_TRAIN=512; FORMAT=olmo_rlzero_code; GEN_BATCH=${OM_GEN_BATCH:-16}; GRAD_MICRO_BATCH=${GRADIENT_MICRO_BATCH:-1} ;;
esac
export OM_PROMPT_FORMAT=$FORMAT OM_GEN_BATCH=$GEN_BATCH OM_ATTN=$ATTN OM_TOP_P=1.0 OM_THINKING=off OM_MATH_VERIFIER=math_verify
export RB_DATASET=$DATASET RB_N_TRAIN=$N_TRAIN RB_N_VAL=$N_VAL RB_FRESH_K=$FRESH_K RB_VAL_K=$VAL_K \
       RB_MICRO_GROUP=$MICRO_GROUP RB_SEED=$SEED RB_MAX_NEW_TOKENS=$MAX_NEW_TOKENS RB_PROJ_DIM=$PROJ_DIM \
       RB_GRAD_LAYERS=$GRAD_LAYERS RB_CLIP_CAP=$CLIP_CAP RB_TOPK_FRAC=$TOPK_FRAC

# Source point of the registered matrix: prompts (identical across seeds) and the run config template.
SRC_POINT=""
for seed in 0 1 2 3 4; do
  candidate="$SOURCE_ROOT/family-$DATASET-s$seed/$TAG-s$seed-$DATASET-d0"
  if [ -s "$candidate/prompts.json" ] && [ -s "$candidate/run_config.json" ]; then SRC_POINT=$candidate; break; fi
done
[ -n "$SRC_POINT" ] || { echo "[abort] no registered d0 point with prompts.json under $SOURCE_ROOT for $DATASET"; exit 1; }

RUN="${RB_RUNS_ROOT:-$OM_WORK/runs/reliability-budget-v1}/$DATASET-fk$FRESH_K-vk$VAL_K-s$SEED"
LOGS="$RUN/logs"
EXPORTS="${RB_EXPORTS_ROOT:-$OM_WORK/exports}"
source_abs=$(realpath -m "$SOURCE_ROOT")
for destination in "$RUN" "$EXPORTS"; do
  destination_abs=$(realpath -m "$destination")
  case "$destination_abs/" in
    "$source_abs/"*) echo "[abort] reference output cannot overwrite primary inputs: $destination"; exit 1 ;;
  esac
done
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$EXPORTS/reliability-budget-run-$DATASET-fk$FRESH_K-vk$VAL_K-s$SEED-$STAMP.txt"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=, read -ra GPUS <<<"$CUDA_VISIBLE_DEVICES"
else
  mapfile -t GPUS < <(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
fi
NGPU=${#GPUS[@]}
[ "$NGPU" -ge 1 ] || { if [ "$DRY" = 1 ]; then echo "[warn] no GPU visible"; NGPU=0; else echo "[abort] no GPU visible"; exit 1; fi; }
# Rough wall-clock from the registered h100 points (about 128-233 s per prompt per shard for 32 responses at the 2048-token cap).
est_hours=$(( (N_TRAIN * FRESH_K / 32 * 180 + N_VAL * VAL_K / 32 * 180) / 3600 / (NGPU > 0 ? NGPU : 1) ))

echo "[rb] dataset=$DATASET fresh_k=$FRESH_K val_k=$VAL_K seed=$SEED  gpus=${GPUS[*]:-none}  model=$MODEL_PATH"
echo "[rb] run=$RUN"
echo "[rb] source point=$SRC_POINT"
echo "[rb] contract: n_train=$N_TRAIN n_val=$N_VAL micro_group=$MICRO_GROUP max_new_tokens=$MAX_NEW_TOKENS proj_dim=$PROJ_DIM grad_layers=$GRAD_LAYERS attn=$ATTN gen_batch=$GEN_BATCH grad_micro_batch=$GRAD_MICRO_BATCH prompt_format=$FORMAT"
echo "[rb] rough generation time on $NGPU GPU(s): about ${est_hours}h (2048-token responses; shorter if responses stop early)"
if [ "$DRY" = 1 ]; then echo "[rb] dry run: nothing started"; exit 0; fi

# Node admission: the same node-local lock the registered launcher and the
# extension launchers hold, so this run never starts on top of another one.
LOCAL_LOCK_DIR="${OM_LOCAL_LOCK_DIR:-/tmp/offpolicy-misranking-$(id -u)}"
mkdir -p "$LOCAL_LOCK_DIR" || { echo "[abort] cannot create $LOCAL_LOCK_DIR"; exit 1; }
exec 8>"$LOCAL_LOCK_DIR/primary.lock"
flock -n 8 || { echo "[abort] another experiment (registered launcher or extension) owns this node's GPUs; use an idle node"; exit 1; }
mkdir -p "$RUN" "$LOGS" "$EXPORTS" || { echo "[abort] cannot create $RUN"; exit 1; }
# Run lease: one launcher per run directory, across nodes.
exec 9>"$RUN/.launcher.lock"
flock -n 9 || { echo "[abort] another launcher already owns $RUN (same dataset, budget and seed on another node?)"; exit 1; }
MAIN_LOG="$LOGS/main.log"
log() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$MAIN_LOG"; }
# Only this invocation's stage subshells and their Python children are stopped.
CHILDREN=()
stop_children() {
  local pid
  for pid in "${CHILDREN[@]}"; do
    pkill -TERM -P "$pid" 2>/dev/null
    kill -TERM "$pid" 2>/dev/null
  done
  for pid in "${CHILDREN[@]}"; do wait "$pid" 2>/dev/null; done
  CHILDREN=()
}
on_signal() { trap - TERM INT; log "[abort] terminated; stopping this run's stage processes"; stop_children; exit 143; }
trap on_signal TERM INT
run_tracked() {  # run_tracked <command...> : run in the background, track it, wait (so TERM/INT still reach it)
  "$@" &
  local pid=$! rc
  CHILDREN+=("$pid")
  wait "$pid"; rc=$?
  CHILDREN=()
  return "$rc"
}

if [ "$DATASET" = math500 ]; then
  MATH_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps") || exit 1
  export PYTHONPATH="$MATH_VERIFY_PATH${PYTHONPATH:+:$PYTHONPATH}"
  "$PY" -c 'from math_verify import parse, verify; assert verify(parse(r"\frac{1}{2}"), parse("0.5"))' \
    || { log "[abort] bundled math verifier failed to import"; exit 1; }
fi

# prompts.json: copied once from the registered point; later runs must see the same file.
if [ -s "$RUN/prompts.json" ]; then
  cmp -s "$RUN/prompts.json" "$SRC_POINT/prompts.json" || { log "[abort] $RUN/prompts.json differs from $SRC_POINT/prompts.json"; exit 1; }
else
  cp "$SRC_POINT/prompts.json" "$RUN/prompts.json.tmp" && mv "$RUN/prompts.json.tmp" "$RUN/prompts.json" || exit 1
fi
# run_config.json: the registered template with the new budget, seed and drift 0.
"$PY" - "$RUN" "$SRC_POINT" "$MODEL_PATH" "$GEN_BATCH" "$GRAD_MICRO_BATCH" <<'PYEOF' || exit 1
import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
run, source = Path(sys.argv[1]), Path(sys.argv[2])
model_path, gen_batch, grad_micro_batch = sys.argv[3], sys.argv[4], int(sys.argv[5])
env = os.environ
# Every argument the stages will see: the resume check compares all of them.
request = {
    "dataset": env["RB_DATASET"], "n_train": int(env["RB_N_TRAIN"]), "n_val": int(env["RB_N_VAL"]),
    "behavior_k": 8, "fresh_k": int(env["RB_FRESH_K"]), "val_k": int(env["RB_VAL_K"]),
    "micro_group": int(env["RB_MICRO_GROUP"]), "seed": int(env["RB_SEED"]), "drift": 0,
    "max_new_tokens": int(env["RB_MAX_NEW_TOKENS"]), "proj_dim": int(env["RB_PROJ_DIM"]),
    "grad_layers": int(env["RB_GRAD_LAYERS"]), "clip_cap": float(env["RB_CLIP_CAP"]),
    "topk_frac": float(env["RB_TOPK_FRAC"]), "temperature": 1.0, "top_p": 1.0, "thinking": "off",
    "prompt_format": env["OM_PROMPT_FORMAT"], "attn": env["OM_ATTN"], "math_verifier": env["OM_MATH_VERIFIER"],
    "model": model_path, "model_resolved": model_path, "gen_batch": gen_batch,
    "gradient_micro_batch": grad_micro_batch,
    "prompts_sha256": hashlib.sha256((run / "prompts.json").read_bytes()).hexdigest(),
}
# Computation identity: the source files that generate, score and analyse.
# Documentation-only commits do not change it; a changed backend does.
COMPUTATION_FILES = [
    "src/experiment.py", "src/rollout.py", "src/rollout_contract.py", "src/grads.py", "src/data.py",
    "src/prompt_format.py", "src/code_sandbox.py", "src/artifact_contract.py", "src/compact_artifacts.py",
    "src/fresh_validation.py", "src/select_rules.py", "src/measurement_ceiling.py", "src/reliability_budget.py",
    "configs/olmo3_rlzero_h100.json",
]
digest = hashlib.sha256()
for name in COMPUTATION_FILES:
    path = Path(name)
    if path.is_file():
        digest.update(name.encode()); digest.update(b"\0"); digest.update(path.read_bytes()); digest.update(b"\0")
request["computation_fingerprint"] = digest.hexdigest()
target = run / "run_config.json"
if target.exists():
    existing = json.loads(target.read_text())
    bad = {k: (existing[k], v) for k, v in request.items() if k in existing and existing[k] != v}
    if "computation_fingerprint" in bad and env.get("RB_ACCEPT_CODE_CHANGE") == "1":
        stored_fp, new_fp = bad.pop("computation_fingerprint")
        lineage = existing.setdefault("reliability_budget", {}).setdefault("code_changes_accepted", [])
        lineage.append({"from": stored_fp, "to": new_fp, "when": datetime.now(timezone.utc).isoformat()})
        existing["computation_fingerprint"] = new_fp
        tmp = target.with_suffix(".json.tmp"); tmp.write_text(json.dumps(existing, indent=1)); tmp.replace(target)
        print("[rb] computation code changed since this run was created; accepted explicitly (RB_ACCEPT_CODE_CHANGE=1) and recorded in the lineage")
    if "computation_fingerprint" in bad:
        print("[abort] the computation code (generation, scoring or analysis sources) changed since this run "
              "directory was created; existing artifacts may not be compatible with the current code.")
        print("  Either rerun on a checkout of the recorded code, start a new run (other seed), or, if the change is "
              "known to be compatible, repeat the command with RB_ACCEPT_CODE_CHANGE=1 (recorded in the run lineage).")
        sys.exit(1)
    if bad:
        print("[abort] this run directory was created with a different effective contract; "
              "not resuming with mixed settings. Differences (stored, requested):")
        for key, (stored, wanted) in sorted(bad.items()):
            print(f"  {key}: {stored!r} -> {wanted!r}")
        print("Use a different seed or budget for a new run, or repeat the original command exactly.")
        sys.exit(1)
    # Keys a run created by an earlier launcher never recorded are execution
    # knobs or identities that do not change stored rollouts; record them now.
    missing = [k for k in request if k not in existing]
    if missing:
        existing.update({k: request[k] for k in missing})
        existing.setdefault("reliability_budget", {})["contract_completed"] = datetime.now(timezone.utc).isoformat()
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing, indent=1))
        tmp.replace(target)
        print(f"[rb] run_config.json kept; recorded previously unrecorded fields: {', '.join(missing)}")
    else:
        print("[rb] run_config.json kept (effective contract matches)")
    sys.exit(0)
config = json.loads((source / "run_config.json").read_text())
config.pop("digest", None)
config.update(request)
try:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "scripts"], text=True).strip()
except Exception:
    head, dirty = None, None
config["reliability_budget"] = {
    "schema": "offpolicy-reliability-budget/v1",
    "purpose": "new on-policy reference groups at a larger budget; no registered label",
    "source_point": str(source),
    "source_run_config_sha256": hashlib.sha256((source / "run_config.json").read_bytes()).hexdigest(),
    "git": head, "git_dirty": bool(dirty), "created": datetime.now(timezone.utc).isoformat(),
}
encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
config["digest"] = hashlib.sha256(encoded).hexdigest()
tmp = target.with_suffix(".json.tmp")
tmp.write_text(json.dumps(config, indent=1))
tmp.replace(target)
print("[rb] run_config.json written")
PYEOF

COMMON=(--run "$RUN" --model "$MODEL_PATH" --dataset "$DATASET"
        --behavior-k 8 --fresh-k "$FRESH_K" --val-k "$VAL_K" --micro-group "$MICRO_GROUP"
        --micro-batch "$GRAD_MICRO_BATCH" --n-train "$N_TRAIN" --n-val "$N_VAL" --seed "$SEED"
        --max-new-tokens "$MAX_NEW_TOKENS" --proj-dim "$PROJ_DIM" --grad-layers "$GRAD_LAYERS"
        --clip-cap "$CLIP_CAP" --temperature 1.0 --topk-frac "$TOPK_FRAC")

run_stage() {  # run_stage <gpu slot> <log file> <experiment.py args...>
  local dev="${GPUS[$1]}" lf="$2"; shift 2
  local t0=$SECONDS
  log "GPU$dev start: $1 ${*:2:1}"
  if CUDA_VISIBLE_DEVICES="$dev" "$PY" src/experiment.py "$@" >> "$lf" 2>&1; then
    log "GPU$dev done: $1 ($((SECONDS - t0))s)"
  else
    local rc=$?
    log "GPU$dev FAILED: $* rc=$rc (see $lf)"; tail -5 "$lf" | tee -a "$MAIN_LOG"; return $rc
  fi
}
wait_all() { local rc=0 p; for p in "$@"; do wait "$p" || rc=1; done; return $rc; }
artifact_ready() {
  "$PY" - "$1" <<'PYEOF'
import sys
from pathlib import Path
from artifact_contract import cached_rollout_ready
sys.exit(0 if cached_rollout_ready(Path(sys.argv[1])) else 1)
PYEOF
}
merge_rollouts() {  # merge_rollouts <base> <responses per prompt>   (same routine as scripts/run_point.sh)
  local base="$1" expected_k="$2"
  "$PY" - "$RUN" "$base" "$expected_k" 2>&1 <<'PYEOF' | tee -a "$MAIN_LOG"; return "${PIPESTATUS[0]}"
import json, sys
from pathlib import Path
from compact_artifacts import compact_rollout_shards
root, base, expected_k = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
merged = root / (base + ".jsonl")
sources = [merged] if merged.exists() else sorted(root.glob(base + ".shard*.jsonl"))
if not sources:
    print(f"[merge-abort] {base}: shard files not found", flush=True); sys.exit(1)
split = "val" if base.endswith("_val") else "train"
n = len(json.loads((root / "prompts.json").read_text())[split])
seen = {}
for s in sources:
    for line in s.open():
        r = json.loads(line)
        seen.setdefault(r["prompt_idx"], []).append((r["rollout_idx"], line))
missing = [i for i in range(n) if i not in seen]
unexpected = sorted(i for i in seen if i < 0 or i >= n)
dup = [i for i, v in seen.items() if len({j for j, _ in v}) != len(v)]
bad_k = [i for i, v in seen.items() if 0 <= i < n if sorted(j for j, _ in v) != list(range(expected_k))]
if missing or unexpected or dup or bad_k:
    print(f"[merge-abort] {base}: missing {len(missing)} (e.g. {missing[:5]}) out-of-range {len(unexpected)} "
          f"duplicate {len(dup)} wrong-K {len(bad_k)} (e.g. {bad_k[:5]}); the shard split changed (GPU count?). "
          f"Remove the shard files of this base and rerun.", flush=True)
    sys.exit(1)
if not merged.exists():
    tmp = root / (base + ".jsonl.tmp")
    with tmp.open("w") as f:
        for i in range(n):
            for _, line in sorted(seen[i]):
                f.write(line)
    tmp.rename(merged)
removed = compact_rollout_shards(root, base)
print(f"[merge] {base}: {n} prompts x K={expected_k}, {sum(len(v) for v in seen.values())} rollouts OK; "
      f"removed {len(removed)} redundant shard files", flush=True)
PYEOF
}

log "=== reliability budget run: $RUN (dataset=$DATASET fresh_k=$FRESH_K val_k=$VAL_K seed=$SEED gpus=${GPUS[*]}) ==="
# 1 rollouts
if artifact_ready "$RUN/rollouts_fresh_train.jsonl" && artifact_ready "$RUN/rollouts_fresh_val.jsonl"; then
  log "stage 1 rollout-fresh: already published and validated; skipped"
else
  log "stage 1 rollout-fresh: ${N_TRAIN}x$FRESH_K + val ${N_VAL}x$VAL_K on $NGPU GPU(s) (longest stage; progress in $LOGS/fresh-shard*.log)"
  pids=(); for i in $(seq 0 $((NGPU - 1))); do
    ( run_stage "$i" "$LOGS/fresh-shard$i.log" --stage rollout-fresh "${COMMON[@]}" --shard "$i:$NGPU" ) & pids+=($!); CHILDREN+=($!)
  done
  wait_all "${pids[@]}" || { log "[abort] a rollout shard failed; rerun the same command to resume"; stop_children; exit 1; }
  CHILDREN=()
  run_tracked merge_rollouts rollouts_fresh_train "$FRESH_K" || exit 1
  run_tracked merge_rollouts rollouts_fresh_val "$VAL_K" || exit 1
fi
# 2+3 gradients
if [ -s "$RUN/oracle_micro_groups.pt" ] && [ -s "$RUN/val_groups.pt" ]; then
  log "stage 2-4 gradients: oracle_micro_groups.pt and val_groups.pt exist; skipped"
else
  log "stage 2-3 val-grads + oracle-grads on $NGPU GPU(s)"
  pids=()
  if [ "$NGPU" -ge 2 ]; then
    NM=$((NGPU - 1))
    ( run_stage "$NM" "$LOGS/val-grads.log" --stage val-grads "${COMMON[@]}" ) & pids+=($!); CHILDREN+=($!)
  else
    NM=1
    run_tracked run_stage 0 "$LOGS/val-grads.log" --stage val-grads "${COMMON[@]}" || exit 1
  fi
  for i in $(seq 0 $((NM - 1))); do
    ( run_stage "$i" "$LOGS/ograds-shard$i.log" --stage oracle-grads "${COMMON[@]}" --shard "$i:$NM" ) & pids+=($!); CHILDREN+=($!)
  done
  wait_all "${pids[@]}" || { log "[abort] a gradient stage failed; rerun the same command to resume"; stop_children; exit 1; }
  CHILDREN=()
  log "stage 4 merge-grads"
  run_tracked run_stage 0 "$LOGS/merge.log" --stage merge-grads "${COMMON[@]}" || exit 1
  [ -s "$RUN/oracle_micro_groups.pt" ] && [ -s "$RUN/val_groups.pt" ] || { log "[abort] gradient artifacts missing after merge"; exit 1; }
fi
# Stored A/B scores from the registered scoring code (src/experiment.py), so the
# analysis can verify its own recomputation against an independent implementation.
write_stored_scores() {
  "$PY" - "$RUN" 2>&1 <<'PYEOF' | tee -a "$MAIN_LOG"; return "${PIPESTATUS[0]}"
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
try:
    import torch
    from experiment import _atomic_text, score_oracle_microgroups, split_validation_directions
except Exception as exc:
    print(f"[rb] stored scores not written (backend lacks the registered scoring functions: {exc})")
    sys.exit(0)
micro = torch.load(run / "oracle_micro_groups.pt", map_location="cpu", weights_only=True)
val_groups = torch.load(run / "val_groups.pt", map_location="cpu", weights_only=True).float()
val_rank, val_a, val_b = split_validation_directions(val_groups)
oracle, halves = {}, {}
for idx in sorted(micro, key=int):
    oracle[idx], halves[idx] = score_oracle_microgroups(micro[idx].float(), val_rank, val_a, val_b)
_atomic_text(run / "scores_splithalf.json", json.dumps(halves, indent=1))
_atomic_text(run / "scores_oracle.json", json.dumps(oracle, indent=1))
print(f"[rb] stored A/B scores written for {len(halves)} prompts with the registered scoring code")
PYEOF
}
if [ -s "$RUN/scores_splithalf.json" ]; then
  log "stored A/B scores present; kept"
else
  run_tracked write_stored_scores || { log "[abort] writing the stored scores failed"; exit 1; }
fi
printf '%s\n' "completed $(date -Is)" > "$RUN/RB_DONE"
# 5 analysis: the new run next to the registered d0 point it was sized from.
log "stage 5 analysis -> $OUT"
analysis() {
  "$PY" src/reliability_budget.py "$RUN" "$SRC_POINT" \
    --label "$DATASET new reference fk$FRESH_K vk$VAL_K s$SEED" --label "$DATASET registered d0 ($(basename "$(dirname "$SRC_POINT")"))" \
    --out "$OUT" --reps "${RB_REPS:-40}" --pairs "${RB_PAIRS:-20}" --target "${RB_TARGET:-0.20}" > "$LOGS/analysis.log" 2>&1
}
run_tracked analysis || { log "[abort] analysis failed (see $LOGS/analysis.log)"; tail -5 "$LOGS/analysis.log"; exit 1; }
log "report: $OUT"
grep "^KEY " "$OUT" | tee -a "$MAIN_LOG"
log "=== reliability budget run complete ==="
