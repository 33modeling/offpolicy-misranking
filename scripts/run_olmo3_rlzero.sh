#!/usr/bin/env bash
# OLMo-3 base RL-Zero experiment: prepare once, then run on every 4xH100 node.
set -uo pipefail

cd "$(dirname "$0")/.."
SUPERVISOR_REPO=$PWD
export OM_REPO="${OM_REPO:-$SUPERVISOR_REPO}"
MODE=${1:-run}
PROFILE=${2:-baseline}
case "$MODE" in
  prepare|check|run|status) ;;
  *) echo "usage: bash scripts/run_olmo3_rlzero.sh [prepare|check|run|status] [baseline|h100]"; exit 2 ;;
esac
case "$PROFILE" in
  baseline)
    DEFAULT_CONFIG="$SUPERVISOR_REPO/configs/olmo3_rlzero.json"
    DEFAULT_MODEL_TAG=olmo3-1025-7b-base-rlzero-grpo-v1
    ;;
  h100)
    DEFAULT_CONFIG="$SUPERVISOR_REPO/configs/olmo3_rlzero_h100.json"
    DEFAULT_MODEL_TAG=olmo3-1025-7b-base-rlzero-grpo-h100-v2
    ;;
  *) echo "[abort] unknown profile=$PROFILE; expected baseline or h100"; exit 2 ;;
esac

export OM_ONLINE=$([ "$MODE" = prepare ] && printf 1 || printf 0)
source scripts/setup_env.sh
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
CONFIG="${OM_RLZERO_CONFIG:-$DEFAULT_CONFIG}"
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv missing: $PY"; exit 1; }
[ -s "$CONFIG" ] || { echo "[abort] experiment config missing: $CONFIG"; exit 1; }
CONFIG=$(realpath "$CONFIG") || exit 1
case "$CONFIG" in
  "$SUPERVISOR_REPO"/configs/*.json) CONFIG_REL="configs/${CONFIG##*/}" ;;
  *) echo "[abort] experiment config must be committed under $SUPERVISOR_REPO/configs"; exit 1 ;;
esac
command -v flock >/dev/null 2>&1 || { echo "[abort] flock missing"; exit 1; }

if [ "$MODE" != status ]; then
  MATH_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py \
    --cache-root "$OM_WORK/runtime-deps") || exit 1
  export PYTHONPATH="$MATH_VERIFY_PATH${PYTHONPATH:+:$PYTHONPATH}"
  "$PY" -c 'from math_verify import parse, verify; assert verify(parse(r"\frac{1}{2}"), parse("0.5"))' \
    || { echo "[abort] bundled math verifier failed to import"; exit 1; }
  echo "[runtime] bundled math-verify ready: $MATH_VERIFY_PATH"
fi

MODEL_KEY=olmo3-7b-base
model_field() {
  "$PY" src/model_matrix.py --config "$CONFIG" --models-dir "$MODELS_DIR" \
    field "$MODEL_KEY" "$1"
}
experiment_field() {
  "$PY" src/model_matrix.py --config "$CONFIG" experiment-field "$1"
}
grpo_field() {
  "$PY" src/model_matrix.py --config "$CONFIG" grpo-field "$1"
}
runtime_field() {
  "$PY" src/model_matrix.py --config "$CONFIG" runtime-field "$1"
}

if [ "$MODE" = prepare ]; then
  export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0
  export HF_HUB_ETAG_TIMEOUT=15 HF_HUB_DOWNLOAD_TIMEOUT=60
  mkdir -p "$MODELS_DIR" "$DATASETS_DIR" "$OM_WORK/locks"
  echo "[prepare] public pinned assets only; inherited Hugging Face tokens are disabled"
  (
    flock 9
    timeout 14400 "$PY" src/model_matrix.py --config "$CONFIG" \
      --models-dir "$MODELS_DIR" download "$MODEL_KEY" || exit 1
    bash scripts/fetch_datasets.sh math500 mbpp || exit 1
    OM_ONLINE=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      OM_MATH_VERIFIER=math_verify "$PY" src/qualify_domain_data.py math500 mbpp \
      --data-root "$DATASETS_DIR" --n-train 512 \
      --dataset-n-train math500=400 --dataset-n-train mbpp=512 \
      --n-val 100 --seeds 0 1 2 3 4 \
      --output "$OM_WORK/preflight/olmo3-rlzero-data.json" || exit 1
  ) 9>"$OM_WORK/locks/olmo3-rlzero-prepare.lock" || exit 1
  echo "[prepare] pinned model and datasets are ready under $GROUP_VOLUME"
  exit 0
fi

export OM_ONLINE=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
[ -d "$GROUP_VOLUME" ] || { echo "[abort] shared GROUP_VOLUME missing: $GROUP_VOLUME"; exit 1; }
[[ "$OM_WORK" == "$GROUP_VOLUME" || "$OM_WORK" == "$GROUP_VOLUME/"* ]] || {
  echo "[abort] OM_WORK must be on GROUP_VOLUME: OM_WORK=$OM_WORK"
  exit 1
}

MODEL_PATH="${OM_OLMO3_MODEL_PATH:-$(model_field path)}"
[ -n "$MODEL_PATH" ] || { echo "[abort] empty OLMo-3 model path"; exit 1; }
MODEL_REVISION=$(model_field revision)
LORA_TARGETS=$(model_field lora_targets)
DATASETS=($(experiment_field datasets))
SEEDS=($(experiment_field seeds))
DRIFTS=($(experiment_field drifts))
N_VAL=$(experiment_field n_val)
CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
MODEL_TAG="${OM_OLMO3_MODEL_TAG:-$DEFAULT_MODEL_TAG}"
ROOT="${OM_OLMO3_ROOT:-$OM_WORK/runs/$MODEL_TAG}"
GLOBAL_RESULTS="${OM_OLMO3_RESULTS:-$OM_WORK/results/$MODEL_TAG}"
QUEUE="$ROOT/.families"
PREFLIGHT="$ROOT/preflight"

family_root() { printf '%s/family-%s-s%s\n' "$ROOT" "$1" "$2"; }
family_result() { printf '%s/family-results/%s-s%s\n' "$ROOT" "$1" "$2"; }
run_dir() {
  printf '%s/%s-s%s-%s-d%s\n' \
    "$(family_root "$1" "$2")" "$MODEL_TAG" "$2" "$1" "$3"
}
family_stamp() { printf '%s/.family-complete\n' "$(family_root "$1" "$2")"; }
family_complete() {
  local dataset=$1 seed=$2 drift stamp expected
  stamp=$(family_stamp "$dataset" "$seed")
  expected="$GENERATION_GIT $CONFIG_SHA $MODEL_REVISION $dataset $seed"
  [ -f "$stamp" ] && [ "$(cat "$stamp" 2>/dev/null)" = "$expected" ] || return 1
  for drift in "${DRIFTS[@]}"; do
    [ -s "$(run_dir "$dataset" "$seed" "$drift")/DONE" ] || return 1
  done
}

if [ "$MODE" = status ]; then
  echo "profile=$PROFILE"
  echo "experiment_root=$ROOT"
  [ -s "$ROOT/.queue/generation.git" ] && \
    echo "generation_git=$(cat "$ROOT/.queue/generation.git")" || \
    echo "generation_git=not-started"
  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      if [ -s "$(family_stamp "$dataset" "$seed")" ]; then
        state=complete
      elif [ -s "$QUEUE/$dataset-s$seed.owner.json" ]; then
        if (flock -n 9) 9>"$QUEUE/$dataset-s$seed.lock"; then
          state="stale-owner $(tr '\n' ' ' < "$QUEUE/$dataset-s$seed.owner.json")"
        else
          state="claimed $(tr '\n' ' ' < "$QUEUE/$dataset-s$seed.owner.json")"
        fi
      else
        state=pending
      fi
      echo "$dataset/s$seed $state"
    done
  done
  [ -s "$GLOBAL_RESULTS/FINAL_REPORT.md" ] && echo "report=$GLOBAL_RESULTS/FINAL_REPORT.md"
  exit 0
fi

DIRTY=$(git status --porcelain -- src scripts configs requirements.txt)
[ -z "$DIRTY" ] || {
  echo "[abort] generation code is dirty; commit and pull one revision before allocating GPUs"
  printf '%s\n' "$DIRTY"
  exit 1
}

CURRENT_GIT=$(git rev-parse HEAD) || exit 1
mkdir -p "$ROOT/.queue" "$QUEUE" "$PREFLIGHT" "$GLOBAL_RESULTS" "$OM_WORK/locks"

if [ "$MODE" = run ]; then
  mapfile -t GPU_NAMES < <(
    timeout 20 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
  )
  GPU_COUNT=${#GPU_NAMES[@]}
  H100_COUNT=$(printf '%s\n' "${GPU_NAMES[@]}" | grep -c H100 || true)
  [ "$GPU_COUNT" -eq 4 ] && [ "$H100_COUNT" -eq 4 ] || {
    echo "[abort] exactly four H100 GPUs required (GPUs=$GPU_COUNT H100=$H100_COUNT)"
    exit 1
  }

  LOCAL_ROOT="${OM_LOCAL_LOCK_DIR:-/tmp/offpolicy-misranking-$(id -u)}"
  local_path=$(realpath -m "$LOCAL_ROOT")
  shared_path=$(realpath -m "$GROUP_VOLUME")
  [[ "$local_path" != "$shared_path" && "$local_path" != "$shared_path/"* ]] || {
    echo "[abort] OM_LOCAL_LOCK_DIR must be node-local"
    exit 1
  }
  mkdir -p "$LOCAL_ROOT/olmo3-preflight" "$ROOT/logs"
  exec 8>"$LOCAL_ROOT/primary.lock"
  flock -n 8 || { echo "[abort] another experiment already owns this physical node"; exit 1; }

  memory=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) || exit 1
  rows=$(printf '%s\n' "$memory" | awk 'NF {n++} END {print n+0}')
  busy=$(printf '%s\n' "$memory" | awk '$1 > 2000 {n++} END {print n+0}')
  [ "$rows" -eq 4 ] && [ "$busy" -eq 0 ] || {
    echo "[abort] GPUs are already in use; refusing to overlap another experiment"
    printf '%s\n' "$memory"
    exit 1
  }

  HOST_TAG=$(hostname 2>/dev/null || printf node)
  WORKER_SUFFIX=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || printf '%s' "$$")
  WORKER_ID=$(printf '%s-%s' "$HOST_TAG" "$WORKER_SUFFIX" | tr -cs 'a-zA-Z0-9._-' '-')
  export WORKER_ID
  LOG="$ROOT/logs/$WORKER_ID.log"
  echo "[worker] id=$WORKER_ID root=$ROOT" | tee -a "$LOG"

  # Keep all allocated GPUs active before model/dataset verification begins.
  # This also spans signal qualification, point transitions, CPU verification,
  # queue waits, and final result collection.
  ACTIVE_OWNER=""
  SUPERVISOR_KEEPALIVE=""
  KEEPALIVE_READY="$LOCAL_ROOT/keepalive-$WORKER_ID.ready"
  stop_supervisor_keepalive() {
    if [ -n "${SUPERVISOR_KEEPALIVE:-}" ]; then
      kill "$SUPERVISOR_KEEPALIVE" 2>/dev/null || true
      wait "$SUPERVISOR_KEEPALIVE" 2>/dev/null || true
    fi
    SUPERVISOR_KEEPALIVE=""
    rm -f -- "$KEEPALIVE_READY"
  }
  start_supervisor_keepalive() {
    [ -z "${SUPERVISOR_KEEPALIVE:-}" ] || return 0
    rm -f -- "$KEEPALIVE_READY"
    OM_GPU_KEEPALIVE_READY_FILE="$KEEPALIVE_READY" CUDA_VISIBLE_DEVICES=0,1,2,3 \
      "$PY" "$SUPERVISOR_REPO/scripts/gpu_keepalive.py" \
      >> "$ROOT/logs/$WORKER_ID-keepalive.log" 2>&1 8>&- 9>&- &
    SUPERVISOR_KEEPALIVE=$!
    for _ in $(seq 1 600); do
      [ -s "$KEEPALIVE_READY" ] && break
      kill -0 "$SUPERVISOR_KEEPALIVE" 2>/dev/null || {
        echo "[abort] GPU keepalive exited during startup" | tee -a "$LOG"
        SUPERVISOR_KEEPALIVE=""
        return 1
      }
      sleep 0.1
    done
    [ -s "$KEEPALIVE_READY" ] || {
      echo "[abort] GPU keepalive was not ready within 60 seconds" | tee -a "$LOG"
      stop_supervisor_keepalive
      return 1
    }
    echo "[worker] four-GPU keepalive ready pid=$SUPERVISOR_KEEPALIVE" | tee -a "$LOG"
  }
  cleanup_worker() {
    [ -z "${ACTIVE_OWNER:-}" ] || rm -f -- "$ACTIVE_OWNER"
    ACTIVE_OWNER=""
    stop_supervisor_keepalive
  }
  trap cleanup_worker EXIT
  trap 'exit 130' INT TERM HUP
  start_supervisor_keepalive || exit 1
  export OM_EXTERNAL_GPU_KEEPALIVE=1
fi

# Adopt separately uploaded assets before entering an older commit-pinned run.
# This creates the standard manifests/paths that the pinned code can read.
(
  flock 9
  if [ ! -s "$MODEL_PATH/.om_snapshot.json" ]; then
    if ! PYTHONPATH="$SUPERVISOR_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$PY" "$SUPERVISOR_REPO/src/model_matrix.py" --config "$CONFIG" \
        --models-dir "$MODELS_DIR" --snapshot-path "$MODEL_PATH" check "$MODEL_KEY"; then
      echo "[model] manifest missing; verifying the uploaded model against pinned official hashes"
      PYTHONPATH="$SUPERVISOR_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$PY" "$SUPERVISOR_REPO/src/model_matrix.py" --config "$CONFIG" \
        --models-dir "$MODELS_DIR" --snapshot-path "$MODEL_PATH" seal "$MODEL_KEY" || exit 1
    fi
  fi
  OM_MATH_VERIFIER=math_verify \
    PYTHONPATH="$SUPERVISOR_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" "$SUPERVISOR_REPO/src/qualify_domain_data.py" "${DATASETS[@]}" \
    --data-root "$DATASETS_DIR" --n-train 512 \
    --dataset-n-train math500=400 --dataset-n-train mbpp=512 \
    --n-val "$N_VAL" --seeds "${SEEDS[@]}" \
    --output "$PREFLIGHT/data-adoption.json" || exit 1
) 9>"$OM_WORK/locks/olmo3-asset-adoption.lock" || exit 1

GENERATION_GIT=$("$PY" src/regime_resume_commit.py "$ROOT" "$CURRENT_GIT" \
  --marker "$ROOT/.queue/generation.git" --advance-empty) || exit 1

GENERATION_REPO=$SUPERVISOR_REPO
if [ "$GENERATION_GIT" != "$CURRENT_GIT" ]; then
  PIPELINE_CACHE="${OM_PIPELINE_CACHE:-/tmp/offpolicy-misranking-$(id -u)/pipelines}"
  GENERATION_REPO="$PIPELINE_CACHE/$GENERATION_GIT"
  mkdir -p "$PIPELINE_CACHE"
  (
    flock 9
    git -C "$SUPERVISOR_REPO" cat-file -e "$GENERATION_GIT^{commit}" 2>/dev/null || {
      echo "[abort] pinned generation commit is unavailable locally: $GENERATION_GIT" >&2
      exit 1
    }
    cached_git=$(git -C "$GENERATION_REPO" rev-parse HEAD 2>/dev/null || true)
    cached_dirty=invalid
    if [ "$cached_git" = "$GENERATION_GIT" ]; then
      cached_dirty=$(git -C "$GENERATION_REPO" status --porcelain \
        -- src scripts configs 2>/dev/null || printf invalid)
    fi
    if [ "$cached_git" != "$GENERATION_GIT" ] || [ -n "$cached_dirty" ]; then
      if [ -e "$GENERATION_REPO" ] || [ -L "$GENERATION_REPO" ]; then
        stale="$PIPELINE_CACHE/.stale-$GENERATION_GIT-$(date +%s)-$$"
        mv -- "$GENERATION_REPO" "$stale" || exit 1
        echo "[worktree] invalid cache quarantined: $stale" >&2
      fi
      git -C "$SUPERVISOR_REPO" worktree prune --expire now || exit 1
      # Two -f flags also override a stale registration that was left locked.
      git -C "$SUPERVISOR_REPO" worktree add -f -f --detach "$GENERATION_REPO" \
        "$GENERATION_GIT" >&2 || exit 1
    fi
    [ "$(git -C "$GENERATION_REPO" rev-parse HEAD 2>/dev/null)" = "$GENERATION_GIT" ] \
      || { echo "[abort] pinned worktree HEAD mismatch after repair" >&2; exit 1; }
    [ -z "$(git -C "$GENERATION_REPO" status --porcelain -- src scripts configs)" ] \
      || { echo "[abort] pinned worktree remains dirty after repair" >&2; exit 1; }
  ) 9>"$PIPELINE_CACHE/.worktree.lock" || exit 1
fi
GENERATION_CONFIG="$GENERATION_REPO/$CONFIG_REL"
[ -s "$GENERATION_CONFIG" ] || {
  echo "[abort] pinned generation commit lacks the OLMo-3 contract: $GENERATION_GIT"
  exit 1
}
[ "$(sha256sum "$GENERATION_CONFIG" | awk '{print $1}')" = "$CONFIG_SHA" ] || {
  echo "[abort] requested config differs from the experiment-wide pinned contract"
  exit 1
}
echo "[contract] git=$GENERATION_GIT model_revision=$MODEL_REVISION config=$CONFIG_SHA"
PINNED_POINT_EXTERNAL_KEEPALIVE=0
grep -Fq 'OM_EXTERNAL_GPU_KEEPALIVE' "$GENERATION_REPO/scripts/run_point.sh" \
  && PINNED_POINT_EXTERNAL_KEEPALIVE=1

# Static qualification is deliberately offline. It hashes every model shard,
# validates both pinned datasets, and executes the actual math/code verifiers.
if ! PYTHONPATH="$GENERATION_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" "$GENERATION_REPO/src/model_matrix.py" --config "$GENERATION_CONFIG" \
    --models-dir "$MODELS_DIR" --snapshot-path "$MODEL_PATH" check "$MODEL_KEY"; then
  echo "[model] manifest missing; verifying the uploaded model against pinned official hashes"
  PYTHONPATH="$GENERATION_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" "$GENERATION_REPO/src/model_matrix.py" --config "$GENERATION_CONFIG" \
    --models-dir "$MODELS_DIR" --snapshot-path "$MODEL_PATH" seal "$MODEL_KEY" || exit 1
fi
OM_MATH_VERIFIER=math_verify PYTHONPATH="$GENERATION_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" "$GENERATION_REPO/src/qualify_domain_data.py" "${DATASETS[@]}" \
  --data-root "$DATASETS_DIR" --n-train 512 \
  --dataset-n-train math500=400 --dataset-n-train mbpp=512 \
  --n-val "$N_VAL" --seeds "${SEEDS[@]}" \
  --output "$PREFLIGHT/data-qualification.json" || exit 1

if [ "$MODE" = check ]; then
  echo "[check] offline model, dataset, prompt, and verifier contracts passed"
  exit 0
fi

export MODEL_PATH OM_MATH_VERIFIER=math_verify OM_TOP_P=1.0 OM_THINKING=off
export OM_ATTN="$(experiment_field attn)" OM_SKIP_HYBRID=1
export OM_LORA_TARGETS="$LORA_TARGETS" OM_GEN_BATCH="$(runtime_field generation_batch)"
export GRPO_LOGPROB_MICRO_BATCH="$(runtime_field logprob_micro_batch)"
export GRPO_GRADIENT_CHECKPOINTING="$(runtime_field gradient_checkpointing)"

signal_qualify() {
  local dataset=$1 report="$PREFLIGHT/$1-signal.json"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$GENERATION_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" "$GENERATION_REPO/src/qualify_rlzero_signal.py" \
    --model "$MODEL_PATH" --dataset "$dataset" --data-root "$DATASETS_DIR" \
    --output "$report" --prompt-count 8 --group-size 8 \
    --max-new-tokens 1024 --generation-batch "$OM_GEN_BATCH"
}
SIGNAL_WAIT_SECONDS="${OM_RLZERO_PREFLIGHT_WAIT_SECONDS:-10}"
case "$SIGNAL_WAIT_SECONDS" in
  ''|*[!0-9]*|0) echo "[abort] OM_RLZERO_PREFLIGHT_WAIT_SECONDS must be a positive integer"; exit 2 ;;
esac

# Nonblocking claims let two clusters qualify math and code concurrently. A
# second locked pass then revalidates the cached artifacts on every worker.
while :; do
  signal_remaining=0
  signal_claimed=0
  for dataset in "${DATASETS[@]}"; do
    [ -s "$PREFLIGHT/$dataset-signal.json" ] && continue
    signal_remaining=$((signal_remaining + 1))
    (
      flock -n 9 || exit 75
      [ -s "$PREFLIGHT/$dataset-signal.json" ] || signal_qualify "$dataset" || exit 1
    ) 9>"$PREFLIGHT/$dataset-signal.lock"
    rc=$?
    [ "$rc" -eq 75 ] && continue
    [ "$rc" -eq 0 ] || exit "$rc"
    signal_claimed=$((signal_claimed + 1))
  done
  [ "$signal_remaining" -eq 0 ] && break
  [ "$signal_claimed" -gt 0 ] || sleep "$SIGNAL_WAIT_SECONDS"
done
for dataset in "${DATASETS[@]}"; do
  (flock 9; signal_qualify "$dataset") \
    9>"$PREFLIGHT/$dataset-signal.lock" || exit 1
done

# Exercise the exact four-rank GRPO launch and an optimizer/adapter resume on
# every physical node before it can claim a long-running family.
SMOKE_KEY=$(printf '%s\n' "$GENERATION_GIT $CONFIG_SHA $MODEL_REVISION ${GPU_NAMES[*]}" \
  | sha256sum | awk '{print $1}')
SMOKE_ROOT="$LOCAL_ROOT/olmo3-preflight/$SMOKE_KEY"
SMOKE_MARKER="$SMOKE_ROOT/PASSED"
if [ ! -s "$SMOKE_MARKER" ]; then
  rm -rf "$SMOKE_ROOT"
  mkdir -p "$SMOKE_ROOT"
  OM_ONLINE=0 MATH500_DIR="$DATASETS_DIR/math500" \
    PYTHONPATH="$GENERATION_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" - "$SMOKE_ROOT/prompts.json" <<'PYEOF' || exit 1
import json, sys
from data import load_prompts
prompts = load_prompts("math500", 4, 1, seed=0)
open(sys.argv[1], "w").write(json.dumps(prompts) + "\n")
PYEOF
  export OM_PROMPT_FORMAT=olmo_rlzero_math
  SMOKE_GROUP_SIZE=2
  [ "$PROFILE" = h100 ] && SMOKE_GROUP_SIZE="$(grpo_field group_size)"
  SMOKE_LOGPROB_MICRO_BATCH="$GRPO_LOGPROB_MICRO_BATCH"
  [ "$SMOKE_LOGPROB_MICRO_BATCH" -le "$SMOKE_GROUP_SIZE" ] \
    || SMOKE_LOGPROB_MICRO_BATCH="$SMOKE_GROUP_SIZE"
  common=(--model "$MODEL_PATH" --objective grpo --prompts "$SMOKE_ROOT/prompts.json"
    --expected-world-size 4 --group-size "$SMOKE_GROUP_SIZE" --clip-epsilon 0.2
    --learning-rate 1e-5 --epochs-per-batch 1 --max-grad-norm 1.0
    --advantage-epsilon 1e-4 --lora-rank 4 --lora-alpha 8
    --checkpoint-every 1 --logprob-micro-batch "$SMOKE_LOGPROB_MICRO_BATCH"
    --max-new-tokens 64 --seed 271828)
  [ "$GRPO_GRADIENT_CHECKPOINTING" = 1 ] \
    || common+=(--disable-gradient-checkpointing)
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$GENERATION_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$GENERATION_REPO/src/train_policy_grpo.py" "${common[@]}" \
    --output "$SMOKE_ROOT/step1" --target-steps 1 || exit 1
  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$GENERATION_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    "$GENERATION_REPO/src/train_policy_grpo.py" "${common[@]}" \
    --output "$SMOKE_ROOT/step2" --target-steps 2 --start-step 1 \
    --resume-adapter "$SMOKE_ROOT/step1" \
    --resume-optimizer "$SMOKE_ROOT/step1/optimizer.pt" || exit 1
  printf '%s\n' "$SMOKE_KEY" > "$SMOKE_MARKER.tmp"
  mv "$SMOKE_MARKER.tmp" "$SMOKE_MARKER"
fi
echo "[preflight] model signal + four-GPU GRPO + checkpoint resume passed" | tee -a "$LOG"

export GRPO_WORLD_SIZE=$(grpo_field world_size)
export GRPO_GROUP_SIZE=$(grpo_field group_size)
export GRPO_CLIP_EPSILON=$(grpo_field clip_epsilon)
export GRPO_LEARNING_RATE=$(grpo_field learning_rate)
export GRPO_EPOCHS_PER_BATCH=$(grpo_field epochs_per_batch)
export GRPO_MAX_GRAD_NORM=$(grpo_field max_grad_norm)
export GRPO_ADVANTAGE_EPSILON=$(grpo_field advantage_epsilon)
export GRPO_LORA_RANK=$(grpo_field lora_rank)
export GRPO_LORA_ALPHA=$(grpo_field lora_alpha)
export GRPO_CHECKPOINT_EVERY=5 RLVR_METHOD=grpo
export REGIME_N_VAL="$N_VAL"
export REGIME_N_TRAIN_BY_DATASET="math500=400 mbpp=512"
export REGIME_BEHAVIOR_K=$(experiment_field behavior_k)
export REGIME_FRESH_K=$(experiment_field fresh_k)
export REGIME_VAL_K=$(experiment_field val_k)
export REGIME_MICRO_GROUP=$(experiment_field micro_group)
export REGIME_MAX_NEW_TOKENS=$(experiment_field max_new_tokens)
export REGIME_PROJ_DIM=$(experiment_field proj_dim)
export REGIME_GRAD_LAYERS=$(experiment_field grad_layers)
export REGIME_CLIP_CAP=$(experiment_field clip_cap)
export REGIME_TOPK_FRAC=$(experiment_field topk_frac)
export REGIME_TEMPERATURE=$(experiment_field temperature)
export REGIME_FIRST_BOOTSTRAP=$(experiment_field first_bootstrap)
export REGIME_MAX_RETRIES=3 OM_STALL_MINUTES=15
export OM_SKIP_GPU_CHECK=0 OM_ALLOW_DIRTY=0 OM_ALLOW_ANALYSIS_UPGRADE=1
FAMILY_ATTEMPTS="${OM_RLZERO_FAMILY_ATTEMPTS:-3}"
case "$FAMILY_ATTEMPTS" in
  ''|*[!0-9]*|0) echo "[abort] OM_RLZERO_FAMILY_ATTEMPTS must be a positive integer"; exit 2 ;;
esac
FAMILY_RETRY_SECONDS="${OM_RLZERO_FAMILY_RETRY_SECONDS:-60}"
case "$FAMILY_RETRY_SECONDS" in
  ''|*[!0-9]*) echo "[abort] OM_RLZERO_FAMILY_RETRY_SECONDS must be a non-negative integer"; exit 2 ;;
esac
QUEUE_WAIT_SECONDS="${OM_RLZERO_QUEUE_WAIT_SECONDS:-60}"
case "$QUEUE_WAIT_SECONDS" in
  ''|*[!0-9]*|0) echo "[abort] OM_RLZERO_QUEUE_WAIT_SECONDS must be a positive integer"; exit 2 ;;
esac
CLAIM_YIELD_SECONDS="${OM_RLZERO_CLAIM_YIELD_SECONDS:-1}"
case "$CLAIM_YIELD_SECONDS" in
  ''|*[!0-9]*) echo "[abort] OM_RLZERO_CLAIM_YIELD_SECONDS must be a non-negative integer"; exit 2 ;;
esac

cleanup_owner() {
  [ -z "$ACTIVE_OWNER" ] || rm -f -- "$ACTIVE_OWNER"
  ACTIVE_OWNER=""
}

run_family() {
  local dataset=$1 seed=$2 root result format owner rc=1 attempt
  root=$(family_root "$dataset" "$seed")
  result=$(family_result "$dataset" "$seed")
  owner="$QUEUE/$dataset-s$seed.owner.json"
  format=olmo_rlzero_math
  [ "$dataset" = mbpp ] && format=olmo_rlzero_code
  ACTIVE_OWNER=$owner
  HOST_TAG="$HOST_TAG" WORKER_ID="$WORKER_ID" GENERATION_GIT="$GENERATION_GIT" \
    DATASET="$dataset" SEED="$seed" "$PY" - "$owner" <<'PYEOF'
import datetime, json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
tmp.write_text(json.dumps({
    "host": os.environ["HOST_TAG"], "worker": os.environ["WORKER_ID"],
    "dataset": os.environ["DATASET"], "seed": int(os.environ["SEED"]),
    "generation_git": os.environ["GENERATION_GIT"],
    "claimed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
}, sort_keys=True) + "\n")
tmp.replace(path)
PYEOF
  [ "$?" -eq 0 ] || { cleanup_owner; return 1; }
  for ((attempt = 1; attempt <= FAMILY_ATTEMPTS; attempt++)); do
    echo "[family] claim=$dataset/s$seed attempt=$attempt/$FAMILY_ATTEMPTS" | tee -a "$LOG"
    # Older pinned generation commits start their own point-local keepalive.
    # Pause the supervisor during those points to avoid duplicate GPU load.
    [ "$PINNED_POINT_EXTERNAL_KEEPALIVE" -eq 1 ] || stop_supervisor_keepalive
    OM_REPO="$GENERATION_REPO" OM_PIPELINE_REPO="$GENERATION_REPO" \
      OM_PIPELINE_SCRIPT="$GENERATION_REPO/scripts/run_point.sh" \
      OM_GENERATION_GIT="$GENERATION_GIT" \
      PYTHONPATH="$GENERATION_REPO/src${PYTHONPATH:+:$PYTHONPATH}" MODEL_PATH="$MODEL_PATH" \
      REGIME_ROOT="$root" REGIME_RESULTS="$result" REGIME_MODEL_TAG="$MODEL_TAG" \
      REGIME_DATASETS="$dataset" REGIME_SEEDS="$seed" \
      REGIME_DRIFTS="${DRIFTS[*]}" REGIME_SKIP_COLLECTION=1 \
      OM_PROMPT_FORMAT="$format" \
      bash "$SUPERVISOR_REPO/scripts/run_matrix.sh" 2>&1 | tee -a "$LOG"
    statuses=("${PIPESTATUS[@]}")
    rc=${statuses[0]}
    if [ "${statuses[1]}" -ne 0 ]; then
      cleanup_owner
      return "${statuses[1]}"
    fi
    [ "$rc" -ne 0 ] || break
    [ "$rc" -ne 43 ] || break
    sleep 30
  done
  if [ "$rc" -ne 0 ]; then
    cleanup_owner
    return "$rc"
  fi
  expected="$GENERATION_GIT $CONFIG_SHA $MODEL_REVISION $dataset $seed"
  temporary="$(family_stamp "$dataset" "$seed").tmp.$$"
  printf '%s\n' "$expected" > "$temporary" || { cleanup_owner; return 1; }
  mv "$temporary" "$(family_stamp "$dataset" "$seed")" || {
    cleanup_owner
    return 1
  }
  cleanup_owner
}

while :; do
  remaining=0
  claimed=0
  retrying=0
  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      family_complete "$dataset" "$seed" && continue
      remaining=$((remaining + 1))
      (
        flock -n 9 || exit 75
        family_complete "$dataset" "$seed" && exit 0
        run_family "$dataset" "$seed"
      ) 9>"$QUEUE/$dataset-s$seed.lock"
      rc=$?
      [ "$PINNED_POINT_EXTERNAL_KEEPALIVE" -eq 1 ] \
        || start_supervisor_keepalive || exit 1
      [ "$rc" -eq 75 ] && continue
      claimed=$((claimed + 1))
      if [ "$rc" -ne 0 ]; then
        cleanup_owner
        echo "[family-retry] $dataset/s$seed rc=$rc; allocation retained, artifacts preserved, automatic retry scheduled" \
          | tee -a "$LOG"
        retrying=$((retrying + 1))
        continue
      fi
      sleep "$CLAIM_YIELD_SECONDS"
    done
  done
  [ "$remaining" -eq 0 ] && break
  if [ "$retrying" -gt 0 ]; then
    echo "[queue] $retrying failed families remain; retrying after ${FAMILY_RETRY_SECONDS}s while GPU keepalive stays active" \
      | tee -a "$LOG"
    sleep "$FAMILY_RETRY_SECONDS"
  elif [ "$claimed" -eq 0 ]; then
    echo "[queue] waiting for $remaining families owned by other clusters" | tee -a "$LOG"
    sleep "$QUEUE_WAIT_SECONDS"
  fi
done

(
  flock 9
  expected_complete="$GENERATION_GIT $CONFIG_SHA $MODEL_REVISION"
  if [ -s "$GLOBAL_RESULTS/COMPLETE" ] \
      && [ "$(cat "$GLOBAL_RESULTS/COMPLETE")" = "$expected_complete" ]; then
    complete_outputs=1
    for output in REGIME.json REGIME.csv REGIME_SUMMARY.csv FINAL_REPORT.md; do
      [ -s "$GLOBAL_RESULTS/$output" ] || complete_outputs=0
    done
    if [ "$complete_outputs" -eq 1 ]; then
      echo "[collect] full-matrix analysis already complete"
      exit 0
    fi
  fi
  runs=()
  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      family_complete "$dataset" "$seed" || exit 1
      for drift in "${DRIFTS[@]}"; do
        runs+=("$(run_dir "$dataset" "$seed" "$drift")")
      done
    done
  done
  PYTHONPATH="$GENERATION_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" "$GENERATION_REPO/src/regime_map.py" \
    "${runs[@]}" --output-dir "$GLOBAL_RESULTS" \
    --first-bootstrap "$REGIME_FIRST_BOOTSTRAP" || exit 1
  for output in REGIME.json REGIME.csv REGIME_SUMMARY.csv FINAL_REPORT.md; do
    [ -s "$GLOBAL_RESULTS/$output" ] || exit 1
  done
  printf '%s %s %s\n' "$GENERATION_GIT" "$CONFIG_SHA" "$MODEL_REVISION" \
    > "$GLOBAL_RESULTS/COMPLETE.tmp.$$"
  mv "$GLOBAL_RESULTS/COMPLETE.tmp.$$" "$GLOBAL_RESULTS/COMPLETE"
) 9>"$QUEUE/collect.lock" || { echo "[collect-abort] global validation failed"; exit 1; }

echo "[complete] all 10 families / 40 points: $GLOBAL_RESULTS/FINAL_REPORT.md" | tee -a "$LOG"
