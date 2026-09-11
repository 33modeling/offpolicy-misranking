#!/usr/bin/env bash
# OLMo-3 base RL-Zero experiment: prepare once, then run on every 4xH100 node.
set -uo pipefail

cd "$(dirname "$0")/.."
SUPERVISOR_REPO=$PWD
export OM_REPO="${OM_REPO:-$SUPERVISOR_REPO}"
MODE=${1:-run}
PROFILE=${2:-baseline}
RUN_ROLE=auto
TARGET_DATASET=""
TARGET_SEED=""
RECOVERY_MIN_GENERATION_BATCH=2
case "$MODE" in
  resume-family|assist)
    [ "$#" -eq 4 ] && [[ "$4" =~ ^[0-9]+$ ]] || {
      echo "usage: bash scripts/run_olmo3_rlzero.sh $MODE h100 <dataset> <seed>"
      exit 2
    }
    RUN_ROLE=$MODE
    TARGET_DATASET=$3
    TARGET_SEED=$4
    MODE=run
    ;;
  prepare|check|run|status) ;;
  *) echo "usage: bash scripts/run_olmo3_rlzero.sh [prepare|check|run|status] [baseline|h100] [verbose|<dataset>]; or [resume-family|assist] h100 <dataset> <seed>"; exit 2 ;;
esac
[ "$RUN_ROLE" = auto ] || export OM_REPO="$SUPERVISOR_REPO"
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

# Status reports the installed revision without changing shared executable files.
if [ "$MODE" = status ]; then
  echo "[code] $(git rev-parse --short HEAD 2>/dev/null || printf unknown) (read-only status; no automatic update)"
fi
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

materialize_local_checkout() {
  local commit=$1 target temporary recorded dirty stale
  local source_repo="${CHECKOUT_SOURCE_REPO:-$SUPERVISOR_REPO}"
  target="$PIPELINE_CACHE/clones/$commit"
  mkdir -p "$PIPELINE_CACHE/clones"
  (
    flock 9
    recorded=$(git -C "$target" rev-parse HEAD 2>/dev/null || true)
    dirty=invalid
    if [ -d "$target/.git" ] && [ "$recorded" = "$commit" ]; then
      dirty=$(git -C "$target" status --porcelain \
        -- src scripts configs requirements.txt 2>/dev/null || printf invalid)
    fi
    if [ ! -d "$target/.git" ] || [ "$recorded" != "$commit" ] || [ -n "$dirty" ]; then
      if [ -e "$target" ] || [ -L "$target" ]; then
        stale="$PIPELINE_CACHE/.stale-$commit-$(date +%s)-$$"
        mv -- "$target" "$stale" || exit 1
        echo "[checkout] invalid node-local cache quarantined: $stale" >&2
      fi
      temporary="$PIPELINE_CACHE/.clone-$commit-$$"
      rm -rf -- "$temporary"
      git clone --quiet --no-hardlinks --no-checkout \
        "$source_repo" "$temporary" >&2 || exit 1
      git -C "$temporary" checkout --quiet --detach "$commit" >&2 || exit 1
      git -C "$temporary" remote remove origin >/dev/null 2>&1 || true
      mv -- "$temporary" "$target" || exit 1
    fi
    [ -d "$target/.git" ] || {
      echo "[abort] runtime checkout must have independent Git metadata: $target" >&2
      exit 1
    }
    [ "$(git -C "$target" rev-parse HEAD 2>/dev/null)" = "$commit" ] || {
      echo "[abort] node-local checkout HEAD mismatch: $target" >&2
      exit 1
    }
    [ -z "$(git -C "$target" status --porcelain -- src scripts configs requirements.txt)" ] || {
      echo "[abort] node-local checkout is dirty: $target" >&2
      exit 1
    }
  ) 9>"$PIPELINE_CACHE/.clone.lock" || return 1
  printf '%s\n' "$target"
}

CONFIG_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
MATRIX_TOOL_REPO=$SUPERVISOR_REPO
if [ "$MODE" = check ] || [ "$MODE" = run ]; then
  DIRTY=$(git status --porcelain -- src scripts configs requirements.txt)
  [ -z "$DIRTY" ] || {
    echo "[abort] generation code is dirty; commit once before allocating GPUs"
    printf '%s\n' "$DIRTY"
    exit 1
  }
  CURRENT_GIT=$(git rev-parse HEAD) || exit 1
  LOCAL_ROOT="${OM_LOCAL_LOCK_DIR:-/tmp/offpolicy-misranking-$(id -u)}"
  export OM_NODE_NAMESPACE="$LOCAL_ROOT"
  local_path=$(realpath -m "$LOCAL_ROOT")
  shared_path=$(realpath -m "$GROUP_VOLUME")
  [[ "$local_path" != "$shared_path" && "$local_path" != "$shared_path/"* ]] || {
    echo "[abort] OM_LOCAL_LOCK_DIR must be node-local"
    exit 1
  }
  PIPELINE_CACHE="${OM_PIPELINE_CACHE:-$LOCAL_ROOT/pipelines}"
  cache_path=$(realpath -m "$PIPELINE_CACHE")
  [[ "$cache_path" != "$shared_path" && "$cache_path" != "$shared_path/"* ]] || {
    echo "[abort] OM_PIPELINE_CACHE must be node-local"
    exit 1
  }
  SUPERVISOR_RUNTIME_REPO=$(materialize_local_checkout "$CURRENT_GIT") || exit 1
  SUPERVISOR_RUNTIME_CONFIG="$SUPERVISOR_RUNTIME_REPO/$CONFIG_REL"
  [ "$(sha256sum "$SUPERVISOR_RUNTIME_CONFIG" | awk '{print $1}')" = "$CONFIG_SHA" ] || {
    echo "[abort] node-local supervisor config differs from requested contract"
    exit 1
  }
  CHECKOUT_SOURCE_REPO=$SUPERVISOR_RUNTIME_REPO
  MATRIX_TOOL_REPO=$SUPERVISOR_RUNTIME_REPO
  CONFIG=$SUPERVISOR_RUNTIME_CONFIG
fi

if [ "$MODE" != status ]; then
  MATH_VERIFY_PATH=$("$PY" "$MATRIX_TOOL_REPO/src/bootstrap_math_verify.py" \
    --cache-root "$OM_WORK/runtime-deps") || exit 1
  export PYTHONPATH="$MATH_VERIFY_PATH${PYTHONPATH:+:$PYTHONPATH}"
  "$PY" -c 'from math_verify import parse, verify; assert verify(parse(r"\frac{1}{2}"), parse("0.5"))' \
    || { echo "[abort] bundled math verifier failed to import"; exit 1; }
  echo "[runtime] bundled math-verify ready: $MATH_VERIFY_PATH"
fi

MODEL_KEY=olmo3-7b-base
model_field() {
  "$PY" "$MATRIX_TOOL_REPO/src/model_matrix.py" --config "$CONFIG" --models-dir "$MODELS_DIR" \
    field "$MODEL_KEY" "$1"
}
experiment_field() {
  "$PY" "$MATRIX_TOOL_REPO/src/model_matrix.py" --config "$CONFIG" experiment-field "$1"
}
grpo_field() {
  "$PY" "$MATRIX_TOOL_REPO/src/model_matrix.py" --config "$CONFIG" grpo-field "$1"
}
runtime_field() {
  "$PY" "$MATRIX_TOOL_REPO/src/model_matrix.py" --config "$CONFIG" runtime-field "$1"
}

# ---- per-dataset runtime overrides (execution-only knobs, not contract fields) --
# Measured on the H100 matrix (2026-09-06): every response runs to the 2048-token
# cap and a batch-8 decode step is overhead-bound (~280 tok/s per GPU), so the
# generation batch is raised to 32 for math500 (short prompts) and 16 for mbpp
# (prompts up to ~2.3k tokens; the fp32 prefill softmax of batch 32 would not
# fit). mbpp validation gradients OOM at micro-batch 4 (fp32 attention softmax,
# 9 GiB per layer at 4.3k tokens); 1 fits with room to spare. Neither field is in
# regime_contract.RUN_CONFIG_FIELDS and the rollout partial manifest does not
# record the batch, so partial rollouts survive the change. A point created with
# other values is repaired before re-entry (src/repair_run_config.py), because the
# pinned run_point.sh refuses a run_config that differs from its environment.
# Set a list to "" to disable, e.g. OM_RLZERO_GEN_BATCH_BY_DATASET="". The
# defaults are sized for the 80 GB H100 profile only.
case "$PROFILE" in
  h100) DEFAULT_GEN_BATCH_BY_DATASET="math500=32 mbpp=16"; DEFAULT_GRADIENT_MICRO_BATCH_BY_DATASET="mbpp=1" ;;
  *) DEFAULT_GEN_BATCH_BY_DATASET=""; DEFAULT_GRADIENT_MICRO_BATCH_BY_DATASET="" ;;
esac
GEN_BATCH_BY_DATASET="${OM_RLZERO_GEN_BATCH_BY_DATASET-$DEFAULT_GEN_BATCH_BY_DATASET}"
GRADIENT_MICRO_BATCH_BY_DATASET="${OM_RLZERO_GRADIENT_MICRO_BATCH_BY_DATASET-$DEFAULT_GRADIENT_MICRO_BATCH_BY_DATASET}"
dataset_override() {  # dataset_override "<name=value ...>" <dataset> -> value (rc 1 if absent)
  local item
  for item in $1; do
    case "$item" in "$2="*) printf '%s\n' "${item#*=}"; return 0 ;; esac
  done
  return 1
}
gen_batch_for() {  # gen_batch_for <dataset>
  dataset_override "$GEN_BATCH_BY_DATASET" "$1" || runtime_field generation_batch
}
gradient_micro_batch_for() {  # gradient_micro_batch_for <dataset>
  dataset_override "$GRADIENT_MICRO_BATCH_BY_DATASET" "$1" || runtime_field gradient_micro_batch
}
validate_dataset_overrides() {  # validate_dataset_overrides <env name> "<list>"
  local item name value
  for item in $2; do
    name=${item%%=*}; value=${item#*=}
    case " ${DATASETS[*]} " in *" $name "*) ;; *) echo "[abort] $1: unknown dataset in '$item' (expected one of: ${DATASETS[*]})"; exit 2 ;; esac
    case "$value" in ''|*[!0-9]*|0) echo "[abort] $1: '$item' must be <dataset>=<positive integer>"; exit 2 ;; esac
  done
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
# Optional static split: OM_RLZERO_ONLY_FAMILIES="math500/s0 mbpp/s0 math500/s1"
# makes this node touch only those families (still lock-protected). Other
# families are neither claimed nor waited for; the final collection still
# requires all of them. Use it when you want one node = one fixed list.
ONLY_FAMILIES="${OM_RLZERO_ONLY_FAMILIES:-}"
PARALLEL_CONTROL=${OM_RLZERO_PARALLEL_CONTROL:-0}
if [ "$RUN_ROLE" != auto ]; then
  ONLY_FAMILIES="$TARGET_DATASET/s$TARGET_SEED"
  PARALLEL_CONTROL=1
fi
case "$PARALLEL_CONTROL" in 0|1) ;; *) echo '[abort] OM_RLZERO_PARALLEL_CONTROL must be 0 or 1'; exit 2 ;; esac
# `run h100 <dataset>` is the phone-typable form of the same split: this node
# handles only that dataset's families.
if [ "$MODE" = run ] && [ "$RUN_ROLE" = auto ] && [ -n "${3:-}" ]; then
  case " ${DATASETS[*]} " in
    *" $3 "*) ;;
    *) echo "[abort] unknown dataset filter: $3 (expected one of: ${DATASETS[*]})"; exit 2 ;;
  esac
  ONLY_FAMILIES=$(for seed in "${SEEDS[@]}"; do printf '%s/s%s ' "$3" "$seed"; done)
fi
validate_dataset_overrides OM_RLZERO_GEN_BATCH_BY_DATASET "$GEN_BATCH_BY_DATASET"
validate_dataset_overrides OM_RLZERO_GRADIENT_MICRO_BATCH_BY_DATASET "$GRADIENT_MICRO_BATCH_BY_DATASET"
family_selected() {  # family_selected <dataset> <seed>
  [ -z "$ONLY_FAMILIES" ] && return 0
  case " $ONLY_FAMILIES " in *" $1/s$2 "*) return 0 ;; esac
  return 1
}
if [ -n "$ONLY_FAMILIES" ]; then
  for fam in $ONLY_FAMILIES; do
    ok=0
    for seed in "${SEEDS[@]}"; do for dataset in "${DATASETS[@]}"; do
      [ "$fam" = "$dataset/s$seed" ] && ok=1
    done; done
    [ "$ok" -eq 1 ] || { echo "[abort] OM_RLZERO_ONLY_FAMILIES has unknown family: $fam (expected e.g. math500/s0)"; exit 2; }
  done
  echo "[queue] this node handles only: $ONLY_FAMILIES"
fi
DRIFTS=($(experiment_field drifts))
N_VAL=$(experiment_field n_val)
MODEL_TAG="${OM_OLMO3_MODEL_TAG:-$DEFAULT_MODEL_TAG}"
ROOT="${OM_OLMO3_ROOT:-$OM_WORK/runs/$MODEL_TAG}"
GLOBAL_RESULTS="${OM_OLMO3_RESULTS:-$OM_WORK/results/$MODEL_TAG}"
QUEUE="$ROOT/.families"
PREFLIGHT="$ROOT/preflight"
if [ "$RUN_ROLE" != auto ]; then
  [ -s "$ROOT/.queue/generation.git" ] && [ -d "$ROOT/family-$TARGET_DATASET-s$TARGET_SEED" ] || {
    echo "[abort] $RUN_ROLE requires an existing family and generation pin; no new matrix will be created"
    exit 2
  }
  if [ "$RUN_ROLE" = assist ] && [ -s "$ROOT/family-$TARGET_DATASET-s$TARGET_SEED/$MODEL_TAG-s$TARGET_SEED-$TARGET_DATASET-d0/DONE" ]; then
    echo "[assist-complete] $ONLY_FAMILIES d0 already has DONE; no GPU work started"
    exit 0
  fi
fi

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
  STATUS_LOG_LINES="${OM_RLZERO_STATUS_LOG_LINES:-20}"
  STATUS_ERROR_LINES="${OM_RLZERO_STATUS_ERROR_LINES:-6}"
  STATUS_PROBE_SECONDS="${OM_RLZERO_STATUS_PROBE_SECONDS:-20}"
  STATUS_STUCK_SECONDS="${OM_RLZERO_STATUS_STUCK_SECONDS:-1800}"
  STATUS_WORKER_STALE_SECONDS="${OM_RLZERO_STATUS_WORKER_STALE_SECONDS:-180}"
  STATUS_HEARTBEAT_STALE_SECONDS="${OM_RLZERO_STATUS_HEARTBEAT_STALE_SECONDS:-300}"
  STATUS_EXPECTED_WORKERS="${OM_RLZERO_STATUS_EXPECTED_WORKERS:-3}"
  # One screen by default; `status <profile> verbose` (or OM_RLZERO_STATUS_VERBOSE=1)
  # appends per-point rows, telemetry and log tails.
  STATUS_VERBOSE_FLAG=()
  if [ "${3:-}" = verbose ] || [ "${OM_RLZERO_STATUS_VERBOSE:-0}" = 1 ]; then
    STATUS_VERBOSE_FLAG=(--verbose)
  fi
  # status must expect the same per-dataset runtime values the launcher applies;
  # otherwise every repaired point reads as a contract mismatch.
  STATUS_DATASET_FLAGS=()
  for dataset in "${DATASETS[@]}"; do
    STATUS_DATASET_FLAGS+=(--dataset-generation-batch "$dataset=$(gen_batch_for "$dataset")")
    STATUS_DATASET_FLAGS+=(--dataset-gradient-micro-batch "$dataset=$(gradient_micro_batch_for "$dataset")")
  done
  for value_name in STATUS_LOG_LINES STATUS_ERROR_LINES STATUS_STUCK_SECONDS \
      STATUS_WORKER_STALE_SECONDS STATUS_HEARTBEAT_STALE_SECONDS \
      STATUS_EXPECTED_WORKERS; do
    value=${!value_name}
    case "$value" in
      ''|*[!0-9]*|0) echo "[abort] invalid $value_name=$value"; exit 2 ;;
    esac
  done
  case "$STATUS_PROBE_SECONDS" in
    ''|*[!0-9]*) echo "[abort] invalid STATUS_PROBE_SECONDS=$STATUS_PROBE_SECONDS"; exit 2 ;;
  esac
  # Every status run is appended to a shared history file so "was it alive at
  # 03:00?" can be answered later from any node: $ROOT/logs/status-history.log
  STATUS_HISTORY="$ROOT/logs/status-history.log"
  mkdir -p "$ROOT/logs" || exit 1
  STATUS_CAPTURE=$(mktemp "$ROOT/logs/.status.XXXXXX") || exit 1
  { printf '\n===== status %s host=%s =====\n' "$(date -u +%FT%TZ)" "$(hostname)";
    printf 'status_checkout=%s primary_generation=%s\n' "$(git -C "$SUPERVISOR_REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)" "$(cat "$ROOT/.queue/generation.git" 2>/dev/null || echo unknown)";
  } | tee "$STATUS_CAPTURE"
  "$PY" "$SUPERVISOR_REPO/src/rlzero_status.py" \
    --profile "$PROFILE" --root "$ROOT" --results "$GLOBAL_RESULTS" \
    --model-tag "$MODEL_TAG" --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" --drifts "${DRIFTS[@]}" \
    --probe-seconds "$STATUS_PROBE_SECONDS" \
    --stuck-seconds "$STATUS_STUCK_SECONDS" \
    --worker-stale-seconds "$STATUS_WORKER_STALE_SECONDS" \
    --heartbeat-stale-seconds "$STATUS_HEARTBEAT_STALE_SECONDS" \
    --expected-workers "$STATUS_EXPECTED_WORKERS" \
    --config-sha "$CONFIG_SHA" --model-revision "$MODEL_REVISION" \
    --generation-batch "$(runtime_field generation_batch)" \
    --gradient-micro-batch "$(runtime_field gradient_micro_batch)" \
    --logprob-micro-batch "$(runtime_field logprob_micro_batch)" \
    --min-recovery-generation-batch "$RECOVERY_MIN_GENERATION_BATCH" \
    "${STATUS_DATASET_FLAGS[@]}" \
    --log-lines "$STATUS_LOG_LINES" --error-lines "$STATUS_ERROR_LINES" \
    "${STATUS_VERBOSE_FLAG[@]}" 2>&1 | tee -a "$STATUS_CAPTURE"
  rc=${PIPESTATUS[0]}
  if [ -f "$SUPERVISOR_REPO/scripts/reference_status.sh" ]; then
    bash "$SUPERVISOR_REPO/scripts/reference_status.sh" "$OM_WORK" 2>&1 | tee -a "$STATUS_CAPTURE"
  else
    echo 'reference_status=UNAVAILABLE (update this status checkout)' | tee -a "$STATUS_CAPTURE"
  fi
  { flock 9 && cat "$STATUS_CAPTURE" >> "$STATUS_HISTORY"; } 9>"$STATUS_HISTORY.lock" || rc=1
  rm -f "$STATUS_CAPTURE"
  echo "history $STATUS_HISTORY"
  exit "$rc"
fi

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

  mkdir -p "$LOCAL_ROOT/olmo3-preflight" "$ROOT/logs"
  PRIMARY_LOCK="$LOCAL_ROOT/primary.lock"
  exec 8>"$PRIMARY_LOCK"
  if ! flock -n 8; then
    if [ "$RUN_ROLE" != auto ]; then
      echo "[abort] this node still has an experiment owner; stop its previous launcher before $RUN_ROLE. No process was terminated."
      exit 75
    fi
    # 2026-09-10: this branch used to terminate whatever held the node lock,
    # which on a node running Qwen on purpose (bash scripts/run_qwen35_9b.sh)
    # killed that launcher and its rollouts. A live launcher is never stopped
    # from here; only orphans (GPU processes whose launcher is gone) are.
    lock_holders=$("$PY" "$SUPERVISOR_RUNTIME_REPO/src/cleanup_run_processes.py" --list \
      --run-prefix "$ROOT" --open-file "$PRIMARY_LOCK") || exit 1
    live_launcher=$(printf '%s\n' "$lock_holders" \
      | grep -E 'scripts/(run_[a-z0-9_]+|go_[a-z0-9_]+)\.sh' | head -3)
    if [ -n "$live_launcher" ]; then
      echo "[abort] this node is running another experiment; nothing was stopped:"
      printf '%s\n' "$live_launcher" | cut -c1-160 | sed 's/^/        pid /'
      echo "        If this node should run OLMo3 instead, Ctrl-C that launcher yourself first."
      exit 75
    fi
    echo "[startup-cleanup] orphaned processes own the node lock (no launcher alive); terminating them"
    "$PY" "$SUPERVISOR_RUNTIME_REPO/src/cleanup_run_processes.py" \
      --run-prefix "$ROOT" --timeout "${OM_RLZERO_STALE_PROCESS_TIMEOUT:-15}" \
      --open-file "$PRIMARY_LOCK" || exit 1
    flock -w 5 8 || {
      echo "[abort] another experiment already owns this physical node; cleanup could not release it"
      exit 1
    }
  fi

  HOST_TAG=$(hostname 2>/dev/null || printf node)
  WORKER_SUFFIX=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || printf '%s' "$$")
  WORKER_ID=$(printf '%s-%s' "$HOST_TAG" "$WORKER_SUFFIX" | tr -cs 'a-zA-Z0-9._-' '-')
  export WORKER_ID
  LOG="$ROOT/logs/$WORKER_ID.log"
  echo "[worker] id=$WORKER_ID root=$ROOT" | tee -a "$LOG"
  echo "[worker-role] role=$RUN_ROLE families=${ONLY_FAMILIES:-all} parallel_control=$PARALLEL_CONTROL supervisor=$CURRENT_GIT" | tee -a "$LOG"

  STALE_PROCESS_TIMEOUT="${OM_RLZERO_STALE_PROCESS_TIMEOUT:-15}"
  GPU_CLEANUP_TIMEOUT="${OM_RLZERO_GPU_CLEANUP_TIMEOUT:-15}"
  for value_name in STALE_PROCESS_TIMEOUT GPU_CLEANUP_TIMEOUT; do
    value=${!value_name}
    case "$value" in
      ''|*[!0-9]*|0) echo "[abort] invalid $value_name=$value" | tee -a "$LOG"; exit 2 ;;
    esac
  done

  cleanup_stale_experiment_processes() {
    if [ "$RUN_ROLE" != auto ]; then
      "$PY" "$SUPERVISOR_RUNTIME_REPO/src/cleanup_run_processes.py" \
        --run-prefix "$(family_root "$TARGET_DATASET" "$TARGET_SEED")" \
        --timeout "$STALE_PROCESS_TIMEOUT" --require-environment "OM_NODE_NAMESPACE=$LOCAL_ROOT"
      return "$?"
    fi
    "$PY" "$SUPERVISOR_RUNTIME_REPO/src/cleanup_run_processes.py" \
      --run-prefix "$ROOT" --timeout "$STALE_PROCESS_TIMEOUT" \
      --require-environment "OM_NODE_NAMESPACE=$LOCAL_ROOT" \
      --command-pattern 'scripts/run_olmo3_rlzero.sh run' \
      --command-pattern 'scripts/run_matrix.sh' \
      --command-pattern 'scripts/run_point.sh' \
      --command-pattern 'src/train_policy_grpo.py' \
      --command-pattern 'src/experiment.py' \
      --command-pattern 'scripts/gpu_keepalive.py'
  }

  gpu_compute_pids() {
    timeout 20 nvidia-smi --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk '$1 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, "", $1); print $1}'
  }

  cleanup_node_gpu_processes() {
    local output pid owner deadline current_uid remaining="" terminated=0
    local pids=()
    current_uid=$(id -u)
    output=$(gpu_compute_pids) || {
      echo "[abort] nvidia-smi compute-process query failed during cleanup" | tee -a "$LOG"
      return 1
    }
    [ -z "$output" ] || mapfile -t pids <<< "$output"
    for pid in "${pids[@]}"; do
      owner=$(stat -c %u "/proc/$pid" 2>/dev/null || true)
      [ "$owner" = "$current_uid" ] || continue
      kill -TERM "$pid" 2>/dev/null || true
      terminated=$((terminated + 1))
    done
    if [ "$terminated" -gt 0 ]; then
      echo "[startup-cleanup] TERM sent to $terminated stale GPU compute processes" | tee -a "$LOG"
    fi

    deadline=$((SECONDS + GPU_CLEANUP_TIMEOUT))
    while [ "$SECONDS" -lt "$deadline" ]; do
      output=$(gpu_compute_pids) || return 1
      remaining=""
      while IFS= read -r pid; do
        [ -n "$pid" ] || continue
        owner=$(stat -c %u "/proc/$pid" 2>/dev/null || true)
        [ "$owner" = "$current_uid" ] && remaining="$remaining $pid"
      done <<< "$output"
      [ -n "$remaining" ] || break
      /bin/sleep 1
    done
    for pid in $remaining; do
      kill -KILL "$pid" 2>/dev/null || true
    done
    [ -z "$remaining" ] || {
      echo "[startup-cleanup] KILL sent to remaining GPU processes:$remaining" | tee -a "$LOG"
      /bin/sleep 1
    }

    output=$(gpu_compute_pids) || return 1
    if [ -n "$output" ]; then
      echo "[abort] GPU compute processes remain after full cleanup: $(printf '%s' "$output" | tr '\n' ' ')" \
        | tee -a "$LOG"
      timeout 20 nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader 2>&1 | tee -a "$LOG" || true
      return 1
    fi
    echo "[startup-cleanup] all GPU compute contexts cleared" | tee -a "$LOG"
  }

  cleanup_stale_experiment_processes 2>&1 | tee -a "$LOG"
  statuses=("${PIPESTATUS[@]}")
  [ "${statuses[0]}" -eq 0 ] && [ "${statuses[1]}" -eq 0 ] || exit 1
  # Explicit roles clean only their target-family orphans, never arbitrary GPU
  # processes. The memory/admission checks below reject a still-busy node.
  if [ "$RUN_ROLE" = auto ]; then
    cleanup_node_gpu_processes || exit 1
  else
    remaining_gpu_pids=$(gpu_compute_pids) || exit 1
    [ -z "$remaining_gpu_pids" ] || {
      echo "[abort] GPU processes remain on this node: $remaining_gpu_pids; explicit roles do not kill unrelated compute"
      exit 75
    }
  fi

  memory=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) || exit 1
  rows=$(printf '%s\n' "$memory" | awk 'NF {n++} END {print n+0}')
  busy=$(printf '%s\n' "$memory" | awk '$1 > 2000 {n++} END {print n+0}')
  [ "$rows" -eq 4 ] && [ "$busy" -eq 0 ] || {
    echo "[abort] GPUs are already in use; refusing to overlap another experiment"
    printf '%s\n' "$memory"
    exit 1
  }

  # Keep all allocated GPUs active before model/dataset verification begins.
  # This also spans signal qualification, point transitions, CPU verification,
  # queue waits, and final result collection.
  ACTIVE_OWNER=""
  SUPERVISOR_KEEPALIVE=""
  KEEPALIVE_READY="$LOCAL_ROOT/keepalive-$WORKER_ID.ready"
  WORKER_HEARTBEAT_PID=""
  WORKER_HEARTBEAT_PATH="$ROOT/.workers/$WORKER_ID.json"
  WORKER_HEARTBEAT_SECONDS="${OM_RLZERO_HEARTBEAT_SECONDS:-15}"
  case "$WORKER_HEARTBEAT_SECONDS" in
    ''|*[!0-9]*|0) echo "[abort] invalid OM_RLZERO_HEARTBEAT_SECONDS=$WORKER_HEARTBEAT_SECONDS"; exit 2 ;;
  esac
  stop_worker_heartbeat() {
    if [ -n "${WORKER_HEARTBEAT_PID:-}" ]; then
      kill "$WORKER_HEARTBEAT_PID" 2>/dev/null || true
      wait "$WORKER_HEARTBEAT_PID" 2>/dev/null || true
    fi
    WORKER_HEARTBEAT_PID=""
    rm -f -- "$WORKER_HEARTBEAT_PATH"
  }
  start_worker_heartbeat() {
    mkdir -p "$ROOT/.workers"
    rm -f -- "$WORKER_HEARTBEAT_PATH"
    # The heartbeat also watches the other workers and prints
    # "[WORKER DEAD] ..." here (terminal + $LOG) and to $ROOT/logs/ALERTS.log.
    # Alerts go to this worker's log, to the shared ALERTS.log and, when the
    # launcher runs on a terminal, straight to that terminal. No pipeline here:
    # $! must stay the heartbeat's own pid.
    # It also prints a [progress] line every OM_RLZERO_PROGRESS_SECONDS (10 min)
    # from durable artifacts only (DONE points, GRPO steps, rollout bytes, last
    # write) and shouts [NOT TRAINING] when nothing durable changed for
    # OM_PROGRESS_STALL_MINUTES (30): a live heartbeat is not progress (2026-09-07).
    "$PY" "$SUPERVISOR_RUNTIME_REPO/src/rlzero_heartbeat.py" \
      --path "$WORKER_HEARTBEAT_PATH" --worker "$WORKER_ID" \
      --host "$HOST_TAG" --launcher-pid "$$" \
      --interval-seconds "$WORKER_HEARTBEAT_SECONDS" \
      --peer-stale-seconds "${OM_RLZERO_PEER_STALE_SECONDS:-300}" \
      --alerts-log "$ROOT/logs/ALERTS.log" --worker-log "$LOG" \
      --terminal "$( { tty; } 2>/dev/null || true)" \
      --progress-root "$ROOT" --progress-seconds "${OM_RLZERO_PROGRESS_SECONDS:-600}" \
      --total-points "$(( ${#SEEDS[@]} * ${#DATASETS[@]} * ${#DRIFTS[@]} ))" \
      >/dev/null 2>&1 &
    WORKER_HEARTBEAT_PID=$!
    for _ in $(seq 1 50); do
      [ -s "$WORKER_HEARTBEAT_PATH" ] && return 0
      kill -0 "$WORKER_HEARTBEAT_PID" 2>/dev/null || break
      sleep 0.1
    done
    echo "[abort] worker heartbeat failed to start" | tee -a "$LOG"
    stop_worker_heartbeat
    return 1
  }
  stop_supervisor_keepalive() {
    if [ -n "${SUPERVISOR_KEEPALIVE:-}" ]; then
      kill "$SUPERVISOR_KEEPALIVE" 2>/dev/null || true
      wait "$SUPERVISOR_KEEPALIVE" 2>/dev/null || true
    fi
    SUPERVISOR_KEEPALIVE=""
    rm -f -- "$KEEPALIVE_READY"
  }
  start_supervisor_keepalive() {
    if [ -n "${SUPERVISOR_KEEPALIVE:-}" ]; then
      kill -0 "$SUPERVISOR_KEEPALIVE" 2>/dev/null && return 0
      wait "$SUPERVISOR_KEEPALIVE" 2>/dev/null || true
      SUPERVISOR_KEEPALIVE=""
    fi
    rm -f -- "$KEEPALIVE_READY"
    # OM_GPU_KEEPALIVE_DUTY (percent, default 15, max 50) is the share of
    # wall time the keepalive keeps an otherwise idle GPU busy; raise it if
    # the cluster judges a job by its utilization.
    OM_GPU_KEEPALIVE_READY_FILE="$KEEPALIVE_READY" CUDA_VISIBLE_DEVICES=0,1,2,3 \
      "$PY" "$SUPERVISOR_RUNTIME_REPO/scripts/gpu_keepalive.py" "${OM_GPU_KEEPALIVE_DUTY:-15}" \
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
    grep -Fq 'gpus=4' "$KEEPALIVE_READY" || {
      echo "[abort] GPU keepalive did not create four healthy CUDA contexts" | tee -a "$LOG"
      stop_supervisor_keepalive
      return 1
    }
    echo "[worker] fresh CUDA contexts passed on all four GPUs; keepalive pid=$SUPERVISOR_KEEPALIVE" \
      | tee -a "$LOG"
  }
  cleanup_worker() {
    local rc=$?
    [ -z "${ACTIVE_OWNER:-}" ] || rm -f -- "$ACTIVE_OWNER"
    ACTIVE_OWNER=""
    stop_supervisor_keepalive
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 130 ] && [ "$rc" -ne 143 ]; then
      # Abnormal exit: keep a "crashed" record so every other worker shouts
      # [WORKER DEAD]. A clean exit or Ctrl-C removes the record silently.
      mark_worker_crashed "$rc"
    else
      stop_worker_heartbeat
    fi
  }
  mark_worker_crashed() {
    if [ -n "${WORKER_HEARTBEAT_PID:-}" ]; then
      kill "$WORKER_HEARTBEAT_PID" 2>/dev/null || true
      wait "$WORKER_HEARTBEAT_PID" 2>/dev/null || true
    fi
    WORKER_HEARTBEAT_PID=""
    "$PY" - "$WORKER_HEARTBEAT_PATH" "$WORKER_ID" "$HOST_TAG" "$1" <<'PYEOF' 2>/dev/null || true
import json, sys, time
from pathlib import Path
path = Path(sys.argv[1])
try:
    record = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    record = {"schema": "offpolicy-worker-heartbeat/v1", "worker": sys.argv[2], "host": sys.argv[3]}
record.update({"state": "crashed", "exit_code": int(sys.argv[4]), "heartbeat_at_ns": time.time_ns()})
path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
PYEOF
  }
  trap cleanup_worker EXIT
  # A dropped SSH session (phone) sends SIGHUP; that must not end a multi-day
  # worker. Only Ctrl-C / kill stop it. Two workers were lost this way.
  trap '' HUP
  trap 'exit 130' INT TERM
  start_worker_heartbeat || exit 1
  start_supervisor_keepalive || exit 1
  export OM_EXTERNAL_GPU_KEEPALIVE=1
fi

# Adopt separately uploaded assets before entering an older commit-pinned run.
# This creates the standard manifests/paths that the pinned code can read.
(
  flock 9
  if [ ! -s "$MODEL_PATH/.om_snapshot.json" ]; then
    if ! PYTHONPATH="$SUPERVISOR_RUNTIME_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$PY" "$SUPERVISOR_RUNTIME_REPO/src/model_matrix.py" --config "$CONFIG" \
        --models-dir "$MODELS_DIR" --snapshot-path "$MODEL_PATH" check "$MODEL_KEY"; then
      echo "[model] manifest missing; verifying the uploaded model against pinned official hashes"
      PYTHONPATH="$SUPERVISOR_RUNTIME_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$PY" "$SUPERVISOR_RUNTIME_REPO/src/model_matrix.py" --config "$CONFIG" \
        --models-dir "$MODELS_DIR" --snapshot-path "$MODEL_PATH" seal "$MODEL_KEY" || exit 1
    fi
  fi
  OM_MATH_VERIFIER=math_verify \
    PYTHONPATH="$SUPERVISOR_RUNTIME_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" "$SUPERVISOR_RUNTIME_REPO/src/qualify_domain_data.py" "${DATASETS[@]}" \
    --data-root "$DATASETS_DIR" --n-train 512 \
    --dataset-n-train math500=400 --dataset-n-train mbpp=512 \
    --n-val "$N_VAL" --seeds "${SEEDS[@]}" \
    --output "$PREFLIGHT/data-adoption.json" || exit 1
) 9>"$OM_WORK/locks/olmo3-asset-adoption.lock" || exit 1

GENERATION_ADVANCE=()
[ "$RUN_ROLE" != auto ] || GENERATION_ADVANCE=(--advance-empty)
GENERATION_GIT=$("$PY" "$SUPERVISOR_RUNTIME_REPO/src/regime_resume_commit.py" \
  "$ROOT" "$CURRENT_GIT" \
  --marker "$ROOT/.queue/generation.git" "${GENERATION_ADVANCE[@]}") || exit 1

git -C "$SUPERVISOR_RUNTIME_REPO" cat-file -e "$GENERATION_GIT^{commit}" 2>/dev/null || {
  echo "[abort] pinned generation commit is unavailable locally: $GENERATION_GIT"
  exit 1
}
GENERATION_REPO=$(materialize_local_checkout "$GENERATION_GIT") || exit 1
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
export GRADIENT_MICRO_BATCH="$(runtime_field gradient_micro_batch)"
export GRPO_LOGPROB_MICRO_BATCH="$(runtime_field logprob_micro_batch)"
export REGIME_RECOVERY_MIN_BATCH="$RECOVERY_MIN_GENERATION_BATCH"
export GRPO_GRADIENT_CHECKPOINTING="$(runtime_field gradient_checkpointing)"

SIGNAL_TIMEOUT_SECONDS="${OM_RLZERO_SIGNAL_TIMEOUT_SECONDS:-1800}"
SMOKE_TIMEOUT_SECONDS="${OM_RLZERO_SMOKE_TIMEOUT_SECONDS:-900}"
PREFLIGHT_KILL_GRACE_SECONDS="${OM_RLZERO_PREFLIGHT_KILL_GRACE_SECONDS:-30}"
for value_name in SIGNAL_TIMEOUT_SECONDS SMOKE_TIMEOUT_SECONDS PREFLIGHT_KILL_GRACE_SECONDS; do
  value=${!value_name}
  case "$value" in
    ''|*[!0-9]*|0) echo "[abort] invalid $value_name=$value" | tee -a "$LOG"; exit 2 ;;
  esac
done

run_timed_preflight() {  # run_timed_preflight <label> <seconds> <command...>
  local label=$1 seconds=$2 rc
  local statuses=()
  shift 2
  echo "[preflight] $label start; timeout=${seconds}s" | tee -a "$LOG"
  timeout --signal=TERM --kill-after="${PREFLIGHT_KILL_GRACE_SECONDS}s" \
    "${seconds}s" "$@" 8>&- 9>&- 2>&1 | tee -a "$LOG" 8>&- 9>&-
  statuses=("${PIPESTATUS[@]}")
  rc=${statuses[0]}
  [ "${statuses[1]}" -eq 0 ] || return "${statuses[1]}"
  if [ "$rc" -eq 124 ]; then
    echo "[preflight-timeout] $label exceeded ${seconds}s; process group terminated" \
      | tee -a "$LOG"
  elif [ "$rc" -ne 0 ]; then
    echo "[preflight-fail] $label rc=$rc" | tee -a "$LOG"
  else
    echo "[preflight] $label passed" | tee -a "$LOG"
  fi
  return "$rc"
}

signal_qualify() {
  local dataset=$1 report="$PREFLIGHT/$1-signal.json"
  run_timed_preflight "signal-$dataset" "$SIGNAL_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES=0 \
      PYTHONPATH="$GENERATION_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY" "$GENERATION_REPO/src/qualify_rlzero_signal.py" \
      --model "$MODEL_PATH" --dataset "$dataset" --data-root "$DATASETS_DIR" \
      --output "$report" --prompt-count 8 --group-size 8 \
      --max-new-tokens 1024 --generation-batch "$OM_GEN_BATCH" \
      --gradient-micro-batch "$GRADIENT_MICRO_BATCH" \
      --grad-layers "$(experiment_field grad_layers)"
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
  run_timed_preflight "grpo-smoke-step1" "$SMOKE_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES=0,1,2,3 \
      PYTHONPATH="$GENERATION_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
      "$GENERATION_REPO/src/train_policy_grpo.py" "${common[@]}" \
      --output "$SMOKE_ROOT/step1" --target-steps 1 || exit 1
  run_timed_preflight "grpo-smoke-step2" "$SMOKE_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES=0,1,2,3 \
      PYTHONPATH="$GENERATION_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
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
export REGIME_MAX_RETRIES="${REGIME_MAX_RETRIES:-3}"
export OM_STALL_MINUTES="${OM_STALL_MINUTES:-15}"
export REGIME_HARD_STALL_SECONDS="${REGIME_HARD_STALL_SECONDS:-${OM_RLZERO_HARD_STALL_SECONDS:-1800}}"
export OM_SKIP_GPU_CHECK=0 OM_ALLOW_DIRTY=0 OM_ALLOW_ANALYSIS_UPGRADE=1
# Point retries are already bounded by run_matrix. Rotate families before
# claiming another full retry budget for the same failure.
FAMILY_ATTEMPTS="${OM_RLZERO_FAMILY_ATTEMPTS:-1}"
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

# ---- failure-loop guard -------------------------------------------------------
# A point that dies of the same error every attempt (e.g. OOM in a stage) used to
# be retried forever: die, restart, die, restart, for days. After
# OM_RLZERO_MAX_FAMILY_FAILURES consecutive failures the family is marked
# `.families/<dataset>-s<seed>.loop` (shared), every worker skips it, status shows
# LOOPING with the error, and the operator decides. Clear with
# OM_RLZERO_CLEAR_LOOPS=1 on the next launch after fixing the cause.
MAX_FAMILY_FAILURES="${OM_RLZERO_MAX_FAMILY_FAILURES:-4}"
# CUDA runtime faults (unspecified launch failure, CUBLAS execution failure,
# device-side assert) hit every node of this cluster a few times a day and
# recover on retry. They are not the same failure repeating: they must not
# count toward the loop guard, or a nearly finished family gets marked LOOPING
# within an hour (math500/s1 and s2, 2026-09-07 03:20Z). Such a failure is
# retried with a growing pause; after MAX_CUDA_FAILURES in a row on this
# worker the family is released for another node instead of being marked.
MAX_CUDA_FAILURES="${OM_RLZERO_MAX_CUDA_FAILURES:-8}"
# Lifetime bound on the CUDA-fault exemption: a fault that reproduces every time
# is not transient, whatever its message says (2026-09-08).
MAX_RUNTIME_FAILURES="${OM_RLZERO_MAX_RUNTIME_FAILURES:-24}"
declare -A FAMILY_FAILURES=()
declare -A FAMILY_CUDA_FAILURES=()
# Earliest time (epoch seconds) at which this worker attempts a family again after
# a failure. A failed family is never retried on the spot: the worker walks on to
# the next family and comes back after this cooldown (2026-09-08).
declare -A FAMILY_NEXT_ATTEMPT=()
declare -A CONTROL_NEXT_ATTEMPT=()
now_seconds() { printf '%s\n' "${EPOCHSECONDS:-$(date +%s)}"; }
failure_backoff_seconds() {  # failure_backoff_seconds <consecutive failures>
  local n=$1 pause
  [ "$n" -ge 1 ] || n=1
  [ "$n" -le 8 ] || n=8
  pause=$(( FAMILY_RETRY_SECONDS * (1 << (n - 1)) ))
  [ "$pause" -le 900 ] || pause=900
  printf '%s\n' "$pause"
}
LAST_FAILURE_KIND=""
loop_marker() { printf '%s/%s-s%s.loop\n' "$QUEUE" "$1" "$2"; }
point_runtime_failures() {  # point_runtime_failures <dataset> <seed> -> failed tries of the point that failed last
  # 2026-09-10: the CUDA-fault bound counted every [point-failed] line of every
  # point in the family for all time, so a family with a long history (mbpp/s4:
  # d0 died on 2026-09-06, d25 lost its node twice) reached the bound on its
  # first fault of the day and was marked LOOPING while its point was healthy.
  # Count only the point whose supervisor log is newest, only lines after its
  # last [point-accepted], and nothing at all for a point that already has DONE.
  local supervisor point
  supervisor=$(ls -t "$(family_root "$1" "$2")"/*/logs/supervisor.log 2>/dev/null | head -1)
  [ -n "$supervisor" ] || { printf '0\n'; return 0; }
  point=$(dirname "$(dirname "$supervisor")")
  [ ! -s "$point/DONE" ] || { printf '0\n'; return 0; }
  awk '/\[point-accepted\]/ {n = 0; next} /\[point-failed\]/ {n++} END {print n + 0}' "$supervisor"
}
remaining_families_summary() {  # who holds each unfinished family this worker may take
  local dataset seed owner host claimed parts=""
  while read -r dataset seed; do
    [ -n "$dataset" ] && [ -n "$seed" ] || continue
    family_selected "$dataset" "$seed" || continue
    family_complete "$dataset" "$seed" && continue
    owner="$QUEUE/$dataset-s$seed.owner.json"
    if family_looping "$dataset" "$seed"; then
      parts+="; $dataset/s$seed marked LOOPING"
    elif [ -s "$owner" ]; then
      host=$(sed -n 's/.*"host": "\([^"]*\)".*/\1/p' "$owner" | head -1)
      claimed=$(sed -n 's/.*"claimed_at_utc": "\([^"]*\)".*/\1/p' "$owner" | head -1 | cut -c1-16)
      parts+="; $dataset/s$seed running on ${host:-?} since ${claimed:-?}Z"
    else
      parts+="; $dataset/s$seed unowned"
    fi
  done < <(ordered_families)
  printf '%s\n' "${parts#; }"
}
family_last_error() {  # family_last_error <dataset> <seed> -> last error line (may be empty)
  # 2026-09-08: judge the attempt that just failed, not the newest error line
  # anywhere in the family's logs. A stale CUDA line from an earlier attempt made
  # every later failure look like a CUDA fault, which is exempt from the loop
  # guard, so a family that failed at its completion check was re-claimed by the
  # same worker for ever. run_matrix.sh writes the failed attempt's own error
  # line to <run>/logs/supervisor.log ([point-failed] ... rc=N: <line>).
  local froot supervisor line
  froot=$(family_root "$1" "$2")
  supervisor=$(ls -t "$froot"/*/logs/supervisor.log 2>/dev/null | head -1)
  if [ -n "$supervisor" ]; then
    line=$(grep -E '\[(point-failed|done-but-incomplete)\]' "$supervisor" 2>/dev/null | tail -1 \
      | sed -E 's/^\[[^]]*\] \[(point-failed|done-but-incomplete)\] //')
    case "$line" in
      '') ;;
      *) printf '%s\n' "$line" | cut -c1-200; return 0 ;;
    esac
  fi
  grep -hE 'OutOfMemoryError|CUDA error|CUBLAS_STATUS|device-side assert|RuntimeError|Error:|\[abort\]|\[config-abort\]' \
    "$froot"/*/logs/*.log 2>/dev/null | tail -1 | cut -c1-200
}
failure_kind() {  # failure_kind "<error line>" -> oom | runtime | other
  "$PY" - "$1" "$SUPERVISOR_RUNTIME_REPO/src" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[2])
from recovery_policy import classify_cuda_failure
print(classify_cuda_failure(sys.argv[1]) or "other")
PYEOF
}
# Every worker clears, at startup, loop markers that recorded a CUDA runtime
# fault: those were written by the old rule that counted such faults, and the
# operator must not have to pick one node to relaunch differently.
# OM_RLZERO_CLEAR_LOOPS=1 clears every marker.
for marker in "$QUEUE"/*.loop; do
  [ -e "$marker" ] || continue
  if [ "${OM_RLZERO_CLEAR_LOOPS:-0}" = 1 ]; then
    rm -f -- "$marker" && echo "[queue] cleared loop marker $(basename "$marker") (OM_RLZERO_CLEAR_LOOPS=1)"
    continue
  fi
  # New permanent contract failures are not legacy CUDA/config-abort markers.
  grep -Eq '(^| )last_rc=43( |$)' "$marker" && continue
  marker_error=$(sed -n 's/^last_error=//p' "$marker" 2>/dev/null | head -1)
  if grep -q '^family=.*runtime_failures_total=' "$marker" 2>/dev/null; then
    marker_family=$(basename "$marker" .loop)
    marker_count=$(point_runtime_failures "${marker_family%-s*}" "${marker_family##*-s}")
    if [ "$marker_count" -ge "$MAX_RUNTIME_FAILURES" ]; then
      echo "[queue] keeping loop marker $(basename "$marker"): its current point failed $marker_count times, every one a CUDA runtime fault (bound $MAX_RUNTIME_FAILURES)" | tee -a "$LOG"
    else
      rm -f -- "$marker" \
        && echo "[queue] cleared loop marker $(basename "$marker"): it was written by the old family-wide count; the current point has failed $marker_count time(s), below the bound of $MAX_RUNTIME_FAILURES" | tee -a "$LOG"
    fi
  elif [ "$(failure_kind "$marker_error" 2>/dev/null || printf other)" = runtime ]; then
    rm -f -- "$marker" \
      && echo "[queue] cleared loop marker $(basename "$marker"): it recorded a CUDA runtime fault, which is retried, not a repeating failure" | tee -a "$LOG"
  elif [ -n "$marker_error" ] && [ -z "${marker_error##*config-abort*}" ]; then
    # 2026-09-08: a finished point re-entered under a different generation batch
    # died in the pinned pipeline; run_matrix.sh now aligns the record first.
    rm -f -- "$marker" \
      && echo "[queue] cleared loop marker $(basename "$marker"): it recorded a re-entry config-abort, which this code repairs before re-entry" | tee -a "$LOG"
  fi
done
family_looping() {  # family_looping <dataset> <seed>
  [ -s "$(loop_marker "$1" "$2")" ]
}
note_family_failure() {  # note_family_failure <dataset> <seed> <rc>; sets LAST_FAILURE_KIND
  local key="$1-s$2" n last_error run
  last_error=$(family_last_error "$1" "$2")
  LAST_FAILURE_KIND=$(failure_kind "$last_error" 2>/dev/null || printf other)
  # rc=43 is a permanent contract/postcondition failure from run_matrix.
  # It must not be reclassified by an older CUDA log or retried on a later claim.
  [ "$3" -ne 43 ] || LAST_FAILURE_KIND=other
  if [ "$LAST_FAILURE_KIND" = runtime ]; then
    n=$(( ${FAMILY_CUDA_FAILURES[$key]:-0} + 1 ))
    FAMILY_CUDA_FAILURES[$key]=$n
    echo "[cuda-flaky] $1/s$2: CUDA runtime fault #$n on this worker (${last_error:-no error line}); not counted as a repeating failure" | tee -a "$LOG"
    return 0
  fi
  FAMILY_CUDA_FAILURES[$key]=0
  n=$(( ${FAMILY_FAILURES[$key]:-0} + 1 ))
  FAMILY_FAILURES[$key]=$n
  [ "$3" -eq 43 ] || [ "$n" -ge "$MAX_FAMILY_FAILURES" ] || return 0
  {
    echo "family=$1/s$2 worker=$WORKER_ID host=$HOST_TAG consecutive_failures=$n last_rc=$3"
    echo "last_error=${last_error:-none captured}"
    echo "marked_at_utc=$(date -u +%FT%TZ)"
  } > "$(loop_marker "$1" "$2")"
  echo "[family-loop] $1/s$2 failed $n times in a row (last: ${last_error:-no error line captured}). Not retrying it any more on any worker. Fix the cause, then relaunch with OM_RLZERO_CLEAR_LOOPS=1." | tee -a "$LOG"
}

run_family() {
  local dataset=$1 seed=$2 control_only=${3:-0} root result format owner rc=1 attempt
  root=$(family_root "$dataset" "$seed")
  result=$(family_result "$dataset" "$seed")
  owner="$QUEUE/$dataset-s$seed.owner.json"
  [ "$control_only" = 0 ] || owner="$QUEUE/$dataset-s$seed.control-owner.json"
  format=olmo_rlzero_math
  [ "$dataset" = mbpp ] && format=olmo_rlzero_code
  ACTIVE_OWNER=$owner
  HOST_TAG="$HOST_TAG" WORKER_ID="$WORKER_ID" GENERATION_GIT="$GENERATION_GIT" \
    SUPERVISOR_GIT="$CURRENT_GIT" PARALLEL_CONTROL="$PARALLEL_CONTROL" \
    DATASET="$dataset" SEED="$seed" CONTROL_ONLY="$control_only" "$PY" - "$owner" <<'PYEOF'
import datetime, json, os, pathlib, sys
path = pathlib.Path(sys.argv[1])
tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
tmp.write_text(json.dumps({
    "host": os.environ["HOST_TAG"], "worker": os.environ["WORKER_ID"],
    "dataset": os.environ["DATASET"], "seed": int(os.environ["SEED"]),
    "generation_git": os.environ["GENERATION_GIT"],
    "supervisor_git": os.environ["SUPERVISOR_GIT"],
    "parallel_control": os.environ["PARALLEL_CONTROL"] == "1",
    "role": "control" if os.environ["CONTROL_ONLY"] == "1" else "family",
    "claimed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
}, sort_keys=True) + "\n")
tmp.replace(path)
PYEOF
  [ "$?" -eq 0 ] || { cleanup_owner; return 1; }
  local family_gen_batch family_micro_batch
  family_gen_batch=$(gen_batch_for "$dataset") || { cleanup_owner; return 1; }
  family_micro_batch=$(gradient_micro_batch_for "$dataset") || { cleanup_owner; return 1; }
  echo "[family] $dataset/s$seed runtime: generation batch=$family_gen_batch gradient micro-batch=$family_micro_batch" | tee -a "$LOG"
  # Points created under other runtime values are updated before re-entry: only
  # unfinished points, only these two fields, every change logged
  # (src/repair_run_config.py).
  if [ "$PARALLEL_CONTROL" = 0 ] && [ -d "$root" ] && compgen -G "$root/*/run_config.json" >/dev/null 2>&1; then
    "$PY" "$SUPERVISOR_RUNTIME_REPO/src/repair_run_config.py" --family-root "$root" \
      --gen-batch "$family_gen_batch" --gradient-micro-batch "$family_micro_batch" --apply \
      2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { cleanup_owner; return 1; }
  fi
  for ((attempt = 1; attempt <= FAMILY_ATTEMPTS; attempt++)); do
    echo "[family] claim=$dataset/s$seed attempt=$attempt/$FAMILY_ATTEMPTS" | tee -a "$LOG"
    # Older pinned generation commits start their own point-local keepalive.
    # Pause the supervisor during those points to avoid duplicate GPU load.
    [ "$PINNED_POINT_EXTERNAL_KEEPALIVE" -eq 1 ] || stop_supervisor_keepalive
    OM_GEN_BATCH="$family_gen_batch" GRADIENT_MICRO_BATCH="$family_micro_batch" \
      OM_REPO="$SUPERVISOR_RUNTIME_REPO" OM_PIPELINE_REPO="$GENERATION_REPO" \
      OM_PIPELINE_SCRIPT="$GENERATION_REPO/scripts/run_point.sh" \
      OM_GENERATION_GIT="$GENERATION_GIT" \
      PYTHONPATH="$SUPERVISOR_RUNTIME_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
      MODEL_PATH="$MODEL_PATH" \
      REGIME_ROOT="$root" REGIME_RESULTS="$result" REGIME_MODEL_TAG="$MODEL_TAG" \
      REGIME_DATASETS="$dataset" REGIME_SEEDS="$seed" \
      REGIME_DRIFTS="${DRIFTS[*]}" REGIME_SKIP_COLLECTION=1 \
      REGIME_PARALLEL_CONTROL="$PARALLEL_CONTROL" REGIME_CONTROL_ONLY="$control_only" \
      REGIME_YIELD_WHEN_BUSY="$PARALLEL_CONTROL" \
      OM_PROMPT_FORMAT="$format" \
      bash "$SUPERVISOR_RUNTIME_REPO/scripts/run_matrix.sh" 8>&- 9>&- 2>&1 \
        | tee -a "$LOG" 8>&- 9>&-
    statuses=("${PIPESTATUS[@]}")
    rc=${statuses[0]}
    if [ "${statuses[1]}" -ne 0 ]; then
      cleanup_owner
      return "${statuses[1]}"
    fi
    [ "$rc" -ne 0 ] || break
    [ "$rc" -ne 43 ] || break
    [ "$rc" -ne 75 ] || break
    [ "$attempt" -lt "$FAMILY_ATTEMPTS" ] || break
    sleep 30
  done
  if [ "$rc" -ne 0 ]; then
    cleanup_owner
    return "$rc"
  fi
  if [ "$control_only" = 1 ]; then
    echo "[control-assist] completed $dataset/s$seed/d0; GRPO chain and family collection remain separate" | tee -a "$LOG"
    cleanup_owner
    return 0
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

# ---- claim order and retry policy -------------------------------------------
# 2026-09-07: math500/s1 (three of four points done, last generation stage) sat
# unowned for nine hours: its worker failed once, moved on to a fresh family and
# stayed busy on it for a day, and every other worker was busy too. Two rules:
# (1) most-progressed family first, so resuming beats starting; (2) a family that
# fails is not abandoned to "the next free worker": this worker keeps it in its
# own rotation and comes back to it.
# 2026-09-08 correction: "keeps it" used to mean retrying the same family
# immediately, in place, for ever. One family that failed at the same step every
# time held a whole node all night. Now a failure moves the worker to the next
# family at once and the failed one is retried after a growing cooldown
# (FAMILY_RETRY_SECONDS doubling, capped at 900s) on a later pass. Work never
# stops because one family is broken.
family_points_done() {  # family_points_done <dataset> <seed> -> number of DONE points
  local drift n=0
  for drift in "${DRIFTS[@]}"; do
    [ -s "$(run_dir "$1" "$2" "$drift")/DONE" ] && n=$((n + 1))
  done
  printf '%s\n' "$n"
}
family_started() {  # family_started <dataset> <seed>: some point directory exists
  local root
  root=$(family_root "$1" "$2")
  [ -d "$root" ] && compgen -G "$root/*/run_config.json" >/dev/null 2>&1
}
family_remaining_work() {  # family_remaining_work <dataset> <seed> -> integer estimate of hours x100 still to run
  # Relative cost of the points still missing. Measured on the h100 matrix
  # (2026-09-08): a d400 point is about 1.7x a d0/d100 point (300 GRPO steps
  # plus the rollout), d25 about 0.8x, and an mbpp point about 1.35x a math500
  # point (512 prompts instead of 400, longer generations). Only the ORDER
  # matters, so rough weights are enough.
  local drift weight total=0 factor=100
  [ "$1" = mbpp ] && factor=135
  for drift in "${DRIFTS[@]}"; do
    [ -s "$(run_dir "$1" "$2" "$drift")/DONE" ] && continue
    case "$drift" in 25) weight=80 ;; 400) weight=170 ;; *) weight=100 ;; esac
    total=$((total + weight * factor / 100))
  done
  printf '%s\n' "$total"
}
# Claim order. Default "remaining": the family with the MOST work left is
# claimed first (longest-processing-time first), ties to a family that already
# has a directory, then registered order. With seven workers for eight families
# and no way to get a node back once the GPU manager takes it, the family that
# waits for a free worker must be the shortest one: leaving a 34-hour family
# unowned while a 16-hour one runs cost the whole matrix 16 hours
# (2026-09-08 evening, mbpp/s4). OM_RLZERO_CLAIM_ORDER=progress restores the
# 2026-09-07 rule (most finished points first).
CLAIM_ORDER="${OM_RLZERO_CLAIM_ORDER:-remaining}"
case "$CLAIM_ORDER" in remaining|progress) ;; *) echo "[abort] OM_RLZERO_CLAIM_ORDER must be remaining or progress, not $CLAIM_ORDER"; exit 2 ;; esac
ordered_families() {  # "<dataset> <seed>" lines in claim order
  local seed dataset started key
  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      started=0
      family_started "$dataset" "$seed" && started=1
      if [ "$CLAIM_ORDER" = progress ]; then
        key=$(family_points_done "$dataset" "$seed")
      else
        key=$(family_remaining_work "$dataset" "$seed")
      fi
      printf '%s %s %s %s\n' "$key" "$started" "$dataset" "$seed"
    done
  done | sort -s -k1,1nr -k2,2nr | awk '{print $3, $4}'
}

while :; do
  remaining=0
  claimed=0
  next_attempt_wait=0
  looping=0
  looping_list=""
  while read -r dataset seed; do
    [ -n "$dataset" ] && [ -n "$seed" ] || continue
    family_selected "$dataset" "$seed" || continue
    family_complete "$dataset" "$seed" && continue
    if [ "$RUN_ROLE" = assist ]; then
      [ -s "$(run_dir "$dataset" "$seed" 0)/DONE" ] || remaining=$((remaining + 1))
      continue
    fi
    if family_looping "$dataset" "$seed"; then
      remaining=$((remaining + 1))
      looping=$((looping + 1))
      looping_list+=" $dataset/s$seed"
      continue
    fi
    remaining=$((remaining + 1))
    # A family this worker failed on is not retried on the spot; another family
    # runs first. Without this a single broken family held a whole node
    # (2026-09-07 night: five workers re-entered the same point until morning).
    cooldown=$(( ${FAMILY_NEXT_ATTEMPT[$dataset-s$seed]:-0} - $(now_seconds) ))
    if [ "$cooldown" -gt 0 ]; then
      if [ "$next_attempt_wait" -eq 0 ] || [ "$cooldown" -lt "$next_attempt_wait" ]; then
        next_attempt_wait=$cooldown
      fi
      continue
    fi
    start_supervisor_keepalive || exit 1
    (
      if [ "$PARALLEL_CONTROL" = 1 ]; then
        # Legacy workers take EX on this same inode: they cannot overlap an
        # upgraded chain or helper. Only upgraded participants share the lease.
        flock -sn 9 || exit 75
        exec {training_lease}>"$QUEUE/$dataset-s$seed.training.lock"
        flock -n "$training_lease" || exit 75
      else
        flock -n 9 || exit 75
      fi
      family_complete "$dataset" "$seed" && exit 0
      run_family "$dataset" "$seed"
    ) 9<>"$QUEUE/$dataset-s$seed.lock" </dev/null   # stdin is the family list; children must not read it
    rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 75 ]; then
      note_family_failure "$dataset" "$seed" "$rc"
      if [ "$LAST_FAILURE_KIND" = runtime ]; then
        echo "[family-retry] $dataset/s$seed rc=$rc (CUDA runtime faults on this worker: ${FAMILY_CUDA_FAILURES[$dataset-s$seed]:-1}/$MAX_CUDA_FAILURES, not a repeating failure); allocation retained, artifacts preserved" \
          | tee -a "$LOG"
      else
        echo "[family-retry] $dataset/s$seed rc=$rc (consecutive failures: ${FAMILY_FAILURES[$dataset-s$seed]:-1}/$MAX_FAMILY_FAILURES); allocation retained, artifacts preserved" \
          | tee -a "$LOG"
      fi
      stop_supervisor_keepalive
      cleanup_stale_experiment_processes 2>&1 | tee -a "$LOG"
      statuses=("${PIPESTATUS[@]}")
      [ "${statuses[0]}" -eq 0 ] && [ "${statuses[1]}" -eq 0 ] || exit 1
      if [ "$RUN_ROLE" = auto ]; then cleanup_node_gpu_processes || exit 1; fi
    fi
    start_supervisor_keepalive || exit 1
    [ "$rc" -eq 75 ] && continue   # held by another worker
    claimed=$((claimed + 1))
    if [ "$rc" -ne 0 ]; then
      cleanup_owner
      family_looping "$dataset" "$seed" && continue   # guard tripped: every worker skips it now
      if [ "$LAST_FAILURE_KIND" = runtime ]; then
        cuda_n=${FAMILY_CUDA_FAILURES[$dataset-s$seed]:-1}
        pause=$(failure_backoff_seconds "$cuda_n")
        # A CUDA runtime fault is exempt from the failure-loop guard because it is
        # usually transient. The exemption had no lifetime bound and the counter
        # dies with the worker, so a fault that reproduces on every attempt (a
        # device-side assert, say) cycled die -> wait -> die for ever and was
        # never marked LOOPING. Count the whole family's failed tries from the
        # durable [point-failed] lines and stop when even a "transient" fault has
        # burned that many attempts.
        runtime_total=$(point_runtime_failures "$dataset" "$seed")
        if [ "${runtime_total:-0}" -ge "$MAX_RUNTIME_FAILURES" ]; then
          {
            echo "family=$dataset/s$seed worker=$WORKER_ID host=$HOST_TAG consecutive_failures=$cuda_n last_rc=$rc runtime_failures_total=$runtime_total"
            echo "last_error=$(family_last_error "$dataset" "$seed")"
            echo "marked_at_utc=$(date -u +%FT%TZ)"
          } > "$(loop_marker "$dataset" "$seed")"
          echo "[family-loop] $dataset/s$seed: $runtime_total failed tries on its current point, every one a CUDA runtime fault. That reproduces, so it is not flaky hardware. No worker retries it until you fix it and relaunch with OM_RLZERO_CLEAR_LOOPS=1." | tee -a "$LOG"
          continue
        fi
        if [ "$cuda_n" -ge "$MAX_CUDA_FAILURES" ]; then
          echo "[cuda-flaky] $dataset/s$seed: $cuda_n CUDA runtime faults in a row here, $runtime_total failed tries on this family in total; this worker moves on (artifacts preserved)" \
            | tee -a "$LOG"
          FAMILY_CUDA_FAILURES[$dataset-s$seed]=0
        fi
      else
        pause=$(failure_backoff_seconds "${FAMILY_FAILURES[$dataset-s$seed]:-1}")
      fi
      FAMILY_NEXT_ATTEMPT[$dataset-s$seed]=$(( $(now_seconds) + pause ))
      echo "[family-next] $dataset/s$seed failed; this worker moves on to the next family now and may come back to this one in ${pause}s (artifacts preserved)" \
        | tee -a "$LOG"
      continue
    fi
    FAMILY_FAILURES[$dataset-s$seed]=0
    FAMILY_CUDA_FAILURES[$dataset-s$seed]=0
    FAMILY_NEXT_ATTEMPT[$dataset-s$seed]=0
    sleep "$CLAIM_YIELD_SECONDS"
  done < <(ordered_families)
  [ "$remaining" -eq 0 ] && break
  if [ "$PARALLEL_CONTROL" = 1 ] && [ "$RUN_ROLE" != resume-family ] && [ "$claimed" -eq 0 ]; then
    while read -r dataset seed; do
      family_selected "$dataset" "$seed" || continue
      family_complete "$dataset" "$seed" && continue
      family_looping "$dataset" "$seed" && continue
      [ ! -s "$(run_dir "$dataset" "$seed" 0)/DONE" ] || continue
      [ ! -s "$QUEUE/$dataset-s$seed.control.loop" ] || continue
      [ "$(now_seconds)" -ge "${CONTROL_NEXT_ATTEMPT[$dataset-s$seed]:-0}" ] || continue
      (
        flock -sn 9 || {
          [ "$RUN_ROLE" != assist ] || echo "[assist-wait] $dataset/s$seed has an exclusive family lease; no helper attached" | tee -a "$LOG"
          exit 75
        }
        # Help an upgraded active training worker, not a stale owner record.
        exec {training_probe}>"$QUEUE/$dataset-s$seed.training.lock"
        if flock -n "$training_probe"; then
          [ "$RUN_ROLE" != assist ] || echo "[assist-wait] no shared training owner for $dataset/s$seed; start resume-family on the owner node" | tee -a "$LOG"
          exit 75
        fi
        exec {training_probe}>&-
        exec {control_lease}>"$QUEUE/$dataset-s$seed.control.lock"
        flock -n "$control_lease" || exit 75
        echo "[control-assist] claim=$dataset/s$seed/d0 worker=$WORKER_ID" | tee -a "$LOG"
        run_family "$dataset" "$seed" 1
      ) 9<>"$QUEUE/$dataset-s$seed.lock" </dev/null
      rc=$?
      [ "$rc" -ne 75 ] || continue
      claimed=$((claimed + 1))
      if [ "$rc" -ne 0 ]; then
        CONTROL_NEXT_ATTEMPT[$dataset-s$seed]=$(( $(now_seconds) + 900 ))
        echo "[control-retry] $dataset/s$seed/d0 rc=$rc; preserved; other OLMo tasks first" | tee -a "$LOG"
        if [ "$rc" = 43 ]; then
          printf 'last_rc=43\nlast_error=control completion or contract failed\n' > "$QUEUE/$dataset-s$seed.control.loop"
        fi
      fi
      break
    done < <(ordered_families)
  fi
  # OLMo3 must finish before any independent model. Keep the worker available
  # for repaired primary families without repeatedly running a known failure.
  if [ "$looping" -gt 0 ] && [ "$looping" -eq "$remaining" ]; then
    echo "[queue] every family left is marked LOOPING:$looping_list" | tee -a "$LOG"
    for fam in $looping_list; do
      echo "[queue]   $fam: $(sed -n 's/^last_error=//p' "$(loop_marker "${fam%/s*}" "${fam#*/s}")" 2>/dev/null | head -1 | cut -c1-200)" | tee -a "$LOG"
    done
    echo "[primary-blocked] OLMo3 is not finished, so this worker keeps waiting for it and starts no other model. Fix the cause above (bash scripts/why.sh), then relaunch with OM_RLZERO_CLEAR_LOOPS=1. For Qwen on this node instead: Ctrl-C, then bash scripts/run_qwen35_9b.sh" | tee -a "$LOG"
    stop_supervisor_keepalive
    sleep "$QUEUE_WAIT_SECONDS"
    continue
  fi
  if [ "$claimed" -eq 0 ]; then
    if [ "$next_attempt_wait" -gt 0 ]; then
      wait_seconds=$next_attempt_wait
      [ "$wait_seconds" -le "$QUEUE_WAIT_SECONDS" ] || wait_seconds=$QUEUE_WAIT_SECONDS
      echo "[queue] $remaining families left; the ones this worker may take are cooling down after a failure; next attempt in ${wait_seconds}s" \
        | tee -a "$LOG"
      sleep "$wait_seconds"
    else
      echo "[queue] waiting for $remaining families owned by other workers or marked LOOPING: $(remaining_families_summary). Nothing to run on this node until then (it stays an idle OLMo3 spare). For Qwen on this node now: Ctrl-C, then bash scripts/run_qwen35_9b.sh" | tee -a "$LOG"
      sleep "$QUEUE_WAIT_SECONDS"
    fi
  fi
done
if [ "$RUN_ROLE" = assist ]; then
  echo "[assist-complete] $ONLY_FAMILIES d0 is complete; no GRPO family or final collection was claimed" | tee -a "$LOG"
  exit 0
fi
if [ -n "$ONLY_FAMILIES" ]; then
  echo "[queue] this node's families are complete: $ONLY_FAMILIES"
  all_done=1
  for seed in "${SEEDS[@]}"; do for dataset in "${DATASETS[@]}"; do
    family_complete "$dataset" "$seed" || all_done=0
  done; done
  if [ "$all_done" -ne 1 ]; then
    echo "[queue] other families are still running elsewhere; the final collection runs on whichever node finishes the last one (or: run h100 again later without OM_RLZERO_ONLY_FAMILIES)"
    exit 0
  fi
fi

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
