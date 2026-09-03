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
  STATUS_LOG_LINES="${OM_RLZERO_STATUS_LOG_LINES:-20}"
  STATUS_ERROR_LINES="${OM_RLZERO_STATUS_ERROR_LINES:-6}"
  STATUS_PROBE_SECONDS="${OM_RLZERO_STATUS_PROBE_SECONDS:-20}"
  STATUS_STUCK_SECONDS="${OM_RLZERO_STATUS_STUCK_SECONDS:-1800}"
  STATUS_WORKER_STALE_SECONDS="${OM_RLZERO_STATUS_WORKER_STALE_SECONDS:-180}"
  STATUS_EXPECTED_WORKERS="${OM_RLZERO_STATUS_EXPECTED_WORKERS:-3}"
  for value_name in STATUS_LOG_LINES STATUS_ERROR_LINES STATUS_STUCK_SECONDS \
      STATUS_WORKER_STALE_SECONDS STATUS_EXPECTED_WORKERS; do
    value=${!value_name}
    case "$value" in
      ''|*[!0-9]*|0) echo "[abort] invalid $value_name=$value"; exit 2 ;;
    esac
  done
  case "$STATUS_PROBE_SECONDS" in
    ''|*[!0-9]*) echo "[abort] invalid STATUS_PROBE_SECONDS=$STATUS_PROBE_SECONDS"; exit 2 ;;
  esac
  "$PY" "$SUPERVISOR_REPO/src/rlzero_status.py" \
    --profile "$PROFILE" --root "$ROOT" --results "$GLOBAL_RESULTS" \
    --model-tag "$MODEL_TAG" --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" --drifts "${DRIFTS[@]}" \
    --probe-seconds "$STATUS_PROBE_SECONDS" \
    --stuck-seconds "$STATUS_STUCK_SECONDS" \
    --worker-stale-seconds "$STATUS_WORKER_STALE_SECONDS" \
    --expected-workers "$STATUS_EXPECTED_WORKERS" \
    --log-lines "$STATUS_LOG_LINES" --error-lines "$STATUS_ERROR_LINES"
  exit $?
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
    echo "[startup-cleanup] previous launcher or orphan owns the node lock; terminating it"
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

  STALE_PROCESS_TIMEOUT="${OM_RLZERO_STALE_PROCESS_TIMEOUT:-15}"
  GPU_CLEANUP_TIMEOUT="${OM_RLZERO_GPU_CLEANUP_TIMEOUT:-15}"
  for value_name in STALE_PROCESS_TIMEOUT GPU_CLEANUP_TIMEOUT; do
    value=${!value_name}
    case "$value" in
      ''|*[!0-9]*|0) echo "[abort] invalid $value_name=$value" | tee -a "$LOG"; exit 2 ;;
    esac
  done

  cleanup_stale_experiment_processes() {
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
  cleanup_node_gpu_processes || exit 1

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
    OM_GPU_KEEPALIVE_READY_FILE="$KEEPALIVE_READY" CUDA_VISIBLE_DEVICES=0,1,2,3 \
      "$PY" "$SUPERVISOR_RUNTIME_REPO/scripts/gpu_keepalive.py" \
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

GENERATION_GIT=$("$PY" "$SUPERVISOR_RUNTIME_REPO/src/regime_resume_commit.py" \
  "$ROOT" "$CURRENT_GIT" \
  --marker "$ROOT/.queue/generation.git" --advance-empty) || exit 1

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
    OM_REPO="$SUPERVISOR_RUNTIME_REPO" OM_PIPELINE_REPO="$GENERATION_REPO" \
      OM_PIPELINE_SCRIPT="$GENERATION_REPO/scripts/run_point.sh" \
      OM_GENERATION_GIT="$GENERATION_GIT" \
      PYTHONPATH="$SUPERVISOR_RUNTIME_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
      MODEL_PATH="$MODEL_PATH" \
      REGIME_ROOT="$root" REGIME_RESULTS="$result" REGIME_MODEL_TAG="$MODEL_TAG" \
      REGIME_DATASETS="$dataset" REGIME_SEEDS="$seed" \
      REGIME_DRIFTS="${DRIFTS[*]}" REGIME_SKIP_COLLECTION=1 \
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
      if [ "$rc" -ne 0 ] && [ "$rc" -ne 75 ]; then
        echo "[family-retry] $dataset/s$seed rc=$rc; allocation retained, artifacts preserved, automatic retry scheduled" \
          | tee -a "$LOG"
        stop_supervisor_keepalive
        cleanup_stale_experiment_processes 2>&1 | tee -a "$LOG"
        statuses=("${PIPESTATUS[@]}")
        [ "${statuses[0]}" -eq 0 ] && [ "${statuses[1]}" -eq 0 ] || exit 1
        cleanup_node_gpu_processes || exit 1
      fi
      start_supervisor_keepalive || exit 1
      [ "$rc" -eq 75 ] && continue
      claimed=$((claimed + 1))
      if [ "$rc" -ne 0 ]; then
        cleanup_owner
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
