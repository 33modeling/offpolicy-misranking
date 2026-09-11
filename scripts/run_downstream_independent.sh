#!/usr/bin/env bash
# E5 with an independent test set: resumable, matched four-GPU GRPO arms.
#
#   bash scripts/run_downstream_independent.sh RUN OUT --eval-prompts TEST.json \
#        [--steps 100] [--eval-k 8] [--dry-run|--prepare-only]
#   DOWNSTREAM_SELECTORS="random passrate_beta fresh_r g11"   arms to train (default; any of
#                                               fresh_r g00 g10 g01 g11 passrate_beta random)
#
# Every arm starts from RUN's policy_step_<drift> adapter+optimizer (at drift 0:
# the base model with a fresh adapter and optimizer), receives
# --steps further GRPO updates on its selected prompts with the point's own
# objective configuration, and is evaluated on TEST.json (never the ranking
# validation prompts). Per-arm leases in OUT let several idle nodes share the
# arms of one seed; a busy arm is skipped, not duplicated.
set -uo pipefail
cd "$(dirname "$0")/.."
# A dropped SSH session (phone) sends SIGHUP; that must not end a multi-hour pass.
trap '' HUP
usage() {
  echo "usage: bash scripts/run_downstream_independent.sh RUN OUT --eval-prompts TEST.json [--steps 100] [--eval-k 8] [--dry-run|--prepare-only]"
}
[ "$#" -ge 2 ] || { usage; exit 2; }
RUN=$(realpath -m "$1"); OUT=$(realpath -m "$2"); shift 2
EVAL_PROMPTS=${DOWNSTREAM_EVAL_PROMPTS:-}; STEPS=${DOWNSTREAM_STEPS:-100}; EVAL_K=${DOWNSTREAM_EVAL_K:-8}
DRY=0; PREPARE_ONLY=0
read -r -a SELECTORS <<< "${DOWNSTREAM_SELECTORS:-random passrate_beta fresh_r g11}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --eval-prompts|--steps|--eval-k)
      [ "$#" -ge 2 ] || { usage; exit 2; }
      case "$1" in
        --eval-prompts) EVAL_PROMPTS=$2 ;; --steps) STEPS=$2 ;; --eval-k) EVAL_K=$2 ;;
      esac; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    *) usage; exit 2 ;;
  esac
done
[ -n "$EVAL_PROMPTS" ] || { echo "[abort] independent --eval-prompts is required; the ranking validation set is not a test set"; exit 2; }
for selector in "${SELECTORS[@]}"; do
  case "$selector" in fresh_r|g00|g10|g01|g11|passrate_beta|random) ;; *) echo "[abort] unknown selector: $selector"; exit 2 ;; esac
done
export OM_ONLINE=0
source scripts/setup_env.sh || exit 1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[abort] venv missing: $PY"; exit 1; }
[ -s "$RUN/run_config.json" ] || { echo "[abort] source run_config.json missing: $RUN"; exit 1; }
# Runtime knobs come from the source point so the arms match the matrix exactly
# (attention kernel, generation batch, LoRA targets, sampling, prompt format).
{ read -r CFG_ATTN; read -r CFG_GEN; read -r CFG_LORA; read -r CFG_TOPP; read -r CFG_THINK; read -r CFG_FMT; } < <(
  "$PY" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(*(str(c.get(k) if c.get(k) is not None else "") for k in sys.argv[2:]), sep="\n")' \
    "$RUN/run_config.json" attn gen_batch lora_targets top_p thinking prompt_format)
export OM_ATTN=${OM_ATTN:-${CFG_ATTN:-eager}} OM_GEN_BATCH=${OM_GEN_BATCH:-${CFG_GEN:-32}} OM_SKIP_HYBRID=1
export OM_LORA_TARGETS="$CFG_LORA" OM_TOP_P=${CFG_TOPP:-1.0} OM_THINKING=${CFG_THINK:-off} OM_PROMPT_FORMAT=${CFG_FMT:-olmo_rlzero_math}
PREPARE=("$PY" src/evidence_downstream.py prepare --run "$RUN" --out "$OUT" --eval-prompts "$EVAL_PROMPTS" \
         --steps "$STEPS" --eval-k "$EVAL_K" --selectors "${SELECTORS[@]}")
if [ "$DRY" = 1 ]; then "${PREPARE[@]}" --dry-run; exit "$?"; fi
"${PREPARE[@]}" >/dev/null || exit 1
echo "[prepared] $OUT (arms: ${SELECTORS[*]}; $STEPS updates; $EVAL_K responses per test prompt)"
[ "$PREPARE_ONLY" = 0 ] || { echo "[prepared] no GPU work launched"; exit 0; }
# Same reward function as the registered matrix (symbolic Math-Verify).
MATH_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps") || exit 1
export PYTHONPATH="$MATH_VERIFY_PATH${PYTHONPATH:+:$PYTHONPATH}" OM_MATH_VERIFIER=math_verify

# Leftover E5 processes on THIS node (an earlier launch whose session dropped or
# was interrupted) still hold the GPUs and the leases. Stop them and resume from
# their checkpoints and partial shards. Only processes that reference the E5
# output root are touched; the OLMo and Qwen launchers never match.
E5_ROOT=$(dirname "$OUT")
source scripts/_e5_node.sh || exit 1
if [ "${OM_E5_CONTROLLER_PID:-}" != "$PPID" ]; then
  e5_cleanup_previous "$E5_ROOT" || exit 1
fi
# Node-local GPU admission: never share a node with the OLMo or Qwen launcher.
if [ "${OM_NODE_LOCK_HELD:-0}" != 1 ]; then
  e5_acquire_node || exit "$?"
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t GPUS < <(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
fi
[ "${#GPUS[@]}" -eq 4 ] || { echo "[abort] E5 requires exactly four allocated GPUs (found ${#GPUS[@]})"; exit 1; }
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"
# Wait for the GPUs to drain after a cleanup (up to 60 s); report if they do not.
for _ in $(seq 1 12); do
  busy_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null | sort -n | tail -1)
  [ -n "$busy_mib" ] && [ "$busy_mib" -gt 4000 ] || break
  sleep 5
done
if [ -n "${busy_mib:-}" ] && [ "$busy_mib" -gt 4000 ]; then
  echo "[busy] GPU memory did not drain after E5 cleanup (${busy_mib} MiB); no new GPU work started"
  exit 75
fi
mkdir -p "$OUT/logs"
exec > >(tee -p -a "$OUT/logs/launcher-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ).log" 7>&- 8>&- 9>&-) 2>&1
echo "[environment] host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES attention=$OM_ATTN generation_batch=$OM_GEN_BATCH arms=${SELECTORS[*]}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true
CHILDREN=()
stop_children() {
  local pid
  for pid in "${CHILDREN[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for pid in "${CHILDREN[@]}"; do wait "$pid" 2>/dev/null || true; done
  CHILDREN=()
}
trap 'trap - INT TERM; stop_children; exit 130' INT
trap 'trap - INT TERM; stop_children; exit 143' TERM
run_tracked() {
  setsid "$@" 7>&- 8>&- 9>&- & local pid=$!
  CHILDREN=("$pid")
  wait "$pid"; local rc=$?
  CHILDREN=()
  return "$rc"
}
shard_progress() {  # one console line per shard while the evaluation runs
  local arm=$1 shard partial log rows
  for shard in 0 1 2 3; do
    partial="$OUT/$arm/evaluation/shard-$shard.jsonl.partial"; log="$OUT/logs/eval-$arm-$shard.log"
    if [ -f "$OUT/$arm/evaluation/shard-$shard.done.json" ]; then echo "  shard $shard: done"
    elif [ -f "$partial" ]; then rows=$(grep -c . "$partial" 2>/dev/null); echo "  shard $shard: ${rows:-0} responses so far | $(tail -n 1 "$log" 2>/dev/null | cut -c1-110)"
    else echo "  shard $shard: loading model | $(tail -n 1 "$log" 2>/dev/null | cut -c1-110)"; fi
  done
}
evaluate_arm() {
  local arm=$1 shard pid failed=0 waited=0
  CHILDREN=()
  echo "[eval] $arm: four shard processes; progress every 5 min here, full logs in $OUT/logs/eval-$arm-<shard>.log"
  for shard in 0 1 2 3; do
    setsid env CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PY" src/evidence_downstream.py evaluate \
      --out "$OUT" --arm "$arm" --shard "$shard" > "$OUT/logs/eval-$arm-$shard.log" 2>&1 7>&- 8>&- 9>&- &
    CHILDREN+=("$!")
  done
  while :; do
    local alive=0
    for pid in "${CHILDREN[@]}"; do kill -0 "$pid" 2>/dev/null && alive=$((alive + 1)); done
    [ "$alive" -gt 0 ] || break
    # Poll completion separately from the five-minute reporting interval.
    sleep 1; waited=$((waited + 1))
    if [ $((waited % 300)) -eq 0 ]; then echo "[eval] $arm: $((waited / 60)) min elapsed, $alive/4 shards running"; shard_progress "$arm"; fi
  done
  for pid in "${CHILDREN[@]}"; do wait "$pid" || failed=1; done
  CHILDREN=()
  if [ "$failed" -ne 0 ]; then
    for shard in 0 1 2 3; do
      [ -f "$OUT/$arm/evaluation/shard-$shard.done.json" ] || echo "  shard $shard failed: $(grep -m1 -E '^\[abort\]|Error|error' "$OUT/logs/eval-$arm-$shard.log" 2>/dev/null | cut -c1-160)"
    done
  fi
  return "$failed"
}
failed=0; busy=0
exec 9>"$OUT/.before.lock"
if flock -n 9; then
  echo "[eval] baseline policy on ${EVAL_K} responses per test prompt (about 60-90 min on four GPUs)"
  evaluate_arm before || { echo "[failed] baseline evaluation; completed shards are retained"; failed=1; }
  flock -u 9
else
  echo "[busy] another node is evaluating the common baseline; continuing with training"
fi
for selector in "${SELECTORS[@]}"; do
  exec 9>"$OUT/.$selector.lock"
  if ! flock -n 9; then echo "[busy] $selector is claimed on another node"; busy=$((busy + 1)); continue; fi
  mapfile -d '' -t ARGS < "$OUT/subsets/train-$selector.args"
  "$PY" src/evidence_downstream.py policy-ready --out "$OUT" --arm "$selector"; ready=$?
  if [ "$ready" -eq 2 ]; then
    if ! "$PY" src/check_downstream_resume.py --out "$OUT" --arm "$selector"; then
      echo "[failed] $selector has no verified repair checkpoint; preserving it for diagnosis"
      failed=1; flock -u 9; continue
    fi
    echo "[repair] $selector: restoring final publication from its verified checkpoint"
  fi
  if [ "$ready" -ne 0 ]; then
    echo "[train] $selector: $STEPS matched GRPO updates (resumes from the newest checkpoint if present)"
    if ! run_tracked "$PY" "${ARGS[@]}" >> "$OUT/logs/train-$selector.log" 2>&1; then
      echo "[failed] $selector training; see $OUT/logs/train-$selector.log; continuing to the next arm"
      failed=1; flock -u 9; continue
    fi
  else
    echo "[done] $selector already trained"
  fi
  echo "[eval] $selector on ${EVAL_K} responses per test prompt"
  evaluate_arm "$selector" || { echo "[failed] $selector evaluation; continuing to the next arm"; failed=1; }
  flock -u 9
done
if [ "${OM_NODE_LOCK_HELD:-0}" != 1 ]; then
  flock -u 8
  if [ "${E5_HOST_LOCK_HELD:-0}" = 1 ]; then flock -u 7; fi
fi
exec 9>"$OUT/.summary.lock"
if flock -n 9; then
  "$PY" src/evidence_downstream.py summarize --out "$OUT" --allow-partial > "$OUT/logs/summary.log" 2>&1 || failed=1
  grep -E '"complete"|"missing' "$OUT/logs/summary.log" || true
fi
echo "[E5] failures=$failed busy_arms=$busy output=$OUT"
[ "$failed" -eq 0 ]
