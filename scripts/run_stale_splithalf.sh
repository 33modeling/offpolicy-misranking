#!/usr/bin/env bash
# Split-half scores of the reuse estimators (g00, g10, g01, g11) on the
# completed MATH-500 points, so that the reuse selectors carry the same
# two-measurement reliability diagnostic as the fresh and difficulty scores.
#
#   bash scripts/run_stale_splithalf.sh          # d400 points of seeds 0 1 2 on THIS idle 4xH100 node
#   bash scripts/run_stale_splithalf.sh d0       # d0 points
#   bash scripts/run_stale_splithalf.sh status   # which points have scores_stale_splithalf.json (no GPU)
#
# Four shards per point (one GPU each), merged on CPU; a point with an existing
# scores_stale_splithalf.json is skipped, finished shards are reused. Writes
# into the point directory (scores_stale_splithalf.json + .protocol.json) and
# logs under <point>/logs/stale-splithalf-*.log. Never shares a node with a
# running launcher (node lock). Afterwards: bash scripts/run_gate_decision.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=run; DRIFT=${E5_DRIFT:-400}
for arg in "$@"; do
  case "$arg" in
    d0) DRIFT=0 ;; d400) DRIFT=400 ;; status) MODE=status ;;
    *) echo "usage: bash scripts/run_stale_splithalf.sh [status] [d0|d400]"; exit 2 ;;
  esac
done
trap '' HUP
trap 'echo "[stale] interrupted"; exit 130' INT TERM
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
ROOT=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
read -r -a SEEDS <<< "${E5_SEEDS:-0 1 2}"
CHECK=${STALE_CHECK_FULL:-4}
run_dir() { printf '%s/family-math500-s%s/%s-s%s-math500-d%s\n' "$ROOT" "$1" "$TAG" "$1" "$DRIFT"; }
export OUT_ROOT="$ROOT/.stale-splithalf-d$DRIFT"   # process marker for cleanup; no directory is created
if [ "$MODE" = status ]; then
  for seed in "${SEEDS[@]}"; do
    run=$(run_dir "$seed")
    if [ -s "$run/scores_stale_splithalf.json" ]; then echo "seed $seed d$DRIFT: ready   $run/scores_stale_splithalf.json"
    else echo "seed $seed d$DRIFT: not yet ($(ls "$run"/scores_stale_splithalf.shard*.json 2>/dev/null | wc -l)/4 shards)"; fi
  done
  exit 0
fi
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
source scripts/_e5_node.sh || exit 1
e5_cleanup_previous "$OUT_ROOT" || exit 1
e5_acquire_node || exit "$?"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else mapfile -t GPUS < <(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); fi
[ "${#GPUS[@]}" -eq 4 ] || { echo "[abort] requires exactly four allocated GPUs (found ${#GPUS[@]})"; exit 1; }
CHILDREN=()
stop_children() { local pid; for pid in "${CHILDREN[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done; for pid in "${CHILDREN[@]}"; do wait "$pid" 2>/dev/null || true; done; CHILDREN=(); }
trap 'trap - INT TERM; stop_children; exit 130' INT
trap 'trap - INT TERM; stop_children; exit 143' TERM
rc_all=0
for seed in "${SEEDS[@]}"; do
  run=$(run_dir "$seed")
  echo "== seed $seed d$DRIFT: $run"
  [ -s "$run/DONE" ] || { echo "  source point not complete; skipped"; continue; }
  [ -s "$run/scores_stale_splithalf.json" ] && { echo "  already scored: $run/scores_stale_splithalf.json"; continue; }
  # runtime knobs of the point (attention kernel, LoRA targets, prompt format)
  { read -r CFG_ATTN; read -r CFG_LORA; read -r CFG_FMT; } < <(
    "$PY" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(*(str(c.get(k) if c.get(k) is not None else "") for k in sys.argv[2:]), sep="\n")' \
      "$run/run_config.json" attn lora_targets prompt_format)
  export OM_ATTN=${OM_ATTN:-${CFG_ATTN:-eager}} OM_LORA_TARGETS="$CFG_LORA" OM_PROMPT_FORMAT=${CFG_FMT:-olmo_rlzero_math}
  mkdir -p "$run/logs"
  exec 9>"$run/.stale-splithalf.lock"
  if ! flock -n 9; then echo "  claimed on another node; skipped"; continue; fi
  CHILDREN=()
  for shard in 0 1 2 3; do
    setsid env CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PY" src/stale_splithalf.py --run "$run" --shard "$shard" --shards 4 --check-full "$CHECK" \
      > "$run/logs/stale-splithalf-shard$shard.log" 2>&1 7>&- 8>&- 9>&- &
    CHILDREN+=("$!")
  done
  echo "  four shard processes; progress every 5 min, logs in $run/logs/stale-splithalf-shard<s>.log"
  waited=0
  while :; do
    alive=0; for pid in "${CHILDREN[@]}"; do kill -0 "$pid" 2>/dev/null && alive=$((alive + 1)); done
    [ "$alive" -gt 0 ] || break
    sleep 1; waited=$((waited + 1))
    if [ $((waited % 300)) -eq 0 ]; then
      echo "  $((waited / 60)) min elapsed, $alive/4 shards running"
      for shard in 0 1 2 3; do echo "    shard $shard: $(tail -n 1 "$run/logs/stale-splithalf-shard$shard.log" 2>/dev/null | cut -c1-110)"; done
    fi
  done
  failed=0; for pid in "${CHILDREN[@]}"; do wait "$pid" || failed=1; done; CHILDREN=()
  if [ "$failed" -ne 0 ]; then
    echo "  [failed] a shard failed; finished shards are kept"
    for shard in 0 1 2 3; do grep -m1 -E '^\[abort\]|Error|error' "$run/logs/stale-splithalf-shard$shard.log" 2>/dev/null | sed "s/^/    shard $shard: /" | cut -c1-160; done
    rc_all=1; flock -u 9; continue
  fi
  CUDA_VISIBLE_DEVICES="" "$PY" src/stale_splithalf.py --run "$run" --merge --shards 4 || rc_all=1
  flock -u 9
done
echo "[stale] pass complete; next:  bash scripts/run_gate_decision.sh   and   bash scripts/run_gain_law.sh"
exit "$rc_all"
