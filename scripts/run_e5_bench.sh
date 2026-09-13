#!/usr/bin/env bash
# Public-benchmark evaluation of the trained E5 policies (AIME24/25, AMC23,
# GSM8K subsample, MATH test outside MATH-500), same sampling, prompt format
# and verifier as the E5 evaluation.
#
#   bash scripts/run_e5_bench.sh            # d400 branch on THIS idle 4xH100 node
#   bash scripts/run_e5_bench.sh d0         # d0 branch
#   bash scripts/run_e5_bench.sh status     # progress per seed, arm and set (no GPU)
#   bash scripts/run_e5_bench.sh results    # finished tables (no GPU); export with run_e5.sh export
#   bash scripts/run_e5_bench.sh plan       # what would run, no GPU
#   Any mode accepts d0 or d400 as an extra word.
#
# The benchmark sets ship with the repository (data/benchmarks); no download step.
# Evaluates the source checkpoint (before) and every arm whose policy is
# complete; arms are leased per seed so several idle nodes can share the work.
# Rerunning resumes from completed shards. Never shares a node with a running
# E5, OLMo or Qwen launcher (node lock).
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=run; DRIFT=${E5_DRIFT:-400}
for arg in "$@"; do
  case "$arg" in
    d0) DRIFT=0 ;; d400) DRIFT=400 ;;
    run|status|results|plan) MODE=$arg ;;
    *) echo "usage: bash scripts/run_e5_bench.sh [run|status|results|plan] [d0|d400]"; exit 2 ;;
  esac
done
trap '' HUP
trap 'echo "[bench] interrupted; nothing else will be started"; exit 130' INT TERM
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
ROOT=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
read -r -a SEEDS <<< "${E5_SEEDS:-0 1 2}"
SETS=${E5_BENCH_SETS:-aime24 aime25 amc23 gsm8k math_rest}
EVAL_K=${E5_BENCH_K:-8}; COUNT=${E5_BENCH_COUNT:-200}
# The five sets are committed under data/benchmarks (fetched on 2026-09-13 with
# scripts/fetch_benchmarks.sh, manifests with revisions and hashes); the cluster
# needs no download. A copy under $DATASETS_DIR/benchmarks takes precedence.
BENCH_DIR="$DATASETS_DIR/benchmarks"
ls "$BENCH_DIR"/*.manifest.json >/dev/null 2>&1 || BENCH_DIR="$PWD/data/benchmarks"
BRANCH="$OM_WORK/runs/e5-reduced/math500-d$DRIFT"
# Marker for this pass's processes (distinct from the E5 marker, which is the branch root).
export OUT_ROOT="$BRANCH/.bench"
run_dir() { printf '%s/family-math500-s%s/%s-s%s-math500-d%s\n' "$ROOT" "$1" "$TAG" "$1" "$DRIFT"; }
echo "[bench] math500 d$DRIFT seeds=${SEEDS[*]} sets=$SETS eval_k=$EVAL_K subsample=$COUNT  branch=$BRANCH"
for seed in "${SEEDS[@]}"; do
  f="$BRANCH/s$seed/benchmark_results.csv"
  if [ -s "$f" ]; then echo "[bench] seed $seed result file (ready): $f"; else echo "[bench] seed $seed result file (not yet): $f"; fi
done
if [ "$MODE" = status ]; then
  for seed in "${SEEDS[@]}"; do
    out="$BRANCH/s$seed"; echo "== seed $seed: $out"
    [ -s "$out/experiment.json" ] || { echo "  E5 seed not prepared"; continue; }
    "$PY" src/benchmark_eval.py status --out "$out" | sed 's/^/  /'
    echo "  logs: $out/logs/"
  done
  exit 0
fi
if [ "$MODE" = results ]; then
  for seed in "${SEEDS[@]}"; do
    out="$BRANCH/s$seed"
    [ -s "$out/benchmarks.json" ] || { echo "seed $seed: no benchmark evaluation yet"; continue; }
    "$PY" src/benchmark_eval.py summarize --out "$out" --allow-partial >/dev/null 2>&1 || true
    "$PY" - "$out" <<'PYEOF'
import csv, sys
from pathlib import Path
out = Path(sys.argv[1]); path = out / "benchmark_results.csv"
if not path.is_file():
    print(f"seed {out.name[1:]}: nothing finished yet"); sys.exit(0)
rows = list(csv.DictReader(path.open()))
f = lambda v: "-".rjust(7) if v in ("", None) else f"{float(v):+.3f}".rjust(7)
print(f"seed {out.name[1:]}: {path}")
print("  arm             set        n  reward | vs before [95% CI]        | vs random [95% CI]        | GPU s")
for r in rows:
    print(f"  {r['selector']:15s} {r['benchmark']:9s} {r['prompts']:>4s} {f(r['reward'])} | {f(r['vs_before'])} [{f(r['before_lower'])},{f(r['before_upper'])}] | "
          f"{f(r['vs_random'])} [{f(r['random_lower'])},{f(r['random_upper'])}] | {float(r['gpu_seconds'] or 0):8.0f}")
PYEOF
  done
  echo "FILES TO UPLOAD: bash scripts/run_e5.sh export   (bundles benchmark_results.csv too)"
  exit 0
fi
[ -d "$BENCH_DIR" ] && ls "$BENCH_DIR"/*.manifest.json >/dev/null 2>&1 || {
  echo "[abort] benchmark sets missing: $BENCH_DIR (expected in the repository under data/benchmarks)"
  exit 1
}
if [ "$MODE" = plan ]; then
  for seed in "${SEEDS[@]}"; do
    out="$BRANCH/s$seed"; echo "== seed $seed: $out"
    [ -s "$out/experiment.json" ] || { echo "  E5 seed not prepared; nothing to evaluate"; continue; }
    "$PY" -c 'import sys; from pathlib import Path; import evidence_downstream as ed
out = Path(sys.argv[1]); arms = ["before"] + ed.arms_of(out)
for arm in arms:
    ready = arm == "before" or (out / arm / "policy" / "policy_train.json").is_file()
    print(f"  {arm:14s} {"evaluate" if ready else "skip (policy not complete)"}")' "$out"
    "$PY" src/benchmark_eval.py status --out "$out" | sed 's/^/  /'
  done
  exit 0
fi
# ---- run
first=""
for seed in "${SEEDS[@]}"; do [ -s "$BRANCH/s$seed/experiment.json" ] && { first=$seed; break; }; done
[ -n "$first" ] || { echo "[abort] no prepared E5 seed under $BRANCH"; exit 1; }
RUN=$(run_dir "$first")
{ read -r CFG_ATTN; read -r CFG_GEN; read -r CFG_LORA; read -r CFG_TOPP; read -r CFG_THINK; read -r CFG_FMT; } < <(
  "$PY" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(*(str(c.get(k) if c.get(k) is not None else "") for k in sys.argv[2:]), sep="\n")' \
    "$RUN/run_config.json" attn gen_batch lora_targets top_p thinking prompt_format)
export OM_ATTN=${OM_ATTN:-${CFG_ATTN:-eager}} OM_GEN_BATCH=${OM_GEN_BATCH:-${CFG_GEN:-32}} OM_SKIP_HYBRID=1
export OM_LORA_TARGETS="$CFG_LORA" OM_TOP_P=${CFG_TOPP:-1.0} OM_THINKING=${CFG_THINK:-off} OM_PROMPT_FORMAT=${CFG_FMT:-olmo_rlzero_math}
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
MATH_VERIFY_PATH=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps") || exit 1
export PYTHONPATH="$MATH_VERIFY_PATH:$PYTHONPATH" OM_MATH_VERIFIER=math_verify
source scripts/_lease.sh
source scripts/_e5_node.sh || exit 1
e5_cleanup_previous "$OUT_ROOT" || exit 1
e5_acquire_node || exit "$?"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else mapfile -t GPUS < <(timeout 20 nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); fi
[ "${#GPUS[@]}" -eq 4 ] || { echo "[abort] benchmark evaluation requires exactly four allocated GPUs (found ${#GPUS[@]})"; exit 1; }
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPUS[*]}")"
CHILDREN=()
stop_children() { local pid; for pid in "${CHILDREN[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done; for pid in "${CHILDREN[@]}"; do wait "$pid" 2>/dev/null || true; done; CHILDREN=(); }
trap 'trap - INT TERM; stop_children; exit 130' INT
trap 'trap - INT TERM; stop_children; exit 143' TERM
rc_all=0
for seed in "${SEEDS[@]}"; do
  out="$BRANCH/s$seed"
  echo "== seed $seed: $out"
  [ -s "$out/experiment.json" ] || { echo "  E5 seed not prepared; skipped"; continue; }
  mkdir -p "$out/logs"
  exec > >(tee -p -a "$out/logs/bench-launcher-$(hostname)-$(date -u +%Y%m%dT%H%M%SZ).log" 7>&- 8>&- 9>&-) 2>&1
  # shellcheck disable=SC2086
  "$PY" src/benchmark_eval.py prepare --out "$out" --datasets-dir "$BENCH_DIR" --sets $SETS --count "$COUNT" --eval-k "$EVAL_K" || { rc_all=1; continue; }
  mapfile -t ARMS < <("$PY" -c 'import sys; from pathlib import Path; import evidence_downstream as ed
out = Path(sys.argv[1]); print("before")
for arm in ed.arms_of(out):
    if (out / arm / "policy" / "policy_train.json").is_file(): print(arm)' "$out")
  failed=0; busy=0
  for arm in "${ARMS[@]}"; do
    if [ "$arm" != before ] && ! "$PY" src/evidence_downstream.py policy-ready --out "$out" --arm "$arm"; then
      echo "[skip] $arm: policy not complete or not valid"; continue
    fi
    exec 9>>"$out/.bench-$arm.lock"
    if ! flock -n 9; then echo "[busy] $arm is being evaluated on another node"; busy=$((busy + 1)); continue; fi
    lease_note "$out/.bench-$arm.lock"
    if "$PY" - "$out" "$arm" <<'PYEOF'
import sys; from pathlib import Path; import benchmark_eval as be
out, arm = Path(sys.argv[1]), sys.argv[2]
sys.exit(0 if all(be.shard_done(out, arm, n, s) for n in be.sets_of(out) for s in range(4)) else 1)
PYEOF
    then echo "[done] $arm already evaluated on every set"; flock -u 9; continue; fi
    echo "[bench] $arm: four shard processes over sets $SETS; progress every 5 min, logs in $out/logs/bench-$arm-<shard>.log"
    CHILDREN=()
    for shard in 0 1 2 3; do
      setsid env CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PY" src/benchmark_eval.py evaluate --out "$out" --arm "$arm" --shard "$shard" \
        > "$out/logs/bench-$arm-$shard.log" 2>&1 7>&- 8>&- 9>&- &
      CHILDREN+=("$!")
    done
    waited=0
    while :; do
      alive=0; for pid in "${CHILDREN[@]}"; do kill -0 "$pid" 2>/dev/null && alive=$((alive + 1)); done
      [ "$alive" -gt 0 ] || break
      sleep 1; waited=$((waited + 1))
      if [ $((waited % 300)) -eq 0 ]; then
        echo "[bench] $arm: $((waited / 60)) min elapsed, $alive/4 shards running"
        for shard in 0 1 2 3; do echo "  shard $shard: $(tail -n 1 "$out/logs/bench-$arm-$shard.log" 2>/dev/null | cut -c1-110)"; done
      fi
    done
    arm_failed=0; for pid in "${CHILDREN[@]}"; do wait "$pid" || arm_failed=1; done; CHILDREN=()
    if [ "$arm_failed" -ne 0 ]; then
      echo "[failed] $arm benchmark evaluation; completed shards are retained"
      for shard in 0 1 2 3; do grep -m1 -E '^\[abort\]|Error|error' "$out/logs/bench-$arm-$shard.log" 2>/dev/null | sed "s/^/  shard $shard: /" | cut -c1-160; done
      failed=1
    fi
    flock -u 9
  done
  "$PY" src/benchmark_eval.py summarize --out "$out" --allow-partial > "$out/logs/bench-summary.log" 2>&1 || failed=1
  grep -E '"complete"|"missing"' "$out/logs/bench-summary.log" || true
  echo "[bench] seed $seed failures=$failed busy_arms=$busy"
  [ -s "$out/benchmark_results.csv" ] && echo "[bench] results file: $out/benchmark_results.csv"
  [ "$failed" -eq 0 ] || rc_all=1
done
echo "[bench] pass complete; check:  bash scripts/run_e5_bench.sh results   (then bash scripts/run_e5.sh export)"
exit "$rc_all"
