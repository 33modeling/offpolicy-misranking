#!/usr/bin/env bash
# Reduced E5 (2026-09-10): does the selected data actually train better?
#   MATH-500, checkpoint d400, seeds 0 1 2, arms random / fresh_r / g11,
#   100 further GRPO updates per arm, evaluated on 300 held-out MATH-train
#   problems x 8 responses (never the ranking validation prompts).
#
#   bash scripts/run_e5.sh          # run the d400 branch on THIS idle 4xH100 node
#   bash scripts/run_e5.sh d0       # run the d0 branch (arms start from the base model)
#   bash scripts/run_e5.sh status   # progress of every branch, seed and arm, no GPU
#   bash scripts/run_e5.sh export   # bundle every finished result file (all branches, all seeds)
#                                   # into one text file under $OM_WORK/exports and print it
#   bash scripts/run_e5.sh results  # finished numbers only: benchmark table per seed and the
#                                   # reliability trajectory per seed (no GPU)
#   bash scripts/run_e5.sh rlog     # reliability-logging run: random arm only, training only,
#                                   # writes reliability_trajectory.csv (split-half reliability
#                                   # of the pass-rate and gradient signals along training)
#   bash scripts/run_e5.sh rlog400  # the same logging run over 400 updates (separate root -rlog400)
#   bash scripts/run_e5.sh gate     # executed gate arm (gate_passrate) added to the seeds of a
#                                   # branch: uniform pilot with reliability logging, one frozen
#                                   # decision (config/gate_rule.json), continuation, evaluation
#   Any mode accepts d0 or d400 as an extra word, e.g.  bash scripts/run_e5.sh d0 stop
#   bash scripts/run_e5.sh plan     # dry run: contracts and commands only
#   bash scripts/run_e5.sh stop     # stop E5 on this node (nothing else)
#   bash scripts/run_e5.sh force    # run, first stopping a non-matrix process that holds this node's GPU lock
#
# Several idle nodes may run the same command: arms are leased per seed, a
# busy arm is skipped, and a node moves on to the next seed. Rerunning after a
# kill resumes from the newest checkpoint or completed evaluation shard.
# Prerequisite once, in an online shell:  bash scripts/fetch_math_train.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=run; DRIFT=${E5_DRIFT:-400}; DRIFT_GIVEN=${E5_DRIFT:+1}
for arg in "$@"; do
  case "$arg" in
    d0) DRIFT=0; DRIFT_GIVEN=1 ;;
    d400) DRIFT=400; DRIFT_GIVEN=1 ;;
    run|status|plan|stop|results|export) MODE=$arg ;;
    force) MODE=run; export E5_FORCE=1 ;;
    rlog) MODE=run; RLOG=1 ;;
    rlog400) MODE=run; RLOG=1; RLOG400=1 ;;
    gate) MODE=run; GATE=1 ;;
    *) echo "usage: bash scripts/run_e5.sh [run|status|plan|stop|force|rlog|rlog400|gate] [d0|d400]"; exit 2 ;;
  esac
done
trap '' HUP
trap 'echo "[e5] interrupted; nothing else will be started"; exit 130' INT TERM
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
TAG=${OM_OLMO3_MODEL_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
ROOT=${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}
DATASET=${E5_DATASET:-math500}
read -r -a SEEDS <<< "${E5_SEEDS:-0 1 2}"
STEPS=${E5_STEPS:-100}; EVAL_K=${E5_EVAL_K:-8}; COUNT=${E5_TEST_COUNT:-300}
SELECTORS=${E5_SELECTORS:-random passrate_beta fresh_r g11}
if [ "${RLOG:-0}" = 1 ]; then
  # Separate root: the logging run must not share arm directories with the benchmark.
  SELECTORS=random; export E5_RELIABILITY_LOG=1 E5_SKIP_EVAL=1; RLOG_SUFFIX="-rlog"
  if [ "${RLOG400:-0}" = 1 ]; then STEPS=400; RLOG_SUFFIX="-rlog400"; fi
fi
if [ "${GATE:-0}" = 1 ]; then
  # The gate arm joins the benchmark root of the branch (arms.json extension);
  # random and passrate_beta there are its comparison arms.
  SELECTORS=${E5_SELECTORS:-gate_passrate}
fi
POOL="$DATASETS_DIR/math_train/math_train.jsonl"; POOL_MANIFEST="$DATASETS_DIR/math_train/dataset_manifest.json"
# E5_TEST_DATASET lets a derived pool (math500mix) reuse the frozen MATH-train test set of the base dataset.
TEST="$OM_WORK/inputs/e5-reduced/test-${E5_TEST_DATASET:-$DATASET}-d$DRIFT.json"
OUT_ROOT="$OM_WORK/runs/e5-reduced/$DATASET-d$DRIFT${RLOG_SUFFIX:-}"
# Exported marker: every process of this pass carries OUT_ROOT in its environment,
# so a later launch on the same node can find and stop the whole earlier pass
# (including this loop), while the matrix launchers never match.
export OUT_ROOT
run_dir() { printf '%s/family-%s-s%s/%s-s%s-%s-d%s\n' "$ROOT" "$DATASET" "$1" "$TAG" "$1" "$DATASET" "$DRIFT"; }

echo "[e5] $DATASET d$DRIFT seeds=${SEEDS[*]} arms=$SELECTORS steps=$STEPS eval_k=$EVAL_K test=$COUNT  out=$OUT_ROOT"
for seed in "${SEEDS[@]}"; do
  if [ "${RLOG:-0}" = 1 ]; then f="$OUT_ROOT/s$seed/random/policy/reliability_trajectory.csv"; else f="$OUT_ROOT/s$seed/downstream_results.csv"; fi
  if [ -s "$f" ]; then echo "[e5] seed $seed result file (ready): $f"; else echo "[e5] seed $seed result file (not yet): $f"; fi
done
if [ "$MODE" = stop ]; then
  source scripts/_e5_node.sh || exit 1
  e5_cleanup_previous "$OUT_ROOT" || exit 1
  echo "[e5] previous E5 processes stopped on $(hostname); checkpoints retained"
  exit 0
fi
if [ "$MODE" = export ]; then
  mkdir -p "$OM_WORK/exports"
  target="$OM_WORK/exports/e5-results-$(date -u +%Y%m%dT%H%M%SZ).txt"
  n=0
  {
    echo "# E5 results export $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) code=$(git rev-parse --short HEAD 2>/dev/null)"
    for f in "$OM_WORK"/runs/e5-reduced/math500*-d*/s*/downstream_results.csv \
             "$OM_WORK"/runs/e5-reduced/math500*-d*/s*/random/policy/reliability_trajectory.csv \
             "$OM_WORK"/runs/e5-reduced/math500*-d*/s*/gate_*/decision.json \
             "$OM_WORK"/runs/e5-reduced/math500*-d*/s*/gate_decision.csv \
             "$OM_WORK"/runs/e5-reduced/math500*-d*/s*/benchmark_results.csv; do
      [ -s "$f" ] || continue
      n=$((n + 1)); echo; echo "### $f"; cat "$f"
    done
    echo; echo "# files: $n"
  } > "$target"
  cat "$target"
  echo; echo "[e5] export written: $target"
  exit 0
fi
if [ "$MODE" = results ]; then
  # Refresh summaries on CPU so older CSVs gain the random-baseline columns.
  for seed_dir in "$OM_WORK"/runs/e5-reduced/math500*-d*/s*; do
    [ -s "$seed_dir/experiment.json" ] && [ -d "$seed_dir/before/evaluation" ] && \
      "$PY" src/evidence_downstream.py summarize --out "$seed_dir" --allow-partial >/dev/null 2>&1 || true
    # recompute reliability trajectories from the raw per-rank logs (adds newer columns)
    ls "$seed_dir"/random/policy/reliability_log.rank*.jsonl >/dev/null 2>&1 && \
      "$PY" src/reliability_trajectory.py --policy "$seed_dir/random/policy" --window "${E5_RELIABILITY_WINDOW:-20}" >/dev/null 2>&1 || true
  done
  "$PY" - "$OM_WORK/runs/e5-reduced" <<'PYEOF'
import csv, sys
from pathlib import Path
root = Path(sys.argv[1])
def f(v, w=6):
    try: return f"{float(v):+.3f}".rjust(w) if v not in ("", None) else "-".rjust(w)
    except ValueError: return str(v).rjust(w)
files = []
for branch in sorted(root.glob("math500*-d*")):
    print(f"== {branch.name}")
    for seed in sorted(branch.glob("s*")):
        results = seed / "downstream_results.csv"
        if results.is_file():
            files.append(results)
            rows = list(csv.DictReader(results.open()))
            print(f"  seed {seed.name[1:]}: {results}")
            print(f"  seed {seed.name[1:]}: reward before {f(rows[0]['reward_before'])}   (after | vs random [95% CI] | vs fresh)")
            for r in rows:
                print(f"    {r['selector']:14s} {f(r['reward_after'])} | {f(r.get('difference_vs_random'))} [{f(r.get('random_lower'))},{f(r.get('random_upper'))}] | {f(r.get('difference_vs_fresh'))}")
                if r.get("gate_decision"):
                    print(f"      gate: decision={r['gate_decision']} ({r['gate_reason']}) r_half={f(r.get('gate_r_half'))} "
                          f"[{f(r.get('gate_lower'))},{f(r.get('gate_upper'))}] pilot={r.get('gate_pilot_steps')} steps "
                          f"{f(r.get('gate_pilot_seconds'), 8)} s | forgone vs selector {f(r.get('forgone_vs_selector'))} "
                          f"[{f(r.get('forgone_lower'))},{f(r.get('forgone_upper'))}]")
        traj = seed / "random" / "policy" / "reliability_trajectory.csv"
        if traj.is_file():
            files.append(traj)
            rows = list(csv.DictReader(traj.open()))
            print(f"  seed {seed.name[1:]}: {traj}")
            print(f"  seed {seed.name[1:]} reliability, half-group correlations (steps: pass | diff score | grad | grad among mixed | mixed frac | mean pass)")
            for r in rows:
                print(f"    {r['step_start']:>4}-{r['step_end']:<4} {f(r['pass_r_half'])} {f(r.get('diff_r_half'))} {f(r['grad_r_half'])} {f(r.get('grad_mixed_r_half'))} {f(r['mixed_fraction'])} {f(r['mean_pass'])}")
        if not results.is_file() and not traj.is_file():
            expected = traj if branch.name.endswith("-rlog") else results
            print(f"  seed {seed.name[1:]}: not finished; will be at {expected}")
print()
print("FILES TO UPLOAD (finished results):")
for path in files:
    print(f"  {path}")
if not files:
    print("  none yet")
PYEOF
  exit 0
fi
if [ "$MODE" = status ]; then
  # Without d0/d400 the status covers every branch that exists.
  if [ -n "${DRIFT_GIVEN:-}" ]; then roots=("$OUT_ROOT"); else mapfile -t roots < <(ls -d "$OM_WORK/runs/e5-reduced/$DATASET-d"* 2>/dev/null); fi
  [ "${#roots[@]}" -gt 0 ] || { echo "no E5 branch prepared yet"; exit 0; }
  for root in "${roots[@]}"; do
    echo "== branch $(basename "$root")"
    for seed in "${SEEDS[@]}"; do
      out="$root/s$seed"
      case "$root" in
        *-rlog) f="$out/random/policy/reliability_trajectory.csv" ;;
        *) f="$out/downstream_results.csv" ;;
      esac
      if [ -s "$f" ]; then echo "  seed $seed RESULT (ready):   $f"; else echo "  seed $seed RESULT (not yet): $f"; fi
      echo "  seed $seed logs:             $out/logs/"
      if [ -s "$out/experiment.json" ]; then
        "$PY" src/downstream_status.py --out "$out"
        # arms added after preparation live in arms.json; show their state too
        [ -s "$out/arms.json" ] && "$PY" src/evidence_downstream.py status --out "$out" | grep -E "^  (passrate_beta|g00|g10|g01|g11|fresh_r|random|gate_passrate) " | grep -vFf <("$PY" -c 'import json,sys; print("\n".join("  "+a+" " for a in json.load(open(sys.argv[1]))["selectors"]))' "$out/experiment.json") | sed 's/^/  [added]/'
        [ -s "$out/downstream_results.csv" ] && { echo "  results:"; sed 's/^/    /' "$out/downstream_results.csv"; }
      else
        echo "seed $seed: not prepared"
      fi
    done
  done
  exit 0
fi

# 0. an earlier E5 launch on this node (dropped session) is stopped and resumed by the launcher itself
# 1. source points must be complete
runs=()
for seed in "${SEEDS[@]}"; do
  run=$(run_dir "$seed")
  [ -s "$run/DONE" ] || { echo "[abort] source point is not complete: $run"; exit 1; }
  runs+=("$run")
done
# 2. held-out pool (fetched once in an online shell)
[ -s "$POOL" ] && [ -s "$POOL_MANIFEST" ] || {
  echo "[abort] held-out MATH-train pool missing: $POOL"
  echo "        run once in an online shell:  bash scripts/fetch_math_train.sh"
  exit 1
}
REVISION=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_revision"])' "$POOL_MANIFEST")
# 3. freeze the independent test set (idempotent; shared by all seeds and nodes)
"$PY" src/evidence_downstream.py prepare-test --candidates "$POOL" --runs "${runs[@]}" --out "$TEST" \
  --count "$COUNT" --dataset EleutherAI/hendrycks_math --revision "$REVISION" --split train || exit 1
if [ "$MODE" = run ]; then
  source scripts/_e5_node.sh || exit 1
  e5_cleanup_previous "$OUT_ROOT" || exit 1
  # Own this node for the entire seed pass, not separately for every child.
  if [ "${OM_NODE_LOCK_HELD:-0}" != 1 ]; then e5_acquire_node || exit "$?"; fi
fi
# 4. one seed after another; arms are leased inside the launcher
rc_all=0
for seed in "${SEEDS[@]}"; do
  run=$(run_dir "$seed"); out="$OUT_ROOT/s$seed"
  echo "== seed $seed: $run"
  if [ "$MODE" = plan ]; then
    DOWNSTREAM_SELECTORS="$SELECTORS" bash scripts/run_downstream_independent.sh "$run" "$out" \
      --eval-prompts "$TEST" --steps "$STEPS" --eval-k "$EVAL_K" --dry-run | head -20 || rc_all=1
    continue
  fi
  OM_NODE_LOCK_HELD=1 OM_E5_CONTROLLER_PID="$$" DOWNSTREAM_SELECTORS="$SELECTORS" \
    bash scripts/run_downstream_independent.sh "$run" "$out" \
    --eval-prompts "$TEST" --steps "$STEPS" --eval-k "$EVAL_K" 7>&- 8>&-
  rc=$?
  if [ "$rc" -eq 75 ]; then echo "[abort] E5 admission failed; see the actual owner or resource error above"; exit 75; fi
  if [ "$rc" -eq 130 ] || [ "$rc" -eq 143 ]; then echo "[e5] stopped by signal; nothing else will be started"; exit "$rc"; fi
  [ "$rc" -eq 0 ] || rc_all=1
done
echo "[e5] pass complete; check:  bash scripts/run_e5.sh status"
exit "$rc_all"
