#!/usr/bin/env bash
# Text-only hand-over for a network that blocks archives: one plain-text file
# (a few hundred KB at most) with everything needed to read a family's result
# and to diagnose its errors. Read-only for the experiment; writes only under
# $OM_WORK/exports and $OM_WORK/readouts.
#
#   bash scripts/digest_family.sh                    # h100: every family with at least one completed point
#   bash scripts/digest_family.sh math500 0          # one family (finished or not)
#   bash scripts/digest_family.sh baseline math500 0
#   DIGEST_READOUT=1 ...                             # also run the slow regime readout (1000 bootstrap per family)
#
# A completed point is one with DONE now, or one whose DONE was parked under
# pinned-scoring/<stamp>/ by scripts/rescore_math500.sh (RESCORE PENDING: the
# pinned scoring is still readable there while a worker recomputes the
# corrected one). Both scorings are shown when both exist, so the pinned versus
# corrected comparison of paper plan §9.1 can be read from one file.
#
# Sections: KEY NUMBERS table for every family and point first (floor, chance,
# fresh and stale precisions, KL, ESS, per scoring), then per family:
# run_config essentials, per-point report.json, divergence_stats.json, GRPO
# statistics (first/last steps), scores summary (count, mean, sign agreement
# with the fresh reference), the regime readout tables, the error census (every
# CUDA/OOM/Runtime error line with the last code frame before it, counted by
# frame), and the tail of each stage log.
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
PROFILE=h100
case "${1:-}" in baseline|h100) PROFILE=$1; shift ;; esac
case "$PROFILE" in
  baseline) TAG=olmo3-1025-7b-base-rlzero-grpo-v1 ;;
  h100)     TAG=olmo3-1025-7b-base-rlzero-grpo-h100-v2 ;;
esac
TAG="${OM_OLMO3_MODEL_TAG:-$TAG}"
ROOT="${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}"
DRIFTS="${DIGEST_DRIFTS:-0 25 100 400}"
TAIL_LINES="${DIGEST_TAIL_LINES:-40}"
[ -d "$ROOT" ] || { echo "[abort] no experiment root: $ROOT"; exit 1; }

point_dir() { echo "$ROOT/family-$1-s$2/$TAG-s$2-$1-d$3"; }
point_done() { [ -s "$(point_dir "$1" "$2" "$3")/DONE" ]; }
point_parked() {  # DONE moved aside by rescoring; the pinned scoring sits under pinned-scoring/<stamp>/
  local f
  for f in "$(point_dir "$1" "$2" "$3")"/pinned-scoring/*/DONE; do [ -s "$f" ] && return 0; done
  return 1
}
latest_parking() {  # latest_parking <run> -> newest pinned-scoring/<stamp> dir, or nothing
  local d last=
  for d in "$1"/pinned-scoring/*/; do [ -d "$d" ] && last=${d%/}; done
  [ -n "$last" ] && echo "$last"
}
point_state() {  # point_state <dataset> <seed> <drift> -> DONE | RESCORE PENDING (...) | not done
  local run; run=$(point_dir "$1" "$2" "$3")
  if [ -s "$run/DONE" ]; then
    if point_parked "$1" "$2" "$3"; then echo "DONE (corrected scoring; pinned scoring parked)"; else echo DONE; fi
  elif point_parked "$1" "$2" "$3"; then
    echo "RESCORE PENDING (pinned scoring parked under $(basename "$(latest_parking "$run")"); waiting for a GPU worker)"
  else
    echo "not done"
  fi
}
family_complete() { local d; for d in $DRIFTS; do point_done "$1" "$2" "$d" || return 1; done; }
family_started() { local d; for d in $DRIFTS; do { point_done "$1" "$2" "$d" || point_parked "$1" "$2" "$d"; } && return 0; done; return 1; }

families=()
if [ "$#" -ge 2 ]; then
  case "$2" in ''|*[!0-9]*) echo "usage: bash scripts/digest_family.sh [h100|baseline] [<dataset> <seed>]"; exit 2 ;; esac
  [ -d "$ROOT/family-$1-s$2" ] || { echo "[abort] no such family: $ROOT/family-$1-s$2"; exit 1; }
  families+=("$1 $2")
elif [ "$#" -ne 0 ]; then
  echo "usage: bash scripts/digest_family.sh [h100|baseline] [<dataset> <seed>]"; exit 2
else
  for dir in "$ROOT"/family-*; do
    [ -d "$dir" ] || continue
    name=${dir##*/family-}; dataset=${name%-s*}; seed=${name##*-s}
    family_started "$dataset" "$seed" && families+=("$dataset $seed")
  done
  [ "${#families[@]}" -gt 0 ] || { echo "[digest] no family has a completed point (DONE, or DONE parked by rescoring) under $ROOT; name one: bash scripts/digest_family.sh <dataset> <seed>"; exit 1; }
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EXPORTS="$OM_WORK/exports"; mkdir -p "$EXPORTS" || { echo "[abort] cannot create $EXPORTS"; exit 1; }
if [ "${#families[@]}" -gt 3 ]; then label="${#families[@]}families"; else label=$(printf '%s\n' "${families[@]}" | awk '{printf "%s%s-s%s", (NR>1?"_":""), $1, $2}'); fi
OUT="$EXPORTS/digest-$TAG-$label-$STAMP.txt"

section() { printf '\n===== %s =====\n' "$*"; }
show_json() {  # show_json <file> [max lines]
  [ -s "$1" ] || { echo "(missing: $1)"; return; }
  "$PY" - "$1" "${2:-200}" <<'PYEOF'
import json, sys
path, limit = sys.argv[1], int(sys.argv[2])
try:
    doc = json.load(open(path))
except Exception as exc:
    print(f"(unreadable: {exc})"); sys.exit(0)
text = json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False)
lines = text.splitlines()
print("\n".join(lines[:limit]))
if len(lines) > limit: print(f"... ({len(lines) - limit} more lines)")
PYEOF
}
key_numbers() {  # key_numbers <run> <scoring dir> <label> -> one line: floor, chance, fresh/stale precisions, KL, ESS
  "$PY" - "$1" "$2" "$3" <<'PYEOF'
import json, sys
from pathlib import Path
run, scoring, label = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None
def fmt(value, digits=3):
    if isinstance(value, bool) or value is None: return "-"
    try: return f"{float(value):.{digits}f}"
    except (TypeError, ValueError): return str(value)
report = load(scoring / "report.json"); div = load(scoring / "divergence_stats.json") or {}
config = load(run / "run_config.json") or {}
if not report:
    print(f"{label:<9} (no report.json)"); sys.exit(0)
n = config.get("n_train"); k = report.get("k")
chance = (k / n) if isinstance(n, (int, float)) and n and isinstance(k, (int, float)) else None
floor = report.get("noise_floor")
gate = "-"
if isinstance(floor, (int, float)) and chance is not None:
    gate = "floor>=2*chance" if floor >= 2 * chance else "floor<2*chance"
fresh = (report.get("certagrad") or {}).get("precision_vs_oracle")
stale = " ".join(f"{e}={fmt((report.get(e) or {}).get('precision'))}" for e in ("g00", "g01", "g10", "g11"))
print(f"{label:<9} floor={fmt(floor)} chance={fmt(chance)} {gate:<15} fresh={fmt(fresh)} {stale} "
      f"KL={fmt(div.get('token_kl_beta_pi'), 6)} ESS={fmt(div.get('traj_ess_frac_g11'))}")
PYEOF
}
scorings() {  # scorings <run> -> lines "<label> <dir>": the current scoring and/or the newest parked pinned scoring
  local park
  [ -s "$1/report.json" ] && echo "current $1"
  park=$(latest_parking "$1")
  [ -n "$park" ] && [ -s "$park/report.json" ] && echo "pinned $park"
  return 0
}
scores_summary() {  # scores_summary <scoring dir>
  "$PY" - "$1" <<'PYEOF'
import json, sys, statistics
from pathlib import Path
run = Path(sys.argv[1])
def load(name):
    p = run / name
    return json.load(open(p)) if p.is_file() else None
split = load("scores_splithalf.json"); off = load("scores_offpolicy.json"); oracle = load("scores_oracle.json")
if not split:
    print("(no scores_splithalf.json)"); sys.exit(0)
fresh = {k: v["r"] for k, v in split.items() if isinstance(v, dict) and "r" in v}
print(f"prompts scored: {len(fresh)}  fresh r mean={statistics.fmean(fresh.values()):.4f}")
if off:
    for est, table in sorted(off.items()):
        vals = {k: (v["score"] if isinstance(v, dict) else v) for k, v in table.items()}
        common = [k for k in vals if k in fresh and vals[k] != 0 and fresh[k] != 0]
        flips = sum(1 for k in common if vals[k] * fresh[k] < 0)
        print(f"  {est}: n={len(vals)} mean={statistics.fmean(vals.values()):.4f} sign-flips vs fresh={flips}/{len(common)}")
if oracle:
    vals = [(v["score"] if isinstance(v, dict) else v) for v in oracle.values()]
    print(f"  oracle: n={len(vals)} mean={statistics.fmean(vals):.4f}")
PYEOF
}
error_census() {  # error_census <family root>
  "$PY" - "$1" <<'PYEOF'
import re, sys, collections
from pathlib import Path
root = Path(sys.argv[1])
pat = re.compile(r"unspecified launch failure|CUBLAS_STATUS|device-side assert|illegal memory access|OutOfMemoryError|CUDA error")
frames = collections.Counter(); stages = collections.Counter(); samples = {}
for log in sorted(root.glob("*/logs/*.log")):
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        continue
    for i, line in enumerate(lines):
        if not pat.search(line) or "Compile with" in line or line.lstrip().startswith("!"):
            continue
        # last code frame before the error line ("  File ..., line N, in fn")
        frame = "?"
        for back in range(i - 1, max(-1, i - 40), -1):
            if lines[back].lstrip().startswith("File "):
                frame = lines[back].strip(); break
        key = (pat.search(line).group(0), frame)
        frames[key] += 1
        stages[log.name.split(".")[0]] += 1
        samples.setdefault(key, f"{log.parent.parent.name}/logs/{log.name}:{i + 1}")
print(f"error lines: {sum(frames.values())}")
print("by stage log:", ", ".join(f"{k}={v}" for k, v in stages.most_common()))
for (kind, frame), n in frames.most_common(25):
    print(f"  {n:4d}  {kind}  <- {frame}   e.g. {samples[(kind, frame)]}")
PYEOF
}

{
  echo "digest=$(basename "$OUT")  created_utc=$STAMP  host=$(hostname 2>/dev/null || echo ?)"
  echo "checkout=$(git rev-parse HEAD 2>/dev/null || echo ?)  generation_git=$(cat "$ROOT/.queue/generation.git" 2>/dev/null || echo none)"
  echo "profile=$PROFILE tag=$TAG root=$ROOT families=$(printf '%s;' "${families[@]}")"
  section "KEY NUMBERS (one line per point and scoring; 'pinned' = parked by rescoring, 'current' = what is on disk now)"
  echo "gate: paper plan §7 needs the one-sided 95% lower bound of floor >= 2*chance; the point estimate here is the optimistic check"
  for fam in "${families[@]}"; do
    set -- $fam; dataset=$1; seed=$2
    for d in $DRIFTS; do
      run=$(point_dir "$dataset" "$seed" "$d")
      [ -d "$run" ] || continue
      state=$(point_state "$dataset" "$seed" "$d")
      any=0
      while read -r kind dir; do
        [ -n "$kind" ] || continue
        any=1
        key_numbers "$run" "$dir" "$dataset/s$seed/d$d $kind"
      done < <(scorings "$run")
      [ "$any" = 1 ] || printf '%-9s %s\n' "$dataset/s$seed/d$d" "(no scoring on disk: $state)"
    done
  done
  for fam in "${families[@]}"; do
    set -- $fam; dataset=$1; seed=$2
    froot="$ROOT/family-$dataset-s$seed"
    section "FAMILY $dataset/s$seed"
    for d in $DRIFTS; do
      run=$(point_dir "$dataset" "$seed" "$d")
      section "point d$d  $(point_state "$dataset" "$seed" "$d")  $(basename "$run")"
      [ -d "$run" ] || { echo "(no directory)"; continue; }
      echo "--- run_config essentials"
      "$PY" - "$run/run_config.json" <<'PYEOF' 2>/dev/null || echo "(no run_config.json)"
import json, sys
c = json.load(open(sys.argv[1]))
keys = ["dataset","seed","drift","n_train","n_val","behavior_k","fresh_k","val_k","max_new_tokens","gen_batch",
        "gradient_micro_batch","grpo_logprob_micro_batch","prompt_format","attn","git","training_objective"]
print(" ".join(f"{k}={c.get(k)}" for k in keys))
PYEOF
      sidecars=$(ls "$run"/*.rescore.json 2>/dev/null | wc -l)
      [ "$sidecars" -eq 0 ] || echo "--- rescore sidecars: $sidecars ($(ls "$run"/*.rescore.json | xargs -n1 basename | tr '\n' ' '))"
      any=0
      while read -r kind dir; do
        [ -n "$kind" ] || continue
        any=1
        echo "--- scoring: $kind  ($dir)"
        echo "--- report.json"; show_json "$dir/report.json" 120
        echo "--- divergence_stats.json"; show_json "$dir/divergence_stats.json" 40
        echo "--- scores"; scores_summary "$dir"
      done < <(scorings "$run")
      [ "$any" = 1 ] || { echo "--- report.json"; echo "(missing: $run/report.json; no parked pinned scoring either)"; }
      for stats in "$run"/policy_step_*/grpo_stats.jsonl; do
        [ -s "$stats" ] || continue
        echo "--- $(basename "$(dirname "$stats")")/grpo_stats.jsonl: $(wc -l < "$stats") steps; first/last:"
        head -1 "$stats" | cut -c1-300; tail -1 "$stats" | cut -c1-300
      done
      for log in "$run"/logs/main.log "$run"/logs/grpo.log; do
        [ -s "$log" ] || continue
        echo "--- $(basename "$log") tail $TAIL_LINES"; tail -n "$TAIL_LINES" "$log" | cut -c1-220
      done
    done
    section "ERROR CENSUS $dataset/s$seed (every error line, grouped by the code frame before it)"
    error_census "$froot"
    section "READOUT $dataset/s$seed"
    if [ "${DIGEST_READOUT:-0}" = 1 ] && family_complete "$dataset" "$seed"; then
      bash scripts/family_readout.sh "$PROFILE" "$dataset" "$seed" "${DIGEST_BOOT:-1000}" 2>&1 | tail -n 120 | cut -c1-220
    elif ! family_complete "$dataset" "$seed"; then
      echo "(no new readout: not all four points have DONE now; earlier readouts, if any, follow)"
    else
      echo "(readout skipped by default; earlier readouts, if any, follow)"
    fi
    for rd in "$OM_WORK/readouts"/family-$dataset-s$seed-*; do
      [ -d "$rd" ] || continue
      echo "--- $(basename "$rd")"
      for f in REGIME_SUMMARY.csv REGIME.csv FINAL_REPORT.md; do
        [ -s "$rd/$f" ] || continue
        echo "--- $f"; head -n 80 "$rd/$f" | cut -c1-220
      done
    done
  done
  section "ALERTS tail"; tail -n 30 "$ROOT/logs/ALERTS.log" 2>/dev/null | cut -c1-200
} > "$OUT" 2>&1
size=$(stat -c %s "$OUT")
echo "[digest] $OUT ($((size / 1024)) KB, $(wc -l < "$OUT") lines)"
echo "[digest] plain text: copy it into the transfer repository and push"
