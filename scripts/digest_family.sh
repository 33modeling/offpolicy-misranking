#!/usr/bin/env bash
# Text-only hand-over for a network that blocks archives: one plain-text file
# (a few hundred KB at most) with everything needed to read a family's result
# and to diagnose its errors. Read-only for the experiment; writes only under
# $OM_WORK/exports and $OM_WORK/readouts.
#
#   bash scripts/digest_family.sh                    # h100: every family with all four points DONE
#   bash scripts/digest_family.sh math500 0          # one family (finished or not)
#   bash scripts/digest_family.sh baseline math500 0
#   DIGEST_READOUT=0 ...                             # skip the regime readout (fast)
#
# Sections per family: run_config essentials, per-point report.json,
# divergence_stats.json, GRPO statistics (first/last steps), scores summary
# (count, mean, sign agreement with the fresh reference), the regime readout
# tables, the error census (every CUDA/OOM/Runtime error line with the last
# code frame before it, counted by frame), and the tail of each stage log.
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

family_complete() {
  local d
  for d in $DRIFTS; do [ -s "$ROOT/family-$1-s$2/$TAG-s$2-$1-d$d/DONE" ] || return 1; done
}
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
    family_complete "$dataset" "$seed" && families+=("$dataset $seed")
  done
  [ "${#families[@]}" -gt 0 ] || { echo "[digest] no family has all four points DONE under $ROOT; name one: bash scripts/digest_family.sh <dataset> <seed>"; exit 1; }
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EXPORTS="$OM_WORK/exports"; mkdir -p "$EXPORTS" || { echo "[abort] cannot create $EXPORTS"; exit 1; }
label=$(printf '%s\n' "${families[@]}" | awk '{printf "%s%s-s%s", (NR>1?"_":""), $1, $2}')
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
scores_summary() {  # scores_summary <run>
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
  for fam in "${families[@]}"; do
    set -- $fam; dataset=$1; seed=$2
    froot="$ROOT/family-$dataset-s$seed"
    section "FAMILY $dataset/s$seed"
    for d in $DRIFTS; do
      run="$froot/$TAG-s$seed-$dataset-d$d"
      section "point d$d  $( [ -s "$run/DONE" ] && echo DONE || echo "not done" )  $(basename "$run")"
      [ -d "$run" ] || { echo "(no directory)"; continue; }
      echo "--- run_config essentials"
      "$PY" - "$run/run_config.json" <<'PYEOF' 2>/dev/null || echo "(no run_config.json)"
import json, sys
c = json.load(open(sys.argv[1]))
keys = ["dataset","seed","drift","n_train","n_val","behavior_k","fresh_k","val_k","max_new_tokens","gen_batch",
        "gradient_micro_batch","grpo_logprob_micro_batch","prompt_format","attn","git","training_objective"]
print(" ".join(f"{k}={c.get(k)}" for k in keys))
PYEOF
      echo "--- report.json"; show_json "$run/report.json" 120
      echo "--- divergence_stats.json"; show_json "$run/divergence_stats.json" 40
      echo "--- scores"; scores_summary "$run"
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
    if [ "${DIGEST_READOUT:-1}" = 1 ] && family_complete "$dataset" "$seed"; then
      bash scripts/family_readout.sh "$PROFILE" "$dataset" "$seed" "${DIGEST_BOOT:-1000}" 2>&1 | tail -n 120 | cut -c1-220
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
