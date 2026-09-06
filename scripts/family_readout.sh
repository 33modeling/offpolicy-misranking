#!/usr/bin/env bash
# Provisional analysis of ONE finished OLMo-3 family (its four points), without
# waiting for the other nine. Prints the regime report to the terminal and
# writes it under $OM_WORK/readouts/. Nothing here touches running workers.
#   bash scripts/family_readout.sh h100 mbpp 0
#   bash scripts/family_readout.sh h100 mbpp 0 10000   # final-quality bootstrap (slow)
set -uo pipefail
cd "$(dirname "$0")/.."
# Arguments: [profile] <dataset> <seed> [bootstrap]. Profile defaults to h100, so
#   bash scripts/family_readout.sh mbpp 0        and
#   bash scripts/family_readout.sh h100 mbpp 0   both work.
PROFILE=auto
case "${1:-}" in baseline|h100) PROFILE=$1; shift ;; esac
DATASET=${1:-}; SEED=${2:-}; BOOT=${3:-1000}
case "$DATASET" in
  math500|mbpp) ;;
  *) echo "usage: bash scripts/family_readout.sh [h100|baseline] <math500|mbpp> <seed 0-4> [bootstrap, default 1000]"; exit 2 ;;
esac
case "$SEED" in ''|*[!0-9]*) echo "usage: seed must be a number 0-4 (got '$SEED')"; exit 2 ;; esac
case "$BOOT" in ''|*[!0-9]*) echo "usage: bootstrap must be a number >= 100 (got '$BOOT')"; exit 2 ;; esac
[ "$BOOT" -ge 100 ] || { echo "usage: bootstrap must be >= 100"; exit 2; }
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
tag_for() { case "$1" in baseline) echo olmo3-1025-7b-base-rlzero-grpo-v1 ;; h100) echo olmo3-1025-7b-base-rlzero-grpo-h100-v2 ;; esac; }
done_points() {  # done_points <root> <tag> -> number of DONE points in this family
  local n=0 d
  for d in 0 25 100 400; do
    [ -s "$1/family-$DATASET-s$SEED/$2-s$SEED-$DATASET-d$d/DONE" ] && n=$((n + 1))
  done
  echo "$n"
}
if [ "$PROFILE" = auto ]; then
  # No profile given: use whichever experiment root actually holds this family
  # (status h100 and status baseline look at different roots; so does this).
  best=h100; best_n=-1
  for cand in h100 baseline; do
    n=$(done_points "$OM_WORK/runs/$(tag_for "$cand")" "$(tag_for "$cand")")
    [ "$n" -gt "$best_n" ] && { best=$cand; best_n=$n; }
  done
  PROFILE=$best
fi
MODEL_TAG=${OM_OLMO3_MODEL_TAG:-$(tag_for "$PROFILE")}
ROOT="${OM_OLMO3_ROOT:-$OM_WORK/runs/$MODEL_TAG}"
FAMILY="$ROOT/family-$DATASET-s$SEED"
runs=(); missing=()
for d in 0 25 100 400; do
  run="$FAMILY/$MODEL_TAG-s$SEED-$DATASET-d$d"
  if [ -s "$run/DONE" ] && [ -s "$run/report.json" ]; then runs+=("$run"); else missing+=("d$d"); fi
done
echo "root     $ROOT  (profile $PROFILE)"
echo "family   $DATASET/s$SEED   points done: ${#runs[@]}/4${missing:+   missing: ${missing[*]}}"
if [ "${#runs[@]}" -eq 0 ]; then
  if [ -d "$FAMILY" ]; then
    echo "family dir exists; contents:"
    for d in "$FAMILY"/*/; do
      [ -d "$d" ] || continue
      printf '   %-50s DONE=%s report.json=%s\n' "$(basename "$d")" "$([ -s "$d/DONE" ] && echo yes || echo no)" "$([ -s "$d/report.json" ] && echo yes || echo no)"
    done
  else
    echo "family dir does not exist: $FAMILY"
  fi
  others=$(ls -d "$OM_WORK"/runs/*/family-"$DATASET"-s"$SEED" 2>/dev/null | grep -v "^$FAMILY$" || true)
  [ -z "$others" ] || { echo "same family under another root (try the other profile: baseline|h100):"; printf '   %s\n' $others; }
  echo "DECISION nothing to analyse yet: no point of $DATASET/s$SEED has DONE+report.json under this root"
  exit 1
fi
OUT="$OM_WORK/readouts/family-$DATASET-s$SEED-$(git rev-parse --short HEAD)-boot$BOOT"
mkdir -p "$OUT"
echo "output   $OUT"
echo "bootstrap $BOOT replicates (final requires 10000; smaller = provisional)"
echo
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" src/regime_map.py "${runs[@]}" \
  --output-dir "$OUT" --first-bootstrap "$BOOT" 2>&1 | tee "$OUT/readout.log"
rc=${PIPESTATUS[0]}
echo
if [ "$rc" -eq 0 ]; then
  echo "DECISION report written: $OUT/FINAL_REPORT.md  (also REGIME.csv, REGIME_SUMMARY.csv, REGIME.json)"
  echo "handover: paste the table above, or copy $OUT off the cluster (scp) - it is text only, a few hundred KB"
  exit 0
fi
echo "regime report failed (rc=$rc, see above). Falling back to the per-point gate numbers, which need no bootstrap:"
echo
printf ' %-6s %-12s %-10s %-10s %-10s %-10s\n' point noise_floor g00 g10 g01 g11
for run in "${runs[@]}"; do
  "$PY" - "$run" <<'PYEOF'
import json, sys
from pathlib import Path
run = Path(sys.argv[1]); r = json.loads((run / "report.json").read_text())
d = run.name.rsplit("-d", 1)[-1]
def p(k):
    v = r.get(k, {}); v = v.get("precision", v) if isinstance(v, dict) else v
    return f"{float(v):.3f}" if isinstance(v, (int, float)) else "-"
print(f" d{d:<5} {float(r.get('noise_floor', float('nan'))):<12.3f} {p('g00'):<10} {p('g10'):<10} {p('g01'):<10} {p('g11'):<10}")
PYEOF
done
echo
echo " noise_floor = split-half ceiling; gNN = top-k precision of each off-policy estimator (higher = better ranking)"
echo "DECISION regime report failed; the table above is the raw per-point readout. Send the failing line from $OUT/readout.log"
exit "$rc"
