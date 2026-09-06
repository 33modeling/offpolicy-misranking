#!/usr/bin/env bash
# One short command, one block of output to copy back. Read-only: touches nothing.
#   bash scripts/info.sh            # OLMo h100 matrix
#   bash scripts/info.sh baseline   # the other profile
set -uo pipefail
cd "$(dirname "$0")/.."
PROFILE=${1:-h100}
case "$PROFILE" in
  baseline) TAG=olmo3-1025-7b-base-rlzero-grpo-v1 ;;
  h100)     TAG=olmo3-1025-7b-base-rlzero-grpo-h100-v2 ;;
  *) echo "usage: bash scripts/info.sh [h100|baseline]"; exit 2 ;;
esac
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
ROOT="${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}"

echo "===== INFO $(date -u +%FT%TZ) host=$(hostname) profile=$PROFILE ====="
echo "CHECKOUT=$(git rev-parse HEAD 2>/dev/null)"
echo "MARKER=$(cat "$ROOT/.queue/generation.git" 2>/dev/null || echo none)"
echo "ROOT=$ROOT"
echo "--- families (points DONE / code that generated them / last error) ---"
for fam in "$ROOT"/family-*; do
  [ -d "$fam" ] || continue
  name=$(basename "$fam" | sed 's/^family-//')
  done_n=0; gits=""
  for d in 0 25 100 400; do
    run="$fam/$TAG-s${name##*-s}-${name%%-s*}-d$d"
    [ -s "$run/DONE" ] && done_n=$((done_n + 1))
    g=$(sed -n 's/.*"git": *"\([0-9a-f]\{7\}\).*/\1/p' "$run/run_config.json" 2>/dev/null | head -1)
    [ -z "$g" ] || gits="$gits $d:$g"
  done
  stamp=no; [ -s "$fam/.family-complete" ] && stamp=yes
  err=$(grep -hoE 'torch\.OutOfMemoryError[^)]*|CUDA error[^)]*|RuntimeError[^)]*' "$fam"/*/logs/*.log 2>/dev/null | tail -1 | cut -c1-70)
  printf '%-12s done=%s/4 stamp=%s code=%s %s\n' "$name" "$done_n" "$stamp" "${gits:- none}" "${err:+| $err}"
done
echo "--- workers (heartbeat age, seconds) ---"
now=$(date +%s)
for w in "$ROOT"/.workers/*.json; do
  [ -f "$w" ] || { echo "none"; break; }
  age=$(( now - $(stat -c %Y "$w") ))
  state=$(sed -n 's/.*"state":"\([a-z-]*\)".*/\1/p' "$w")
  printf '%-40s %ss %s\n' "$(basename "$w" .json)" "$age" "${state:-?}"
done
echo "--- oom count per family (all logs) ---"
grep -rlc "OutOfMemoryError" "$ROOT"/family-*/*/logs/*.log 2>/dev/null | sed "s|$ROOT/family-||" | cut -d/ -f1 | sort | uniq -c | sed 's/^/  /' || echo "  none"
echo "===== END ====="
