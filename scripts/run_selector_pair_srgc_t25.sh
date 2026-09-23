#!/usr/bin/env bash
# Resume the single t25 repeated check and write a fresh read-only result.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

node=$(hostname) || exit 1
log="$HOME/srgc-repeat-$node.log"
snapshot="$HOME/srgc-repeat-$node.txt"
latest="$HOME/step-latest.txt"

bash scripts/run_selector_pair_srgc_repeat.sh measure --interval 25 \
  --out "$snapshot" 2>&1 | tee -a "$log"
measure_status=${PIPESTATUS[0]}

bash scripts/run_selector_pair_srgc_repeat.sh results --interval 25 \
  --out "$latest"
results_status=$?
printf '\n[SR-GC t25] latest result: %s\n[SR-GC t25] log: %s\n' "$latest" "$log"
if [ "$results_status" -eq 0 ]; then
  grep -E '^s[0-9]+-t25' "$latest" || true
else
  exit "$results_status"
fi
exit "$measure_status"
