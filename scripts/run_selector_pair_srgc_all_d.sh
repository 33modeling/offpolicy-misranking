#!/usr/bin/env bash
# Diagnostic D measurements on every saved 25-step On checkpoint; no training.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
log="$HOME/srgc-all-d-$(hostname).log"
output="$HOME/srgc-all-d.txt"
bash scripts/run_selector_pair_srgc_repeat.sh all-measure --out "$output" "$@" 2>&1 | tee -a "$log"
measure_status=${PIPESTATUS[0]}
bash scripts/run_selector_pair_srgc_repeat.sh all-results --out "$output" "$@"
results_status=$?
printf '[SR-GC all-D] result: %s\n[SR-GC all-D] log: %s\n' "$output" "$log"
if [ "$results_status" -ne 0 ]; then
  exit "$results_status"
fi
exit "$measure_status"
