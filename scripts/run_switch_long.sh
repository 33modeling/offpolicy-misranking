#!/usr/bin/env bash
# Longer-horizon variant of the selection-switch experiment: the same five
# certified prefixes and the same 15 states, with three times the continuation
# allocation (87,120 GPU-seconds, a 300-update equivalent) so that fresh
# gradient scoring is a smaller share of each branch. Its own root, ledgers,
# development labels and gate. One node launcher per node; the MoPPS pass is
# skipped. Same modes as run_selection_switch.sh (run|stop|status|why|waive|...).
#
#   bash scripts/run_switch_long.sh          prepare on first use, then run
#   bash scripts/run_switch_long.sh status
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export SWITCH_ROOT=${SWITCH_LONG_ROOT:-$WORK/runs/selection-switch-long-v1}
export SWITCH_PREFIX_SOURCE=${SWITCH_PREFIX_SOURCE:-$WORK/runs/selection-switch-v1}
export SWITCH_BUDGET_GPU_SECONDS=${SWITCH_BUDGET_GPU_SECONDS:-87120}
export EXPERIMENTS_SKIP_MOPPS=1
exec bash scripts/run_selection_switch.sh "$@"
