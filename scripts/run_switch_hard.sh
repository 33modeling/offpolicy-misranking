#!/usr/bin/env bash
# Cached-hard variant of the selection-switch experiment: the same five
# certified prefixes, the same 15 states and the same 29,040 GPU-second
# allocation, but the selection arms rank the pool from the pre-continuation
# reward cache (the 10% of the pool with the lowest cached success rate among prompts solved at least once) instead of
# rescoring with fresh responses and gradients. Selection then costs a metered
# read, so the arms complete about as many updates as random and the gate's
# decision rests on the per-update gain alone. Its own root, ledgers,
# development labels and gate. One node launcher per node; the MoPPS pass is
# skipped. Same modes as run_selection_switch.sh (run|stop|status|why|waive|...).
#
#   bash scripts/run_switch_hard.sh          prepare on first use, then run
#   bash scripts/run_switch_hard.sh status
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export SWITCH_ROOT=${SWITCH_HARD_ROOT:-$WORK/runs/selection-switch-hard-v1}
export SWITCH_PREFIX_SOURCE=${SWITCH_PREFIX_SOURCE:-$WORK/runs/selection-switch-v1}
export SWITCH_SELECTOR=hard
export EXPERIMENTS_SKIP_MOPPS=1
exec bash scripts/run_selection_switch.sh "$@"
