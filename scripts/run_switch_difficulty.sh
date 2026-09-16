#!/usr/bin/env bash
# Cached-difficulty variant of the selection-switch experiment: the same five
# certified prefixes, the same 15 states and the same 29,040 GPU-second
# allocation, but the selection arms rank the pool from the pre-continuation
# reward cache (the 10% of the pool closest to a 0.5 cached success rate) instead of
# rescoring with fresh responses and gradients. Selection then costs a metered
# read, so the arms complete about as many updates as random and the gate's
# decision rests on the per-update gain alone. Its own root, ledgers,
# development labels and gate. One node launcher per node; the MoPPS pass is
# skipped. Same modes as run_selection_switch.sh (run|stop|status|why|waive|...).
#
#   bash scripts/run_switch_difficulty.sh          prepare on first use, then run
#   bash scripts/run_switch_difficulty.sh status
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export SWITCH_ROOT=${SWITCH_DIFFICULTY_ROOT:-$WORK/runs/selection-switch-difficulty-v1}
export SWITCH_PREFIX_SOURCE=${SWITCH_PREFIX_SOURCE:-$WORK/runs/selection-switch-v1}
export SWITCH_SELECTOR=difficulty
# Convergence gate: every branch also evaluates held-out reward at three archived
# checkpoints (SWITCH_CURVE_POINTS, SWITCH_CURVE_K responses per question) and the
# gate selects when selection reaches the common target reward with fewer updates
# than random after paying its scoring cost in update units.
export SWITCH_GATE=${SWITCH_GATE:-convergence}
export EXPERIMENTS_SKIP_MOPPS=1
exec bash scripts/run_selection_switch.sh "$@"
