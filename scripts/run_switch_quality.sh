#!/usr/bin/env bash
# Data-quality variant of the selection-switch experiment: the same five
# certified prefixes, the same 15 states and evaluation set, fresh gradient
# selection as in the primary run, but matched-update accounting: fresh scoring
# is metered on a separate "scoring" ledger that the allocation does not count,
# so the selection arms train as many updates as random. The comparison is
# then how much better fresh-selected data learns per update, and the
# convergence gate learns at which states that advantage still saves updates.
# Scoring cost is recorded, reported, and subtracted from the gate label in update
# units (not from the training allocation). Own root, labels,
# gate and ledgers; the MoPPS pass is skipped. Same modes as the switch launcher.
#
#   bash scripts/run_switch_quality.sh          prepare on first use, then run
#   bash scripts/run_switch_quality.sh status
#   bash scripts/run_switch_quality.sh pilot    held-out fresh vs random only
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export SWITCH_ROOT=${SWITCH_QUALITY_ROOT:-$WORK/runs/selection-switch-quality-v1}
export SWITCH_PREFIX_SOURCE=${SWITCH_PREFIX_SOURCE:-$WORK/runs/selection-switch-v1}
export SWITCH_SELECTOR=fresh_r
export SWITCH_ACCOUNTING=matched
export SWITCH_GATE=${SWITCH_GATE:-convergence}
export EXPERIMENTS_SKIP_MOPPS=1
if [ "${1:-run}" = pilot ]; then
  export SWITCH_ONLY_SEEDS=3,4 SWITCH_ONLY_ARMS=selection_full,random_full
  shift
  set -- run "$@"
fi
exec bash scripts/run_selection_switch.sh "$@"
