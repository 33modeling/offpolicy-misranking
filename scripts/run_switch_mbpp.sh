#!/usr/bin/env bash
# MBPP (code) variant of the selection-switch experiment: the same protocol on
# the OLMo MBPP matrix family (512-prompt pool, top 10% = 51, execution-verified
# rewards): five on-policy-gradient-selected prefixes, 18 development and 30
# held-out continuations, one budget per branch derived from the MBPP seed-0
# d100 update timings, an independent MBPP test set disjoint from the runs'
# prompts. Its own root, ledgers, gate and status; the MoPPS pass is skipped.
#
#   bash scripts/run_switch_mbpp.sh          prepare on first use, then run
#   bash scripts/run_switch_mbpp.sh status
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
export SWITCH_ROOT=${SWITCH_MBPP_ROOT:-$WORK/runs/selection-switch-mbpp-v1}
export SWITCH_DATASET=mbpp
export EXPERIMENTS_SKIP_MOPPS=1
exec bash scripts/run_selection_switch.sh "$@"
