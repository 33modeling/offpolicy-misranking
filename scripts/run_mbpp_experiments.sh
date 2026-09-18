#!/usr/bin/env bash
# MBPP counterpart of the fresh / quality / difficulty switch suites, as one
# command per node. This script chooses the roots and then hands the node to
# scripts/run_experiments.sh, which already owns everything a node needs to run
# unattended: stale cost recovery, one queue pass per root, sibling help in
# priority order, the stall watchdog, the GPU-fault cooldown, the between-pass
# 'git pull' with an in-place restart, and a hold that ends the moment a branch
# becomes claimable. Nothing here loops, holds, or decides when to stop.
#
# The three suites are one queue, not three stages. quality and difficulty reuse
# the fresh root's certified prefixes, so they stay unclaimable until those
# prefixes exist; until then a node takes whatever fresh work is left instead of
# waiting for a stage to end. Every node runs the same command at the same time,
# and a node that loses its GPUs rejoins with the same command.
#
#   bash scripts/run_mbpp_experiments.sh            this node joins the MBPP queue
#   bash scripts/run_mbpp_experiments.sh stop       stop this node's launcher and workers
#   bash scripts/run_mbpp_experiments.sh progress   per-root progress, running branches, node names
#   bash scripts/run_mbpp_experiments.sh results    one results file per suite
#
# MBPP_HOLD_SECONDS (default 15) is the pause between passes.
# EXPERIMENTS_HELP_SIBLINGS=0 keeps this node on the first suite's root only.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
SUITE=${2:-all}
usage() {
  echo 'usage: bash scripts/run_mbpp_experiments.sh [run|stop|plan|check|status|progress|results|why] [all|fresh|quality|difficulty]'
}
[ "$#" -le 2 ] || { usage; exit 2; }
case "$MODE" in run|stop|plan|check|status|progress|results|why) ;; -h|--help) usage; exit 0 ;; *) usage; exit 2 ;; esac
case "$SUITE" in all|fresh|quality|difficulty) ;; *) usage; exit 2 ;; esac

export OM_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
HOLD=${MBPP_HOLD_SECONDS:-15}
[[ "$HOLD" =~ ^[0-9]+$ ]] && [ "$HOLD" -gt 0 ] || { echo '[abort] MBPP_HOLD_SECONDS must be positive'; exit 2; }

# Root names, their prerequisites and per-suite settings live in one file that
# the node launcher sources too, so both agree on what "the MBPP queue" is.
export EXPERIMENTS_MBPP_SUITE="$SUITE"
source scripts/_mbpp_experiments.sh
mbpp_queue_init

if [ "$MODE" = why ]; then
  PY=${SWITCH_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
  [ -x "$PY" ] || PY=python3
  ROOT_ARGS=()
  for root in "${MBPP_ROOTS[@]}"; do ROOT_ARGS+=(--root "$root"); done
  exec env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
    "$PY" scripts/mbpp_failure_summary.py --work "$OM_WORK" "${ROOT_ARGS[@]}"
fi

for root in "${MBPP_ROOTS[@]}"; do
  mbpp_queue_settings "$root"
  printf '[mbpp:%s] selector=%s accounting=%s gate=%s\n  root=%s\n' \
    "$MBPP_SUITE" "$MBPP_SELECTOR" "$MBPP_ACCOUNTING" "$MBPP_GATE" "$MBPP_ROOT"
  [ -z "$MBPP_PREFIX" ] || printf '  shared prefixes and evaluation=%s\n' "$MBPP_PREFIX"
done

if [ "$MODE" = plan ]; then
  echo '[plan] seeds 0..4; fresh-selected states at 25/50/100 updates; 18 development + 30 held-out continuations per suite'
  echo '[plan] MBPP execution rewards; final evaluation K=8; convergence curves: 3 checkpoints, K=4'
  echo '[plan] evaluation excludes every source train/validation prompt; available count checked before launch'
  echo "[plan] one queue over ${MBPP_SUITES[*]}: a node takes whichever root has claimable work, in that order"
  echo '[plan] quality and difficulty stay unclaimable until the fresh root has certified all fifteen prefixes'
  echo '[plan] no files written or GPU work started; use check to validate local inputs'
  exit 0
fi

if [ "$MODE" = check ]; then
  # Read-only preflight: pool manifest, source completeness, disjoint evaluation
  # and any already-frozen root's protocol. No GPU, no node, no files written.
  mbpp_queue_check "$SUITE"
  exit 0
fi

# Read-only views, one per suite root. Report errors without skipping other roots.
if [ "$MODE" = status ] || [ "$MODE" = results ]; then
  failed=0
  for root in "${MBPP_ROOTS[@]}"; do
    mbpp_queue_settings "$root"
    echo "[mbpp:$MBPP_SUITE] $MODE"
    rc=0
    env -u OUT_ROOT -u SWITCH_PREFIX_SOURCE -u SWITCH_ONLY_SEEDS -u SWITCH_ONLY_ARMS \
      -u SWITCH_BUDGET_GPU_SECONDS -u SWITCH_RUNTIME_REPO -u SWITCH_DETACHED -u EXPERIMENTS_DETACHED \
      SWITCH_ROOT="$MBPP_ROOT" EXPERIMENTS_COMBINED=0 EXPERIMENTS_SKIP_MOPPS=1 \
      bash scripts/run_selection_switch.sh "$MODE" || rc=$?
    [ "$rc" -eq 0 ] || { echo "[mbpp:$MBPP_SUITE] $MODE rc=$rc"; failed=1; }
  done
  exit "$failed"
fi

# run, stop, progress: the node launcher owns all three. It re-reads the suite
# from EXPERIMENTS_MBPP_SUITE, takes the first root as its own and the rest as
# siblings, prepares a root when its prerequisites are met, and retries ordinary
# task failures. A math root's settings must not reach an MBPP branch, so they are
# dropped here. Input checks run inside the node's passes, after cleanup, so
# unfinished prerequisites do not prevent a stopped node from rejoining.
exec env -u OUT_ROOT -u SWITCH_PREFIX_SOURCE -u SWITCH_DATASET -u SWITCH_SELECTOR \
  -u SWITCH_ACCOUNTING -u SWITCH_GATE -u SWITCH_ONLY_SEEDS -u SWITCH_ONLY_ARMS \
  -u SWITCH_BUDGET_GPU_SECONDS -u OM_NODE_LOCK_HELD -u SWITCH_RUNTIME_REPO \
  -u SWITCH_DETACHED -u EXPERIMENTS_DETACHED \
  SWITCH_ROOT="${MBPP_ROOTS[0]}" EXPERIMENTS_SKIP_MOPPS=1 \
  EXPERIMENTS_HELP_SIBLINGS="${EXPERIMENTS_HELP_SIBLINGS:-1}" \
  EXPERIMENTS_HOLD_SECONDS="${EXPERIMENTS_HOLD_SECONDS:-$HOLD}" \
  EXPERIMENTS_HOLD_POLL_SECONDS="${EXPERIMENTS_HOLD_POLL_SECONDS:-5}" \
  bash scripts/run_experiments.sh "$MODE"
