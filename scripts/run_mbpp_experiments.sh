#!/usr/bin/env bash
# MBPP selector/cost-accounting comparison, as one
# command per node. This script chooses the roots and then hands the node to
# scripts/run_experiments.sh, which already owns everything a node needs to run
# unattended: stale cost recovery, one queue pass per root, sibling help in
# priority order, the stall watchdog, the GPU-fault cooldown, the between-pass
# 'git pull' with an in-place restart, and a hold that ends the moment a branch
# becomes claimable. Nothing here loops, holds, or decides when to stop.
#
# The default queue is the existing matched/convergence MBPP condition (quality),
# with 48 continuations. It reuses the primary on-policy prefixes; missing prefixes
# do not authorize starting the legacy fresh continuation suite. Every node runs the same command,
# and a node that loses its GPUs rejoins with the same command.
#
#   bash scripts/run_mbpp_experiments.sh            update, reload changed code, or follow this node's log
#   bash scripts/run_mbpp_experiments.sh stop       stop this node's launcher and workers
#   bash scripts/run_mbpp_experiments.sh restart    load fixes; retain checkpoints and fault receipts
#   bash scripts/run_mbpp_experiments.sh progress   MBPP suites only: branch counts, running branches, node names
#   bash scripts/run_mbpp_experiments.sh status --watch  MBPP-only live status; never the math view
#   bash scripts/run_mbpp_experiments.sh results    one results file per suite
#
# MBPP_HOLD_SECONDS (default 15) is the pause between passes.
# EXPERIMENTS_HELP_SIBLINGS=0 keeps this node on the first suite's root only.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-run}
[ "$#" -eq 0 ] || shift
SUITE=all
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then SUITE=$1; shift; fi
STATUS_ARGS=()
STATUS_WATCH=
usage() {
  echo 'usage: bash scripts/run_mbpp_experiments.sh [run|restart|stop|plan|check|status|progress|results|saved|why] [all|fresh|difficulty|long|quality]'
  echo '       bash scripts/run_mbpp_experiments.sh status [all|fresh|difficulty|long|quality] [--all] [--watch [SECONDS]]'
  echo '       default all = On-policy · 선택비용 별도; matched/convergence; 48 continuations'
  echo '       selection cost is recorded separately; existing training allocation is unchanged'
  echo '       fresh, difficulty, long are explicit legacy commands; saved work remains visible'
}
case "$MODE" in run|restart|stop|plan|check|status|progress|results|saved|why) ;; -h|--help) usage; exit 0 ;; *) usage; exit 2 ;; esac
case "$SUITE" in all|fresh|quality|difficulty|long) ;; *) usage; exit 2 ;; esac
while [ "$#" -gt 0 ]; do
  [ "$MODE" = status ] || { usage; exit 2; }
  case "$1" in
    --all) STATUS_ARGS+=(--all); shift ;;
    --watch)
      STATUS_WATCH=15; shift
      if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then STATUS_WATCH=$1; shift; fi
      [[ "$STATUS_WATCH" =~ ^[1-9][0-9]*$ ]] || { echo '[abort] watch interval must be a positive integer'; exit 2; }
      ;;
    *) usage; exit 2 ;;
  esac
done

export OM_WORK=${OM_WORK:-/group-volume/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
HOLD=${MBPP_HOLD_SECONDS:-15}
[[ "$HOLD" =~ ^[0-9]+$ ]] && [ "$HOLD" -gt 0 ] || { echo '[abort] MBPP_HOLD_SECONDS must be positive'; exit 2; }

# Root names, their prerequisites and per-suite settings live in one file that
# the node launcher sources too, so both agree on what "the MBPP queue" is.
export EXPERIMENTS_MBPP_SUITE="$SUITE"
source scripts/_mbpp_experiments.sh
mbpp_queue_init

# Labels come from _mbpp_experiments.sh. Frozen keys and saved paths are unchanged.

if [ "$MODE" = why ] || [ "$MODE" = saved ]; then
  PY=${SWITCH_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
  [ -x "$PY" ] || PY=python3
  ROOT_ARGS=()
  while IFS= read -r root; do ROOT_ARGS+=(--root "$root"); done < <(mbpp_observation_roots)
  [ "$MODE" != saved ] || ROOT_ARGS+=(--storage)
  exec env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
    "$PY" scripts/mbpp_failure_summary.py --work "$OM_WORK" "${ROOT_ARGS[@]}"
fi

# One view over all requested MBPP roots. Never enter a worker launcher or let
# inherited math settings choose the status root. Reload viewer code each frame.
if [ "$MODE" = status ]; then
  PY=${SWITCH_PYTHON:-${VENV_DIR:-$OM_WORK/.venv-cu126}/bin/python}
  [ -x "$PY" ] || PY=python3
  ROOT_ARGS=()
  for root in "${MBPP_ROOTS[@]}"; do ROOT_ARGS+=(--root "$root"); done
  if [ "$SUITE" = all ]; then
    while IFS= read -r root; do
      [ "$root" = "$SWITCH_MBPP_QUALITY_ROOT" ] || ROOT_ARGS+=(--retained-root "$root")
    done < <(mbpp_observation_roots)
  fi
  while :; do
    if [ -n "$STATUS_WATCH" ] && [ -t 1 ]; then printf '\033[2J\033[H'; fi
    rc=0
    env -u OUT_ROOT -u SWITCH_ROOT -u SWITCH_PREFIX_SOURCE -u SWITCH_DATASET \
      -u SWITCH_SELECTOR -u SWITCH_ACCOUNTING -u SWITCH_GATE \
      -u SWITCH_ONLY_SEEDS -u SWITCH_ONLY_ARMS -u SWITCH_RUNTIME_REPO \
      CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
      EXPERIMENTS_COMBINED=0 EXPERIMENTS_SKIP_MOPPS=1 \
      "$PY" scripts/mbpp_status.py "${ROOT_ARGS[@]}" "${STATUS_ARGS[@]}" || rc=$?
    [ -n "$STATUS_WATCH" ] || exit "$rc"
    sleep "$STATUS_WATCH"
  done
fi

for root in "${MBPP_ROOTS[@]}"; do
  mbpp_queue_settings "$root"
  printf '[mbpp:%s] selector=%s accounting=%s gate=%s\n  root=%s\n' \
    "$(mbpp_suite_label "$MBPP_SUITE")" "$(mbpp_selector_label "$MBPP_SELECTOR")" \
    "$(mbpp_accounting_label "$MBPP_ACCOUNTING")" "$(mbpp_gate_label "$MBPP_GATE")" "$MBPP_ROOT"
  [ -z "$MBPP_PREFIX" ] || printf '  shared prefixes and evaluation=%s\n' "$MBPP_PREFIX"
  [ -z "$MBPP_BUDGET" ] || printf '  continuation budget=%s GPU-seconds (same as MATH long)\n' "$MBPP_BUDGET"
done
if [ "$SUITE" = all ]; then
  echo '[mbpp] 기본 실행: On-policy · 선택비용 별도, 48개. 다른 조건의 결과·노드 기록은 보존하며 자동 재실행하지 않습니다.'
  echo '[mbpp] 그냥 실행하면 업데이트 확인 후 구버전만 자동 재시작합니다. 같은 코드로 실행 중이면 로그를 표시합니다.'
fi

if [ "$MODE" = plan ]; then
  echo '[plan] seeds 0..4; on-policy-selected states at 25/50/100 updates; 18 development + 30 held-out continuations per suite'
  echo '[plan] MBPP execution rewards; final evaluation K=8; convergence curves: 3 checkpoints, K=4'
  echo '[plan] evaluation excludes every source train/validation prompt; available count checked before launch'
  echo "[plan] ${#MBPP_SUITES[@]} condition(s), $((48 * ${#MBPP_SUITES[@]})) continuation branches; shared prefix preparation and evaluation are additional work"
  echo '[plan] default: On-policy · 선택비용 별도; matched accounting, convergence gate; existing frozen training allocation is retained'
  echo '[plan] selection GPU cost is recorded on reporting, outside diagnostic+training allocation, and included in actual total cost; evaluation is also reported separately'
  echo '[plan] all fifteen certified MBPP prefixes are reused; missing prefixes wait for separate preparation, never automatic legacy fresh continuation training'
  echo '[plan] fresh/difficulty/long require explicit selection; their policies, results and ledgers are preserved; explicit long retains its 87120 GPU-second cap'
  echo '[plan] no files written or GPU work started; use check to validate local inputs'
  exit 0
fi

if [ "$MODE" = check ]; then
  # Read-only preflight: pool manifest, source completeness, disjoint evaluation
  # and any already-frozen root's protocol. No GPU, no node, no files written.
  mbpp_queue_check "$SUITE"
  exit 0
fi

# Results are exported per suite root; status above is one consolidated view.
if [ "$MODE" = results ]; then
  failed=0
  while IFS= read -r root; do
    mbpp_queue_settings "$root"
    echo "[mbpp:$(mbpp_suite_label "$MBPP_SUITE")] results"
    rc=0
    env -u OUT_ROOT -u SWITCH_PREFIX_SOURCE -u SWITCH_ONLY_SEEDS -u SWITCH_ONLY_ARMS \
      -u SWITCH_BUDGET_GPU_SECONDS -u SWITCH_RUNTIME_REPO -u SWITCH_DETACHED -u EXPERIMENTS_DETACHED \
      SWITCH_ROOT="$MBPP_ROOT" EXPERIMENTS_COMBINED=0 EXPERIMENTS_SKIP_MOPPS=1 \
      bash scripts/run_selection_switch.sh results || rc=$?
    [ "$rc" -eq 0 ] || { echo "[mbpp:$(mbpp_suite_label "$MBPP_SUITE")] results rc=$rc"; failed=1; }
  done < <(mbpp_observation_roots)
  exit "$failed"
fi

# Inspect the shared storage before handing control to anything that can stop a
# controller, recover an attempt, prepare roots or launch GPU work. In
# particular, a blocked restart must leave the current controller untouched.
if [ "$MODE" = run ] || [ "$MODE" = restart ]; then
  if ! env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 MBPP_STORAGE_AUDIT_AUTOMATIC=1 \
    bash scripts/check_mbpp_storage.sh "$SUITE"; then
    echo '[abort] MBPP storage audit blocked startup; no controller was started or stopped.' >&2
    echo '[abort] Review the storage audit before resuming; existing files were not changed by this launcher.' >&2
    exit 2
  fi
fi

# run, restart, stop, progress: the node launcher owns their lifecycle. It re-reads the suite
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
