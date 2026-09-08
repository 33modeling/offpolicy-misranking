#!/usr/bin/env bash
# Rescore the finished MATH-500 families with the corrected verifier
# (registered handling, paper plan §9.1, 2026-09-08).
#
#   bash scripts/rescore_math500.sh              # dry run: counts only, changes nothing
#   bash scripts/rescore_math500.sh apply        # rewrite every complete math500 family
#   bash scripts/rescore_math500.sh apply 0 1    # only seeds 0 and 1
#
# No GPU, no regeneration, no retraining. Rewards are recomputed from the stored
# responses; the pinned value stays in each row as `reward_pinned`, the pinned
# hashes in `<prefix>.rescore.json`, and the pinned gradients/scores/report are
# moved into `<point>/pinned-scoring/<stamp>/`. The family is then incomplete on
# purpose: a worker started with the usual command recomputes gradients, scores
# and the report (generation and GRPO are skipped, their artifacts are intact).
#
# Run the apply step only on a family no worker holds. On the idle node, follow
# it immediately with a worker restricted to those families so no busy node is
# tempted away from the main matrix:
#   OM_RLZERO_ONLY_FAMILIES="math500/s0 math500/s1" bash scripts/run_olmo3_rlzero.sh run h100
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
MODE=dry
[ "${1:-}" = apply ] && { MODE=apply; shift; }
PROFILE=h100
case "${1:-}" in baseline|h100) PROFILE=$1; shift ;; esac
case "$PROFILE" in
  baseline) TAG=olmo3-1025-7b-base-rlzero-grpo-v1 ;;
  h100)     TAG=olmo3-1025-7b-base-rlzero-grpo-h100-v2 ;;
esac
TAG="${OM_OLMO3_MODEL_TAG:-$TAG}"
ROOT="${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}"
[ -d "$ROOT" ] || { echo "[abort] no experiment root: $ROOT"; exit 1; }

DEPS=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps" 2>/dev/null | tail -1)
[ -z "$DEPS" ] || export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'from math_verify import parse, verify' 2>/dev/null \
  || { echo "[abort] bundled math-verify is not importable; run the launcher once so it is cached"; exit 2; }
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

seeds=()
for s in "$@"; do
  [[ "$s" =~ ^[0-9]+$ ]] || { echo "[abort] seeds are integers, not '$s'"; exit 2; }
  seeds+=(--seed "$s")
done
GEN_GIT=$(cat "$ROOT/.queue/generation.git" 2>/dev/null | tr -d '[:space:]')
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EXPORTS="$OM_WORK/exports"; mkdir -p "$EXPORTS"
OUT="$EXPORTS/rescore-math500-$MODE-$STAMP.txt"
args=(--root "$ROOT" --tag "$TAG" --dataset math500 ${GEN_GIT:+--pinned "$GEN_GIT"})
[ "$MODE" = apply ] && args+=(--apply)
nice -n 19 "$PY" src/rescore_rollouts.py "${args[@]}" "${seeds[@]}" 2>&1 | tee "$OUT"
status=${PIPESTATUS[0]}
echo "[rescore] $OUT"
exit "$status"
