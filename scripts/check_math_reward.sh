#!/usr/bin/env bash
# Would the corrected math verifier change the math500 rewards already on disk?
#
#   bash scripts/check_math_reward.sh
#
# No arguments, no GPU, no regeneration: it decodes the responses stored in the
# finished math500 points, scores each one with the pinned verifier and with the
# corrected one, and prints how many rewards move. Read-only for the experiment;
# writes one small text file under $OM_WORK/exports so it can be handed over.
#
# Knobs (only if the full scan is too slow on a busy node):
#   OM_MATH_CHECK_POINTS=1   only the first finished point
#   OM_MATH_CHECK_ROWS=2000  only the first N rows of each rollout file
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
PROFILE=h100
case "${1:-}" in baseline|h100) PROFILE=$1; shift ;; esac
case "$PROFILE" in
  baseline) TAG=olmo3-1025-7b-base-rlzero-grpo-v1 ;;
  h100)     TAG=olmo3-1025-7b-base-rlzero-grpo-h100-v2 ;;
esac
TAG="${OM_OLMO3_MODEL_TAG:-$TAG}"
ROOT="${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}"
[ -d "$ROOT" ] || { echo "[abort] no experiment root: $ROOT"; exit 1; }

# math-verify lives in the bundled runtime dependencies, like the launcher's own
# preflight arranges; without it the measurement cannot run at all.
DEPS=$("$PY" src/bootstrap_math_verify.py --cache-root "$OM_WORK/runtime-deps" 2>/dev/null | tail -1)
[ -z "$DEPS" ] || export PYTHONPATH="$DEPS${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'from math_verify import parse, verify' 2>/dev/null \
  || { echo "[abort] bundled math-verify is not importable; run the launcher once so it is cached under $OM_WORK/runtime-deps"; exit 2; }

GEN_GIT=$(cat "$ROOT/.queue/generation.git" 2>/dev/null | tr -d '[:space:]')
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EXPORTS="$OM_WORK/exports"; mkdir -p "$EXPORTS" || { echo "[abort] cannot create $EXPORTS"; exit 1; }
OUT="$EXPORTS/math-reward-check-$TAG-$STAMP.txt"

# The running jobs own the GPUs; this is CPU work and must not compete for them.
nice -n 19 "$PY" src/measure_math_reward.py \
  --root "$ROOT" --dataset math500 \
  ${GEN_GIT:+--pinned "$GEN_GIT"} 2>&1 | tee "$OUT"
status=${PIPESTATUS[0]}
echo "[math-reward-check] $OUT"
echo "[math-reward-check] plain text: copy it into the transfer repository and push"
exit "$status"
