#!/usr/bin/env bash
# Complete Qwen status from shared artifacts; no GPU or checkout updates.
#   bash scripts/run_qwen35_9b.sh status
#   bash scripts/run_qwen35_9b.sh status verbose
# The default includes every registered point and every retained launcher.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
HISTORY="$OM_WORK/console-logs/status-qwen35-history.log"
if [ -z "${STATUS_HISTORY_ACTIVE:-}" ] && mkdir -p "$(dirname "$HISTORY")" 2>/dev/null; then
  printf '\n===== status %s host=%s =====\n' "$(date -u +%FT%TZ)" "$(hostname)" >> "$HISTORY"
  STATUS_HISTORY_ACTIVE=1 bash "$0" "$@" 2>&1 | tee -a "$HISTORY"
  status_codes=("${PIPESTATUS[@]}")
  echo "history : $HISTORY"
  if [ "${status_codes[0]}" -ne 0 ]; then exit "${status_codes[0]}"; fi
  exit "${status_codes[1]}"
fi

RUN_ID=${RUN_ID:-qwen35-9b-posttrained-math-code-grpo-v1}
RUNS="$OM_WORK/runs/$RUN_ID"
RES="$OM_WORK/results/$RUN_ID"
PY="${VENV_DIR:-}/bin/python"; [ -x "$PY" ] || PY=python3
FULL_TOOL="src/matrix_status.py"
CONFIG="configs/qwen35_9b_grpo.json"
echo "Qwen3.5-9B GRPO   $(date -u +%FT%TZ)   code $(git rev-parse --short HEAD 2>/dev/null || printf unknown)   host $(hostname)"
echo "work     $OM_WORK"
full_args=(--root "$RUNS" --console-logs "$OM_WORK/console-logs" --log-glob 'additional-qwen35-*.log')
[ ! -f "$CONFIG" ] || full_args+=(--config "$CONFIG")
[ "${1:-}" != verbose ] && [ "${OM_QWEN_STATUS_VERBOSE:-0}" != 1 ] || full_args+=(--verbose)
if [ ! -f "$FULL_TOOL" ]; then
  echo "[status-error] missing $FULL_TOOL; full status is unavailable from this checkout" >&2
  exit 2
fi
# One renderer owns both DECISION and the tables, including on idle nodes.
# Do not silently replace a failed full matrix with a six-point sample.
if "$PY" "$FULL_TOOL" "${full_args[@]}"; then
  [ ! -d "$RES" ] || echo "results  $RES"
else
  rc=$?
  echo "[status-error] full matrix renderer failed (rc=$rc); see the error above. No experiment was changed." >&2
  exit "$rc"
fi
