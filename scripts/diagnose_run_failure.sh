#!/usr/bin/env bash
# Print and persist actionable diagnostics for a failed smoke or full run.
set -uo pipefail

RUN="${1:?usage: diagnose_run_failure.sh RUN_DIR [CONSOLE_LOG] [EXIT_CODE]}"
CONSOLE_LOG="${2:-}"
EXIT_CODE="${3:-unknown}"
mkdir -p "$RUN"
OUT="$RUN/FAILURE_DIAGNOSTIC.txt"
TMP="$OUT.tmp.$$"

logs=()
[ -n "$CONSOLE_LOG" ] && [ -f "$CONSOLE_LOG" ] && logs+=("$CONSOLE_LOG")
if [ -d "$RUN/logs" ]; then
  while IFS= read -r path; do logs+=("$path"); done < <(
    find "$RUN/logs" -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | head -8 | cut -d' ' -f2-
  )
fi

{
  echo "== run failure diagnostic =="
  echo "time=$(date -Is)"
  echo "run=$RUN"
  echo "exit_code=$EXIT_CODE"
  echo

  echo "== error signatures =="
  if [ "${#logs[@]}" -eq 0 ]; then
    echo "no log files found"
  elif ! grep -Ein \
      'traceback|error|exception|out of memory|oom|killed|abort|fail|timeout|nccl|cuda|watchdog|워처' \
      "${logs[@]}" 2>/dev/null | tail -80; then
    echo "no known error signature found"
  fi
  echo

  echo "== recent log tails =="
  if [ "${#logs[@]}" -eq 0 ]; then
    echo "no log files found"
  else
    for log in "${logs[@]}"; do
      echo "--- $log"
      tail -30 "$log" 2>/dev/null || true
    done
  fi
  echo

  echo "== required artifacts =="
  for artifact in prompts.json run_config.json manifest.json \
      rollouts_behavior_train.jsonl rollouts_fresh_train.jsonl \
      score_protocol.json oracle_protocol.json report.json DONE; do
    if [ -s "$RUN/$artifact" ]; then
      printf 'ok      %s\n' "$artifact"
    else
      printf 'missing %s\n' "$artifact"
    fi
  done
  echo

  echo "== remaining run processes =="
  pgrep -af -- "--run $RUN" 2>/dev/null || echo "none"
  echo

  echo "== GPU state =="
  timeout 20 nvidia-smi \
    --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader 2>&1 || echo "nvidia-smi unavailable"
} > "$TMP"

mv "$TMP" "$OUT"
cat "$OUT"
echo "== diagnostic saved: $OUT"
