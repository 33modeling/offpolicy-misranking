#!/usr/bin/env bash
# One screen, no GPU, no locks: where is the Qwen3.5-9B run right now?
#   bash scripts/run_qwen35_9b.sh status
# Reads the newest session log and the run/result directories only.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
RUN_ID=${RUN_ID:-qwen35-9b-posttrained-math-code-grpo-v1}
RUNS="$OM_WORK/runs/$RUN_ID"
RES="$OM_WORK/results/$RUN_ID"
LOG=$(ls -t "$OM_WORK"/console-logs/additional-qwen35-*.log 2>/dev/null | head -1)

echo "code    : $(git rev-parse --short HEAD 2>/dev/null) $(git log -1 --format=%s 2>/dev/null | cut -c1-50)"
echo "host    : $(hostname)  now=$(date -u +%FT%TZ)"
if [ -z "$LOG" ]; then
  echo "session : none (no console-logs/additional-qwen35-*.log under $OM_WORK)"
else
  echo "session : $LOG"
  echo "started : $(grep -m1 '^\[launch\]' "$LOG" | grep -o 'utc=[^ ]*' | cut -c5-)"
  echo "stage   : $(grep '^\[stage\]' "$LOG" | tail -1 | sed 's/^\[stage\] //')"
  exit_line=$(grep '^\[exit\]' "$LOG" | tail -1)
  if [ -n "$exit_line" ]; then
    echo "exit    : $(printf '%s' "$exit_line" | grep -o 'rc=[^ ]* stage=[^ ]*')  (finished)"
  elif pgrep -f "run_additional_experiments.sh --run qwen35" >/dev/null 2>&1; then
    echo "exit    : running (launcher process alive)"
  else
    echo "exit    : NO EXIT RECORD and no launcher process -> killed or node lost"
  fi
  last_err=$(grep -E '^(\[[0-9: -]+\] )?\[(abort|recovery-abort|family-fail|regime-hard-stall|qualification-abort|signal-abort)\]|Traceback|Error' "$LOG" | tail -1 | cut -c1-160)
  [ -z "$last_err" ] || echo "last err: $last_err"
fi

if [ -d "$RUNS" ]; then
  total=$(find "$RUNS" -mindepth 2 -maxdepth 2 -type d -name '*-s*-d*' 2>/dev/null | wc -l)
  done_n=$(find "$RUNS" -mindepth 3 -maxdepth 3 -name DONE 2>/dev/null | wc -l)
  echo "points  : $done_n done / $total started / 40 in matrix"
  echo "--- in progress (newest first) ---"
  find "$RUNS" -mindepth 4 -maxdepth 4 -path '*/logs/main.log' 2>/dev/null | xargs -r ls -t 2>/dev/null | head -6 | while read -r main; do
    run=$(dirname "$(dirname "$main")")
    [ -f "$run/DONE" ] && continue
    printf '%-44s %s\n' "$(basename "$run")" \
      "$(grep -F '[progress]' "$main" | tail -1 | sed 's/.*\[progress\] //' | cut -c1-90)"
    g="$run/logs/grpo.log"
    [ -f "$g" ] && grep -E '\] step [0-9]+/' "$g" | tail -1 | sed 's/^/    /' | cut -c1-140
    last=$(grep -E '✘|\[abort\]|cuda-recovery|oom-backoff' "$main" | tail -1 | cut -c1-140)
    [ -z "$last" ] || echo "    ! $last"
  done
  [ -d "$RES" ] && echo "results : $RES ($(ls "$RES" 2>/dev/null | wc -l) entries)"
else
  echo "points  : none started ($RUNS missing)"
fi
[ -z "$LOG" ] || echo "follow  : tail -F $LOG"
