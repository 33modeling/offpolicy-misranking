#!/usr/bin/env bash
# No GPU, no locks: is the Qwen3.5-9B run fine, and if not, what to do?
#   bash scripts/run_qwen35_9b.sh status            # DECISION + whole 40-point matrix + launchers + KEY NUMBERS
#   bash scripts/run_qwen35_9b.sh status verbose    # + per-point rows and the newest stage-log lines
# Line 2 (DECISION) is the answer. Everything is appended to a history file.
# The matrix table comes from src/matrix_status.py (flat run_matrix layout);
# it is the Qwen counterpart of the OLMo `status h100` screen.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh >/dev/null 2>&1
HISTORY="$OM_WORK/console-logs/status-qwen35-history.log"
if [ -z "${STATUS_HISTORY_ACTIVE:-}" ] && mkdir -p "$(dirname "$HISTORY")" 2>/dev/null; then
  printf '\n===== status %s host=%s =====\n' "$(date -u +%FT%TZ)" "$(hostname)" >> "$HISTORY"
  STATUS_HISTORY_ACTIVE=1 bash "$0" "$@" | tee -a "$HISTORY"
  status_codes=("${PIPESTATUS[@]}")
  echo "history : $HISTORY"
  if [ "${status_codes[0]}" -ne 0 ]; then exit "${status_codes[0]}"; fi
  exit "${status_codes[1]}"
fi

RUN_ID=${RUN_ID:-qwen35-9b-posttrained-math-code-grpo-v1}
RUNS="$OM_WORK/runs/$RUN_ID"
RES="$OM_WORK/results/$RUN_ID"
# The DECISION reads one session log. With several nodes the newest file can be
# an old launcher's exit record rewritten late; prefer the newest log that has
# no [exit] line (a session still running somewhere), else the newest overall.
LOG=""
for candidate in $(ls -t "$OM_WORK"/console-logs/additional-qwen35-*.log 2>/dev/null); do
  [ -n "$LOG" ] || LOG=$candidate
  if ! grep -q '^\[exit\]' "$candidate" 2>/dev/null; then LOG=$candidate; break; fi
done
NOW=$(date +%s)
ERROR_PATTERN='✘|\[abort\]'

fmt_age() {  # seconds -> 3m / 2h05m / 4d
  local a=$1
  if [ "$a" -lt 90 ]; then printf '%ss' "$a"
  elif [ "$a" -lt 5400 ]; then printf '%sm' "$((a / 60))"
  elif [ "$a" -lt 172800 ]; then printf '%sh%02dm' "$((a / 3600))" "$(((a % 3600) / 60))"
  else printf '%sd' "$((a / 86400))"; fi
}
newest_epoch() {  # newest_epoch <dir> [find-args...] -> epoch of newest file (keepalive excluded)
  local d=$1; shift
  find "$d" -type f ! -name 'keepalive.log' ! -name '.pipeline-activity.json*' "$@" \
    -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1
}

# ---- gather facts ----
launcher_alive=0; pgrep -f "run_additional_experiments.sh --run qwen35" >/dev/null 2>&1 && launcher_alive=1
stage=""; exit_line=""; started=""; fails=0; log_age=""
if [ -n "$LOG" ]; then
  started=$(grep -m1 '^\[launch\]' "$LOG" | grep -o 'utc=[^ ]*' | cut -c5-)
  stage=$(grep '^\[stage\]' "$LOG" | tail -1 | sed 's/^\[stage\] //')
  exit_line=$(grep '^\[exit\]' "$LOG" | tail -1)
  fails=$(grep -c '^\[family-fail\]' "$LOG" 2>/dev/null || true)
  fails=${fails:-0}
  log_age=$((NOW - $(stat -c %Y "$LOG" 2>/dev/null || echo "$NOW")))
fi
total=0; done_n=0; write_age=""; current=""; current_stage=""; current_err=""
if [ -d "$RUNS" ]; then
  total=$(find "$RUNS" -mindepth 2 -maxdepth 2 -type d -name '*-s*-d*' 2>/dev/null | wc -l)
  done_n=$(find "$RUNS" -mindepth 3 -maxdepth 3 -type f -name DONE -size +0c 2>/dev/null | wc -l)
  newest=$(newest_epoch "$RUNS"); [ -z "$newest" ] || write_age=$((NOW - newest))
  main=$(find "$RUNS" -mindepth 4 -maxdepth 4 -path '*/logs/main.log' 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
  if [ -n "$main" ]; then
    current=$(basename "$(dirname "$(dirname "$main")")")
    current_stage=$(grep -F '[progress]' "$main" | tail -1 | sed 's/.*\[progress\] //' | cut -d' ' -f3- | cut -c1-60)
    current_err=$(grep -E "$ERROR_PATTERN" "$main" | tail -1 | cut -c1-120)
  fi
fi

# ---- training progress from durable artifacts (DONE, GRPO steps, rollout bytes, last write) ----
PY="${VENV_DIR:-}/bin/python"; [ -x "$PY" ] || PY=python3
PROGRESS_TOOL="$(dirname "$0")/../src/training_progress.py"
progress=""; progress_word=""
if [ -f "$PROGRESS_TOOL" ] && [ -d "$RUNS" ]; then
  progress=$("$PY" "$PROGRESS_TOOL" --root "$RUNS" --total-points 40 --record 2>/dev/null)
  progress_word=$(printf '%s' "$progress" | awk '{print $1}')
  [ "$progress_word" != NOT ] || progress_word="NOT $(printf '%s' "$progress" | awk '{print $2}')"
fi

# ---- decide ----
if [ -z "$LOG" ]; then
  decision="NOT STARTED: no session log. Start: bash scripts/run_qwen35_9b.sh run"
elif [ -n "$exit_line" ]; then
  rc=$(printf '%s' "$exit_line" | grep -o 'rc=[0-9]*' | cut -d= -f2)
  if [ "${rc:-1}" = 0 ]; then
    if [ "$done_n" -eq 40 ]; then
      decision="DONE: launcher finished rc=0 ($done_n/40 points). Nothing to do."
    else
      decision="WARNING: launcher finished rc=0 but only $done_n/40 points have nonempty DONE records. Check the selected matrix and completion artifacts before treating this run as complete."
    fi
  else
    decision="ERROR: launcher exited rc=$rc at stage '$stage'. Read the ! lines below, fix, then run again (finished points and .partial rollouts resume)."
  fi
elif [ "$launcher_alive" -eq 0 ]; then
  decision="ERROR: launcher is gone with no exit record (node lost or killed). Run again: bash scripts/run_qwen35_9b.sh run (finished work resumes)."
elif [ "$progress_word" = "NOT TRAINING" ]; then
  decision="ERROR: $progress. The launcher is alive but nothing durable has been written; this is not a running experiment. Ctrl-C, then run again (finished work resumes)."
elif [ "$total" -eq 0 ]; then
  if [ "${log_age:-0}" -lt 1800 ]; then
    decision="NO ERROR: preparing, stage '$stage' (log written $(fmt_age "$log_age") ago). Nothing to do."
  else
    decision="ERROR: stuck in stage '$stage' for $(fmt_age "$log_age") with no output. Ctrl-C and run again."
  fi
elif [ -n "$write_age" ] && [ "$write_age" -ge 10800 ]; then
  decision="ERROR: alive but nothing written for $(fmt_age "$write_age") (last point $current, $current_stage). Hung. Ctrl-C, then run again (finished work resumes)."
elif [ -n "$write_age" ] && [ "$write_age" -ge 2700 ]; then
  decision="WARNING: nothing written for $(fmt_age "$write_age") at $current ($current_stage). A 2048-token rollout stage can be quiet this long. Check again in 30 min; if still quiet, treat as hung."
elif [ "$fails" -gt 0 ] && [ "$done_n" -eq 0 ]; then
  decision="WARNING: running, but $fails family failures so far and 0 points done. Read the ! lines; if the same error repeats on the next status, fix it before it burns GPU time."
else
  decision="NO ERROR. Running: $current $current_stage (written $(fmt_age "${write_age:-0}") ago), $done_n/40 points done. Nothing to do."
fi

# ---- print ----
echo "Qwen3.5-9B GRPO   $(date -u +%FT%TZ)   code $(git rev-parse --short HEAD 2>/dev/null)   host $(hostname)"
echo "DECISION $decision"
[ -z "$progress" ] || echo "PROGRESS $progress"
if [ -n "$LOG" ]; then
  echo "session  started $started   stage $stage   launcher $([ "$launcher_alive" -eq 1 ] && echo alive || echo not-running)"
fi
echo "points   $done_n done / $total started / 40 in matrix   family failures this session: $fails"
full_status_printed=0
if [ -d "$RUNS" ] || [ -n "$LOG" ]; then
  # Whole matrix, OLMo-style: every family, every point, launcher sessions on
  # all nodes, KEY NUMBERS, overall verdict. Read-only. `status verbose` adds
  # per-point rows and the newest stage-log lines. Falls back to the short
  # six-point table when the renderer or python is missing.
  FULL_TOOL="$(dirname "$0")/../src/matrix_status.py"
  CONFIG="$(dirname "$0")/../configs/qwen35_9b_grpo.json"
  full_args=(--root "$RUNS" --console-logs "$OM_WORK/console-logs" --log-glob 'additional-qwen35-*.log')
  [ -f "$CONFIG" ] && full_args+=(--config "$CONFIG")
  [ "${1:-}" != verbose ] && [ "${OM_QWEN_STATUS_VERBOSE:-0}" != 1 ] || full_args+=(--verbose)
  if [ -f "$FULL_TOOL" ]; then
    echo
    if PYTHONPATH="$(dirname "$0")/../src${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$FULL_TOOL" "${full_args[@]}"; then
      full_status_printed=1
    else
      echo " (full status renderer failed; showing the short table)"
    fi
  fi
fi
if [ -d "$RUNS" ] && [ "$full_status_printed" -eq 0 ]; then
  echo
  echo " stage = k/8 of the point pipeline: 1 prep  2 behavior-rollout  3 grpo  4 fresh-rollout  5 gradients  6 scores  7 merge+report  8 DONE;  +Nmin = time in this point"
  echo " note:  ok = fine   ok (earlier attempt failed: ...) = recovered   ERROR (current): = this attempt is failing, read it"
  echo " point                  stage                         last write   note"
  find "$RUNS" -mindepth 4 -maxdepth 4 -path '*/logs/main.log' 2>/dev/null | xargs -r ls -t 2>/dev/null | head -6 | while read -r m; do
    run=$(dirname "$(dirname "$m")")
    [ -s "$run/DONE" ] && continue
    age=$(newest_epoch "$run"); [ -n "$age" ] && age=$(fmt_age $((NOW - age))) || age="-"
    st=$(grep -F '[progress]' "$m" | tail -1 | sed 's/.*\[progress\] //' | cut -d' ' -f3- | cut -c1-28)
    prog_n=$(grep -nF '[progress]' "$m" | tail -1 | cut -d: -f1); prog_n=${prog_n:-0}
    note="ok"
    last=$(grep -nE "$ERROR_PATTERN|cuda-recovery|oom-backoff" "$m" | tail -1)
    if [ -n "$last" ]; then
      n=${last%%:*}
      text=$(sed -n "$((n)),$((n + 8))p" "$m" | grep -vE '^\s*$|^\[' | tail -1 | cut -c1-90)
      if [ "$n" -gt "$prog_n" ]; then
        note="ERROR (current): $text"
      else
        note="ok (earlier attempt failed: $text)"
      fi
    fi
    short=$(basename "$run" | sed "s/^$RUN_ID-//")
    printf ' %-22s %-29s %-12s %s\n' "${short:0:22}" "${st:-starting}" "$age" "$note"
  done
  echo
  "$PY" "$(dirname "$0")/../src/point_key_numbers.py" --root "$RUNS" 2>/dev/null \
    || echo " KEY NUMBERS unavailable (python or point_key_numbers.py missing)"
fi
[ ! -d "$RES" ] || echo "results  $RES ($(ls "$RES" 2>/dev/null | wc -l) entries)"
echo
echo " DECISION words:  ERROR = you act   WARNING = check again in 30 min   NO ERROR = leave it      last write = time since that point wrote any file"
echo " point name = s<seed>-<dataset>-d<drift>; 10 families (2 datasets x 5 seeds) x 4 points (d0 d25 d100 d400) = 40"
[ -z "$LOG" ] || echo " log      $LOG"
