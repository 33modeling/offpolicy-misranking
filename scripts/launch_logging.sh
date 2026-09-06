#!/usr/bin/env bash
# Source after setup_env: persist both streams to one session log, and show the
# terminal only what a human needs — stage transitions, pass/fail, abort/error
# lines, family/point progress, and on failure a short excerpt of the log.
# ADDITIONAL_VERBOSE=1 restores the full stream on the terminal.
mkdir -p "$OM_WORK/console-logs"
SESSION_LOG=$(mktemp "$OM_WORK/console-logs/additional-${PROFILE}-${MODE#--}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")
export SESSION_LOG
exec {LAUNCH_STDOUT}>&1 {LAUNCH_STDERR}>&2
# Lines worth a human's attention. Everything else goes only to the file.
TERMINAL_PATTERN='\[progress\]|^\[stage\]|^\[exit\]|abort\]|\[error\]|Traceback|Error:|Error\b|✔|✘|try [0-9]+/[0-9]+ ->|\[family|\[queue\]|\[cuda-recovery\]|\[contract|\[additional\]|\[runtime\]|\[regime-contract\]|\[transfer-smoke|\[27b-runtime\]|\[check\]|\[download\]|\[seal\]|complete|passed|PASS|FAIL'
if [ "${ADDITIONAL_VERBOSE:-0}" = 1 ]; then
  exec > >(tee -a "$SESSION_LOG") 2>&1
else
  exec > >(tee -a "$SESSION_LOG" | { grep --line-buffered -E "$TERMINAL_PATTERN" >&"$LAUNCH_STDOUT" || { filter_rc=$?; [ "$filter_rc" -eq 1 ] || exit "$filter_rc"; }; }) 2>&1
fi
LAUNCH_LOGGER_PID=$!
LAUNCH_STAGE=admission
log_stage() {
  LAUNCH_STAGE=$1
  printf '[stage] %s  %s\n' "$(date -u +%H:%M:%SZ)" "$LAUNCH_STAGE"
}
failure_excerpt() {  # terminal-only: one diagnosis + one action; raw lines only if unknown
  {
    echo
    # Advisory only: never let the diagnoser change the launcher's exit status.
    python3 "$(dirname "${BASH_SOURCE[0]}")/../src/diagnose_launch_failure.py" "$SESSION_LOG" 2>/dev/null \
      || echo "DIAGNOSIS: (diagnoser unavailable) last log line: $(tail -n 1 "$SESSION_LOG" 2>/dev/null)"
    echo "--- last error lines ---"
    grep -vE '^\[(stage|launch|exit|error|additional|runtime)\]|^\s*$' "$SESSION_LOG" 2>/dev/null | tail -n 6 | cut -c1-200
    if [ "${PROFILE:-}" = qwen35 ] && [ -x "$(dirname "${BASH_SOURCE[0]}")/doctor_qwen35.sh" ]; then
      echo "--- state ---"
      bash "$(dirname "${BASH_SOURCE[0]}")/doctor_qwen35.sh" 2>&1 | cut -c1-200 || true
    fi
    echo "(full log: $SESSION_LOG)"
  } >&"$LAUNCH_STDOUT"
}
finish_launch_log() {
  local rc=$? logger_rc=0
  trap - EXIT ERR INT TERM
  printf '[work-exit] utc=%s rc=%s stage=%s\n' "$(date -u +%FT%TZ)" "$rc" "$LAUNCH_STAGE"
  exec 1>&"$LAUNCH_STDOUT" 2>&"$LAUNCH_STDERR"
  wait "$LAUNCH_LOGGER_PID" || logger_rc=$?
  # Only the grep stage normalizes its no-match status; tee errors must propagate.
  [ "$logger_rc" -eq 0 ] || { echo "[abort] log writer failed rc=$logger_rc" >&2; rc=$logger_rc; }
  # The authoritative exit record is written only after the writer has drained.
  if ! printf '[exit] utc=%s rc=%s stage=%s log=%s\n' "$(date -u +%FT%TZ)" "$rc" "$LAUNCH_STAGE" "$SESSION_LOG" >> "$SESSION_LOG"; then
    echo "[abort] cannot persist final exit record" >&2
    rc=1
  fi
  if [ "$rc" -eq 0 ]; then
    printf 'OK  %s %s finished (log: %s)\n' "$PROFILE" "${MODE#--}" "$SESSION_LOG"
  else
    printf 'FAILED  %s %s  stage=%s rc=%s code=%s\n' "$PROFILE" "${MODE#--}" "$LAUNCH_STAGE" "$rc" "$(git rev-parse --short HEAD 2>/dev/null || printf '?')"
    failure_excerpt
  fi
  exit "$rc"
}
trap finish_launch_log EXIT
# rc here is unreliable inside pipelines (PIPESTATUS); the [exit] line carries the real one.
trap 'printf "[error] utc=%s line=%s stage=%s\n" "$(date -u +%FT%TZ)" "$LINENO" "$LAUNCH_STAGE" >&2' ERR
trap 'exit 130' INT
trap 'exit 143' TERM
printf '[launch] utc=%s profile=%s mode=%s host=%s pid=%s git=%s log=%s\n' \
  "$(date -u +%FT%TZ)" "$PROFILE" "$MODE" "$(hostname)" "$$" \
  "$(git rev-parse HEAD 2>/dev/null || printf unknown)" "$SESSION_LOG"
printf 'START  %s %s  code=%s  (terminal shows stages/pass-fail/errors only; full log: %s)\n' \
  "$PROFILE" "${MODE#--}" "$(git rev-parse --short HEAD 2>/dev/null || printf '?')" "$SESSION_LOG" >&"$LAUNCH_STDOUT"
