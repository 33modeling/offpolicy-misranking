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
TERMINAL_PATTERN='^\[stage\]|^\[exit\]|abort\]|\[error\]|Traceback|Error:|Error\b|✔|✘|try [0-9]+/[0-9]+ ->|\[family|\[queue\]|\[cuda-recovery\]|\[contract|\[additional\]|\[runtime\]|\[regime-contract\]|\[transfer-smoke|\[27b-runtime\]|\[check\]|\[download\]|\[seal\]|complete|passed|PASS|FAIL'
if [ "${ADDITIONAL_VERBOSE:-0}" = 1 ]; then
  exec > >(tee -a "$SESSION_LOG") 2>&1
else
  exec > >(tee -a "$SESSION_LOG" | grep --line-buffered -E "$TERMINAL_PATTERN" >&"$LAUNCH_STDOUT") 2>&1
fi
LAUNCH_LOGGER_PID=$!
LAUNCH_STAGE=admission
log_stage() {
  LAUNCH_STAGE=$1
  printf '[stage] %s  %s\n' "$(date -u +%H:%M:%SZ)" "$LAUNCH_STAGE"
}
failure_excerpt() {  # terminal-only: the last error-ish lines, then the tail
  local hits
  hits=$(grep -nE 'abort\]|Traceback|Error:|Error\b|✘|traceback' "$SESSION_LOG" 2>/dev/null | grep -vE '^[0-9]+:\[(error|exit)\]' | tail -8)
  {
    echo
    echo "── 원인 (로그의 에러 줄, 마지막 8개) ──"
    [ -n "$hits" ] && printf '%s\n' "$hits" || echo "(에러 패턴 없음 — 아래 tail 참고)"
    echo "── 로그 마지막 8줄 ──"
    tail -n 8 "$SESSION_LOG" 2>/dev/null
    echo "── 전체: tail -n 60 $SESSION_LOG"
  } >&"$LAUNCH_STDOUT"
}
finish_launch_log() {
  local rc=$? logger_rc=0
  trap - EXIT ERR INT TERM
  printf '[exit] utc=%s rc=%s stage=%s log=%s\n' "$(date -u +%FT%TZ)" "$rc" "$LAUNCH_STAGE" "$SESSION_LOG"
  exec 1>&"$LAUNCH_STDOUT" 2>&"$LAUNCH_STDERR"
  wait "$LAUNCH_LOGGER_PID" || logger_rc=$?
  # grep exits 1 when nothing matched; that is not a writer failure.
  [ "$logger_rc" -eq 0 ] || [ "$logger_rc" -eq 1 ] || { echo "[abort] log writer failed rc=$logger_rc" >&2; rc=$logger_rc; }
  if [ "$rc" -eq 0 ]; then
    printf '✔ %s %s 완료 (log: %s)\n' "$PROFILE" "${MODE#--}" "$SESSION_LOG"
  else
    printf '✘ %s %s 실패 — stage=%s rc=%s\n' "$PROFILE" "${MODE#--}" "$LAUNCH_STAGE" "$rc"
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
printf '▶ %s %s  (터미널엔 단계·성패·에러만; 전체 출력은 %s, ADDITIONAL_VERBOSE=1이면 전부)\n' \
  "$PROFILE" "${MODE#--}" "$SESSION_LOG" >&"$LAUNCH_STDOUT"
