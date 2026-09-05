#!/usr/bin/env bash
# Source after setup_env: persist both streams, including preflight failures.
mkdir -p "$OM_WORK/console-logs"
SESSION_LOG=$(mktemp "$OM_WORK/console-logs/additional-${PROFILE}-${MODE#--}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX.log")
export SESSION_LOG
exec {LAUNCH_STDOUT}>&1 {LAUNCH_STDERR}>&2
exec > >(tee -a "$SESSION_LOG") 2>&1
LAUNCH_LOGGER_PID=$!
LAUNCH_STAGE=admission
log_stage() {
  LAUNCH_STAGE=$1
  printf '[stage] utc=%s name=%s\n' "$(date -u +%FT%TZ)" "$LAUNCH_STAGE"
}
finish_launch_log() {
  local rc=$? logger_rc=0
  trap - EXIT ERR INT TERM
  printf '[exit] utc=%s rc=%s stage=%s log=%s\n' "$(date -u +%FT%TZ)" "$rc" "$LAUNCH_STAGE" "$SESSION_LOG"
  exec 1>&$LAUNCH_STDOUT 2>&$LAUNCH_STDERR
  wait "$LAUNCH_LOGGER_PID" || logger_rc=$?
  [ "$logger_rc" -eq 0 ] || { echo "[abort] log writer failed rc=$logger_rc" >&2; rc=$logger_rc; }
  exit "$rc"
}
trap finish_launch_log EXIT
trap 'printf "[error] utc=%s rc=%s line=%s stage=%s\n" "$(date -u +%FT%TZ)" "$?" "$LINENO" "$LAUNCH_STAGE" >&2' ERR
trap 'exit 130' INT
trap 'exit 143' TERM
printf '[launch] utc=%s profile=%s mode=%s host=%s pid=%s git=%s log=%s\n' \
  "$(date -u +%FT%TZ)" "$PROFILE" "$MODE" "$(hostname)" "$$" \
  "$(git rev-parse HEAD 2>/dev/null || printf unknown)" "$SESSION_LOG"
