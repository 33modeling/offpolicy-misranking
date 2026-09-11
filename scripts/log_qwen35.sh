#!/usr/bin/env bash
# Read-only live session output; prefer this node on shared storage.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${OM_WORK:-}" ]; then
  if [ -d "${GROUP_VOLUME:-/group-volume}" ]; then
    OM_WORK="${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking"
  else
    OM_WORK="${OM_REPO:-$PWD}/.work"
  fi
fi
host=$(hostname)
latest= local_log=
shopt -s nullglob
for file in "$OM_WORK"/console-logs/additional-qwen35-run-*.log; do
  [ -f "$file" ] && [ -r "$file" ] || continue
  if [[ -z "$latest" || "$file" -nt "$latest" ]]; then latest=$file; fi
  IFS= read -r header < "$file" || true
  if [[ " $header " == *" host=$host "* ]] && [[ -z "$local_log" || "$file" -nt "$local_log" ]]; then
    local_log=$file
  fi
done
log=${local_log:-$latest}
if [ -z "$log" ]; then
  echo "[log] no Qwen 9B run log found in $OM_WORK/console-logs" >&2
  exit 1
fi
[ -n "$local_log" ] || echo "[log] no session for $host; following the newest shared Qwen session"
echo "[log] $log"
echo "[log] Ctrl+C closes this viewer only."
exec tail -n 80 -F -- "$log"
