#!/usr/bin/env bash
# Source-only helpers for collision-free, atomic Markdown report publication.

make_report_dir() {
  local suffix=$1
  local root="$OM_WORK/readouts"
  mkdir -p "$root" || return 1
  REPORT_DIR=$(mktemp -d "$root/$(date '+%Y-%m-%d_%H%M%S')-$suffix.XXXXXX")
  export REPORT_DIR
}

publish_report() {
  local output=$1 allowed=$2 display=$3
  shift 3
  local tmp="$output.tmp"
  local stem="${output%.md}"
  local err="$stem.err"
  local partial="$stem.partial.md"
  local rc

  rm -f "$tmp" "$err" "$partial"
  "$@" > "$tmp" 2> "$err"
  rc=$?
  if [[ " $allowed " == *" $rc "* ]] && [ -s "$tmp" ]; then
    mv "$tmp" "$output" || return 1
    [ -s "$err" ] || rm -f "$err"
    [ "$display" = "yes" ] && cat "$output"
    return 0
  fi
  [ -e "$tmp" ] && mv "$tmp" "$partial"
  echo "[report-abort] $(basename "$output"): exit=$rc size=$(wc -c < "$partial" 2>/dev/null || echo 0) stderr=$err" >&2
  return 1
}
