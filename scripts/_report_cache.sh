#!/usr/bin/env bash
# Shared, fail-closed cache helpers for CPU-only report generation.
# Code files are content-hashed. Immutable run artifacts use size + nanosecond
# mtime so checking a cache does not reread multi-gigabyte tensors/rollouts.

report_cache_key() {
  local mode=code path
  {
    printf 'offpolicy-report-cache-v1\0'
    for path in "$@"; do
      if [ "$path" = "--" ]; then
        mode=data
        continue
      fi
      printf '%s\0%s\0' "$mode" "$path"
      if [ ! -e "$path" ]; then
        printf 'missing\0'
      elif [ "$mode" = code ]; then
        sha256sum -- "$path"
      else
        stat -Lc '%F\0%s\0%y\0' -- "$path"
      fi
    done
  } | sha256sum | cut -d' ' -f1
}

report_cache_key_values() {
  local base=$1
  shift
  {
    printf '%s\0' "$base"
    printf '%s\0' "$@"
  } | sha256sum | cut -d' ' -f1
}

report_cache_hit() {
  local marker=$1 expected=$2 output
  shift 2
  [ -s "$marker" ] || return 1
  [ "$(sed -n '1p' "$marker")" = "$expected" ] || return 1
  for output in "$@"; do
    [ -s "$output" ] || return 1
  done
  [ "$(wc -l < "$marker")" -eq $((1 + $#)) ] || return 1
  tail -n +2 "$marker" | sha256sum -c --status
}

report_cache_write() {
  local marker=$1 key=$2 tmp="$1.tmp.$$" output
  shift 2
  mkdir -p "$(dirname "$marker")" || return 1
  for output in "$@"; do
    [ -s "$output" ] || return 1
  done
  {
    printf '%s\n' "$key"
    for output in "$@"; do
      sha256sum -- "$output"
    done
  } > "$tmp" || return 1
  mv -- "$tmp" "$marker"
}
