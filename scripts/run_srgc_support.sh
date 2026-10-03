#!/bin/sh
# Matched selection/retention/cache controls; existing runs are never redefined.
set -eu
cd "$(dirname "$0")/.."
[ "$#" -ge 1 ] && [ "$#" -le 2 ] || {
    echo "usage: sh scripts/run_srgc_support.sh math|mbpp|all [run|status|results]" >&2
    exit 2
}
case "$1" in math|mbpp|all) ;; *) echo "dataset must be math, mbpp or all" >&2; exit 2 ;; esac
case "${2:-run}" in
    run) exec sh scripts/run_srgc_sr_refresh.sh "$1" support ;;
    status) exec sh scripts/run_srgc_sr_refresh.sh "$1" status support ;;
    results) exec sh scripts/run_srgc_sr_refresh.sh "$1" support_results ;;
    *) echo "action must be run, status or results" >&2; exit 2 ;;
esac
