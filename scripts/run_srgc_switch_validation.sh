#!/bin/sh
# Additional timing and own-path rule controls; no P0 restart.
set -eu
cd "$(dirname "$0")/.."
[ "$#" -ge 1 ] && [ "$#" -le 2 ] || {
    echo "usage: sh scripts/run_srgc_switch_validation.sh math|mbpp|all [run|timing|rules|status|results|json]" >&2
    exit 2
}
case "$1" in math|mbpp|all) ;; *) echo "dataset must be math, mbpp or all" >&2; exit 2 ;; esac
case "${2:-run}" in
    run) exec sh scripts/run_srgc_sr_refresh.sh "$1" switch_validation ;;
    timing|rules) exec sh scripts/run_srgc_sr_refresh.sh "$1" "$2" ;;
    status) exec sh scripts/run_srgc_sr_refresh.sh "$1" status switch_validation ;;
    results) exec sh scripts/run_srgc_sr_refresh.sh "$1" switch_validation_results ;;
    json) exec sh scripts/run_srgc_sr_refresh.sh "$1" switch_validation_json ;;
    *) echo "action must be run, timing, rules, status, results or json" >&2; exit 2 ;;
esac
