#!/bin/sh
# Fixed-state GRPO interventions, using the existing four-GPU supervisor.
set -eu
cd "$(dirname "$0")/.."
[ "$#" -ge 1 ] && [ "$#" -le 2 ] || {
    echo "usage: sh scripts/run_srgc_mechanism.sh math|mbpp|all [run|status|results|json]" >&2
    exit 2
}
case "$1" in math|mbpp|all) ;; *) echo "dataset must be math, mbpp or all" >&2; exit 2 ;; esac
case "${2:-run}" in
    run) exec sh scripts/run_srgc_sr_refresh.sh "$1" mechanism ;;
    status) exec sh scripts/run_srgc_sr_refresh.sh "$1" status mechanism ;;
    results) exec sh scripts/run_srgc_sr_refresh.sh "$1" mechanism_results ;;
    json) exec sh scripts/run_srgc_sr_refresh.sh "$1" mechanism_json ;;
    *) echo "action must be run, status, results or json" >&2; exit 2 ;;
esac
