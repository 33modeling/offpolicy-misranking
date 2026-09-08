#!/usr/bin/env bash
# Read-only admission gate for post-primary compute; source after setup_env.sh.
require_olmo3_complete() {
  local profile=${OM_PRIMARY_PROFILE:-h100} tag config root results binding
  local generation config_sha revision extra dataset seed drift family output
  case "$profile" in
    h100) tag=olmo3-1025-7b-base-rlzero-grpo-h100-v2 ;;
    baseline) tag=olmo3-1025-7b-base-rlzero-grpo-v1 ;;
    *) echo "[primary-pending] unknown OM_PRIMARY_PROFILE=$profile" >&2; return 75 ;;
  esac
  config="$(dirname "${BASH_SOURCE[0]}")/../configs/olmo3_rlzero$([ "$profile" != h100 ] || printf _h100).json"
  config=${OM_RLZERO_CONFIG:-$config}
  tag=${OM_OLMO3_MODEL_TAG:-$tag}
  root=${OM_OLMO3_ROOT:-${OM_WORK:?}/runs/$tag}
  results=${OM_OLMO3_RESULTS:-$OM_WORK/results/$tag}
  binding=$(cat "$results/COMPLETE" 2>/dev/null) || binding=""
  read -r generation config_sha revision extra <<< "$binding"
  if [[ ! "$generation" =~ ^[0-9a-f]{40}$ || ! "$config_sha" =~ ^[0-9a-f]{64}$ \
      || ! "$revision" =~ ^[0-9a-f]{40}$ || -n "$extra" ]] \
      || [ "$generation" != "$(cat "$root/.queue/generation.git" 2>/dev/null)" ] \
      || [ ! -s "$config" ] \
      || [ "$config_sha" != "$(sha256sum "$config" 2>/dev/null | awk '{print $1}')" ]; then
    echo "[primary-pending] OLMo3 full completion is missing or stale: $results/COMPLETE; no additional model will start" >&2
    return 75
  fi
  for output in REGIME.json REGIME.csv REGIME_SUMMARY.csv FINAL_REPORT.md; do
    [ -s "$results/$output" ] || {
      echo "[primary-pending] missing OLMo3 collection: $results/$output" >&2; return 75;
    }
  done
  for dataset in math500 mbpp; do
    for seed in 0 1 2 3 4; do
      family="$root/family-$dataset-s$seed"
      [ "$(cat "$family/.family-complete" 2>/dev/null)" = "$binding $dataset $seed" ] || {
        echo "[primary-pending] OLMo3 $dataset/s$seed is not complete" >&2; return 75;
      }
      for drift in 0 25 100 400; do
        [ -s "$family/$tag-s$seed-$dataset-d$drift/DONE" ] || {
          echo "[primary-pending] OLMo3 $dataset/s$seed/d$drift is not DONE" >&2; return 75;
        }
      done
    done
  done
  echo '[primary-complete] OLMo3 40/40 points and full collection verified'
}
