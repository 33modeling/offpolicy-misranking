#!/usr/bin/env bash
# Rotate registered independent matrices after primary work becomes blocked.
# No downloads, artifact deletion or promotion of a partial primary matrix.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ONCE=0
case "${1:-}" in --once) ONCE=1 ;; '') ;; *) exit 2 ;; esac
read -r -a PROFILES <<< "${OM_RLZERO_FALLBACK_PROFILES:-olmo3_domains qwen35_2b qwen35_4b qwen35 qwen38}"
[ "${#PROFILES[@]}" -gt 0 ] || { echo '[fallback] no registered profiles selected'; exit 2; }
for profile in "${PROFILES[@]}"; do
  case "$profile" in olmo3_domains|qwen35_2b|qwen35_4b|qwen35|qwen38) ;;
    *) echo "[fallback] unknown profile: $profile"; exit 2 ;;
  esac
done
COOLDOWN=${OM_RLZERO_FALLBACK_COOLDOWN_SECONDS:-900}
[[ "$COOLDOWN" =~ ^[1-9][0-9]*$ ]] || exit 2
LOCAL_LOCK_DIR=${OM_LOCAL_LOCK_DIR:-/tmp/offpolicy-misranking-$(id -u)}
mkdir -p "$LOCAL_LOCK_DIR" || exit 1
exec 7>"$LOCAL_LOCK_DIR/fallback.lock"
flock -n 7 || { echo '[fallback] another rotation worker owns this node'; exit 1; }
export OM_LOCAL_LOCK_DIR="$LOCAL_LOCK_DIR"
export OM_REPO="$PWD"
export ADDITIONAL_MAX_RESTARTS=0 ADDITIONAL_REGIME_MAX_RETRIES=1 OM_WAIT_PRIMARY=0
export REGIME_YIELD_WHEN_BUSY=1
export ADDITIONAL_GPU_WAIT_SECONDS="${ADDITIONAL_GPU_WAIT_SECONDS:-60}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 OM_ONLINE=0
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
# A fallback is a separate matrix, not a continuation of the OLMo generation.
unset OM_PIPELINE_REPO OM_PIPELINE_SCRIPT OM_GENERATION_GIT
unset REGIME_SKIP_COLLECTION REGIME_MATRIX MODEL_PATH
unset OM_EXTERNAL_GPU_KEEPALIVE
declare -A READY_AT=() COMPLETE=()
CHILD=""
cleanup() {
  if [ -n "$CHILD" ]; then
    kill -TERM -- "-$CHILD" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 -- "-$CHILD" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap '' HUP
while :; do
  for profile in "${PROFILES[@]}"; do
    [ "${COMPLETE[$profile]:-0}" = 0 ] || continue
    now=$(date +%s)
    [ "$now" -ge "${READY_AT[$profile]:-0}" ] || continue
    echo "[fallback] starting registered profile=$profile (offline, one matrix attempt)"
    setsid bash scripts/run_additional_experiments.sh --run "$profile" 7>&- &
    CHILD=$!
    wait "$CHILD"
    rc=$?
    # A failed launcher can leave a progress watcher holding descriptors.
    cleanup
    CHILD=""
    if [ "$rc" -eq 0 ]; then
      COMPLETE[$profile]=1
      echo "[fallback] profile=$profile complete; selecting next experiment"
    else
      READY_AT[$profile]=$(( $(date +%s) + COOLDOWN ))
      echo "[fallback] profile=$profile unavailable/failed rc=$rc; preserved for retry; selecting next experiment"
    fi
  done
  [ "$ONCE" -eq 0 ] || break
  # Waiting is only reached after every selected independent profile was tried.
  # Do not fake GPU work when models/data are missing or all work is complete.
  echo '[fallback] no other ready selected work; retaining worker and checking again'
  sleep 30
done
