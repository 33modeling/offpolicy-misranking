#!/usr/bin/env bash
# Why is a family not finishing? One small plain-text file with the evidence the
# status table does not show: for every family that has an owner or was written
# in the last day, the worker log's decision lines (try / family-plan /
# done-but-incomplete / point-failed / contract-fail / abort / quarantine), the
# tail of the current point's newest attempt logs, main.log, supervisor.log,
# and the queue markers. Read-only; writes only under $OM_WORK/exports.
#
#   bash scripts/why.sh              # h100 matrix, every active family
#   bash scripts/why.sh mbpp 0       # one family
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
PROFILE=h100
case "${1:-}" in baseline|h100) PROFILE=$1; shift ;; esac
case "$PROFILE" in
  baseline) TAG=olmo3-1025-7b-base-rlzero-grpo-v1 ;;
  h100)     TAG=olmo3-1025-7b-base-rlzero-grpo-h100-v2 ;;
esac
TAG="${OM_OLMO3_MODEL_TAG:-$TAG}"
ROOT="${OM_OLMO3_ROOT:-$OM_WORK/runs/$TAG}"
[ -d "$ROOT" ] || { echo "[abort] no experiment root: $ROOT"; exit 1; }
PATTERN='try [0-9]+/|family-plan|family-order|done-but-incomplete|point-failed|contract-fail|config-abort|\[abort\]|quarantin|family-retry|family-loop|cuda-flaky|cuda-recovery|\[repair|claimed|released|incomplete'
TAIL="${WHY_TAIL_LINES:-30}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EXPORTS="$OM_WORK/exports"; mkdir -p "$EXPORTS" || { echo "[abort] cannot create $EXPORTS"; exit 1; }
OUT="$EXPORTS/why-$TAG-$STAMP.txt"

families=()
if [ "$#" -ge 2 ]; then
  [ -d "$ROOT/family-$1-s$2" ] || { echo "[abort] no such family: $ROOT/family-$1-s$2"; exit 1; }
  families+=("$1 $2")
else
  for dir in "$ROOT"/family-*; do
    [ -d "$dir" ] || continue
    name=${dir##*/family-}; dataset=${name%-s*}; seed=${name##*-s}
    if [ -s "$ROOT/.families/$dataset-s$seed.owner" ] \
        || [ -n "$(find "$dir" -maxdepth 3 -newermt '-1 day' -print -quit 2>/dev/null)" ]; then
      families+=("$dataset $seed")
    fi
  done
fi

section() { printf '\n===== %s =====\n' "$*"; }
{
  echo "why=$(basename "$OUT")  created_utc=$STAMP  host=$(hostname 2>/dev/null || echo ?)  checkout=$(git rev-parse --short HEAD 2>/dev/null || echo ?)"
  echo "generation_git=$(cat "$ROOT/.queue/generation.git" 2>/dev/null || echo none)  families=$(printf '%s;' "${families[@]}")"
  section "queue markers (.families)"
  for m in "$ROOT"/.families/*.owner "$ROOT"/.families/*.loop "$ROOT"/.families/*.fail*; do
    [ -f "$m" ] || continue
    echo "--- $(basename "$m")"; head -c 600 "$m"; echo
  done
  for fam in "${families[@]}"; do
    set -- $fam; dataset=$1; seed=$2
    froot="$ROOT/family-$dataset-s$seed"
    section "FAMILY $dataset/s$seed"
    for run in "$froot"/$TAG-s$seed-$dataset-d*; do
      [ -d "$run" ] || continue
      echo "$(basename "$run"): $( [ -s "$run/DONE" ] && echo DONE || echo "no DONE" )  attempts=$(ls "$run"/logs/regime-attempt-*.log 2>/dev/null | wc -l)  last-write=$(find "$run" -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM\n' 2>/dev/null | sort | tail -1)"
    done
    owner=$(sed -n 's/^worker=//p' "$ROOT/.families/$dataset-s$seed.owner" 2>/dev/null | head -1)
    for wlog in $( { [ -n "$owner" ] && ls "$ROOT/logs/$owner"*.log 2>/dev/null; grep -ls "$dataset/s$seed" "$ROOT"/logs/run*.log 2>/dev/null | xargs -r ls -t | head -2; } | awk '!seen[$0]++'); do
      echo "--- worker log $(basename "$wlog") (decision lines, last 40)"
      grep -E "$PATTERN" "$wlog" 2>/dev/null | grep -F "$dataset/s$seed" | tail -40 | cut -c1-240
      echo "--- worker log $(basename "$wlog") (last 8 lines)"
      tail -n 8 "$wlog" | cut -c1-200
    done
    # the point being worked on: newest logs directory
    run=$(ls -td "$froot"/$TAG-s$seed-$dataset-d*/logs 2>/dev/null | head -1); run=${run%/logs}
    [ -n "$run" ] || continue
    echo "--- current point: $(basename "$run")"
    for f in "$run/logs/supervisor.log" "$run/logs/main.log"; do
      [ -s "$f" ] || continue
      echo "--- $(basename "$f") tail $TAIL"; tail -n "$TAIL" "$f" | cut -c1-220
    done
    for f in $(ls -t "$run"/logs/regime-attempt-*.log 2>/dev/null | head -3); do
      echo "--- $(basename "$f") ($(wc -l < "$f") lines) tail $TAIL"; tail -n "$TAIL" "$f" | cut -c1-220
      echo "--- $(basename "$f") error lines"; grep -nE 'config-abort|\[abort\]|Error|Traceback|누락' "$f" | tail -8 | cut -c1-220
    done
  done
  section "ALERTS tail"; tail -n 20 "$ROOT/logs/ALERTS.log" 2>/dev/null | cut -c1-200
} > "$OUT" 2>&1
echo "[why] $OUT ($(( $(stat -c %s "$OUT") / 1024 )) KB)"
echo "[why] plain text: copy it into the transfer repository and push"
