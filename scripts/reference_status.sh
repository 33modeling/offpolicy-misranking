#!/usr/bin/env bash
# Read shared reference metadata/logs only; never probes CUDA or starts work.
set -uo pipefail
WORK=${1:?usage: reference_status.sh WORK}
echo '=== REFERENCE EXPERIMENTS (separate from primary 40 points) ==='
now=$(date +%s); host_here=$(hostname); found=0
while IFS= read -r entry; do
  [ -s "$entry" ] || continue
  IFS=$'\t' read -r state host pid dataset replicate fk vk seed run console started < "$entry" || continue
  case "$state" in STARTING|RUNNING|COMPLETE|FAILED|INTERRUPTED) ;; *) continue ;; esac
  found=1; liveness=remote-unverified
  if [ "$host" = "$host_here" ] && [[ "$pid" =~ ^[0-9]+$ ]]; then
    if kill -0 "$pid" 2>/dev/null; then liveness=local-pid-present
    else liveness=local-pid-absent; [ "$state" != RUNNING ] && [ "$state" != STARTING ] || state=EXITED_UNCLEAN; fi
  fi
  echo "reference dataset=$dataset replicate=$replicate host=$host pid=$pid reported_state=$state liveness=$liveness fresh_k=$fk val_k=$vk seed=$seed"
  echo "  run=$run"
  echo "  console=$console"
  if [ "$state" = FAILED ] && [ -s "$console" ]; then
    echo '  failed conditions from this invocation:'
    grep -E '^\[failed\]|^\[abort\]' "$console" | tail -5
  fi
  [ "$run" != - ] && [ -d "$run" ] || continue
  if [ -s "$run/RB_DONE" ]; then echo '  completion=RB_DONE (marker, not an independent artifact audit)'; fi
  for log in "$run"/logs/*.log; do
    [ -s "$log" ] || continue
    modified=$(stat -c %Y "$log" 2>/dev/null) || continue
    scope=modified-since-condition-start
    [[ "$started" =~ ^[0-9]+$ ]] && [ "$modified" -lt "$started" ] && scope=prior-condition-history
    line=$(tail -1 "$log")
    printf '  stage_log=%s age=%ss scope=%s last=%s\n' "${log##*/}" "$((now - modified))" "$scope" "$line"
  done
  echo '  Stage tails are activity evidence, not attribution of errors to the current attempt.'
  echo '  ETA shown in a stage log applies to that stage/shard only, not the entire experiment.'
done < <(find "$WORK/reference-workers" -maxdepth 1 -name '*.state' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -10 | cut -d' ' -f2-)
if [ "$found" -eq 0 ]; then
  echo "reference launcher records=none under $WORK/reference-workers; execution status UNKNOWN (not proof no job exists)"
fi
echo "reference_outputs=$WORK/runs/reference-axes"
