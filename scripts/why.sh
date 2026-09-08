#!/usr/bin/env bash
# One small text file that answers "why is this family not finishing?".
#
#   bash scripts/why.sh              # every family that is running or was touched today
#   bash scripts/why.sh mbpp 0       # one family
#   bash scripts/why.sh all          # every family, finished ones included
#
# Part 1 is a table: one line per family with its state and the reason, taken
# from durable evidence (DONE markers, the supervisor's own decision lines, the
# failing attempt's error line), not from guesswork.
# Part 2 is the evidence behind each line, so the diagnosis can be checked.
# Read-only for the experiment; writes only under $OM_WORK/exports.
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
DRIFTS="${WHY_DRIFTS:-0 25 100 400}"
TAIL="${WHY_TAIL_LINES:-25}"
[ -d "$ROOT" ] || { echo "[abort] no experiment root: $ROOT"; exit 1; }

ALL=0
[ "${1:-}" = all ] && { ALL=1; shift; }
families=()
if [ "$#" -ge 2 ]; then
  [ -d "$ROOT/family-$1-s$2" ] || { echo "[abort] no such family: $ROOT/family-$1-s$2"; exit 1; }
  families+=("$1 $2")
else
  for dir in "$ROOT"/family-*; do
    [ -d "$dir" ] || continue
    name=${dir##*/family-}; dataset=${name%-s*}; seed=${name##*-s}
    # Every family that is not finished, however long it has been silent. The
    # first version required a write in the last day, which hid exactly the two
    # families that had been stuck for 26 hours: the ones most needing a look
    # (2026-09-08, math500/s4 and mbpp/s4 absent from the report).
    complete=1
    for d in $DRIFTS; do
      [ -s "$dir/$TAG-s$seed-$dataset-d$d/DONE" ] || complete=0
    done
    if [ "$ALL" = 1 ] || [ "$complete" = 0 ]; then
      families+=("$dataset $seed")
    fi
  done
fi
[ "${#families[@]}" -gt 0 ] || { echo "[why] no family is running or was written to in the last day under $ROOT"; exit 1; }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EXPORTS="$OM_WORK/exports"; mkdir -p "$EXPORTS" || { echo "[abort] cannot create $EXPORTS"; exit 1; }
OUT="$EXPORTS/why-$TAG-$STAMP.txt"
DECISIONS='\[(point-failed|done-but-incomplete|family-plan|family-order|family-next|family-retry|family-loop|cuda-flaky|cuda-recovery|contract-fail|repair|repair-failed|config-abort|abort|queue)\]|try [0-9]+/[0-9]+ ->'
ERRORS='config-abort|\[abort\]|OutOfMemoryError|CUDA error|CUBLAS_STATUS|device-side assert|RuntimeError|Error:|누락'

run_dir_of() { printf '%s/family-%s-s%s/%s-s%s-%s-d%s\n' "$ROOT" "$1" "$2" "$TAG" "$2" "$1" "$3"; }
age_secs() {  # age_secs <file> -> seconds, or -1 when there is no such file
  [ -e "$1" ] || { printf -- '-1\n'; return; }
  printf '%s\n' $(( $(date +%s) - $(stat -c %Y "$1" 2>/dev/null || date +%s) ))
}
age_of() {  # age_of <file> -> "12s" / "4m" / "3h07m" / "-"
  [ -e "$1" ] || { printf -- '-\n'; return; }
  local s=$(( $(date +%s) - $(stat -c %Y "$1" 2>/dev/null || date +%s) ))
  if [ "$s" -lt 60 ]; then printf '%ds\n' "$s"
  elif [ "$s" -lt 3600 ]; then printf '%dm\n' $((s / 60))
  elif [ "$s" -lt 86400 ]; then printf '%dh%02dm\n' $((s / 3600)) $(((s % 3600) / 60))
  else printf '%dd\n' $((s / 86400)); fi
}
# -d matters: without it `ls -t <dir>` lists the directory CONTENTS, so the
# newest point directory came back as one of its files (e.g. "DONE").
newest_in() { ls -td "$@" 2>/dev/null | head -1; }
refused_count() {  # refused_count <point dir>: rejections since the point was last accepted
  awk '/\[point-accepted\]/ { n = 0; next } /\[done-but-incomplete\]/ { n++ } END { print n + 0 }' \
    "$1/logs/supervisor.log" 2>/dev/null || echo 0
}
refused_reason() {  # refused_reason <point dir>: the newest rejection reason
  awk '/\[point-accepted\]/ { last = ""; next } /\[done-but-incomplete\]/ { sub(/^.*\[done-but-incomplete\] /, ""); last = $0 } END { print last }' \
    "$1/logs/supervisor.log" 2>/dev/null
}
short_reason() {  # short_reason <width>: shorten each line but keep BOTH ends
  # The tail of a message names the mismatch (`config={...}, expected {...}`), so
  # head-only truncation hid the one differing field for a whole night
  # (2026-09-07). Keep the head and the tail, drop the middle.
  awk -v w="${1:-200}" '{
    gsub(/[[:space:]]+/, " ")
    if (length($0) <= w) { print; next }
    k = w - 5; h = int((k + 1) / 2)
    print substr($0, 1, h) " ... " substr($0, length($0) - (k - h) + 1)
  }'
}

# --- per family: points, current point, state, reason ------------------------
declare -A F_POINTS F_CURRENT F_STATE F_WHY F_AGE F_STAGE
for fam in "${families[@]}"; do
  set -- $fam; dataset=$1; seed=$2; key="$dataset/s$seed"
  froot="$ROOT/family-$dataset-s$seed"
  points=""; done_count=0
  for d in $DRIFTS; do
    run=$(run_dir_of "$dataset" "$seed" "$d")
    if [ -s "$run/DONE" ]; then points+="+"; done_count=$((done_count + 1))
    elif [ -n "$(newest_in "$run"/logs/*.log)" ]; then points+="*"
    else points+="."; fi   # a directory with no log was created, never worked on
  done
  # The point being worked on is the one being WRITTEN, which is often a finished
  # point that the completion check refused and the worker is re-running. Picking
  # "the last drift without DONE" pointed at an idle directory and every age,
  # stage and reason below then described the wrong point.
  current=""; newest_ns=0
  for d in $DRIFTS; do
    run=$(run_dir_of "$dataset" "$seed" "$d")
    # a point with no log was never worked on; its directory mtime is just when
    # it was created and would beat the point that is actually being written
    candidate=$(newest_in "$run"/logs/*.log)
    [ -n "$candidate" ] || continue
    ns=$(stat -c %Y "$candidate" 2>/dev/null || echo 0)
    if [ "$ns" -gt "$newest_ns" ]; then newest_ns=$ns; current=$run; fi
  done
  [ -n "$current" ] || current=$(newest_in "$froot"/$TAG-s$seed-$dataset-d*)
  F_POINTS[$key]=$points
  F_CURRENT[$key]=${current:-none}
  newest_log=$(newest_in "$current"/logs/*.log)
  F_AGE[$key]=$(age_of "${newest_log:-$current}")
  F_STAGE[$key]=$(basename "${newest_log:-none}" .log)
  # the supervisor's own last word about this family, if any
  supervisor=$(newest_in "$froot"/*/logs/supervisor.log)
  reason=""
  [ -z "$supervisor" ] || reason=$(grep -E '\[(point-failed|done-but-incomplete)\]' "$supervisor" 2>/dev/null | tail -1 | short_reason 180)
  [ -n "$reason" ] || reason=$(grep -hE "$ERRORS" "$current"/logs/*.log 2>/dev/null | tail -1 | short_reason 180)
  refused=$(refused_count "$current")
  if [ "${refused:-0}" -gt 0 ] && [ ! -f "$ROOT/.families/$dataset-s$seed.loop" ]; then
    # the GPUs are busy on a point that already finished and was refused: the
    # single state that reads as healthy everywhere else (2026-09-07 night)
    F_STATE[$key]="REDOING A REFUSED POINT ${refused}x - it finished and the completion check refused it"
    F_WHY[$key]=$(refused_reason "$current" | short_reason 200)
  elif [ -f "$ROOT/.families/$dataset-s$seed.loop" ]; then
    F_STATE[$key]="LOOPING (every worker skips it until you clear it)"
    F_WHY[$key]=$(sed -n 's/^last_error=//p' "$ROOT/.families/$dataset-s$seed.loop" | head -1 | short_reason 180)
  elif [ "$done_count" -eq "$(printf '%s\n' $DRIFTS | wc -l)" ]; then
    F_STATE[$key]="COMPLETE"
    F_WHY[$key]="all $done_count points done"
  elif [ -f "$ROOT/.families/$dataset-s$seed.owner.json" ]; then
    # The launcher writes .owner.json (run_olmo3_rlzero.sh:1001), as status and
    # the heartbeat both read. A test for ".owner" is never true, and every
    # running family was reported as "QUEUED (no worker)" - the one symptom that
    # makes an operator restart a healthy node.
    owner_file="$ROOT/.families/$dataset-s$seed.owner.json"
    owner_age=$(age_of "$owner_file")
    owner_who=$(sed -n 's/.*"worker": *"\([^"]*\)".*/\1/p' "$owner_file" | head -1)
    owner_host=$(sed -n 's/.*"host": *"\([^"]*\)".*/\1/p' "$owner_file" | head -1)
    owner_who="worker=${owner_who:-?} host=${owner_host:-?}"
    # compare seconds, never the formatted age: "3h07m" ends in "m" and a glob
    # test for minutes called a family silent for hours RUNNING.
    write_secs=$(age_secs "${newest_log:-$current}")
    if [ "$write_secs" -ge 0 ] && [ "$write_secs" -le "${WHY_RUNNING_SECONDS:-300}" ]; then
      F_STATE[$key]="RUNNING ($owner_who, claimed $owner_age ago)"
      F_WHY[$key]=$(tail -1 "${newest_log:-/dev/null}" 2>/dev/null | short_reason 180)
    else
      F_STATE[$key]="OWNED BUT SILENT for ${F_AGE[$key]} - $owner_who holds it and writes nothing"
      F_WHY[$key]=${reason:-no error line; look at the stage log below}
    fi
  else
    F_STATE[$key]="QUEUED (no worker; last write ${F_AGE[$key]} ago)"
    F_WHY[$key]=${reason:-no error line; the worker was killed or moved on}
  fi
done

{
  echo "why=$(basename "$OUT")  created_utc=$STAMP  host=$(hostname 2>/dev/null || echo ?)"
  echo "checkout=$(git rev-parse --short HEAD 2>/dev/null || echo ?)  generation=$(cat "$ROOT/.queue/generation.git" 2>/dev/null || echo none)  root=$ROOT"
  echo
  echo "================ 1. DIAGNOSIS (one line per family) ================"
  echo "points: d0 d25 d100 d400   + done   * in progress   . not started"
  echo
  for fam in "${families[@]}"; do
    set -- $fam; key="$1/s$2"
    printf '%-12s %-6s %-58s\n' "$key" "${F_POINTS[$key]}" "${F_STATE[$key]}"
    point_name=$(basename "${F_CURRENT[$key]}"); point_name=${point_name##*-}
    printf '%-12s %-6s point=%s stage=%s last write=%s\n' "" "" "$point_name" "${F_STAGE[$key]}" "${F_AGE[$key]}"
    printf '%-12s %-6s why: %s\n\n' "" "" "${F_WHY[$key]:-(nothing recorded)}"
  done
  echo "================ 2. EVIDENCE ================"
  echo
  echo "--- queue markers ($ROOT/.families)"
  for m in "$ROOT"/.families/*.owner.json "$ROOT"/.families/*.loop; do
    [ -f "$m" ] || continue
    echo "  $(basename "$m") (written $(age_of "$m") ago): $(tr '\n' ' ' < "$m" | short_reason 220)"
  done
  echo
  echo "--- workers seen in $ROOT/logs (newest first)"
  for wl in $(ls -t "$ROOT"/logs/run*.log 2>/dev/null | head -8); do
    echo "  $(basename "$wl") (last line $(age_of "$wl") ago): $(tail -1 "$wl" | short_reason 160)"
  done
  for fam in "${families[@]}"; do
    set -- $fam; dataset=$1; seed=$2; key="$dataset/s$seed"
    froot="$ROOT/family-$dataset-s$seed"; current=${F_CURRENT[$key]}
    echo
    echo "===== $key ====="
    echo "state: ${F_STATE[$key]}   points ${F_POINTS[$key]}   why: ${F_WHY[$key]}"
    for d in $DRIFTS; do
      run=$(run_dir_of "$dataset" "$seed" "$d")
      [ -d "$run" ] || { echo "  d$d: not started"; continue; }
      echo "  d$d: $( [ -s "$run/DONE" ] && echo DONE || echo "no DONE" ), attempts=$(ls "$run"/logs/regime-attempt-*.log 2>/dev/null | wc -l), last write $(age_of "$(newest_in "$run"/logs/*.log)") ago, rollouts=$(ls "$run"/rollouts_*.jsonl 2>/dev/null | wc -l) merged +$(ls "$run"/*.partial 2>/dev/null | wc -l) partial, grpo steps=$(cat "$run"/policy_step_*/grpo_stats.jsonl 2>/dev/null | wc -l)"
    done
    echo "  --- decision lines for $key from the worker logs (last 25)"
    grep -hE "$DECISIONS" "$ROOT"/logs/run*.log 2>/dev/null | grep -F "$key" | tail -25 | sed 's/^/    /' | short_reason 230
    [ -d "$current" ] || continue
    for f in "$current/logs/supervisor.log" "$current/logs/main.log"; do
      [ -s "$f" ] || continue
      echo "  --- $(basename "$current")/logs/$(basename "$f") tail $TAIL"
      tail -n "$TAIL" "$f" | sed 's/^/    /' | short_reason 230
    done
    for f in $(ls -t "$current"/logs/regime-attempt-*.log 2>/dev/null | head -2); do
      echo "  --- $(basename "$f") ($(wc -l < "$f") lines): error lines"
      grep -nE "$ERRORS" "$f" 2>/dev/null | tail -6 | sed 's/^/    /' | short_reason 230
      echo "  --- $(basename "$f") tail $TAIL"
      tail -n "$TAIL" "$f" | sed 's/^/    /' | short_reason 230
    done
    echo "  --- last error line of every stage log of $(basename "$current")"
    for f in "$current"/logs/*.log; do
      [ -s "$f" ] || continue
      line=$(grep -hE "$ERRORS" "$f" 2>/dev/null | tail -1)
      [ -n "$line" ] && printf '    %-34s %s\n' "$(basename "$f")" "$(printf '%s' "$line" | short_reason 180)"
    done
  done
  echo
  echo "--- ALERTS.log tail"
  tail -n 20 "$ROOT/logs/ALERTS.log" 2>/dev/null | sed 's/^/  /' | short_reason 200
} > "$OUT" 2>&1
echo "[why] $OUT ($(( $(stat -c %s "$OUT") / 1024 )) KB, $(wc -l < "$OUT") lines)"
sed -n '/1. DIAGNOSIS/,/2. EVIDENCE/p' "$OUT" | head -60
echo "[why] plain text: copy it into the transfer repository and push"
