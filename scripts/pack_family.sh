#!/usr/bin/env bash
# Pack finished families into ONE archive that fits through git (parts < 90 MB):
# every log, config, manifest, score, report, GRPO statistic, recovery record,
# queue marker and readout of the family. Raw rollouts (GB) and weights /
# optimizer / projected-gradient tensors are left out; the manifest inside the
# archive lists what was excluded. Read-only for the experiment: it writes only
# under $OM_WORK/exports and $OM_WORK/readouts.
#
#   bash scripts/pack_family.sh                      # h100: every family with all four points DONE
#   bash scripts/pack_family.sh math500 0            # one family (finished or not)
#   bash scripts/pack_family.sh baseline math500 0   # other profile
#   PACK_READOUT=0 ...                               # skip the regime readout (fast)
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
DRIFTS="${PACK_DRIFTS:-0 25 100 400}"
PART_MB="${PACK_PART_MB:-90}"
[ -d "$ROOT" ] || { echo "[abort] no experiment root: $ROOT"; exit 1; }

family_complete() {  # family_complete <dataset> <seed>
  local d
  for d in $DRIFTS; do
    [ -s "$ROOT/family-$1-s$2/$TAG-s$2-$1-d$d/DONE" ] || return 1
  done
}

families=()
if [ "$#" -ge 2 ]; then
  case "$2" in ''|*[!0-9]*) echo "usage: bash scripts/pack_family.sh [h100|baseline] [<dataset> <seed>]"; exit 2 ;; esac
  [ -d "$ROOT/family-$1-s$2" ] || { echo "[abort] no such family: $ROOT/family-$1-s$2"; exit 1; }
  families+=("$1 $2")
elif [ "$#" -ne 0 ]; then
  echo "usage: bash scripts/pack_family.sh [h100|baseline] [<dataset> <seed>]"; exit 2
else
  for dir in "$ROOT"/family-*; do
    [ -d "$dir" ] || continue
    name=${dir##*/family-}; dataset=${name%-s*}; seed=${name##*-s}
    family_complete "$dataset" "$seed" && families+=("$dataset $seed")
  done
  [ "${#families[@]}" -gt 0 ] || { echo "[pack] no family has all four points DONE under $ROOT; name one: bash scripts/pack_family.sh <dataset> <seed>"; exit 1; }
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EXPORTS="$OM_WORK/exports"
mkdir -p "$EXPORTS" || { echo "[abort] cannot create $EXPORTS"; exit 1; }
label=$(printf '%s\n' "${families[@]}" | awk '{printf "%s%s-s%s", (NR>1?"_":""), $1, $2}')
ARCHIVE="$EXPORTS/$TAG-$label-$STAMP.tar.gz"
MANIFEST="$EXPORTS/.manifest-$STAMP.txt"

# Regime readout (registered labels need the full matrix; this is the one-family preview).
if [ "${PACK_READOUT:-1}" = 1 ]; then
  for fam in "${families[@]}"; do
    set -- $fam
    if family_complete "$1" "$2"; then
      echo "[pack] readout $1/s$2 (bootstrap ${PACK_BOOT:-1000})"
      bash scripts/family_readout.sh "$PROFILE" "$1" "$2" "${PACK_BOOT:-1000}" >"$EXPORTS/.readout-$1-s$2-$STAMP.log" 2>&1 \
        || echo "[pack] readout failed for $1/s$2 (log kept in the archive)"
    fi
  done
fi

# Paths relative to $OM_WORK so the archive unpacks into one tree.
paths=()
add() { [ -e "$OM_WORK/$1" ] && paths+=("$1"); return 0; }
for fam in "${families[@]}"; do
  set -- $fam
  add "runs/$TAG/family-$1-s$2"
  add "runs/$TAG/family-results/$1-s$2"
  for f in "$OM_WORK/readouts"/family-$1-s$2-*; do [ -e "$f" ] && add "${f#"$OM_WORK"/}"; done
  for f in "$ROOT/.families/$1-s$2".*; do [ -e "$f" ] && add "${f#"$OM_WORK"/}"; done
  for f in "$EXPORTS"/.readout-$1-s$2-$STAMP.log; do [ -e "$f" ] && add "${f#"$OM_WORK"/}"; done
done
add "runs/$TAG/.queue/generation.git"
add "runs/$TAG/logs"
add "runs/$TAG/preflight"
[ "${#paths[@]}" -gt 0 ] || { echo "[abort] nothing to pack"; exit 1; }

EXCLUDES=(
  --exclude='rollouts_*.jsonl' --exclude='rollouts_*.partial' --exclude='rollouts_*.shard*'
  --exclude='*.safetensors' --exclude='optimizer.pt' --exclude='*.pt'
  --exclude='.stale-*' --exclude='.pipeline-activity*' --exclude='keepalive.log' --exclude='*.lock'
)
{
  echo "archive=$(basename "$ARCHIVE")"
  echo "created_utc=$STAMP host=$(hostname 2>/dev/null || echo ?)"
  echo "checkout=$(git rev-parse HEAD 2>/dev/null || echo ?)"
  echo "generation_git=$(cat "$ROOT/.queue/generation.git" 2>/dev/null || echo none)"
  echo "profile=$PROFILE tag=$TAG root=$ROOT"
  echo "families=$(printf '%s;' "${families[@]}")"
  echo "excluded_patterns=rollouts_*.jsonl rollouts_*.partial rollouts_*.shard* *.safetensors optimizer.pt *.pt .stale-* .pipeline-activity* keepalive.log *.lock"
  echo "--- included files (bytes path) ---"
  (cd "$OM_WORK" && find "${paths[@]}" -type f \
     ! -name 'rollouts_*.jsonl' ! -name 'rollouts_*.partial' ! -name 'rollouts_*.shard*' \
     ! -name '*.safetensors' ! -name 'optimizer.pt' ! -name '*.pt' ! -name 'keepalive.log' ! -name '*.lock' \
     ! -path '*/.stale-*' ! -name '.pipeline-activity*' -printf '%s %p\n' 2>/dev/null | sort -k2)
  echo "--- excluded files (bytes path) ---"
  (cd "$OM_WORK" && find "${paths[@]}" -type f \
     \( -name 'rollouts_*.jsonl' -o -name 'rollouts_*.partial' -o -name 'rollouts_*.shard*' \
        -o -name '*.safetensors' -o -name 'optimizer.pt' -o -name '*.pt' \) -printf '%s %p\n' 2>/dev/null | sort -k2)
} > "$MANIFEST"

if ! tar -czf "$ARCHIVE" "${EXCLUDES[@]}" -C "$OM_WORK" "${paths[@]}" -C "$EXPORTS" --transform "s|^\.manifest-$STAMP\.txt$|MANIFEST.txt|" ".manifest-$STAMP.txt"; then
  echo "[abort] tar failed"; rm -f -- "$ARCHIVE"; exit 1
fi
rm -f -- "$MANIFEST"
size=$(stat -c %s "$ARCHIVE")
echo "[pack] $ARCHIVE ($((size / 1048576)) MB)"
echo "[pack] included: $(tar -tzf "$ARCHIVE" | grep -c -v '/$') files; excluded raw rollouts, weights and tensors (see MANIFEST.txt inside)"
if [ "$size" -gt $((PART_MB * 1048576)) ]; then
  split -b "${PART_MB}m" -d -a 2 -- "$ARCHIVE" "$ARCHIVE.part-" || { echo "[abort] split failed"; exit 1; }
  rm -f -- "$ARCHIVE"
  echo "[pack] larger than ${PART_MB} MB: split into parts (each fits through git). Upload every part; rebuild with:"
  echo "       cat $(basename "$ARCHIVE").part-* > $(basename "$ARCHIVE")"
  ls -l "$ARCHIVE".part-* | awk '{print "[pack]   " $5 " " $9}'
fi
echo "[pack] to hand over: copy the file(s) above into the transfer repository and push"
