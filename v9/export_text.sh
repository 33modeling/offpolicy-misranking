#!/bin/sh
# V9 key results as one text file, no arguments: sh v9/export_text.sh
#   -> $WORK/v9-export/v9-results-<node>-<time>.txt (+ .sha256)
# Runs `sh v9/run.sh collect` first, then writes: node, time, code commit, collect exit code, the status screen,
# and every file of $V9_ROOT/paper (tables .tex, results.json, .csv, .dat, .json), each under a header line
#   ===== FILE paper/<name> bytes=<n> sha256=<hash> =====
# No checkpoints and no raw records. A failed collect still writes the file (with the previous paper/ outputs).
set -u
export PYTHONDONTWRITEBYTECODE=1
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$HERE/.." && pwd)
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
V9=${V9_ROOT:-$WORK/v9}
HOST=$(hostname 2>/dev/null || echo node)
OUTDIR=${V9_EXPORT_OUT:-$WORK/v9-export}
mkdir -p "$OUTDIR" 2>/dev/null || { OUTDIR=$HOME/v9-export; mkdir -p "$OUTDIR"; }
OUT="$OUTDIR/v9-results-$HOST-$(date +%Y%m%d-%H%M%S).txt"
[ -d "$V9" ] || { echo "[v9-text] no V9 results at $V9" >&2; exit 2; }
echo "[v9-text] collect ..."
LOG="$OUT.collect.log"
sh "$HERE/run.sh" collect > "$LOG" 2>&1
CRC=$?
{
    echo "===== V9 RESULTS host=$HOST at=$(date '+%Y-%m-%d %H:%M:%S %Z') v9_root=$V9"
    echo "code=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown) collect_exit=$CRC"
    echo "===== COLLECT LOG (last 40 lines) ====="
    tail -40 "$LOG"
    echo "===== STATUS ====="
    sh "$HERE/run.sh" status 2>&1
    for f in "$V9"/paper/*; do
        [ -f "$f" ] || continue
        n=$(basename "$f")
        echo "===== FILE paper/$n bytes=$(wc -c < "$f" | tr -d ' ') sha256=$(sha256sum "$f" | cut -d' ' -f1) ====="
        cat "$f"
        echo
    done
    echo "===== END ====="
} > "$OUT"
rm -f "$LOG"
(cd "$OUTDIR" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256")
echo "[v9-text] collect exit=$CRC"
echo "[v9-text] file: $OUT ($(du -h "$OUT" | cut -f1))"
[ "$CRC" -eq 0 ] || { echo "[v9-text] collect failed, the file holds the previous tables"; exit 1; }
echo "[v9-text] ok"
