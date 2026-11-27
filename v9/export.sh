#!/bin/sh
# V9 results into one file, no arguments: sh v9/export.sh
#   -> $WORK/v9-export/v9-export-<node>-<time>.tar.gz (+ .sha256)
# Contents:
#   steps.txt                 node, time, code commit, exit code of each step
#   logs/collect.log          output of `sh v9/run.sh collect` (run first, so paper/ is current)
#   status.txt                output of `sh v9/run.sh status`
#   v9/...                    every file below $V9_ROOT up to 50 MB (paper/ tables, results.json, run records,
#                             per-25-update evaluations, bench results, costs, task state); checkpoints and
#                             model weights (*.pt, *.safetensors, *.bin, *.pth, *.ckpt) and larger files are left out
#   excluded.txt              path and size of every file left out
# $V9_ROOT is only read (collect writes its usual paper/ outputs). A failed step does not stop the rest.
set -u
export PYTHONDONTWRITEBYTECODE=1
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO=$(CDPATH= cd -- "$HERE/.." && pwd)
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
V9=${V9_ROOT:-$WORK/v9}
HOST=$(hostname 2>/dev/null || echo node)
STAMP=$(date +%Y%m%d-%H%M%S)
NAME="v9-export-$HOST-$STAMP"
OUTDIR=${V9_EXPORT_OUT:-$WORK/v9-export}
mkdir -p "$OUTDIR" 2>/dev/null || OUTDIR=$HOME/v9-export
mkdir -p "$OUTDIR/$NAME/logs" || { echo "[v9-export] cannot create $OUTDIR/$NAME" >&2; exit 2; }
D="$OUTDIR/$NAME"
FAILED=
[ -d "$V9" ] || { echo "[v9-export] no V9 results at $V9" >&2; exit 2; }
{
    echo "host=$HOST at=$(date '+%Y-%m-%d %H:%M:%S %Z') v9_root=$V9"
    echo "code=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$D/steps.txt"

echo "[v9-export] collect ..."
sh "$HERE/run.sh" collect > "$D/logs/collect.log" 2>&1
rc=$?; echo "collect exit=$rc" >> "$D/steps.txt"
if [ "$rc" -ne 0 ]; then FAILED="$FAILED collect"; echo "[v9-export] collect FAILED (exit $rc)"; tail -5 "$D/logs/collect.log"; fi

echo "[v9-export] status ..."
sh "$HERE/run.sh" status > "$D/status.txt" 2>&1
rc=$?; echo "status exit=$rc" >> "$D/steps.txt"
if [ "$rc" -ne 0 ]; then FAILED="$FAILED status"; fi

echo "[v9-export] copy results ..."
LIST="$D/logs/included.txt"
SKIP='.*\.\(pt\|pth\|ckpt\|bin\|safetensors\|lock\)$'
(cd "$V9" && find . -type f ! -regex "$SKIP" -size -52428801c -print) | sort > "$LIST"
(cd "$V9" && find . -type f \( -regex "$SKIP" -o ! -size -52428801c \) -exec ls -l {} + 2>/dev/null \
    | awk '{print $5, $NF}') | sort -k2 > "$D/excluded.txt"
mkdir -p "$D/v9"
if (cd "$V9" && tar -cf - -T "$LIST") | (cd "$D/v9" && tar -xf -); then
    echo "copy exit=0 files=$(wc -l < "$LIST") excluded=$(wc -l < "$D/excluded.txt")" >> "$D/steps.txt"
else
    echo "copy exit=1" >> "$D/steps.txt"; FAILED="$FAILED copy"
fi

FINAL="$OUTDIR/$NAME.tar.gz"
if ! tar -czf "$FINAL" -C "$OUTDIR" "$NAME"; then
    echo "[v9-export] packing failed: $FINAL" >&2; exit 2
fi
(cd "$OUTDIR" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo
cat "$D/steps.txt"
rm -rf "$D"
cat "$OUTDIR/$NAME.tar.gz.sha256"
echo "[v9-export] file: $FINAL ($(du -h "$FINAL" | cut -f1))"
if [ -n "$FAILED" ]; then echo "[v9-export] failed steps:$FAILED (the file still holds everything else)"; exit 1; fi
echo "[v9-export] all steps ok"
