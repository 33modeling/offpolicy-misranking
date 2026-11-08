#!/bin/sh
# Pack every v8 result needed for analysis into ONE file (no checkpoints, no rollout caches):
#   sh v8/experiments/harvest.sh            -> $V8_ROOT/harvest/v8-harvest-<host>-<stamp>.tar.gz
# Included: locator, matrices, manifests, ledgers (events), decisions/signals/training stats,
# snapshot indexes (hashes only), evaluation records, frozen artifacts, audit, analysis, logs.
set -eu
WORK=${OM_WORK:-${GROUP_VOLUME:-/group-volume}/${OM_USER:-minsoo3.kim}/offpolicy-misranking}
V8_ROOT=${V8_ROOT:-$WORK/v8}
STAMP=$(date +%Y%m%dT%H%M%S)
OUT="$V8_ROOT/harvest/v8-harvest-$(hostname)-$STAMP.tar.gz"
mkdir -p "$V8_ROOT/harvest"
cd "$V8_ROOT"
LIST=$(mktemp "${TMPDIR:-$V8_ROOT}/v8-harvest-list.XXXXXX")
find . -path ./harvest -prune -o -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' \
     -o -name '*.png' -o -name '*.log' \) ! -path '*/caches/committed/*/rewards.json' -print > "$LIST"
# Workers keep appending logs and replacing JSON while this runs. GNU tar exits 1 for "file changed
# as we read it" (the archive is still complete); only exit >= 2 is a real failure.
set +e
tar --ignore-failed-read --warning=no-file-changed --warning=no-file-removed -czf "$OUT" -T "$LIST"
rc=$?
set -e
rm -f "$LIST"
if [ "$rc" -gt 1 ]; then echo "[v8] harvest failed (tar exit $rc): $OUT" >&2; exit "$rc"; fi
sha256sum "$OUT" > "$OUT.sha256"
echo "$OUT"
cat "$OUT.sha256"
