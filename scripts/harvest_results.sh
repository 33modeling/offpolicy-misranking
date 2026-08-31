#!/usr/bin/env bash
# Package validated 7B and 27B matrix reports without recomputing experiments.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[harvest-abort] venv missing: $PY"; exit 1; }

RESULTS_7B="$OM_WORK/results/regime-qwen2.5-7b-grpo-v1"
RESULTS_27B="$OM_WORK/results/regime-qwen3.8-27b-grpo-v1"
READOUTS="$OM_WORK/readouts"
mkdir -p "$READOUTS" "$OM_WORK/locks" "$OM_WORK/results"
command -v flock >/dev/null 2>&1 || { echo "[harvest-abort] flock missing"; exit 1; }
exec 8>"$OM_WORK/locks/rlvr-harvest.lock"
flock 8

inputs=()
for root in "$RESULTS_7B" "$RESULTS_27B"; do
  for name in REGIME.json REGIME.csv REGIME_SUMMARY.csv FINAL_REPORT.md .regime_analysis.key; do
    path="$root/$name"
    [ -s "$path" ] || { echo "[harvest-abort] missing or empty: $path"; exit 1; }
    inputs+=("$path")
  done
done

KEY=$(sha256sum "${inputs[@]}" | sha256sum | cut -d' ' -f1)
CURRENT="$OM_WORK/results/.rlvr-harvest-current"
if [ -s "$CURRENT" ]; then
  old_key=$(sed -n '1p' "$CURRENT")
  old_dir=$(sed -n '2p' "$CURRENT")
  if [ "$old_key" = "$KEY" ] && [[ "$old_dir" == "$READOUTS/"* ]] \
      && [ -s "$old_dir/MANIFEST.sha256" ] \
      && (cd "$old_dir" && sha256sum -c MANIFEST.sha256 >/dev/null 2>&1); then
    echo "[harvest] inputs unchanged; reuse $old_dir"
    exit 0
  fi
fi

temporary=$(mktemp -d "$READOUTS/.rlvr-grpo.XXXXXX")
GIT_HEAD=$(git rev-parse HEAD)
"$PY" - "$RESULTS_27B" "$RESULTS_7B" "$temporary" "$GIT_HEAD" "$KEY" <<'PYEOF'
import csv
import json
import sys
from pathlib import Path

primary, replication, output = map(Path, sys.argv[1:4])
git, digest = sys.argv[4:6]
(output / "REPORT.md").write_text(
    "# RLVR Experiment Results\n\n"
    "## Primary: Qwen3.8-27B\n\n"
    + (primary / "FINAL_REPORT.md").read_text().strip()
    + "\n\n## Scale replication: Qwen2.5-7B\n\n"
    + (replication / "FINAL_REPORT.md").read_text().strip()
    + "\n"
)
(output / "RESULTS.json").write_text(json.dumps({
    "schema": "offpolicy-rlvr-harvest/v1",
    "git": git,
    "input_digest": digest,
    "primary_27b": json.loads((primary / "REGIME.json").read_text()),
    "replication_7b": json.loads((replication / "REGIME.json").read_text()),
}, indent=2, sort_keys=True) + "\n")

rows = []
fieldnames = ["experiment"]
for label, root in (("primary_27b", primary), ("replication_7b", replication)):
    with (root / "REGIME.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append({"experiment": label, **row})
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
with (output / "RESULTS.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
PYEOF
(
  cd "$temporary"
  sha256sum REPORT.md RESULTS.json RESULTS.csv > MANIFEST.sha256
  sha256sum -c MANIFEST.sha256 >/dev/null
)

target="$READOUTS/rlvr-grpo-$(date +%Y%m%d-%H%M%S)-${GIT_HEAD:0:8}"
mv "$temporary" "$target"
pointer="$CURRENT.tmp"
printf '%s\n%s\n' "$KEY" "$target" > "$pointer"
mv "$pointer" "$CURRENT"
echo "[harvest] published $target"
