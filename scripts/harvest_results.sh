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
target="$READOUTS/rlvr-grpo"
if [ -s "$target/RESULTS.json" ] && [ -s "$target/MANIFEST.sha256" ] \
    && (cd "$target" && sha256sum -c MANIFEST.sha256 >/dev/null 2>&1); then
  old_key=$("$PY" - "$target/RESULTS.json" <<'PYEOF'
import json
import sys

try:
    print(json.load(open(sys.argv[1], encoding="utf-8"))["input_digest"])
except (OSError, KeyError, TypeError, ValueError):
    raise SystemExit(1)
PYEOF
  ) || old_key=""
  if [ "$old_key" = "$KEY" ]; then
    echo "[harvest] inputs unchanged; reuse $target"
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

previous="$READOUTS/.rlvr-grpo.previous.$$"
trap 'rm -rf "$temporary"; [ ! -e "$previous" ] || { rm -rf "$target"; mv "$previous" "$target"; }' ERR
[ ! -e "$target" ] || mv "$target" "$previous"
mv "$temporary" "$target"
rm -rf "$previous"
trap - ERR

# Remove bundles and the pointer created by the superseded timestamped layout.
find "$READOUTS" -mindepth 1 -maxdepth 1 -type d \
  -name 'rlvr-grpo-[0-9]*' -exec rm -rf -- {} +
rm -f "$OM_WORK/results/.rlvr-harvest-current"
echo "[harvest] published $target"
