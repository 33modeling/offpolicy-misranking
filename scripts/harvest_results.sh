#!/usr/bin/env bash
# Package validated 7B and 27B matrix reports without recomputing experiments.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"
[ -x "$PY" ] || { echo "[harvest-abort] venv missing: $PY"; exit 1; }

RESULTS_7B="${RLVR_RESULTS_7B:-$OM_WORK/results/regime-qwen2.5-7b-grpo-v1}"
RESULTS_27B="${RLVR_RESULTS_27B:-$OM_WORK/results/regime-qwen3.8-27b-grpo-v1}"
READOUT_ID="${RLVR_READOUT_ID:-rlvr-grpo}"
[[ "$READOUT_ID" =~ ^[a-zA-Z0-9._-]+$ ]] || {
  echo "[harvest-abort] RLVR_READOUT_ID must be one safe path component"
  exit 1
}
READOUTS="$OM_WORK/readouts"
mkdir -p "$READOUTS" "$OM_WORK/locks" "$OM_WORK/results"
command -v flock >/dev/null 2>&1 || { echo "[harvest-abort] flock missing"; exit 1; }
exec 8>"$OM_WORK/locks/rlvr-harvest.lock"
flock 8

target="$READOUTS/$READOUT_ID"
temporary=$(mktemp -d "$READOUTS/.$READOUT_ID.XXXXXX")
previous=""
published=0
publish_cleanup() {
  rc=$?
  trap - EXIT HUP INT TERM
  rm -rf "$temporary"
  if [ -n "$previous" ] && { [ -e "$previous" ] || [ -L "$previous" ]; }; then
    if [ "$published" -eq 1 ] && { [ -e "$target" ] || [ -L "$target" ]; }; then
      rm -rf "$previous"
    elif [ ! -e "$target" ] && [ ! -L "$target" ]; then
      mv -T -- "$previous" "$target"
    else
      echo "[harvest-abort] prior bundle retained at $previous" >&2
    fi
  fi
  exit "$rc"
}
trap publish_cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
GIT_HEAD=$(git rev-parse HEAD)
"$PY" src/harvest_results.py \
  --primary "$RESULTS_27B" \
  --replication "$RESULTS_7B" \
  --output "$temporary" \
  --git "$GIT_HEAD" \
  --code scripts/harvest_results.sh \
  --code src/harvest_results.py
(
  cd "$temporary"
  sha256sum REPORT.md RESULTS.json RESULTS.csv > MANIFEST.sha256
  sha256sum -c MANIFEST.sha256 >/dev/null
)

bundle_current() {
  local expected actual name manifest_names
  [ -d "$target" ] && [ ! -L "$target" ] || return 1
  expected=$(printf '%s\n' MANIFEST.sha256 REPORT.md RESULTS.csv RESULTS.json)
  actual=$(find "$target" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)
  [ "$actual" = "$expected" ] || return 1
  for name in MANIFEST.sha256 REPORT.md RESULTS.csv RESULTS.json; do
    [ -f "$target/$name" ] && [ ! -L "$target/$name" ] && [ -s "$target/$name" ] \
      || return 1
  done
  manifest_names=$(awk 'NF == 2 {print $2}' "$target/MANIFEST.sha256" | sort)
  [ "$manifest_names" = "$(printf '%s\n' REPORT.md RESULTS.csv RESULTS.json)" ] \
    || return 1
  (cd "$target" && sha256sum -c MANIFEST.sha256 >/dev/null 2>&1) || return 1
  cmp -s "$target/MANIFEST.sha256" "$temporary/MANIFEST.sha256"
}

cleanup_legacy() {
  [ "$READOUT_ID" = "rlvr-grpo" ] || return 0
  find "$READOUTS" -mindepth 1 -maxdepth 1 -type d \
    -name 'rlvr-grpo-[0-9]*' -exec rm -rf -- {} +
  rm -f "$OM_WORK/results/.rlvr-harvest-current"
}

if bundle_current; then
  cleanup_legacy
  echo "[harvest] inputs unchanged; reuse $target"
  exit 0
fi

previous=$(mktemp -d "$READOUTS/.$READOUT_ID.previous.XXXXXX")
rmdir "$previous"
[ ! -e "$target" ] && [ ! -L "$target" ] || mv -T -- "$target" "$previous"
if ! mv -T -- "$temporary" "$target"; then
  exit 1
fi
published=1
rm -rf "$previous"
previous=""
trap - EXIT HUP INT TERM

# Remove bundles and the pointer created by the superseded timestamped layout.
cleanup_legacy
echo "[harvest] published $target"
