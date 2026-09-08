#!/usr/bin/env bash
# Put the Qwen3.5-9B matrix root aside so the next launch starts a fresh matrix.
#
#   bash scripts/reset_qwen35_root.sh
#
# When to use: `run_qwen35_9b.sh` stops with "[regime-contract-abort] matrix
# contract mismatch". That means the root still holds a contract from an earlier
# launch made under a different config, model snapshot or code, and the launcher
# refuses to mix two matrices in one root. Nothing is deleted: the run root, its
# results directory and its contract files are renamed with a timestamp, and the
# relaunch line is printed. Refuses to touch a root that holds a finished point
# or one a worker claimed in the last 30 minutes.
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
[ -n "${OM_WORK:-}" ] || { echo "[abort] OM_WORK is not set (scripts/setup_env.sh)"; exit 1; }
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3

RUN_ID=qwen35-9b-posttrained-math-code-grpo-v1           # MATRIX_IDS for profile qwen35
CONFIG=configs/qwen35_9b_grpo.json
MODEL_KEY=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["models"][0]["key"])' "$CONFIG") \
  || { echo "[abort] cannot read the model key from $CONFIG"; exit 1; }
ROOT="$OM_WORK/runs/$RUN_ID/$MODEL_KEY"
RESULTS="$OM_WORK/results/$RUN_ID/$MODEL_KEY"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

if [ ! -d "$ROOT" ] && ! compgen -G "$OM_WORK/contracts/$RUN_ID-$MODEL_KEY-*.json" >/dev/null; then
  echo "[reset] nothing to reset: no root at $ROOT and no contract for $RUN_ID"; exit 0
fi

# Never move finished work aside by accident.
if [ "${OM_FORCE:-0}" != 1 ] && compgen -G "$ROOT/*/DONE" >/dev/null; then
  echo "[abort] $ROOT holds finished points:"; ls -d "$ROOT"/*/DONE | sed 's/^/   /'
  echo "        this root is a real matrix; not touching it (OM_FORCE=1 overrides)"; exit 1
fi
for owner in "$ROOT"/.families/*.owner.json; do
  [ -f "$owner" ] || continue
  if [ "$(( $(date +%s) - $(stat -c %Y "$owner") ))" -lt 1800 ]; then
    echo "[abort] a worker claimed $(basename "$owner" .owner.json) $(( ( $(date +%s) - $(stat -c %Y "$owner") ) / 60 ))m ago; stop it first"; exit 1
  fi
done

moved=0
if [ -d "$ROOT" ]; then
  echo "[reset] root: $(ls "$ROOT" 2>/dev/null | wc -l) entries, generation=$(cat "$ROOT/.queue/generation.git" 2>/dev/null | cut -c1-12 || echo none), points with run_config=$(ls "$ROOT"/*/run_config.json 2>/dev/null | wc -l), DONE=$(ls "$ROOT"/*/DONE 2>/dev/null | wc -l)"
  mv -- "$ROOT" "$ROOT.stale-$STAMP" && { echo "[reset] moved $ROOT -> $ROOT.stale-$STAMP"; moved=$((moved + 1)); }
fi
if [ -d "$RESULTS" ]; then
  mv -- "$RESULTS" "$RESULTS.stale-$STAMP" && { echo "[reset] moved $RESULTS -> $RESULTS.stale-$STAMP"; moved=$((moved + 1)); }
fi
for contract in "$OM_WORK/contracts/$RUN_ID-$MODEL_KEY-"*.json "$OM_WORK/contracts/$RUN_ID-$MODEL_KEY-"*.json.lock; do
  [ -e "$contract" ] || continue
  mv -- "$contract" "$contract.stale-$STAMP" && { echo "[reset] moved $(basename "$contract")"; moved=$((moved + 1)); }
done
echo "[reset] $moved item(s) put aside; nothing deleted. Next launch starts a fresh matrix:"
echo "        bash scripts/run_qwen35_9b.sh"
