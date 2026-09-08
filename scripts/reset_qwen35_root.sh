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
# relaunch line is printed. By default even unfinished point directories and
# partial checkpoints are preserved. Live locks always prevent a reset.
set -euo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
[ -n "${OM_WORK:-}" ] || { echo "[abort] OM_WORK is not set (scripts/setup_env.sh)"; exit 1; }
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3

RUN_ID=qwen35-9b-posttrained-math-code-grpo-v1           # MATRIX_IDS for profile qwen35
CONFIG=configs/qwen35_9b_grpo.json
MODEL_KEY=$("$PY" src/model_matrix.py --config "$CONFIG" list-models) \
  || { echo "[abort] cannot read the model key from $CONFIG"; exit 1; }
[[ "$MODEL_KEY" =~ ^[A-Za-z0-9._-]+$ ]] || { echo '[abort] exactly one model key is required'; exit 1; }
ROOT="$OM_WORK/runs/$RUN_ID/$MODEL_KEY"
RESULTS="$OM_WORK/results/$RUN_ID/$MODEL_KEY"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOCAL_LOCK_DIR=${OM_LOCAL_LOCK_DIR:-/tmp/offpolicy-misranking-$(id -u)}
mkdir -p "$LOCAL_LOCK_DIR" "$OM_WORK/locks"
exec 8>>"$LOCAL_LOCK_DIR/primary.lock"
flock -n 8 || { echo '[abort] this node has a live primary/additional launcher; stop only that launcher before resetting'; exit 1; }
exec 7>>"$OM_WORK/locks/$RUN_ID-$MODEL_KEY.lifecycle.lock"
flock -n 7 || { echo '[abort] a worker on another node is using this matrix (lifecycle lock)'; exit 1; }
[ ! -L "$ROOT" ] && [ ! -L "$RESULTS" ] || { echo '[abort] refusing a symlinked run/results root'; exit 1; }

# Older run_matrix workers use .queue locks, not .families/*.owner.json.
# Keep their actual lock descriptors held until every move has finished.
for lock in "$ROOT"/.queue/*.lock "$OM_WORK/contracts/$RUN_ID-$MODEL_KEY-"*.json.lock; do
  [ -e "$lock" ] || continue
  exec {queue_fd}>>"$lock"
  flock -n "$queue_fd" || { echo "[abort] live matrix lock: $lock"; exit 1; }
done

if [ ! -d "$ROOT" ] && ! compgen -G "$OM_WORK/contracts/$RUN_ID-$MODEL_KEY-*.json" >/dev/null; then
  echo "[reset] nothing to reset: no root at $ROOT and no contract for $RUN_ID"; exit 0
fi

if [ "${OM_FORCE:-0}" != 1 ]; then
  if [ -d "$ROOT" ]; then
    payload=$(find "$ROOT" -mindepth 1 -maxdepth 1 ! -name .queue ! -name .progress ! -name logs -print -quit)
    [ -z "$payload" ] || {
      echo "[abort] preserving existing point/artifact: $payload"
      echo '[abort] unfinished checkpoints count as work too; inspect the contract difference before considering a new root'
      exit 1
    }
  fi
  if [ -d "$RESULTS" ] && [ -n "$(find "$RESULTS" -mindepth 1 -print -quit)" ]; then
    echo "[abort] preserving existing results: $RESULTS"; exit 1
  fi
fi

moved=0
mkdir -p "$OM_WORK/quarantine"
BACKUP=$(mktemp -d "$OM_WORK/quarantine/$RUN_ID-reset-$STAMP-XXXXXX")
mkdir "$BACKUP/contracts"
echo "[reset] archive=$BACKUP"
if [ -d "$ROOT" ]; then
  mv -- "$ROOT" "$BACKUP/run"
  echo "[reset] moved $ROOT -> $BACKUP/run"; moved=$((moved + 1))
fi
if [ -d "$RESULTS" ]; then
  mv -- "$RESULTS" "$BACKUP/results"
  echo "[reset] moved $RESULTS -> $BACKUP/results"; moved=$((moved + 1))
fi
for contract in "$OM_WORK/contracts/$RUN_ID-$MODEL_KEY-"*.json; do
  [ -e "$contract" ] || continue
  mv -- "$contract" "$BACKUP/contracts/$(basename "$contract")"
  echo "[reset] moved $(basename "$contract")"; moved=$((moved + 1))
done
echo "[reset] $moved item(s) put aside; nothing deleted. Next launch starts a fresh matrix:"
echo "        bash scripts/run_qwen35_9b.sh"
