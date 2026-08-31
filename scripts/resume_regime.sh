#!/usr/bin/env bash
# Resume the canonical 7B regime matrix at the generation commit recorded by
# its existing run configs while retaining the current supervisor/collector.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh

PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3
CURRENT_REPO=$PWD
current=$(git rev-parse HEAD) || exit 1
regime_root="$OM_WORK/runs/regime-qwen2.5-7b-instruct"
target=$("$PY" src/regime_resume_commit.py "$regime_root" "$current") || exit 1

if [ "$target" = "$current" ]; then
  snapshot="$CURRENT_REPO"
else
  snapshot_root="${OM_LOCAL_SNAPSHOT_ROOT:-/tmp/${USER:-user}-offpolicy-code-snapshots}"
  mkdir -p "$snapshot_root" || exit 1
  exec 8>"$snapshot_root/.regime-resume.lock"
  flock 8 || exit 1
  if ! git cat-file -e "$target^{commit}" 2>/dev/null; then
    echo "== 기록된 regime commit 자동 fetch: ${target:0:12}"
    git fetch --no-tags origin "$target" || exit 1
  fi
  snapshot="$snapshot_root/offpolicy-misranking-${target:0:12}"
  if [ -e "$snapshot/.git" ]; then
    snapshot_git=$(git -C "$snapshot" rev-parse HEAD 2>/dev/null || true)
    [ "$snapshot_git" = "$target" ] || {
      echo "[resume-regime-abort] snapshot commit mismatch: $snapshot" >&2
      exit 1
    }
  elif [ -e "$snapshot" ]; then
    echo "[resume-regime-abort] snapshot path is not a worktree: $snapshot" >&2
    exit 1
  else
    git worktree add --detach "$snapshot" "$target" || exit 1
  fi
  flock -u 8
fi

echo "== regime generation commit: ${target:0:12}"
OM_REGIME_RESUME_WRAPPED=1 \
OM_PIPELINE_REPO="$snapshot" \
OM_PIPELINE_SCRIPT="$snapshot/scripts/run_14b.sh" \
  exec bash "$CURRENT_REPO/scripts/go_additional.sh"
