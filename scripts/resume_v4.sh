#!/usr/bin/env bash
# Resume a v4 worker from the generation commit recorded in existing run configs.
# Analysis-only git pulls therefore cannot quarantine or restart multi-day artifacts.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh
PY="$VENV_DIR/bin/python"; [ -x "$PY" ] || PY=python3

slot=${1:-}
case "$slot" in 1|2|3) ;; *) echo "usage: bash scripts/resume_v4.sh <1|2|3>" >&2; exit 2;; esac

current=$(git rev-parse HEAD) || exit 1
target=$("$PY" src/v4_resume_commit.py "$OM_WORK/runs" "$current") || exit 1
git cat-file -e "$target^{commit}" 2>/dev/null || {
  echo "[resume-v4-abort] recorded commit is unavailable locally: $target" >&2
  exit 1
}

if [ "$target" = "$current" ]; then
  echo "== v4 resume: 현재 commit ${current:0:12} 사용"
  exec env -u OM_REPO -u PYTHONPATH OM_V4_RESUME_WRAPPED=1 \
    bash scripts/go_v4.sh "$slot"
fi

snapshot_root="$OM_WORK/code-snapshots"
snapshot="$snapshot_root/offpolicy-misranking-${target:0:12}"
mkdir -p "$snapshot_root" || exit 1
exec 8>"$snapshot_root/.v4-resume.lock"
flock 8 || exit 1
if [ -e "$snapshot/.git" ]; then
  snapshot_git=$(git -C "$snapshot" rev-parse HEAD 2>/dev/null || true)
  [ "$snapshot_git" = "$target" ] || {
    echo "[resume-v4-abort] snapshot commit mismatch: $snapshot" >&2
    exit 1
  }
elif [ -e "$snapshot" ]; then
  echo "[resume-v4-abort] snapshot path exists but is not a git worktree: $snapshot" >&2
  exit 1
else
  git worktree add --detach "$snapshot" "$target" || exit 1
fi
flock -u 8

echo "== v4 resume: run_config commit ${target:0:12}의 격리 worktree 사용"
echo "   완료 run은 스킵하고 미완료 shard/.partial부터 재개"
cd "$snapshot" || exit 1
# setup_env.sh must resolve OM_REPO and PYTHONPATH from the snapshot. Keeping
# values exported by the caller would mix the current src/ with the old Git tree.
exec env -u OM_REPO -u PYTHONPATH OM_V4_RESUME_WRAPPED=1 \
  bash scripts/go_v4.sh "$slot"
