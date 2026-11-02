#!/usr/bin/env bash
# Rebuild the memory-only hotfix on top of whatever commit the running matrix is
# pinned to, in case it is not the commit the current hotfix was built on.
#   bash scripts/build_mbpp_hotfix.sh              # show what it would do
#   bash scripts/build_mbpp_hotfix.sh go           # build and push the branch
# The new commit changes src/grads.py only: chunked LM-head log-probs and an
# in-process OOM back-off. No reward, verifier, sampling or contract change.
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
V2_TAG=${OM_OLMO3_V2_TAG:-olmo3-1025-7b-base-rlzero-grpo-h100-v2}
BRANCH=${OM_MBPP_HOTFIX_BRANCH:-hotfix/valgrads-memory}
BASE=${1:-}
case "$BASE" in go|"") BASE=$(tr -d '[:space:]' < "$OM_WORK/runs/$V2_TAG/.queue/generation.git" 2>/dev/null) ;; esac
[ -n "$BASE" ] || { echo "[abort] cannot read the pinned commit; pass it as the first argument"; exit 1; }
git rev-parse -q --verify "$BASE^{commit}" >/dev/null || { echo "[abort] unknown commit: $BASE"; exit 1; }
echo "base (v2 pin) : $BASE"
echo "branch        : $BRANCH"
EXISTING=$(git rev-parse -q --verify "$BRANCH^{commit}" 2>/dev/null || true)
if [ -n "$EXISTING" ] && [ "$(git rev-parse "$EXISTING^")" = "$(git rev-parse "$BASE")" ]; then
  echo "already built on this base: $EXISTING"; exit 0
fi
if [ "${1:-}" != go ] && [ "${2:-}" != go ]; then
  echo; echo "dry run. to build and push:  bash scripts/build_mbpp_hotfix.sh go"; exit 0
fi
tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT
git show "$BASE:src/grads.py" > "$tmp/grads.py" || { echo "[abort] base has no src/grads.py"; exit 1; }
python3 scripts/apply_valgrads_memory_patch.py "$tmp/grads.py" || exit 1
blob=$(git hash-object -w "$tmp/grads.py") || exit 1
export GIT_INDEX_FILE="$tmp/index"
git read-tree "$BASE" && git update-index --cacheinfo 100644 "$blob" src/grads.py || exit 1
tree=$(git write-tree) || exit 1
commit=$(git commit-tree "$tree" -p "$BASE" -m "Generation hotfix on $BASE: memory-only fix for the validation-gradient OOM

Chunked LM-head log-probs (512 positions, activation checkpointing) and an
in-process OOM back-off in prompt_gradient. Nothing else changes: no reward,
no verifier, no sampling, no contract field.") || exit 1
unset GIT_INDEX_FILE
git branch -f "$BRANCH" "$commit" && git push -q -f origin "$BRANCH" \
  && echo "built and pushed: $commit ($BRANCH)"
