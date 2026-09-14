# shellcheck shell=bash
# pin_checkout <commit> : print the path of a clean node-local checkout of
# <commit>, cloned from the current repository (the same mechanism as the
# matrix supervisor's materialize_local_checkout). run_point.sh records the
# commit that initialized a point in run_config.json and refuses every later
# stage from a different revision ([code-abort]); a point that is resumed after
# `git pull` here must therefore re-enter its own commit, not this checkout.
# Cache: $OM_PIPELINE_CACHE (default /tmp/offpolicy-misranking-<uid>/pipelines).
pin_checkout() {
  local commit=$1 cache="${OM_PIPELINE_CACHE:-/tmp/offpolicy-misranking-$(id -u)/pipelines}"
  local target temporary recorded dirty stale
  target="$cache/clones/$commit"
  mkdir -p "$cache/clones" || return 1
  (
    flock 9
    recorded=$(git -C "$target" rev-parse HEAD 2>/dev/null || true)
    dirty=invalid
    if [ -d "$target/.git" ] && [ "$recorded" = "$commit" ]; then
      dirty=$(git -C "$target" status --porcelain -- src scripts configs requirements.txt 2>/dev/null || printf invalid)
    fi
    if [ ! -d "$target/.git" ] || [ "$recorded" != "$commit" ] || [ -n "$dirty" ]; then
      if [ -e "$target" ] || [ -L "$target" ]; then
        stale="$cache/.stale-$commit-$(date +%s)-$$"
        mv -- "$target" "$stale" || exit 1
        echo "[checkout] invalid node-local cache quarantined: $stale" >&2
      fi
      git cat-file -e "$commit^{commit}" 2>/dev/null \
        || { echo "[abort] pinned commit is not in this repository (git fetch first): $commit" >&2; exit 1; }
      temporary="$cache/.clone-$commit-$$"
      rm -rf -- "$temporary"
      git clone --quiet --no-hardlinks --no-checkout "$PWD" "$temporary" >&2 || exit 1
      git -C "$temporary" checkout --quiet --detach "$commit" >&2 || exit 1
      git -C "$temporary" remote remove origin >/dev/null 2>&1 || true
      mv -- "$temporary" "$target" || exit 1
    fi
    [ "$(git -C "$target" rev-parse HEAD 2>/dev/null)" = "$commit" ] \
      || { echo "[abort] pinned checkout HEAD mismatch: $target" >&2; exit 1; }
    [ -z "$(git -C "$target" status --porcelain -- src scripts configs requirements.txt)" ] \
      || { echo "[abort] pinned checkout is dirty: $target" >&2; exit 1; }
  ) 9>"$cache/.clone.lock" || return 1
  printf '%s\n' "$target"
}
