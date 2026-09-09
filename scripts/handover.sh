#!/usr/bin/env bash
# Hand plain-text files over through the transfer repository without typing a
# path: copy them into <transfer clone>/offpolicy-misranking/, commit, push.
#   bash scripts/handover.sh <file>...
# Called by scripts/digest_family.sh for its output. Read-only for the
# experiment; touches only the transfer clone.
#
# The transfer clone is found, in order: $OM_TRANSFER_DIR, $HOME/transfer,
# $HOME/dev/transfer, $OM_WORK/../transfer, $GROUP_VOLUME/$OM_USER/transfer,
# then any git checkout up to three levels under $HOME. A candidate counts only
# if its origin URL names a "transfer" repository. Nothing is cloned here.
set -uo pipefail
cd "$(dirname "$0")/.."
export OM_ONLINE=0
source scripts/setup_env.sh >/dev/null 2>&1
[ "$#" -ge 1 ] || { echo "usage: bash scripts/handover.sh <file>..."; exit 2; }

is_transfer_clone() { [ -d "$1/.git" ] && git -C "$1" remote get-url origin 2>/dev/null | grep -qi "transfer"; }
find_transfer() {
  local cand
  for cand in "${OM_TRANSFER_DIR:-}" "$HOME/transfer" "$HOME/dev/transfer" "${OM_WORK:-/nonexistent}/../transfer" \
              "${GROUP_VOLUME:-/nonexistent}/${OM_USER:-nobody}/transfer"; do
    [ -n "$cand" ] && is_transfer_clone "$cand" && { echo "$cand"; return 0; }
  done
  while IFS= read -r cand; do
    cand=${cand%/.git}
    is_transfer_clone "$cand" && { echo "$cand"; return 0; }
  done < <(find "$HOME" -mindepth 2 -maxdepth 4 -type d -name .git 2>/dev/null)
  return 1
}

repo=$(find_transfer) || {
  echo "[handover] no transfer clone found (looked in \$OM_TRANSFER_DIR, \$HOME/transfer, \$HOME/dev/transfer, \$OM_WORK/../transfer, \$GROUP_VOLUME/\$OM_USER/transfer, and under \$HOME)."
  echo "[handover] copy by hand into the transfer repository and push:"
  printf '           %s\n' "$@"
  exit 1
}
dest="$repo/offpolicy-misranking"; mkdir -p "$dest" || { echo "[handover] cannot create $dest"; exit 1; }
names=()
for f in "$@"; do
  [ -s "$f" ] || { echo "[handover] skip (missing or empty): $f"; continue; }
  cp -f -- "$f" "$dest/" && names+=("offpolicy-misranking/$(basename "$f")")
done
[ "${#names[@]}" -gt 0 ] || { echo "[handover] nothing to hand over"; exit 1; }

# Bring the clone up to date first so the push is a fast-forward; a failed pull
# (no network, conflicting local edits) is reported and the push still tried.
if ! out=$(git -C "$repo" pull --rebase --autostash 2>&1); then
  echo "[handover] pull failed (continuing): $(printf '%s\n' "$out" | tail -1)"
fi
git -C "$repo" add -- "${names[@]}"
if git -C "$repo" -c user.name=33modeling -c user.email=33modeling@gmail.com commit -q -m "update" -- "${names[@]}"; then
  if out=$(git -C "$repo" push 2>&1); then
    echo "[handover] pushed to $(git -C "$repo" remote get-url origin 2>/dev/null): ${names[*]#offpolicy-misranking/}"
  else
    printf '%s\n' "$out" | tail -3
    echo "[handover] committed in $repo but the push failed; run 'git push' there when the network is back"
    exit 1
  fi
else
  echo "[handover] nothing new to commit in $repo (already pushed?)"
fi
