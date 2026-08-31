#!/usr/bin/env bash
# Canonical end-to-end entrypoint for one cluster slot:
# corrected v4 -> 7B regime discovery -> one CPU-only final collection.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/setup_env.sh

slot=${1:-}
case "$slot" in
  1|2|3) ;;
  *) echo "usage: bash scripts/go_v4.sh <cluster: 1|2|3>" >&2; exit 2 ;;
esac
command -v flock >/dev/null 2>&1 || {
  echo "[abort] flock command missing" >&2
  exit 1
}

mkdir -p "$OM_WORK/locks" || exit 1
node_tag=$(hostname 2>/dev/null || printf node)
node_tag=$(printf '%s' "$node_tag" | tr -cs 'a-zA-Z0-9._-' '-')

# One 4-GPU workflow per node and one owner per global slot. The three distinct
# slots remain parallel across clusters.
exec 8>"$OM_WORK/locks/offpolicy-node-$node_tag.lock"
flock -n 8 || {
  echo "[abort] an offpolicy workflow is already running on this node" >&2
  exit 1
}
exec 9>"$OM_WORK/locks/offpolicy-slot-$slot.lock"
flock -n 9 || {
  echo "[abort] offpolicy slot $slot is already running on another node" >&2
  exit 1
}

echo "== [1/3] corrected v4: slot $slot (완료 run 자동 생략)"
OM_OFFPOLICY_WRAPPED=1 bash scripts/go_v4.sh "$slot"

echo "== [2/3] Qwen2.5-7B regime discovery (완료 point 자동 생략)"
bash scripts/go_additional.sh

echo "== [3/3] model reports + harvest (동일 입력 재계산 생략)"
bash scripts/collect_v4.sh

echo "== offpolicy workflow complete: slot $slot"
