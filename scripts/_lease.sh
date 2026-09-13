# shellcheck shell=bash
# Lease bookkeeping shared by the queue scripts. Open a lease file with >>
# (append: opening must not truncate another node's note), take `flock -n`,
# and on success call `lease_note FILE` so that `run_queue.sh status` can
# say which node holds the lease and since when. Stale notes are harmless:
# the status shows a note only while the flock is actually held.
lease_note() {
  printf 'host=%s pid=%s since=%s\n' "$(hostname)" "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$1" 2>/dev/null || true
}
