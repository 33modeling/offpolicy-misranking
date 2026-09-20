#!/usr/bin/env bash
# Operational ownership helpers, separate from E5's scientific code.

e5_cleanup_previous() {
  local root=$1
  echo "[cleanup] stopping previous E5 processes on $(hostname): $root"
  # The shell's late OUT_ROOT export is visible in its children, not necessarily
  # in its own /proc/environ. Include the old seed loop, not just one seed child.
  "$PY" src/cleanup_run_processes.py --run-prefix "$root" \
    --command-pattern "$root" --command-pattern 'scripts/run_e5.sh ' \
    --require-environment "OUT_ROOT=$root" --launcher-environment-from-child \
    --timeout 15 --compact || return 1
  "$PY" src/cleanup_run_processes.py --run-prefix "$root" \
    --command-pattern "$root" --timeout 15 --compact || return 1
}

e5_cleanup_lock_helpers() {
  "$PY" src/cleanup_run_processes.py --run-prefix "$1.unused-scope" \
    --open-file "$1" --orphan-lock-helpers-only --timeout 2 --compact
}

e5_show_lock_owners() {
  "$PY" src/cleanup_run_processes.py --run-prefix "$1.unused-scope" \
    --open-file "$1" --describe-lock-owners
}

e5_physical_node_id() {
  local boot
  read -r boot < /proc/sys/kernel/random/boot_id || return 1
  [[ "$boot" =~ ^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$ ]] || return 1
  printf '%s' "$boot"
}

e5_acquire_shared_pair_node() {
  local directory=$1 host physical holders
  physical=$(e5_physical_node_id) || {
    echo '[abort] physical node identity unavailable; refusing hostname-only admission'
    return 75
  }
  host=$(hostname) || return 75
  host=$(printf '%s' "$host" | tr -c 'a-zA-Z0-9._-' '_')
  # A visible legacy owner is local regardless of its hostname or GPU label.
  holders=$("$PY" src/cleanup_run_processes.py --list --run-prefix "$directory/none" \
    --open-file "$directory/primary.lock") || return 1
  if [ -n "$holders" ]; then
    echo '[busy] local legacy node owner is still running; nothing stopped'
    e5_show_lock_owners "$directory/primary.lock"
    return 75
  fi
  # New same-name nodes share the legacy guard; old exclusive-host launchers
  # cannot enter alongside them. Never bypass an existing legacy host owner.
  exec 7>"$directory/primary.$host.lock" || return 1
  if ! flock -s -n 7; then
    echo "[busy] legacy hostname lock is held: $directory/primary.$host.lock; existing owner preserved"
    exec 7>&-
    return 75
  fi
  LOCK_FILE="$directory/physical.$physical.lock"
  exec 8>"$LOCK_FILE" || { exec 7>&-; return 1; }
  if ! flock -n 8; then
    echo "[busy] physical node lock is held: $LOCK_FILE; existing owner preserved"
    exec 7>&- 8>&-
    return 75
  fi
  E5_HOST_LOCK_HELD=1
  echo "[node] node ownership acquired for this controller: id=$EXPERIMENTS_NODE_ID physical=$physical"
}

e5_acquire_node() {
  local directory="${OM_LOCAL_LOCK_DIR:-/tmp/offpolicy-misranking-$(id -u)}"
  local holders filesystem error_file node
  E5_HOST_LOCK_HELD=0
  if [ -n "${PAIR_ROOT:-}" ]; then
    "$PY" "$(dirname "${BASH_SOURCE[0]}")/_pair_gpu_cleanup.py" --root "$PAIR_ROOT" || return $?
  fi
  mkdir -p "$directory" || return 1
  LOCK_FILE="$directory/primary.lock"
  filesystem=$(stat -f -c %T "$directory" 2>/dev/null || printf unknown)
  if [ -n "${PAIR_ROOT:-}" ] || [ -n "${RLOO_ROOT:-}" ]; then
    source "$(dirname "${BASH_SOURCE[0]}")/_node_id.sh" || return $?
    case "$filesystem" in nfs*|cifs|smb*|ceph|lustre|gpfs)
      e5_acquire_shared_pair_node "$directory"
      return $? ;;
    esac
  fi
  echo "[node] host=$(hostname) controller=$$ lock=$LOCK_FILE fs=$filesystem"
  e5_cleanup_lock_helpers "$LOCK_FILE" || return 1
  error_file=$(mktemp "$directory/.e5-flock-XXXXXX") || return 1
  exec 8>"$LOCK_FILE" || { rm -f "$error_file"; return 1; }
  if ! flock -n 8 2>"$error_file"; then
    holders=$("$PY" src/cleanup_run_processes.py --list --run-prefix "$directory/none" \
      --open-file "$LOCK_FILE") || { rm -f "$error_file"; exec 8>&-; return 1; }
    if [ -s "$error_file" ]; then
      echo "[abort] node locking failed; GPU work was not started: $(cat "$error_file")"
      rm -f "$error_file"; exec 8>&-; return 75
    fi
    rm -f "$error_file"
    # Missing /proc evidence alone does not establish a remote owner. Retain
    # the per-host fallback only for an identified shared filesystem.
    case "$filesystem" in nfs*|cifs|smb*|ceph|lustre|gpfs) ;;
      *) filesystem=local ;;
    esac
    if [ -z "$holders" ] && [ "$filesystem" != local ]; then
      node=$(hostname) || return 1
      node=$(printf '%s' "$node" | tr -c 'a-zA-Z0-9._-' '_')
      LOCK_FILE="$directory/primary.$node.lock"
      e5_cleanup_lock_helpers "$LOCK_FILE" || { exec 8>&-; return 1; }
      exec 8>"$LOCK_FILE" || return 1
      if ! flock -n 8; then
        echo "[busy] per-host node lock is held: $LOCK_FILE"
        e5_show_lock_owners "$LOCK_FILE"
        exec 8>&-; return 75
      fi
      echo "[node] acquired per-host lock on shared filesystem: $LOCK_FILE"
    elif [ -n "$holders" ] && [ "${E5_FORCE:-0}" = 1 ] && ! printf '%s\n' "$holders" | grep -Eq \
        'scripts/(run_additional_experiments|run_olmo3_rlzero|run_qwen35_9b|run_matrix|run_point|run_reliability_budget|run_reference_axes|go_[a-z0-9_]+)\.sh|src/(experiment|train_policy_grpo)\.py'; then
      echo "[force] stopping the explicitly authorized non-matrix lock holders:"
      e5_show_lock_owners "$LOCK_FILE"
      "$PY" src/cleanup_run_processes.py --run-prefix "$directory/none" \
        --open-file "$LOCK_FILE" --timeout 15 --compact || { exec 8>&-; return 1; }
      flock -w 5 8 || { echo "[busy] lock still held after force"; exec 8>&-; return 75; }
    else
      echo "[busy] node=$(hostname) lock=$LOCK_FILE; actual visible owners:"
      e5_show_lock_owners "$LOCK_FILE"
      [ -n "$holders" ] || echo "owner not visible; refusing to bypass a busy local lock"
      exec 8>&-; return 75
    fi
  else
    rm -f "$error_file"
  fi
  if [ "$LOCK_FILE" = "$directory/primary.lock" ]; then
    # Keep the same host lock even when the shared legacy lock was free. A
    # remote controller releasing primary.lock must not admit a second local
    # controller while the first still owns primary.<host>.lock.
    case "$filesystem" in nfs*|cifs|smb*|ceph|lustre|gpfs)
      node=$(hostname) || { exec 8>&-; return 1; }
      node=$(printf '%s' "$node" | tr -c 'a-zA-Z0-9._-' '_')
      e5_cleanup_lock_helpers "$directory/primary.$node.lock" || { exec 8>&-; return 1; }
      exec 7>"$directory/primary.$node.lock" || { exec 8>&-; return 1; }
      if ! flock -n 7; then
        echo "[busy] per-host node lock is held: $directory/primary.$node.lock"
        e5_show_lock_owners "$directory/primary.$node.lock"
        exec 7>&- 8>&-; return 75
      fi
      E5_HOST_LOCK_HELD=1
      ;;
    esac
  fi
  echo "[node] node ownership acquired for this E5 controller"
}
