# Node identity for launcher files, fault records and cost events. Two containers of
# one cluster job can share a hostname; their GPUs (and container ids) do not, so the
# identity is the hostname plus four hex digits of a hash over the visible GPU UUIDs,
# CUDA_VISIBLE_DEVICES, the container's cgroup and its first address. Sourced by the
# launchers; EXPERIMENTS_NODE_ID set in the environment wins (tests, operators).
node_identity() {
  local host suffix material
  host=$(hostname 2>/dev/null || cat /etc/hostname 2>/dev/null || echo unknown-host)
  material=$( { nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null; printf '%s\n' "${CUDA_VISIBLE_DEVICES:-}";
                cat /proc/1/cgroup 2>/dev/null; hostname -I 2>/dev/null; } | tr -d '[:space:]' || true)
  if [ -n "$material" ] && command -v sha256sum >/dev/null 2>&1; then
    suffix=$(printf '%s' "$material" | sha256sum | cut -c1-4)
    printf '%s-g%s\n' "$host" "$suffix"
  else
    printf '%s\n' "$host"
  fi
}
export EXPERIMENTS_NODE_ID=${EXPERIMENTS_NODE_ID:-$(node_identity)}
