"""Kill orphaned SRGC child processes that keep GPUs and execution locks after an interrupted worker.

torchrun starts every rank in its own session, so terminating the launcher's
process group leaves the ranks alive when the launcher is killed. Those ranks
hold ``.<task>.execution.lock`` / ``.manifest.lock`` and GPU memory, and every
later attempt of the same task fails or blocks. This guard lives outside the
hashed ``srgc_rebuttal`` package: it patches ``cluster.run_child`` at runtime so
that (a) stale processes of this plan without a live launcher ancestor are
reaped before GPU admission/leases and before any child starts, and (b) the whole descendant tree of a finished
or interrupted child is terminated, not only its process group.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import sys
import time

TARGET_MARKERS = ("srgc_rebuttal.run_experiment", "srgc_rebuttal.build_cache", "torch.distributed.run",
                  "srgc_step_checkpoints.py")
OWNER_MARKERS = ("run_srgc_rebuttal.py", "srgc_rebuttal.cluster")


def _read(path):
    try:
        return Path(path).read_bytes()
    except OSError:
        return b""


def process_table():
    """{pid: (ppid, uid, cmdline)} for every readable process."""
    table = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        stat = _read(f"/proc/{pid}/stat").decode(errors="replace")
        if ") " not in stat:
            continue
        fields = stat.rsplit(") ", 1)[1].split()
        try:
            ppid = int(fields[1])
            uid = os.stat(f"/proc/{pid}").st_uid
        except (IndexError, ValueError, OSError):
            continue
        cmdline = _read(f"/proc/{pid}/cmdline").replace(b"\0", b" ").decode(errors="replace").strip()
        table[pid] = (ppid, uid, cmdline)
    return table


def descendants(pid, table=None):
    table = process_table() if table is None else table
    children = {}
    for child, (parent, _, _) in table.items():
        children.setdefault(parent, []).append(child)
    found, stack = [], [pid]
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            found.append(child)
            stack.append(child)
    return found


def _has_live_owner(pid, table):
    seen = set()
    parent = table.get(pid, (0, 0, ""))[0]
    while parent > 1 and parent not in seen:
        seen.add(parent)
        cmdline = table.get(parent, (0, 0, ""))[2]
        if any(marker in cmdline for marker in OWNER_MARKERS):
            return True
        parent = table.get(parent, (0, 0, ""))[0]
    return False


def is_target(cmdline, plan):
    """A Python interpreter running this plan's launcher, rank or cache module (never a shell wrapper)."""
    argv0 = os.path.basename(cmdline.split(" ", 1)[0]) if cmdline else ""
    return (argv0.startswith("python") and plan in cmdline and
            any(marker in cmdline for marker in TARGET_MARKERS) and
            not any(marker in cmdline for marker in OWNER_MARKERS))


def orphan_pids(plan_path, table=None):
    """Processes of this plan (same user) whose ancestors no longer include a launcher."""
    table = process_table() if table is None else table
    plan = str(Path(plan_path).resolve())
    me = os.getpid()
    orphans = []
    for pid, (_, uid, cmdline) in table.items():
        if pid == me or uid != os.getuid() or not is_target(cmdline, plan):
            continue
        if not _has_live_owner(pid, table):
            orphans.append(pid)
    return sorted(orphans)


def _signal_all(pids, sig):
    groups = set()
    for pid in pids:
        try:
            groups.add(os.getpgid(pid))
        except ProcessLookupError:
            continue
    for group in groups:
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            pass
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _alive(pids):
    alive = []
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            alive.append(pid)
            continue
        if _read(f"/proc/{pid}/stat").decode(errors="replace").rsplit(") ", 1)[-1][:1] != "Z":
            alive.append(pid)
    return alive


def terminate(pids, *, grace=30.0, label="descendant"):
    """SIGTERM, wait up to ``grace`` seconds, then SIGKILL; returns the pids that were alive."""
    pids = _alive(list(pids))
    if not pids:
        return []
    table = process_table()
    for pid in pids:
        print(f"GUARD terminating {label} pid={pid} cmd={table.get(pid, (0, 0, '?'))[2][:160]}", flush=True)
    _signal_all(pids, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while _alive(pids) and time.monotonic() < deadline:
        time.sleep(0.5)
    remaining = _alive(pids)
    if remaining:
        _signal_all(remaining, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while _alive(remaining) and time.monotonic() < deadline:
            time.sleep(0.2)
        print(f"GUARD killed {len(remaining)} {label}(s) that ignored SIGTERM", flush=True)
    return pids


def reap_orphans(plan_path, *, grace=30.0):
    pids = orphan_pids(plan_path)
    if pids:
        print(f"GUARD found {len(pids)} orphaned process(es) of {plan_path}; reaping before start", flush=True)
        terminate(pids, grace=grace, label="orphan")
    return pids


def guarded_run_child(original, plan_path):
    def run_child(command, log_path, environment, **kwargs):
        reap_orphans(plan_path)
        seen = set()
        inner = kwargs.get("heartbeat", lambda pid: None)
        def heartbeat(pid):
            # Record the tree while it is still attached; setsid ranks are reparented once the launcher dies.
            seen.update(descendants(pid))
            inner(pid)
        kwargs["heartbeat"] = heartbeat
        try:
            return original(command, log_path, environment, **kwargs)
        finally:
            terminate(seen)
            reap_orphans(plan_path, grace=10.0)
    return run_child


@contextmanager
def process_guard(plan_path):
    """Patch ``srgc_rebuttal.cluster.run_child`` for the lifetime of a worker."""
    from srgc_rebuttal import cluster
    # worker() checks GPU occupancy and acquires device leases before its first
    # run_child(). An orphan can retain either resource, so cleaning only in
    # run_child() leaves restart blocked before cleanup can ever run.
    reap_orphans(plan_path)
    original = cluster.run_child
    cluster.run_child = guarded_run_child(original, plan_path)
    try:
        yield
    finally:
        cluster.run_child = original


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="Reap orphaned SRGC processes of a plan on this node")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(args)
    pids = orphan_pids(options.plan)
    table = process_table()
    for pid in pids:
        print(f"orphan pid={pid} cmd={table.get(pid, (0, 0, '?'))[2][:200]}")
    if not options.dry_run:
        terminate(pids, label="orphan")
    print(f"{len(pids)} orphan(s){' (dry run)' if options.dry_run else ' terminated'}", flush=True)
    return 0 if not pids or not options.dry_run else 1


if __name__ == "__main__":
    sys.exit(main())
