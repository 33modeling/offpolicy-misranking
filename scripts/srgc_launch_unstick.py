"""End this node's Gemma IT launch children that wait on a startup lock.

A child started before the 2026-10-31 startup fix waits in the shared plan
preparation lock (.start-prepare.lock) and never reaches newer code; the node
shows LAUNCH indefinitely. This ends only such children: a Gemma IT startup
process of the current user, waiting on a startup lock, in a process group
launched by a model-chain coordinator, with no torch.distributed.run process
in that group or below it. SIGKILL makes the coordinator record exit 137 and
launch the same model again from the current checkout. Training, admission
and the coordinator itself are never signalled; no queue file is changed.
"""

import os
import signal
import socket
import sys
from pathlib import Path

STARTUP = {"srgc_research.dispatch.gemma_it_run", "srgc_research.dispatch.gemma4_it.entry",
           "srgc_research.dispatch.gemma4_it.cli"}
COORDINATOR = "srgc_research.dispatch.model_chain"
# The shared session lock stays open for a child's whole life, so it is not
# evidence of a startup wait.
LOCKS = (".start-prepare.lock", ".auto-prepare.lock", ".download.lock")


def read(proc, pid, uid):
    path = proc / str(pid)
    if path.stat().st_uid != uid:
        return None
    tail = (path / "stat").read_text().rsplit(") ", 1)[1].split()
    argv = (path / "cmdline").read_bytes().decode(errors="replace").split("\0")
    module = argv[argv.index("-m") + 1] if "-m" in argv[:-1] else None
    return {"pid": pid, "ppid": int(tail[1]), "pgid": int(tail[2]), "argv": argv, "module": module}


def startup_lock(proc, pid):
    for fd in (proc / str(pid) / "fd").iterdir():
        try:
            target = os.readlink(fd).removesuffix(" (deleted)")
        except OSError:
            continue
        name = Path(target).name
        if name.endswith(LOCKS) or name.endswith(".lock") and "/gemma-runtime-cache/" in target:
            return target
    return None


def waiting(proc, pid):
    """Kernel wait channel of a lock wait (local flock, NFS or Lustre)."""
    try:
        channel = (proc / str(pid) / "wchan").read_text().strip()
    except OSError:
        return None
    return channel if "lock" in channel or "ldlm" in channel else None


def candidates(proc=Path("/proc"), uid=None):
    uid = os.getuid() if uid is None else uid
    rows = {}
    for path in proc.iterdir():
        if path.name.isdecimal():
            try:
                row = read(proc, int(path.name), uid)
            except (OSError, ValueError, IndexError):
                continue
            if row:
                rows[row["pid"]] = row
    children = {}
    for row in rows.values():
        children.setdefault(row["ppid"], []).append(row["pid"])

    def below(pid):
        stack, seen = [pid], set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(children.get(current, ()))
        return seen

    found = {}
    for row in rows.values():
        if row["module"] not in STARTUP:
            continue
        leader = rows.get(row["pgid"])
        coordinator = rows.get(leader["ppid"]) if leader else None
        if not coordinator or coordinator["module"] != COORDINATOR:
            continue
        try:
            lock = startup_lock(proc, row["pid"])
            channel = waiting(proc, row["pid"])
        except OSError:
            continue
        if not lock or not channel:
            continue
        group = {pid for pid, other in rows.items() if other["pgid"] == row["pgid"]}
        tree = set().union(*(below(pid) for pid in group))
        if any("torch.distributed.run" in rows[pid]["argv"] for pid in tree if pid in rows):
            continue
        found[row["pgid"]] = {"pgid": row["pgid"], "pid": row["pid"], "lock": lock,
                              "wait": channel, "coordinator": coordinator["pid"]}
    return list(found.values())


def main():
    host = socket.gethostname()
    rows = candidates()
    for row in rows:
        try:
            os.killpg(row["pgid"], signal.SIGKILL)
            result = "ended; coordinator relaunches it with the current code"
        except ProcessLookupError:
            result = "already gone"
        print(f"{host} launch child pid={row['pid']} waited on {Path(row['lock']).name} "
              f"({row['wait']}): {result}", flush=True)
    if not rows:
        print(f"{host} no Gemma launch child is waiting on a startup lock; nothing changed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
