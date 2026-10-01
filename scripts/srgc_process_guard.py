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

try:
    from srgc_log_tail import tail_lines
except ImportError:
    from scripts.srgc_log_tail import tail_lines

TARGET_MARKERS = ("srgc_rebuttal.run_experiment", "srgc_rebuttal.build_cache",
                  "srgc_step_checkpoints.py", "srgc_qwen35_rank.py", "srgc_sr_refresh.py")
OWNER_MARKERS = ("run_srgc_rebuttal.py", "srgc_rebuttal.cluster", "run_srgc_qwen35.py",
                 "srgc_extra_worker.py", "run_srgc_sr_refresh.sh")


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


def is_target(cmdline, plan=None):
    """A Python interpreter running an SRGC launcher, rank or cache module (never a shell wrapper).

    ``plan`` restricts the match to one plan path; ``None`` matches any SRGC child, which is
    what orphan reaping wants: a rank without a live launcher is garbage whichever plan it ran.
    """
    argv0 = os.path.basename(cmdline.split(" ", 1)[0]) if cmdline else ""
    return (argv0.startswith("python") and (plan is None or plan in cmdline) and
            any(marker in cmdline for marker in TARGET_MARKERS) and
            not any(marker in cmdline for marker in OWNER_MARKERS))


def orphan_pids(plan_path=None, table=None):
    """SRGC child processes (same user) whose ancestors no longer include a launcher.

    ``plan_path`` is accepted for compatibility and logging only; every orphaned SRGC child
    on the node is reaped because each one keeps GPU memory and execution locks.
    """
    table = process_table() if table is None else table
    me = os.getpid()
    orphans = []
    for pid, (_, uid, cmdline) in table.items():
        if pid == me or uid != os.getuid() or not is_target(cmdline):
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


def gpu_memory_summary():
    """'0:1200MiB 1:0MiB ...' from nvidia-smi, or None when it is unavailable."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    cells = []
    for line in out.splitlines():
        if "," in line:
            index, used = (part.strip() for part in line.split(",", 1))
            cells.append(f"{index}:{used}MiB")
    return " ".join(cells) or None


def reap_orphans(plan_path=None, *, grace=30.0, report_gpus=True):
    pids = orphan_pids(plan_path)
    if pids:
        print(f"GUARD found {len(pids)} orphaned SRGC process(es) on this node; reaping before start", flush=True)
        terminate(pids, grace=grace, label="orphan")
        if report_gpus:
            summary = gpu_memory_summary()
            if summary:
                print(f"GUARD gpu memory after reaping: {summary}", flush=True)
    return pids


GPU_FREE_MIB = int(os.environ.get("SRGC_GPU_FREE_MIB", "2000"))
GPU_WAIT_SECONDS = float(os.environ.get("SRGC_GPU_WAIT_SECONDS", "600"))
SHM_DIR = Path("/dev/shm")


def _nvidia_smi(args):
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return None
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_usage(environment=None):
    """[(index, used MiB)] for the GPUs visible to the child, or None when nvidia-smi is unavailable."""
    devices = (environment or os.environ).get("CUDA_VISIBLE_DEVICES")
    args = ["--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    if devices:
        args += ["-i", devices]
    out = _nvidia_smi(args)
    if out is None:
        return None
    usage = []
    for line in out.splitlines():
        if "," in line:
            index, used = (part.strip() for part in line.split(",", 1))
            try:
                usage.append((index, int(float(used))))
            except ValueError:
                continue
    return usage


def gpu_compute_processes():
    """[(pid, used MiB)] of every compute process nvidia-smi can see (host pids)."""
    out = _nvidia_smi(["--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"]) or ""
    processes = []
    for line in out.splitlines():
        if "," in line:
            pid, used = (part.strip() for part in line.split(",", 1))
            if pid.isdigit():
                try:
                    processes.append((int(pid), int(float(used))))
                except ValueError:
                    processes.append((int(pid), -1))
    return processes


def wait_for_free_gpus(environment=None, *, threshold_mib=None, timeout=None, poll=5.0, heartbeat=lambda: None,
                       usage=gpu_usage, processes=gpu_compute_processes, clock=time.monotonic, sleep=time.sleep):
    """Block until every visible GPU is below ``threshold_mib`` used; reap SRGC orphans meanwhile.

    Returns True when the GPUs are free, False after ``timeout`` seconds. A child launched onto
    GPUs that a dying or foreign process still occupies fails with CUDA OOM or an NCCL error
    minutes later; waiting here turns that into either a clean start or a fast, explained failure.
    """
    threshold = GPU_FREE_MIB if threshold_mib is None else threshold_mib
    limit = GPU_WAIT_SECONDS if timeout is None else timeout
    started, last_report = clock(), -1e9
    while True:
        current = usage(environment)
        if current is None:
            return True  # no nvidia-smi: nothing to measure
        busy = [(index, used) for index, used in current if used > threshold]
        if not busy:
            return True
        now = clock()
        if now - last_report >= 60:
            holders = ", ".join(f"pid {pid} {used}MiB" for pid, used in processes()) or "no compute process listed"
            print(f"GUARD waiting for GPU memory to free: " + " ".join(f"{i}:{u}MiB" for i, u in current)
                  + f" (threshold {threshold}MiB; {holders})", flush=True)
            last_report = now
            reap_orphans(None, grace=10.0, report_gpus=False)
        if now - started >= limit:
            return False
        heartbeat()
        sleep(poll)


def clean_shm(*, shm_dir=None, table=None):
    """Remove this user's leftover NCCL/torch shared-memory files once no SRGC child is alive.

    Ranks killed with SIGKILL leave ``/dev/shm/nccl-*`` segments behind; a full ``/dev/shm``
    makes the next NCCL init fail with DistBackendError.
    """
    shm_dir = SHM_DIR if shm_dir is None else Path(shm_dir)
    table = process_table() if table is None else table
    alive = [pid for pid, (_, uid, cmdline) in table.items()
             if uid == os.getuid() and pid != os.getpid() and is_target(cmdline)]
    if alive or not shm_dir.is_dir():
        return []
    removed = []
    for path in list(shm_dir.glob("nccl-*")) + list(shm_dir.glob("torch_*")):
        try:
            if path.is_symlink() or path.stat().st_uid != os.getuid():
                continue
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path.name)
        except OSError:
            continue
    if removed:
        print(f"GUARD removed {len(removed)} leftover shared-memory file(s) from {shm_dir}", flush=True)
    return removed


def guarded_run_child(original, plan_path):
    def run_child(command, log_path, environment, **kwargs):
        reap_orphans(plan_path)
        clean_shm()
        seen = set()
        inner = kwargs.get("heartbeat", lambda pid: None)
        if not wait_for_free_gpus(environment, heartbeat=lambda: inner(None)):
            usage = gpu_usage(environment) or []
            holders = ", ".join(f"pid {pid} {used}MiB" for pid, used in gpu_compute_processes()) or "none listed"
            message = (f"GPUs still busy after {GPU_WAIT_SECONDS:.0f}s; not starting this attempt. "
                       f"memory used: {' '.join(f'{i}:{u}MiB' for i, u in usage)}; compute processes: {holders}")
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with Path(log_path).open("a") as handle:
                handle.write(f"GUARD {message}\n")
            report_failure(log_path, 75)
            return 75
        def heartbeat(pid):
            # Record the tree while it is still attached; setsid ranks are reparented once the launcher dies.
            seen.update(descendants(pid))
            inner(pid)
        kwargs["heartbeat"] = heartbeat
        try:
            code = original(command, log_path, environment, **kwargs)
        finally:
            terminate(seen)
            reap_orphans(plan_path, grace=10.0)
        if code not in (0, None):
            report_failure(log_path, code)
        return code
    return run_child


def failure_tail(log_path, lines=40):
    """The last ``lines`` of a child log, trimmed to the last Traceback when one is present."""
    try:
        tail = tail_lines(log_path, lines)
    except OSError:
        return ["(log file not readable)"]
    for index in range(len(tail) - 1, -1, -1):
        if tail[index].startswith("Traceback"):
            return tail[index:]
    return tail


def report_failure(log_path, code, lines=40):
    name = Path(log_path).name.removesuffix(".log")
    print(f"FAILED {name} exit={code} log={log_path}", flush=True)
    for line in failure_tail(log_path, lines):
        print(f"FAILED {name} | {line}", flush=True)


CANONICAL_LOCK_DIR = ".srgc-gpu-node-locks"


def canonical_lock_root(environment=None):
    """``<group volume>/.srgc-gpu-node-locks``: the lease namespace every SRGC launcher (OLMo, Qwen, manual) shares."""
    try:
        from srgc_shared_storage import storage_root
    except ImportError:
        from scripts.srgc_shared_storage import storage_root
    try:
        group, _ = storage_root(environment or os.environ)
    except ValueError:
        return None
    return group / CANONICAL_LOCK_DIR


def shared_device_leases(original, environment=None):
    """Wrap ``cluster.device_leases`` so a lease is also taken in the canonical namespace.

    OLMo workers, manual runs and Qwen workers each used their own lock directory, so two of
    them could hold "exclusive" leases on the same GPUs and collide (CUDA OOM, NCCL errors).
    """
    from contextlib import ExitStack

    @contextmanager
    def device_leases(root, uuids):
        roots = {Path(root)}
        canonical = canonical_lock_root(environment)
        if canonical is not None:
            roots.add(canonical)
        with ExitStack() as stack:
            fds = []
            for each in sorted(roots):
                fds.extend(stack.enter_context(original(each, uuids)))
            yield tuple(fds)
    return device_leases


@contextmanager
def process_guard(plan_path):
    """Patch ``srgc_rebuttal.cluster.run_child`` (and device leases) for the lifetime of a worker."""
    from srgc_rebuttal import cluster
    original_leases = cluster.device_leases
    cluster.device_leases = shared_device_leases(original_leases)
    # worker() checks GPU occupancy and acquires device leases before its first
    # run_child(). An orphan can retain either resource, so cleaning only in
    # run_child() leaves restart blocked before cleanup can ever run.
    reap_orphans(plan_path)
    summary = gpu_memory_summary()
    if summary:
        print(f"GUARD gpu memory at start: {summary}", flush=True)
    original = cluster.run_child
    cluster.run_child = guarded_run_child(original, plan_path)
    try:
        yield
    finally:
        cluster.run_child = original
        cluster.device_leases = original_leases


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
