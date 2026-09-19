"""Own one MBPP node controller; reap only children carrying its unique token.

The guard, not its GPU workers, owns the lease. A killed guard leaves a durable
token so the next invocation can reclaim its own children without sweeping the
node, deleting lock files, or stopping another experiment.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cleanup_run_processes as cleanup

TOKEN = "OM_MBPP_CONTROLLER_TOKEN"


def runtime_fingerprint():
    """CPU-only identity of executable checkout contents, including local fixes."""
    repo = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("scripts", "src"):
        for path in sorted((repo / folder).iterdir()):
            if path.is_file() and path.suffix in (".py", ".sh"):
                digest.update(str(path.relative_to(repo)).encode() + b"\0")
                digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def runtime_record(lock_path, pid, fingerprint, guard_hash=None):
    start = identity(pid)
    if start is None:
        raise ValueError("MBPP controller is no longer alive")
    return {"schema": "mbpp-controller-runtime-v1", "pid": pid, "start_time": start,
            "lock": str(lock_path.resolve()), "fingerprint": fingerprint,
            "guard_hash": guard_hash or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def runtime_current(lock_path, pid, fingerprint):
    try:
        recorded = json.loads(lock_path.with_suffix(".runtime.json").read_text())
        return recorded == runtime_record(lock_path, pid, fingerprint)
    except (OSError, ValueError):
        # A legacy controller without a bound version receipt is upgraded once.
        return False


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return None if fields[0] == "Z" else int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def wait_gpu_release(targets, timeout=10.):
    """Wait for the driver to drop only proven old CUDA owners; never reset GPUs."""
    owned = {p.pid for p in targets if p.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1")
             or "_gpu_keepalive.py" in p.command}
    if not owned:
        return
    deadline = time.monotonic() + timeout
    while True:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
                timeout=max(.1, min(3., deadline - time.monotonic())))
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("cannot verify old CUDA memory release; refusing restart") from exc
        rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
        if any(not row.isdigit() for row in rows):
            raise RuntimeError("invalid GPU process report; refusing restart")
        remaining = owned & {int(row) for row in rows}
        if not remaining:
            print('[mbpp-clean] driver confirms previous owned CUDA PIDs released', flush=True)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU still reports previous CUDA PIDs {sorted(remaining)}; refusing restart")
        print(f"[mbpp-clean] waiting for CUDA release: pids={sorted(remaining)}", flush=True)
        time.sleep(min(.5, max(0., deadline - time.monotonic())))


def inspect_cleanup(lock_path, pid):
    """Bounded, read-only stop evidence; never signals or prints environments."""
    print(f"[cleanup-status] controller pid={pid} alive={identity(pid) is not None}", flush=True)
    try:
        owner = json.loads(lock_path.with_suffix('.owner.json').read_text())
        if (owner.get('schema') != 'mbpp-node-owner-v1' or owner.get('pid') != pid
                or owner.get('lock') != str(lock_path.resolve())):
            raise ValueError('owner receipt does not match this controller')
        token = owner['token']
        if not isinstance(token, str) or len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
            raise ValueError('invalid owner token')
        children = cleanup.list_processes('/unused-mbpp-token-scope', command_patterns=('',),
                                         required_environment=((TOKEN, token),))
        print(f"[cleanup-status] owned processes remaining={len(children)}", flush=True)
        for process in children[:12]:
            scripts = [Path(arg).name for arg in process.argv if arg.endswith(('.py', '.sh'))]
            role = 'GPU 유지용' if '_gpu_keepalive.py' in scripts else 'MBPP 작업'
            if any(name.startswith('train_') for name in scripts):
                role = '학습'
            details = []
            for key in ('--phase', '--arm'):
                if key in process.argv and process.argv.index(key) + 1 < len(process.argv):
                    details.append(f"{key[2:]}={process.argv[process.argv.index(key)+1][:60]}")
            print(f"  pid={process.pid} parent={process.ppid} {role} {' '.join(scripts) or 'child process'} {' '.join(details)}", flush=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'[cleanup-status] owner details unavailable: {exc}', flush=True)
    for label, query in (
            ('GPU index, used MiB, utilization %', '--query-gpu=index,memory.used,utilization.gpu'),
            ('GPU owner PID, used MiB, process', '--query-compute-apps=pid,used_memory,process_name')):
        try:
            value = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=2, check=True)
            print(f'[cleanup-status] {label}', flush=True)
            print('\n'.join(value.stdout.splitlines()[:12])[:1600] or 'none reported', flush=True)
        except (OSError, subprocess.SubprocessError):
            print(f'[cleanup-status] {label}: unavailable (not proof of free memory)', flush=True)
    print('[cleanup-status] GPU PID visibility can differ inside containers; unknown owners are not killed.', flush=True)


def reap(token):
    # No root wildcard, GPU PID sweep or group signal. The fresh unguessable
    # marker is required on every initial target; their descendants are included.
    if not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("invalid MBPP owner token; refusing cleanup")
    targets = cleanup.terminate("/unused-mbpp-token-scope", timeout=10,
        command_patterns=("",), required_environment=((TOKEN, token),), compact=True)
    remaining = cleanup.list_processes("/unused-mbpp-token-scope",
        command_patterns=("",), required_environment=((TOKEN, token),))
    if remaining:
        raise RuntimeError(f"MBPP children still alive; refusing new GPU work: {[p.pid for p in remaining]}")
    wait_gpu_release(targets)
    print(f"[mbpp-clean] previous owned processes stopped={len(targets)}; remaining=0", flush=True)
    return targets


def publish(path, value):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run(lock_path, command):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owner_path = lock_path.with_suffix(".owner.json")
    with lock_path.open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[already running] MBPP node controller owns {lock_path}; "
                  "existing work was not stopped. Use the existing console.", flush=True)
            return 0
        if owner_path.exists():
            old = json.loads(owner_path.read_text())
            if (old.get("schema") != "mbpp-node-owner-v1"
                    or old.get("lock") != str(lock_path.resolve())):
                raise ValueError(f"unrecognized MBPP owner receipt: {owner_path}")
            if old.get("state") != "released" and identity(old["pid"]) == old["start_time"]:
                raise ValueError("previous MBPP guard is still alive; refusing takeover")
            print("[mbpp-recover] checking children of the previous controller only", flush=True)
            reap(old["token"])
        token = uuid.uuid4().hex
        owner = {"schema": "mbpp-node-owner-v1", "lock": str(lock_path.resolve()),
                 "pid": os.getpid(), "start_time": identity(os.getpid()), "token": token, "state": "active"}
        publish(owner_path, owner)
        child = None
        stopped = 0
        rc = 75

        def stop(signum, _frame):
            nonlocal stopped
            stopped = stopped or signum
            # Raising once takes us to the bounded, token-scoped finalizer.
            raise InterruptedError("MBPP controller stop requested")

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, stop)
        try:
            env = {**os.environ, TOKEN: token, "MBPP_GUARD_PID": str(os.getpid()),
                   "MBPP_GUARD_CODE_HASH": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "EXPERIMENTS_DETACHED": "1"}
            # close_fds prevents inheriting this guard's flock into any worker.
            child = subprocess.Popen(command, env=env, start_new_session=True, close_fds=True)
            rc = child.wait()
            rc = rc if rc >= 0 else 128 - rc
            return rc
        except InterruptedError:
            rc = 128 + (stopped or signal.SIGTERM)
            return rc
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            # Also runs after ordinary worker failure. Keep the lease until all
            # owned descendants have exited, including detached GPU ranks.
            reap(token)
            if child is not None:
                child.wait(timeout=5)
            # The old PID may still be exiting after flock closes. This receipt
            # proves teardown finished, so an immediate relaunch need not fail.
            publish(owner_path, {**owner, "state": "released"})
            print(f"[node-launcher-exit] pid={os.getpid()} rc={rc} owner=mbpp-guard", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--fingerprint", action="store_true")
    parser.add_argument("--runtime-current", type=int)
    parser.add_argument("--record-runtime", type=int)
    parser.add_argument("--inspect-cleanup", type=int)
    parser.add_argument("--loaded-fingerprint")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.fingerprint:
        print(runtime_fingerprint())
        return 0
    if args.lock is None:
        parser.error("--lock required")
    if args.inspect_cleanup is not None:
        inspect_cleanup(args.lock, args.inspect_cleanup)
        return 0
    if args.runtime_current is not None or args.record_runtime is not None:
        if not args.loaded_fingerprint:
            parser.error("--loaded-fingerprint required")
        if args.runtime_current is not None:
            return 0 if runtime_current(args.lock, args.runtime_current, args.loaded_fingerprint) else 1
        publish(args.lock.with_suffix(".runtime.json"),
                runtime_record(args.lock, args.record_runtime, args.loaded_fingerprint,
                               os.environ.get("MBPP_GUARD_CODE_HASH")))
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("controller command required")
    try:
        return run(args.lock, command)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"[mbpp-guard] {exc}; no unrelated process was stopped", file=sys.stderr, flush=True)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
