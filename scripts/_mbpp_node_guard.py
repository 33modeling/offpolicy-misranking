"""Own one MBPP node controller; reap only children carrying its unique token.

The guard, not its GPU workers, owns the lease. A killed guard leaves a durable
token so the next invocation can reclaim its own children without sweeping the
node, deleting lock files, or stopping another experiment.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cleanup_run_processes as cleanup

TOKEN = "OM_MBPP_CONTROLLER_TOKEN"


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return None if fields[0] == "Z" else int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def reap(token):
    # No root wildcard, GPU PID sweep or group signal. The fresh unguessable
    # marker is required on every initial target; their descendants are included.
    if not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("invalid MBPP owner token; refusing cleanup")
    return cleanup.terminate("/unused-mbpp-token-scope", timeout=10,
        command_patterns=("",), required_environment=((TOKEN, token),), compact=True)


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
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
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
