"""Start the pinned Llama runner after the requested allocated-GPU cleanup."""

import hashlib
import os
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

OWNER_MARKERS = (
    "srgc_research.dispatch.llama_run",
    "srgc_research.dispatch.llama31.cli",
    "srgc_research.dispatch.information_queue",
    "srgc_research.dispatch.information_run",
    "srgc_research.information_cli",
    "srgc_research.dispatch.qwen_run",
)


def allocated_gpus():
    """Validate the same four full H100 devices, without rejecting occupancy."""
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    ids = devices.split(",")
    if len(ids) != 4 or len(set(ids)) != 4 or not all(ids):
        raise ValueError("each worker needs four distinct allocated GPUs")
    result = subprocess.run(
        ["nvidia-smi", f"--id={devices}", "--query-gpu=uuid,memory.used,name,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    rows = [[value.strip() for value in line.split(",")] for line in result.stdout.strip().splitlines()]
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("nvidia-smi did not return four valid GPU records")
    if any("H100" not in row[2] or int(row[3]) < 75000 for row in rows):
        raise ValueError("this allocation requires four full H100 GPUs with at least 75,000 MiB each")
    uuids = tuple(sorted(row[0] for row in rows))
    if len(set(uuids)) != 4:
        raise ValueError("allocated GPU identifiers alias the same device")
    return devices, uuids, tuple(int(row[1]) for row in rows)


def gpu_processes():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    rows = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 2 or not fields[1].isdigit():
            raise ValueError("cannot identify allocated GPU processes")
        rows.setdefault(int(fields[1]), set()).add(fields[0])
    return rows


def process_token(pid):
    """Bind a signal to one UID/start time, so a recycled PID is never killed."""
    try:
        path = Path(f"/proc/{pid}")
        fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return path.stat().st_uid, fields[19]
    except (OSError, IndexError):
        return None


def cleanup_processes(uuids, *, grace_seconds=5):
    """Stop this user's jobs on this allocation, including their retrying owners."""
    from scripts import srgc_process_guard as guard
    table, uid = guard.process_table(), os.getuid()
    # Snapshot start times before the NVML query, not after a PID could have
    # exited and been reassigned to a new process while nvidia-smi was running.
    existing = {pid: process_token(pid) for pid, (_, owner, _) in table.items() if owner == uid}
    applications = gpu_processes()
    protected, parent = set(), os.getpid()
    while parent > 1 and parent not in protected:
        protected.add(parent)
        parent = table.get(parent, (0, None, ""))[0]
    chosen = {pid for pid, devices in applications.items() if devices & set(uuids)
              and pid not in protected and table.get(pid, (0, None, ""))[1] == uid}
    if any(applications[pid] - set(uuids) for pid in chosen):
        raise ValueError("GPU process also uses devices outside this allocation")
    targets = set(chosen)
    markers = (*guard.OWNER_MARKERS, *OWNER_MARKERS, "torch.distributed.run")
    for pid in chosen:
        seen, parent = set(), table[pid][0]
        while parent > 1 and parent not in seen and parent not in protected:
            seen.add(parent)
            ancestor, owner, command = table.get(parent, (0, None, ""))
            if owner != uid:
                break
            if command.split(" ", 1)[0].rsplit("/", 1)[-1].startswith("python") and any(
                marker in command for marker in markers
            ):
                descendants = set(guard.descendants(parent, table))
                if any(devices - set(uuids) for child, devices in applications.items()
                       if child in descendants):
                    raise ValueError("GPU launcher also owns jobs outside this allocation")
                targets.add(parent)
            parent = ancestor
    tokens = {pid: existing.get(pid) for pid in targets}
    tokens = {pid: token for pid, token in tokens.items() if token and token[0] == uid}
    if not tokens:
        raise RuntimeError("busy allocated GPUs have no visible processes owned by this user")
    print("LLAMA GPU cleanup: stopping owned GPU processes and launchers: "
          + ",".join(map(str, sorted(tokens))), flush=True)

    def send(pid, sig):
        if process_token(pid) == tokens[pid]:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    # Stop retrying coordinators along with their GPU ranks.
    for pid in sorted(tokens):
        send(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and any(process_token(pid) == token for pid, token in tokens.items()):
        time.sleep(.2)
    for pid in sorted(tokens):
        send(pid, signal.SIGKILL)
    return tuple(sorted(tokens))


@contextmanager
def clean_start(*, identity=None):
    from scripts import srgc_process_guard as guard
    from srgc_rebuttal import cluster
    from srgc_rebuttal.runtime import Busy, lease
    identity = cluster.gpu_identity if identity is None else identity
    devices, uuids, memory = allocated_gpus()
    root = guard.canonical_lock_root()
    if root is None:
        raise ValueError("Llama cleanup requires shared group storage")
    key = hashlib.sha256("\0".join(uuids).encode()).hexdigest()
    # Held through the runner lifetime: another local Llama start must not
    # mistake this launcher's active ranks for the previous experiment.
    with lease(root / f"llama-start-{key}.lock"):
        if any(value > 4000 for value in memory):
            cleanup_processes(uuids)
            deadline = time.monotonic() + 30
            while True:
                try:
                    identity()
                    break
                except Busy:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("allocated GPUs remain busy after owned-process cleanup") from None
                    time.sleep(1)
            print(f"LLAMA GPU cleanup: complete; devices={devices}", flush=True)
        with patch.object(guard, "OWNER_MARKERS", (*guard.OWNER_MARKERS, *OWNER_MARKERS)):
            yield


def released_gpu_identity(identity, *, timeout=30):
    """Allow delayed CUDA teardown between tasks without killing another job."""
    from srgc_rebuttal.runtime import Busy

    try:
        return identity()
    except Busy as error:
        reason = str(error)
    started = time.monotonic()
    print(f"LLAMA waiting for GPU memory release (up to {timeout}s): {reason}", flush=True)
    while time.monotonic() - started < timeout:
        time.sleep(1)
        try:
            result = identity()
        except Busy:
            continue
        print("LLAMA GPU memory release complete; continuing existing queue", flush=True)
        return result
    # The worker records this as a failed attempt and applies its bounded
    # retry/resume policy, instead of exiting on a transient occupancy check.
    raise TimeoutError(f"GPU memory release did not finish within {timeout}s; allocated devices remain busy")


def admission_failure_footer(error, root):
    """Print the actual admission traceback after backup shutdown messages."""
    from scripts.srgc_log_tail import tail_lines

    message = str(error)
    if message.startswith("Llama generation/backward admission failed: "):
        name = message.split(": ", 1)[1]
    elif message.startswith("four-GPU admission failed (") and "; inspect " in message:
        name = message.split("; inspect ", 1)[1]
    else:
        return
    path = Path(name)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        return
    print("\nLLAMA ADMISSION FAILURE DETAILS", flush=True)
    try:
        for line in tail_lines(path, lines=40):
            print(line, flush=True)
    except OSError as read_error:
        print(f"log unavailable: {read_error}", flush=True)


def main(argv=None):
    from scripts import srgc_process_guard as guard
    from srgc_rebuttal import cluster
    from srgc_research.dispatch.llama31 import cli, resume
    from srgc_research.dispatch.model_resume import LlamaResumeFirst, failure_footer
    args = list(sys.argv[1:] if argv is None else argv) or ["all"]
    original, started = cluster.gpu_identity, False
    with ExitStack() as stack:
        def gpu_identity():
            nonlocal started
            if not started:
                # The pinned worker checks stopped/completed queues first.
                # Cleanup starts only when it actually requests GPU admission.
                stack.enter_context(clean_start(identity=original))
                started = True
            return released_gpu_identity(original)
        stack.enter_context(patch.object(guard, "OWNER_MARKERS", (*guard.OWNER_MARKERS, *OWNER_MARKERS)))
        stack.enter_context(patch.object(cluster, "gpu_identity", gpu_identity))
        stack.enter_context(patch.object(resume, "ResumeFirst", LlamaResumeFirst))
        try:
            return cli.main(args)
        except (OSError, ValueError, TypeError, RuntimeError, ImportError, subprocess.CalledProcessError) as error:
            from srgc_research.dispatch.llama31.storage import default_root

            root = default_root(os.environ)
            if "--root" in args and args.index("--root") + 1 < len(args):
                root = Path(args[args.index("--root") + 1])
            root = next((Path(arg.split("=", 1)[1]) for arg in args if arg.startswith("--root=")), root)
            # A dataset-only worker also respects the other dataset's resume
            # backlog, so print its failures when they are the blocking cause.
            plans = [root / "experiments" / f"llama31-8b-{name}.json" for name in ("math", "mbpp")]
            failure_footer(root, plans, "LLAMA")
            admission_failure_footer(error, root)
            raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode if error.returncode > 0 else 128 - error.returncode) from None
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
        print(f"LLAMA refused: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None
