"""Multi-node cache/prefix/arm execution with device leases and worker receipts."""

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

from .plan import DEFAULT_PLAN, input_path, load_plan
from .runtime import Busy, atomic_json, finalize_seed, lease, run_root
from .cluster_queue import Task, TaskQueue
from .admission import admit
from .existing_runtime import runtime_packages
from .progress import signature as progress_signature


def child_environment():
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "OPENBLAS_DEFAULT_NUM_THREADS", "GOTO_NUM_THREADS", "BLIS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS",
                 "OMP_THREAD_LIMIT", "RAYON_NUM_THREADS"):
        env[name] = "1"
    env.update(TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    repo = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo), str(repo / "src"), env.get("PYTHONPATH"))))
    return env


def gpu_identity():
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    if len(devices.split(",")) != 4 or len(set(devices.split(","))) != 4 or not all(devices.split(",")):
        raise ValueError("each worker needs four distinct allocated GPUs")
    result = subprocess.run(["nvidia-smi", f"--id={devices}", "--query-gpu=uuid,memory.used,name,memory.total",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=20)
    rows = [line.split(",") for line in result.stdout.strip().splitlines()]
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("nvidia-smi did not return four valid GPU records")
    if any("H100" not in row[2] or int(row[3]) < 75000 for row in rows):
        raise ValueError("this allocation requires four full H100 GPUs with at least 75,000 MiB each")
    if any(int(row[1]) > 4000 for row in rows):
        raise Busy("allocated GPUs are busy; no existing process was stopped")
    uuids = tuple(sorted(row[0].strip() for row in rows))
    if len(set(uuids)) != 4:
        raise ValueError("allocated GPU identifiers alias the same device")
    return devices, uuids


@contextmanager
def device_leases(root, uuids):
    # Lock each UUID, not just the set: partially overlapping allocations conflict too.
    with ExitStack() as stack:
        handles = [stack.enter_context(lease(root / f"{hashlib.sha256(u.encode()).hexdigest()}.lock"))
                   for u in sorted(uuids)]
        yield tuple(h.fileno() for h in handles)


def run_child(command, log_path, environment, *, pass_fds=(), heartbeat=lambda pid: None,
              should_stop=lambda: False, interval=5, timeout=None, progress=None, stall_seconds=1800):
    """Keep the real child process group; release leases only after it exits."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log, log_path.open("r", errors="replace") as reader:
        reader.seek(0, os.SEEK_END)
        def relay():
            while chunk := reader.read(65536):
                sys.stdout.write(chunk)
                sys.stdout.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True, pass_fds=pass_fds)
        old_handlers = {}
        def terminate_group(sig):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        def stop(signum, frame):
            if process.poll() is None:
                terminate_group(signal.SIGTERM)
            raise KeyboardInterrupt
        started = changed = time.monotonic()
        previous = progress() if progress else None
        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                old_handlers[signum] = signal.signal(signum, stop)
            while True:
                relay()
                heartbeat(process.pid)
                if should_stop():
                    raise KeyboardInterrupt
                now = time.monotonic()
                if timeout is not None and now - started > timeout:
                    raise TimeoutError(f"child exceeded {timeout}s; log: {log_path}")
                if progress:
                    current = progress()
                    if current != previous:
                        changed, previous = now, current
                    if now - changed > stall_seconds:
                        raise TimeoutError(f"no task progress for {stall_seconds}s; log: {log_path}")
                try:
                    return process.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # Ranks can outlive their launcher. Tear down our session even after
            # the leader exits, before releasing task/device leases.
            terminate_group(signal.SIGTERM)
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    terminate_group(signal.SIGKILL)
                    process.wait()
            terminate_group(signal.SIGKILL)
            relay()
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def task_command(queue, task):
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "--max_restarts=0", "-m"]
    if task.arm == "cache":
        return [*command, "srgc_rebuttal.build_cache", "--plan", str(queue.plan_path),
                "--bundle", str(input_path(queue.plan_path, queue.plan, task.seed)),
                "--cache-seed", str(task.seed), "--max-new-tokens", str(queue.plan["max_new_tokens"])]
    return [*command, "srgc_rebuttal.run_experiment", "--plan", str(queue.plan_path),
            "--seed", str(task.seed), "--task", task.arm, "--resume"]


def direct(action, plan_path, arguments):
    """Manual run/cache uses the same device admission and teardown as a worker."""
    plan = load_plan(plan_path)
    root = run_root(plan_path, plan)
    devices, uuids = gpu_identity()
    environment = child_environment()
    environment["CUDA_VISIBLE_DEVICES"] = devices
    session = root / ".queue" / "admission" / f"manual-{uuid.uuid4().hex}"
    with device_leases(root.parent / "gpu-node-locks", uuids) as fds:
        admit(session, environment, run_child, pass_fds=fds, plan=plan)
        progress_dir = session / "progress"
        environment["SRGC_PROGRESS_DIR"] = str(progress_dir)
        module = "run_experiment" if action == "run" else "build_cache"
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                   "--max_restarts=0", "-m", f"srgc_rebuttal.{module}", *arguments]
        print(f"ADMISSION PASSED; task log: {session / 'task.log'}", flush=True)
        return run_child(command, session / "task.log", environment, pass_fds=fds,
                         progress=lambda: progress_signature(progress_dir))


def publish_reports(queue):
    from .cost_report import compare
    from .summarize import summarize
    with lease(queue.directory / "analysis.lock", wait=True):
        for seed in queue.plan["seeds"]:
            finalize_seed(queue.root / f"seed-{seed}", queue.identities[seed],
                          queue.plan["arms"], queue.plan["total_updates"])
        atomic_json(queue.root / "cost-comparison.json", compare(queue.plan_path))
        atomic_json(queue.root / "results-summary.json", summarize(queue.plan_path))


def stop_requested(queue, *, immediate=False):
    path = queue.directory / "stop.json"
    try:
        record = json.loads(path.read_text())
        return not immediate or record["immediate"]
    except FileNotFoundError:
        return False  # A concurrent resume removed the stop receipt.


def worker(args):
    queue = TaskQueue(args.plan)
    queue.bind()
    lock_root = args.node_lock_root or queue.root.parent / "gpu-node-locks"
    environment = child_environment()
    worker_id = args.worker_id or uuid.uuid4().hex
    if not worker_id.isalnum():
        raise ValueError("worker id must be alphanumeric")
    path = queue.directory / "workers" / f"{worker_id}.json"
    base = {"worker_id": worker_id, "host": socket.gethostname(), "pid": os.getpid(),
            "started": time.time(), "dataset": queue.plan["dataset"]}
    def update(state, task=None, child_pid=None, error=None):
        atomic_json(path, {**base, "status": state, "task": task.key if task else None,
                          "child_pid": child_pid, "heartbeat": time.time(), "error": error})
    update("preflight")
    try:
        if stop_requested(queue):
            update("stopped")
            return
        devices, uuids = gpu_identity()
        base.update(devices=devices, gpu_uuids=uuids)
        environment["CUDA_VISIBLE_DEVICES"] = devices
        with device_leases(lock_root, uuids) as gpu_fds:
            base["admission"] = admit(queue.directory / "admission" / worker_id, environment, run_child,
                pass_fds=gpu_fds, plan=queue.plan, heartbeat=lambda pid: update("preflight", child_pid=pid),
                should_stop=lambda: stop_requested(queue))
            update("idle")
            run_worker(queue, args, environment, gpu_fds, worker_id, update)
    except BaseException as exc:
        update("stopped" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}")
        raise


def run_worker(queue, args, environment, gpu_fds, worker_id, update):
    while True:
        if stop_requested(queue):
            update("stopped")
            return
        with queue.claim(retry_failed=args.retry_failed, max_attempts=args.max_attempts,
                         retry_delay=args.retry_delay, worker_id=worker_id) as task:
            if task is None:
                status = queue.status(max_attempts=args.max_attempts)
                if all(row["status"] == "complete" for row in status):
                    publish_reports(queue)
                    update("complete")
                    return
                retryable = args.retry_failed and any(row["status"] == "failed" and
                    row.get("attempt", 0) < args.max_attempts for row in status)
                if not retryable and not any(row["status"] in {"running", "ready", "recoverable", "interrupted"}
                                             for row in status):
                    raise RuntimeError("failed task blocks remaining work; inspect logs before retry")
                update("idle")
            else:
                print(f"RUN {task.key}", flush=True)
                try:
                    gpu_identity()
                    receipt = json.loads(queue.receipt(task).read_text())
                    progress_dir = queue.directory / "progress" / receipt["attempt_id"]
                    code = run_child(task_command(queue, task), queue.directory / "logs" / f"{task.key}.log",
                        {**environment, "SRGC_PROGRESS_DIR": str(progress_dir)}, pass_fds=(*gpu_fds, *queue.claim_fds),
                        heartbeat=lambda pid: update("running", task, pid),
                        should_stop=lambda: stop_requested(queue, immediate=True), interval=args.heartbeat_seconds,
                        progress=lambda: progress_signature(progress_dir), stall_seconds=getattr(args, "stall_seconds", 1800))
                except KeyboardInterrupt:
                    queue.finish(task, 130, interrupted=True)
                    update("stopped", task)
                    return
                except BaseException:
                    queue.finish(task, 130)
                    raise
                code = queue.finish(task, code)
                print(f"DONE {task.key} exit={code}", flush=True)
                update("idle")
        if task is None:
            time.sleep(min(args.poll_seconds, 30))


def ssh_command(host, repo, python, plan, *, action="worker", options=(), worker_id=None):
    if not host or host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError("invalid SSH host")
    command = [python, str(Path(repo) / "scripts/run_srgc_rebuttal.py"),
               "cluster", action, "--plan", str(plan), *options]
    if worker_id:
        command += ["--worker-id", worker_id]
    # Each argument is shell-quoted once; no user string becomes shell syntax.
    if action == "worker":
        log = f".rebuttal-worker-logs/{worker_id or 'worker'}.log"
        remote = (f"cd {shlex.quote(repo)} && mkdir -p .rebuttal-worker-logs && "
                  f"{{ nohup {shlex.join(command)} >> {shlex.quote(log)} 2>&1 < /dev/null & echo $!; }}")
    else:
        remote = f"cd {shlex.quote(repo)} && {shlex.join(command)}"
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
            shlex.join(["bash", "-lc", remote])]


def probe(queue, token):
    if not token.isalnum():
        raise ValueError("invalid preflight token")
    path = queue.directory / "preflight" / token
    if not path.with_suffix(".json").exists():
        raise ValueError("node cannot see the coordinator's shared queue")
    try:
        with lease(path.with_suffix(".lock")):
            raise ValueError("shared filesystem does not enforce the coordinator's lock")
    except Busy:
        pass
    devices, uuids = gpu_identity()
    lock_root = queue.root.parent / "gpu-node-locks"
    with device_leases(lock_root, uuids):
        pass
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise ValueError("PyTorch cannot access four allocated CUDA devices")
    return {"host": socket.gethostname(), "devices": devices, "gpu_uuids": uuids,
            "gpu_models": sorted(torch.cuda.get_device_name(i) for i in range(4)),
            "python": list(sys.version_info[:2]),
            "packages": runtime_packages(),
            "code_verifier_environment": {k: os.environ.get(k) for k in ("SRGC_CODE_TIMEOUT", "SRGC_CODE_MEMORY_MB")}}


def remote_calls(commands, timeout=90):
    processes = []
    try:
        for command in commands:
            processes.append(subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        results = []
        for p in processes:
            stdout, stderr = p.communicate(timeout=timeout)
            if p.returncode:
                raise RuntimeError(f"remote command failed ({p.returncode}): {stderr[-4000:]}")
            results.append(stdout)
        return results
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.communicate()


def remote_plan(args):
    path = Path(args.plan)
    if path.is_absolute():
        try:
            path = path.relative_to(Path(__file__).resolve().parents[1])
        except ValueError:
            return str(path)
    return str(Path(args.repo) / path)


def launch(args):
    queue = TaskQueue(Path(args.plan))
    queue.bind()
    if stop_requested(queue):
        raise ValueError("queue is stopped; run cluster resume before launching workers")
    token = uuid.uuid4().hex
    challenge = queue.directory / "preflight" / token
    atomic_json(challenge.with_suffix(".json"), {"token": token})
    try:
        with lease(challenge.with_suffix(".lock")):
            results = remote_calls([ssh_command(h, args.repo, args.python, remote_plan(args), action="probe",
                options=["--probe-token", token]) for h in args.hosts])
        probes = [json.loads(r) for r in results]
        signature = lambda p: {k: p[k] for k in ("gpu_models", "packages", "python", "code_verifier_environment")}
        if any(signature(p) != signature(probes[0]) for p in probes):
            raise ValueError("nodes differ in GPU model, packages or verifier settings")
        devices = [u for p in probes for u in p["gpu_uuids"]]
        if len(set(devices)) != len(devices):
            raise ValueError("hosts alias overlapping physical GPU allocations")
        workers = {h: uuid.uuid4().hex for h in args.hosts}
        options = ["--max-attempts", str(args.max_attempts), "--retry-delay", str(args.retry_delay)]
        if args.retry_failed:
            options += ["--retry-failed"]
        remote_calls([ssh_command(h, args.repo, args.python, remote_plan(args), options=options, worker_id=w)
                      for h, w in workers.items()])
        deadline = time.monotonic() + args.startup_timeout
        while True:
            records = {h: json.loads((queue.directory / "workers" / f"{w}.json").read_text())
                       for h, w in workers.items() if (queue.directory / "workers" / f"{w}.json").exists()}
            failed = {h: r for h, r in records.items() if r["status"] in {"failed", "stopped"}}
            if failed:
                raise RuntimeError(f"worker startup failed: {failed}")
            if len(records) == len(workers) and all(r["status"] != "preflight" for r in records.values()):
                atomic_json(queue.directory / "launches" / f"{token}.json", {"workers": workers, "nodes": probes})
                print(json.dumps({"status": "workers acknowledged", "workers": workers, "nodes": probes}, indent=2))
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"workers did not acknowledge startup: {set(workers) - set(records)}; inspect .rebuttal-worker-logs; acknowledged workers continue")
            time.sleep(0.5)
    finally:
        challenge.with_suffix(".json").unlink(missing_ok=True)


def cluster_status(queue):
    tasks = queue.status()
    workers = []
    for path in sorted((queue.directory / "workers").glob("*.json")):
        row = json.loads(path.read_text())
        row["heartbeat_age_seconds"] = max(0, time.time() - row["heartbeat"])
        if row["status"] in {"running", "idle", "preflight"} and row["heartbeat_age_seconds"] > 60:
            row["status"] = "heartbeat_stale"
        workers.append(row)
    counts = {s: sum(r["status"] == s for r in tasks) for s in sorted({r["status"] for r in tasks})}
    return {"dataset": queue.plan["dataset"], "output_root": str(queue.root), "stop_requested": stop_requested(queue),
            "counts": counts, "tasks": tasks, "workers": workers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("worker", "status", "stop", "resume", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
        if name == "worker":
            p.add_argument("--poll-seconds", type=float, default=10)
            p.add_argument("--retry-failed", action="store_true")
            p.add_argument("--node-lock-root", type=Path)
            p.add_argument("--worker-id")
            p.add_argument("--heartbeat-seconds", type=float, default=5)
            p.add_argument("--max-attempts", type=int, default=3)
            p.add_argument("--retry-delay", type=float, default=60)
            p.add_argument("--stall-seconds", type=float, default=1800)
        elif name == "stop":
            p.add_argument("--now", action="store_true", help="interrupt owned child groups; otherwise drain current tasks")
        elif name == "probe":
            p.add_argument("--probe-token", required=True)
    for name in ("commands", "launch"):
        p = sub.add_parser(name)
        p.add_argument("--hosts", nargs="+", required=True)
        p.add_argument("--repo", required=True)
        p.add_argument("--python", default="python3")
        p.add_argument("--plan", default="srgc_rebuttal/experiments/additional_seeds.json")
        p.add_argument("--retry-failed", action="store_true")
        p.add_argument("--max-attempts", type=int, default=3)
        p.add_argument("--retry-delay", type=float, default=60)
        p.add_argument("--startup-timeout", type=float, default=660)
    args = parser.parse_args()
    if args.action == "worker":
        if (not all(math.isfinite(v) for v in (args.poll_seconds, args.heartbeat_seconds, args.retry_delay, args.stall_seconds))
                or args.poll_seconds <= 0 or args.heartbeat_seconds <= 0 or args.max_attempts < 1
                or args.retry_delay < 0 or args.stall_seconds <= 0):
            parser.error("intervals and attempts must be positive; retry delay must be nonnegative")
        worker(args)
    elif args.action == "status":
        print(json.dumps(cluster_status(TaskQueue(args.plan)), indent=2))
    elif args.action in {"stop", "resume", "probe"}:
        queue = TaskQueue(args.plan)
        queue.bind()
        if args.action == "stop":
            atomic_json(queue.directory / "stop.json", {"immediate": args.now, "requested": time.time()})
        elif args.action == "resume":
            (queue.directory / "stop.json").unlink(missing_ok=True)
        else:
            print(json.dumps(probe(queue, args.probe_token)))
    else:
        if len(set(args.hosts)) != len(args.hosts):
            parser.error("hosts must be distinct")
        options = ["--max-attempts", str(args.max_attempts), "--retry-delay", str(args.retry_delay)]
        if args.retry_failed:
            options += ["--retry-failed"]
        commands = [ssh_command(host, args.repo, args.python, remote_plan(args), options=options) for host in args.hosts]
        if args.action == "commands":
            for command in commands:
                print(shlex.join(command))
        else:
            if args.max_attempts < 1 or args.retry_delay < 0 or args.startup_timeout <= 0:
                parser.error("invalid launch retry or timeout settings")
            launch(args)


if __name__ == "__main__":
    main()
