"""Shared-filesystem queue: one four-GPU worker per node, prefix then independent arms."""

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

from .plan import DEFAULT_PLAN, input_path, load_plan, validate_inputs
from .runtime import (Busy, arm_complete, atomic_json, finalize_seed, identity, lease,
                      prefix_ready, run_root)


@dataclass(frozen=True)
class Task:
    seed: int
    arm: str

    @property
    def key(self):
        return f"seed-{self.seed}.{self.arm}"


class TaskQueue:
    def __init__(self, plan_path: Path):
        self.plan_path = plan_path.resolve()
        self.plan = load_plan(self.plan_path)
        self.root = run_root(self.plan_path, self.plan)
        self.directory = self.root / ".queue"
        self.identities = {s: identity(self.plan_path, self.plan, s) for s in self.plan["seeds"]}
        for seed in self.plan["seeds"]:
            validate_inputs(json.loads(input_path(self.plan_path, self.plan, seed).read_text()))
        self.tasks = [Task(s, "prefix") for s in self.plan["seeds"]]
        # Start the costly continuations early, without changing any learning rule.
        self.tasks += [Task(s, arm) for arm in ("on_policy", "switch", "sr", "random")
                       for s in self.plan["seeds"]]

    def bind(self):
        record = {"schema": "srgc-shared-queue-v1",
                  "identities": {str(s): value for s, value in self.identities.items()}}
        with lease(self.directory / "bind.lock", wait=True):
            path = self.directory / "protocol.json"
            if path.exists() and json.loads(path.read_text()) != record:
                raise ValueError("shared queue has different code, plan or input files")
            if not path.exists():
                atomic_json(path, record)

    def verify(self):
        if any(identity(self.plan_path, self.plan, s) != expected for s, expected in self.identities.items()):
            raise ValueError("code, plan or inputs changed after queue initialization")

    def complete(self, task):
        folder = self.root / f"seed-{task.seed}"
        if task.arm == "prefix":
            return prefix_ready(folder, self.identities[task.seed], self.plan["shared_prefix_updates"])
        return arm_complete(folder, self.identities[task.seed], task.arm, self.plan["total_updates"])

    def ready(self, task):
        return task.arm == "prefix" or self.complete(Task(task.seed, "prefix"))

    def receipt(self, task):
        return self.directory / "tasks" / f"{task.key}.json"

    @contextmanager
    def claim(self, *, retry_failed=False):
        self.verify()
        for task in self.tasks:
            if self.complete(task) or not self.ready(task):
                continue
            path = self.receipt(task)
            if path.exists() and json.loads(path.read_text()).get("status") == "failed" and not retry_failed:
                continue
            with ExitStack() as locks:
                try:
                    locks.enter_context(lease(self.directory / "leases" / f"{task.key}.lock"))
                except Busy:
                    continue
                else:
                    if self.complete(task):
                        continue
                    if path.exists() and json.loads(path.read_text()).get("status") == "failed" and not retry_failed:
                        continue
                    atomic_json(path, {"task": task.key, "status": "running", "host": socket.gethostname(),
                                       "pid": os.getpid(), "started": time.time()})
                    yield task
                    return
        yield None

    def finish(self, task, exit_code):
        if exit_code == 0 and not self.complete(task):
            exit_code = 2
        previous = json.loads(self.receipt(task).read_text()) if self.receipt(task).exists() else {}
        atomic_json(self.receipt(task), {**previous, "task": task.key,
            "status": "complete" if exit_code == 0 else "failed", "exit_code": exit_code,
            "host": socket.gethostname(), "finished": time.time()})
        return exit_code

    def status(self):
        rows = []
        for task in self.tasks:
            if self.complete(task):
                state = "complete"
            elif not self.ready(task):
                state = "waiting_for_prefix"
            elif self.receipt(task).exists():
                state = json.loads(self.receipt(task).read_text())["status"]
            else:
                state = "ready"
            rows.append({"task": task.key, "status": state})
        return rows


def child_environment():
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "OPENBLAS_DEFAULT_NUM_THREADS", "GOTO_NUM_THREADS", "BLIS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS",
                 "OMP_THREAD_LIMIT", "RAYON_NUM_THREADS"):
        env[name] = "1"
    env.update(TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def gpu_identity():
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    if len(devices.split(",")) != 4 or len(set(devices.split(","))) != 4:
        raise ValueError("each worker needs four distinct allocated GPUs")
    result = subprocess.run(["nvidia-smi", f"--id={devices}", "--query-gpu=uuid,memory.used",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=20)
    rows = [line.split(",") for line in result.stdout.strip().splitlines()]
    if len(rows) != 4 or any(int(row[1]) > 4000 for row in rows):
        raise Busy("allocated GPUs are busy; no existing process was stopped")
    key = hashlib.sha256(",".join(sorted(row[0].strip() for row in rows)).encode()).hexdigest()[:24]
    return devices, key


def run_child(command, log_path, environment):
    """Keep the real child process group; release leases only after it exits."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        old_handlers = {}
        def stop(signum, frame):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            raise KeyboardInterrupt
        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                old_handlers[signum] = signal.signal(signum, stop)
            return process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def worker(args):
    queue = TaskQueue(args.plan)
    queue.bind()
    devices, node_key = gpu_identity()
    lock_root = args.node_lock_root or queue.root.parent / "gpu-node-locks"
    environment = child_environment()
    environment["CUDA_VISIBLE_DEVICES"] = devices
    # GPU UUIDs identify the same devices even across container hostnames.
    with lease(lock_root / f"{node_key}.lock"):
        while True:
            with queue.claim(retry_failed=args.retry_failed) as task:
                if task is None:
                    status = queue.status()
                    if all(row["status"] == "complete" for row in status):
                        for seed in queue.plan["seeds"]:
                            finalize_seed(queue.root / f"seed-{seed}", queue.identities[seed],
                                          queue.plan["arms"], queue.plan["total_updates"])
                        return
                    if any(row["status"] == "failed" for row in status) and not any(
                            row["status"] in {"running", "ready"} for row in status):
                        raise RuntimeError("failed task blocks remaining work; inspect logs before retry")
                else:
                    gpu_identity()  # Re-admit before each child; never kill unrelated jobs.
                    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                        "--nproc_per_node=4", "-m", "srgc_rebuttal.run_experiment",
                        "--plan", str(queue.plan_path), "--seed", str(task.seed), "--task", task.arm, "--resume"]
                    print(f"RUN {task.key}", flush=True)
                    try:
                        code = run_child(command, queue.directory / "logs" / f"{task.key}.log", environment)
                    except BaseException:
                        queue.finish(task, 130)
                        raise
                    code = queue.finish(task, code)
                    print(f"DONE {task.key} exit={code}", flush=True)
                    if code:
                        raise RuntimeError("task failed; checkpoint and log retained")
            if task is None:
                time.sleep(min(args.poll_seconds, 30))


def ssh_command(host, repo, python, plan):
    if not host or host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError("invalid SSH host")
    command = [python, "-m", "srgc_rebuttal.cluster", "worker", "--plan", plan]
    # Each argument is shell-quoted once; no user string becomes shell syntax.
    remote = (f"cd {shlex.quote(repo)} && mkdir -p .rebuttal-worker-logs && "
              f"nohup {shlex.join(command)} >> .rebuttal-worker-logs/worker.log 2>&1 < /dev/null &")
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
            shlex.join(["bash", "-lc", remote])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("worker", "status"):
        p = sub.add_parser(name)
        p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
        if name == "worker":
            p.add_argument("--poll-seconds", type=float, default=10)
            p.add_argument("--retry-failed", action="store_true")
            p.add_argument("--node-lock-root", type=Path)
    for name in ("commands", "launch"):
        p = sub.add_parser(name)
        p.add_argument("--hosts", nargs="+", required=True)
        p.add_argument("--repo", required=True)
        p.add_argument("--python", default="python3")
        p.add_argument("--plan", default="srgc_rebuttal/experiments/additional_seeds.json")
    args = parser.parse_args()
    if args.action == "worker":
        if args.poll_seconds <= 0:
            parser.error("poll interval must be positive")
        worker(args)
    elif args.action == "status":
        print(json.dumps(TaskQueue(args.plan).status(), indent=2))
    else:
        if len(set(args.hosts)) != len(args.hosts):
            parser.error("hosts must be distinct")
        commands = [ssh_command(host, args.repo, args.python, args.plan) for host in args.hosts]
        if args.action == "commands":
            for command in commands:
                print(shlex.join(command))
        else:
            queue = TaskQueue(Path(args.plan))
            queue.bind()  # Validate locally before starting remote workers.
            children = [subprocess.Popen(command) for command in commands]
            codes = [p.wait(timeout=30) for p in children]
            if any(codes):
                raise SystemExit("one or more worker launches failed; inspect node logs")


if __name__ == "__main__":
    main()
