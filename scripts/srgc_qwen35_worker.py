"""Qwen workers: shared leases, dataset-correct receipts and runtime consistency."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import socket
import time
import uuid

from srgc_rebuttal.runtime import atomic_json, lease


def bind_runtime(queues, signature):
    for queue in queues:
        path = queue.directory / "runtime.json"
        with lease(path.with_suffix(".lock"), wait=True):
            if path.exists():
                if json.loads(path.read_text()) != signature:
                    raise ValueError(f"node runtime differs from this experiment: {path}")
            else:
                atomic_json(path, signature)


def runtime_signature():
    import platform
    import torch
    from srgc_qwen35 import runtime_packages
    return {"packages": runtime_packages(), "python": platform.python_version(),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


def write_worker(queues, worker_id, state, *, active=None, task=None, **details):
    for queue in queues:
        actual = active is None or queue.plan_path == active.plan_path
        atomic_json(queue.directory / "workers" / f"{worker_id}.json", {
            "worker_id": worker_id, "host": socket.gethostname(), "pid": os.getpid(),
            "dataset": queue.plan["dataset"], "heartbeat": time.time(),
            "status": state if actual or state != "running" else "idle",
            "task": task.key if task is not None and actual else None,
            "active_dataset": active.plan["dataset"] if active else None,
            "active_plan": str(active.plan_path) if active else None, **details})


def drain(queues, args, environment, gpu_fds, worker_id, update):
    """A failed/stopped dataset cannot make another dataset's receipts ambiguous."""
    from srgc_rebuttal import cluster
    published = set()
    while True:
        claimed, ended, blocked = False, 0, 0
        for queue in queues:
            if cluster.stop_requested(queue):
                ended += 1
                continue
            with queue.claim(retry_failed=args.retry_failed, max_attempts=args.max_attempts,
                             retry_delay=args.retry_delay, worker_id=worker_id) as task:
                if task is None:
                    status = queue.status(max_attempts=args.max_attempts)
                    if all(row["status"] == "complete" for row in status):
                        if queue.plan_path not in published:
                            cluster.publish_reports(queue)
                            published.add(queue.plan_path)
                        ended += 1
                    else:
                        retryable = args.retry_failed and any(row["status"] == "failed" and
                            row.get("attempt", 0) < args.max_attempts for row in status)
                        if not retryable and not any(row["status"] in {"running", "ready", "recoverable", "interrupted"}
                                                    for row in status):
                            blocked += 1
                    continue
                claimed = True
                label = f"{queue.plan['dataset']}:{task.key}"
                log = queue.directory / "logs" / f"{task.key}.log"
                print(f"RUN {label} log={log}", flush=True)
                try:
                    cluster.gpu_identity()
                    receipt = json.loads(queue.receipt(task).read_text())
                    progress_dir = queue.directory / "progress" / receipt["attempt_id"]
                    code = cluster.run_child(cluster.task_command(queue, task), log,
                        {**environment, "SRGC_PROGRESS_DIR": str(progress_dir)},
                        pass_fds=(*gpu_fds, *queue.claim_fds),
                        heartbeat=lambda pid: update("running", active=queue, task=task, child_pid=pid),
                        should_stop=lambda: cluster.stop_requested(queue, immediate=True),
                        interval=args.heartbeat_seconds, progress=lambda: cluster.progress_signature(progress_dir),
                        stall_seconds=args.stall_seconds)
                except KeyboardInterrupt:
                    queue.finish(task, 130, interrupted=True)
                    update("stopped", active=queue, task=task)
                    return
                except BaseException:
                    queue.finish(task, 130)
                    raise
                code = queue.finish(task, code)
                print(f"DONE {label} exit={code}", flush=True)
                update("idle")
            if claimed:
                break
        if ended == len(queues):
            update("stopped" if any(cluster.stop_requested(q) for q in queues) else "complete")
            return
        if blocked and ended + blocked == len(queues):
            raise RuntimeError("failed task blocks remaining work; inspect dataset-specific task logs")
        if not claimed:
            update("idle")
            time.sleep(min(args.poll_seconds, 30))


def worker(plans, args, group, common_root, admission):
    from srgc_rebuttal import cluster
    from srgc_rebuttal.cluster_queue import TaskQueue
    from srgc_seed_order import seed_first
    queues = [TaskQueue(plan) for plan in plans]
    for queue in queues:
        queue.bind()
        queue.tasks = seed_first(queue.tasks, queue.plan["seeds"])
    worker_id = uuid.uuid4().hex
    metadata = {"started": time.time()}

    def update(state, **details):
        write_worker(queues, worker_id, state, **metadata, **details)

    update("preflight")
    try:
        if all(cluster.stop_requested(q) or all(r["status"] == "complete" for r in q.status()) for q in queues):
            update("stopped" if any(cluster.stop_requested(q) for q in queues) else "complete")
            return
        environment = cluster.child_environment()
        environment["SRGC_QWEN_PLAN"] = str(plans[0])
        environment["SRGC_QWEN_PLANS"] = json.dumps([str(p) for p in plans])
        devices, uuids = cluster.gpu_identity()
        environment["CUDA_VISIBLE_DEVICES"] = devices
        metadata.update(devices=devices, gpu_uuids=uuids)
        with ExitStack() as stack:
            # A canonical lock also prevents two Qwen roots from bypassing each
            # other. Keep the legacy namespace to interoperate with OLMo workers.
            roots = {Path(common_root) / "gpu-node-locks", Path(group) / ".srgc-gpu-node-locks"}
            fds = tuple(fd for root in sorted(roots) for fd in stack.enter_context(cluster.device_leases(root, uuids)))
            bind_runtime(queues, runtime_signature())
            metadata["admission"] = admission(cluster.admit,
                queues[0].directory / "admission" / worker_id, environment, cluster.run_child,
                pass_fds=fds, plan=queues[0].plan,
                heartbeat=lambda pid: update("preflight", child_pid=pid),
                should_stop=lambda: all(cluster.stop_requested(q) for q in queues))
            drain(queues, args, environment, fds, worker_id, update)
    except BaseException as exc:
        update("stopped" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}")
        raise
