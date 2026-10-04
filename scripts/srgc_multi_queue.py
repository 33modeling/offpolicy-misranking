"""One worker serves several plans: it takes a task from the first queue that has one.

``run_srgc.sh all run`` starts a worker on the MATH plan with the MBPP plan as
a second queue. Each claim pass asks the queues in order, so MATH work is
preferred and MBPP fills the node when MATH has nothing claimable (or is
finished). Dependencies, leases, attempts, receipts and the hashed experiment
package are untouched: this module manages multi-queue scheduling and receipts.
"""

from contextlib import ExitStack, contextmanager
import json
import os
import socket
import time
import uuid

from srgc_rebuttal.runtime import atomic_json


@contextmanager
def available_devices(queues, args, environment, base, update):
    """Wait before admission without claiming work or disturbing the GPU owner."""
    from srgc_rebuttal import cluster
    while True:
        if all(cluster.stop_requested(q) or all(r["status"] == "complete" for r in q.status()) for q in queues):
            update("stopped" if any(cluster.stop_requested(q) for q in queues) else "complete")
            yield None
            return
        stack = ExitStack()
        try:
            devices, uuids = cluster.gpu_identity()
            lock_root = args.node_lock_root or queues[0].root.parent / "gpu-node-locks"
            fds = stack.enter_context(cluster.device_leases(lock_root, uuids))
        except cluster.Busy as exc:
            stack.close()
            update("idle", error=str(exc))
            print(f"NODE idle (GPU allocation already owned): {exc}", flush=True)
            time.sleep(min(args.poll_seconds, 30))
            continue
        except BaseException:
            stack.close()
            raise
        with stack:
            environment["CUDA_VISIBLE_DEVICES"] = devices
            base.update(devices=devices, gpu_uuids=uuids)
            yield fds
        return


def report_failure(queue, task, code):
    from scripts.srgc_log_tail import tail_lines
    receipt = json.loads(queue.receipt(task).read_text())
    log = queue.directory / "logs" / f"{task.key}.log"
    print(f"FAILED {queue.plan['dataset']}:{task.key} attempt={receipt['attempt']} "
          f"exit={code} validation_error={receipt.get('validation_error') or '-'} log={log}", flush=True)
    try:
        lines = tail_lines(log, 40)
    except OSError:
        lines = ["(task log not readable)"]
    for line in lines:
        print(f"FAILED {task.key} | {line}", flush=True)


def worker_multi(args, extra_plans):
    """Admit one GPU allocation and keep a truthful worker receipt in each queue."""
    from srgc_rebuttal import cluster
    queues = [cluster.TaskQueue(plan) for plan in (args.plan, *extra_plans)]
    if len({q.plan_path for q in queues}) != len(queues):
        raise ValueError("a multi-queue worker requires distinct plans")
    for queue in queues:
        queue.bind()
    primary = queues[0]
    primary._worker_queues = queues
    primary._active_queue = None
    worker_id = args.worker_id or uuid.uuid4().hex
    if not worker_id.isalnum():
        raise ValueError("worker id must be alphanumeric")
    base = {"worker_id": worker_id, "host": socket.gethostname(), "pid": os.getpid(),
            "started": time.time()}

    def update(state, task=None, child_pid=None, error=None):
        active = primary._active_queue
        for queue in queues:
            actual = active is None or queue.plan_path == active.plan_path
            atomic_json(queue.directory / "workers" / f"{worker_id}.json", {
                **base, "dataset": queue.plan["dataset"], "heartbeat": time.time(),
                "status": state if actual or state != "running" else "idle",
                "task": task.key if task is not None and actual else None,
                "active_plan": str(active.plan_path) if active else None,
                "active_dataset": active.plan["dataset"] if active else None,
                "active_task": task.key if task is not None else None,
                "child_pid": child_pid, "error": error})

    update("preflight")
    try:
        # A stop belongs to its dataset. It must not hide runnable work in the
        # other queue, including when the preferred dataset was already stopped.
        if all(cluster.stop_requested(q) or all(r["status"] == "complete" for r in q.status()) for q in queues):
            for queue in queues:
                if not cluster.stop_requested(queue):
                    cluster.publish_reports(queue)
            update("stopped" if any(cluster.stop_requested(q) for q in queues) else "complete")
            return
        environment = cluster.child_environment()
        with available_devices(queues, args, environment, base, update) as gpu_fds:
            if gpu_fds is None:
                return
            base["admission"] = cluster.admit(primary.directory / "admission" / worker_id,
                environment, cluster.run_child, pass_fds=gpu_fds, plan=primary.plan,
                heartbeat=lambda pid: update("preflight", child_pid=pid),
                should_stop=lambda: all(cluster.stop_requested(q) for q in queues))
            update("idle")
            # Retain wrappers installed by the checkpoint/status entry point.
            cluster.run_worker(primary, args, environment, gpu_fds, worker_id, update)
    except BaseException as exc:
        update("stopped" if isinstance(exc, KeyboardInterrupt) else "failed",
               error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        primary._active_queue = None


def run_worker_multi(queues, args, environment, gpu_fds, worker_id, update):
    from srgc_rebuttal import cluster
    published = set()
    queues[0]._worker_queues = queues
    while True:
        claimed, finished, blocked = False, 0, 0
        for queue in queues:
            if cluster.stop_requested(queue):
                finished += 1
                continue
            with queue.claim(retry_failed=args.retry_failed, max_attempts=args.max_attempts,
                             retry_delay=args.retry_delay, worker_id=worker_id) as task:
                if task is None:
                    status = queue.status(max_attempts=args.max_attempts)
                    if all(row["status"] == "complete" for row in status):
                        if queue.plan_path not in published:
                            cluster.publish_reports(queue)
                            published.add(queue.plan_path)
                        finished += 1
                    else:
                        retryable = args.retry_failed and any(row["status"] == "failed" and
                            row.get("retry_attempt", row.get("attempt", 0)) < args.max_attempts for row in status)
                        if not retryable and not any(row["status"] in {"running", "ready", "recoverable", "interrupted"}
                                                     for row in status):
                            blocked += 1
                    continue
                claimed = True
                queues[0]._active_queue = queue
                print(f"RUN {queue.plan['dataset']}:{task.key}", flush=True)
                try:
                    cluster.gpu_identity()
                    receipt = json.loads(queue.receipt(task).read_text())
                    progress_dir = queue.directory / "progress" / receipt["attempt_id"]
                    code = cluster.run_child(cluster.task_command(queue, task),
                        queue.directory / "logs" / f"{task.key}.log",
                        {**environment, "SRGC_PROGRESS_DIR": str(progress_dir)}, pass_fds=(*gpu_fds, *queue.claim_fds),
                        heartbeat=lambda pid: update("running", task, pid),
                        should_stop=lambda: cluster.stop_requested(queue, immediate=True),
                        interval=args.heartbeat_seconds,
                        progress=lambda: cluster.progress_signature(progress_dir),
                        stall_seconds=getattr(args, "stall_seconds", 1800))
                except KeyboardInterrupt:
                    queue.finish(task, 130, interrupted=True)
                    update("stopped", task)
                    raise
                except TimeoutError as exc:
                    # run_child has already terminated its process group. Keep
                    # this worker alive so other tasks and bounded retries run.
                    code = 124
                    log = queue.directory / "logs" / f"{task.key}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open("a") as handle:
                        handle.write(f"\nTimeoutError: {exc}\n")
                except BaseException:
                    queue.finish(task, 1)
                    raise
                code = queue.finish(task, code)
                print(f"DONE {queue.plan['dataset']}:{task.key} exit={code}", flush=True)
                if code:
                    report_failure(queue, task, code)
                queues[0]._active_queue = None
                update("idle")
            if claimed:
                break  # re-scan from the first queue so the preferred plan keeps priority
        if finished == len(queues):
            update("stopped" if any(cluster.stop_requested(q) for q in queues) else "complete")
            return
        if blocked and finished + blocked == len(queues):
            raise RuntimeError("failed task blocks remaining work; inspect logs before retry")
        if not claimed:
            update("idle")
            time.sleep(min(args.poll_seconds, 30))


@contextmanager
def multi_queue(extra_plans):
    """Use the same ownership-aware admission for one or several queues."""
    from srgc_rebuttal import cluster
    original = cluster.run_worker
    original_entry = cluster.worker
    extra_plans = [plan for plan in extra_plans]

    def run_worker(queue, args, environment, gpu_fds, worker_id, update):
        if hasattr(queue, "_worker_queues"):
            return run_worker_multi(queue._worker_queues, args, environment, gpu_fds, worker_id, update)
        queues = [queue]
        for plan in extra_plans:
            extra = cluster.TaskQueue(plan)
            extra.bind()
            queues.append(extra)
        return run_worker_multi(queues, args, environment, gpu_fds, worker_id, update)

    cluster.run_worker = run_worker
    cluster.worker = lambda args: worker_multi(args, extra_plans)
    try:
        yield
    finally:
        cluster.run_worker = original
        cluster.worker = original_entry
