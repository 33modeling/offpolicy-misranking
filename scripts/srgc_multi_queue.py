"""One worker serves several plans: it takes a task from the first queue that has one.

``run_srgc.sh all run`` starts a worker on the MATH plan with the MBPP plan as
a second queue. Each claim pass asks the queues in order, so MATH work is
preferred and MBPP fills the node when MATH has nothing claimable (or is
finished). Dependencies, leases, attempts, receipts and the hashed experiment
package are untouched: this module only replaces the worker's claim loop.
"""

from contextlib import contextmanager
import json
import time


def run_worker_multi(queues, args, environment, gpu_fds, worker_id, update):
    from srgc_rebuttal import cluster
    published = set()
    queues[0]._worker_queues = queues
    while True:
        if any(cluster.stop_requested(queue) for queue in queues):
            update("stopped")
            return
        claimed, finished, blocked = False, 0, 0
        for queue in queues:
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
                            row.get("attempt", 0) < args.max_attempts for row in status)
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
                    return
                except BaseException:
                    queue.finish(task, 130)
                    raise
                code = queue.finish(task, code)
                print(f"DONE {queue.plan['dataset']}:{task.key} exit={code}", flush=True)
                queues[0]._active_queue = None
                update("idle")
            if claimed:
                break  # re-scan from the first queue so the preferred plan keeps priority
        if finished == len(queues):
            update("complete")
            return
        if blocked and finished + blocked == len(queues):
            raise RuntimeError("failed task blocks remaining work; inspect logs before retry")
        if not claimed:
            update("idle")
            time.sleep(min(args.poll_seconds, 30))


@contextmanager
def multi_queue(extra_plans):
    """Patch ``cluster.run_worker`` so the worker also drains ``extra_plans`` (in order)."""
    from srgc_rebuttal import cluster
    original = cluster.run_worker
    extra_plans = [plan for plan in extra_plans]

    def run_worker(queue, args, environment, gpu_fds, worker_id, update):
        queues = [queue]
        for plan in extra_plans:
            extra = cluster.TaskQueue(plan)
            extra.bind()
            queues.append(extra)
        return run_worker_multi(queues, args, environment, gpu_fds, worker_id, update)

    cluster.run_worker = run_worker
    try:
        yield
    finally:
        cluster.run_worker = original
