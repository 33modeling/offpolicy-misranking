"""Run a sibling dataset's existing backlog on otherwise idle model workers."""

import copy
import json
import time
from contextlib import contextmanager
from unittest.mock import patch

from srgc_rebuttal.runtime import lease


def drain(worker, factory, pattern, env_key, label, requested, args, environment,
          gpu_fds, worker_id, update):
    from srgc_rebuttal import cluster

    cohort = list(requested)
    for name in ("math", "mbpp"):
        path = requested[0].plan_path.parent / pattern.format(dataset=name)
        if path.is_file() and all(q.plan_path.resolve() != path.resolve() for q in cohort):
            cohort.append(type(requested[0])(path))
    manager = factory(cohort)
    cohort = list(manager.queues.values())
    siblings = [q for q in cohort if q not in requested]
    for queue in siblings:
        queue.bind()
    runtime = requested[0].directory / "runtime.json"
    if siblings and runtime.is_file():
        worker.bind_runtime(cohort, json.loads(runtime.read_text()))
    snapshots = {q.plan.get("model_snapshot_sha256") for q in cohort}
    if len(snapshots) != 1:
        raise ValueError("sibling datasets use different local model snapshots")
    environment = {**environment, env_key: json.dumps([str(q.plan_path) for q in cohort])}
    metadata_path = requested[0].directory / "workers" / f"{worker_id}.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    excluded = {"worker_id", "host", "pid", "dataset", "heartbeat", "status", "task",
                "active_dataset", "active_plan", "idle_reason", "child_pid"}
    metadata = {key: value for key, value in metadata.items() if key not in excluded}

    def progress(state, **details):
        update(state, **details)
        if siblings:
            worker.write_worker(siblings, worker_id, state, **metadata, **details)

    published = set()
    while True:
        if all(cluster.stop_requested(q) for q in requested):
            progress("stopped")
            return
        pending = None
        claimed = False
        for queue in cohort:
            if cluster.stop_requested(queue):
                continue
            # A dataset-only command can rescue old sibling work, but cannot
            # open a new sibling seed or continuation after the backlog clears.
            tasks = queue.tasks
            if queue in siblings:
                if pending is None:
                    with lease(manager.marker.with_suffix(".lock"), wait=True):
                        pending = manager.pending()
                queue.tasks = [task for task in tasks
                               if (str(queue.plan_path.resolve()), task.key) in pending]
            try:
                if not queue.tasks:
                    continue
                with manager.claim(queue, retry_failed=True, max_attempts=args.max_attempts,
                                   retry_delay=args.retry_delay, worker_id=worker_id) as task:
                    if task is None:
                        continue
                    claimed = True
                    task_label = f"{label} {queue.plan['dataset']}:{task.key}"
                    log = queue.directory / "logs" / f"{task.key}.log"
                    print(f"RUN {task_label} log={log}", flush=True)
                    try:
                        cluster.gpu_identity()
                        row = manager.receipt(queue, task)
                        directory = queue.directory / "progress" / row["attempt_id"]
                        code = cluster.run_child(cluster.task_command(queue, task), log,
                            {**environment, "SRGC_PROGRESS_DIR": str(directory)},
                            pass_fds=(*gpu_fds, *queue.claim_fds),
                            heartbeat=lambda pid, active=queue, job=task: progress("running", active=active, task=job, child_pid=pid),
                            should_stop=lambda active=queue: cluster.stop_requested(active, immediate=True),
                            interval=args.heartbeat_seconds,
                            progress=lambda path=directory: cluster.progress_signature(path),
                            stall_seconds=args.stall_seconds)
                    except TimeoutError as error:
                        code = 124
                        log.parent.mkdir(parents=True, exist_ok=True)
                        with log.open("a") as handle:
                            handle.write(f"TIMEOUT: {error}\n")
                        print(f"TIMEOUT {task_label}: {error}", flush=True)
                    except KeyboardInterrupt:
                        queue.finish(task, 130, interrupted=True)
                        progress("stopped", active=queue, task=task)
                        raise
                    except BaseException:
                        queue.finish(task, 1)
                        raise
                    interrupted = code in (130, 143)
                    code = queue.finish(task, code, interrupted=interrupted)
                    print(f"DONE {task_label} exit={code}", flush=True)
                    if interrupted:
                        progress("stopped", active=queue, task=task)
                        return
                    if code:
                        from scripts.srgc_log_tail import tail_lines
                        try:
                            print("\n".join(tail_lines(log, 25)), flush=True)
                        except OSError:
                            pass
                    progress("idle")
            finally:
                queue.tasks = tasks
            if claimed:
                break
        if claimed:
            continue
        with lease(manager.marker.with_suffix(".lock"), wait=True):
            pending = manager.pending()
        complete = True
        for queue in requested:
            if cluster.stop_requested(queue):
                continue
            if all(queue.complete(task) for task in queue.tasks):
                if queue.plan_path not in published:
                    cluster.publish_reports(queue)
                    published.add(queue.plan_path)
            else:
                complete = False
        if complete and not pending:
            progress("stopped" if any(cluster.stop_requested(q) for q in requested) else "complete")
            return
        owned = sum(manager.owned(*manager.tasks[key]) for key in pending)
        if pending:
            jobs = ", ".join(f"{manager.tasks[key][0].plan['dataset']}:{key[1]}" for key in sorted(pending))
            reason = f"resume backlog={len(pending)} owned={owned}; {jobs}"
            cooling = []
            for key in pending:
                row = manager.receipt(*manager.tasks[key])
                if row.get("status") == "failed":
                    remaining = row.get("finished", 0) + args.retry_delay - time.time()
                    if remaining > 0:
                        cooling.append(remaining)
            if cooling:
                reason += f"; retry in {min(cooling):.0f}s"
        else:
            reason = "waiting for task leases, cache or prefix completion"
        progress("idle", idle_reason=reason)
        print(f"IDLE {label}: {reason}", flush=True)
        time.sleep(min(args.poll_seconds, 30))


@contextmanager
def resume_worker(worker, factory, *, pattern, env_key, label):
    """Replace operational dispatch while leaving every frozen adapter intact."""
    def run(queues, args, environment, gpu_fds, worker_id, update):
        return drain(worker, factory, pattern, env_key, label, queues, copy.copy(args),
                     environment, gpu_fds, worker_id, update)

    with patch.object(worker, "drain", run):
        yield
