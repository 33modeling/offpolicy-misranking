"""Run a sibling dataset's existing backlog on otherwise idle model workers."""

import copy
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from srgc_rebuttal.runtime import Busy, lease

from .datasets import dataset_name
from .preemption import interruption_code, log_position


def active_dataset_phase(environment=None):
    environment = os.environ if environment is None else environment
    key = "SRGC_CACHE_DATASET" if environment.get("SRGC_CACHE_ONLY") == "1" else "SRGC_DATASET_PHASE"
    phase = environment.get(key)
    if phase is not None and phase not in {"math", "mbpp"}:
        raise ValueError(f"{key} must be math or mbpp")
    return phase


def phase_queues(queues):
    phase = active_dataset_phase()
    if phase is None:
        return queues
    selected = [queue for queue in queues if dataset_name(queue.plan) == phase]
    if not selected:
        raise ValueError(f"no {phase} plan in requested queues")
    return selected


def phase_pending(manager, pending):
    phase = active_dataset_phase()
    cache_only = os.environ.get("SRGC_CACHE_ONLY") == "1"
    return {key for key in pending
            if (not cache_only or manager.tasks[key][1].arm == "cache")
            and (phase is None or dataset_name(manager.tasks[key][0].plan) == phase)}


def stop_code():
    from .model_chain import PAUSED, enabled

    return PAUSED if enabled() else None


def cohort_for(requested, factory, pattern):
    cohort = list(requested)
    for name in ("math", "mbpp"):
        path = requested[0].plan_path.parent / pattern.format(dataset=name)
        if path.is_file() and all(q.plan_path.resolve() != path.resolve() for q in cohort):
            cohort.append(type(requested[0])(path))
    manager = factory(cohort)
    cohort = list(manager.queues.values())
    if len({q.plan.get("model_snapshot_sha256") for q in cohort}) != 1:
        raise ValueError("sibling datasets use different local model snapshots")
    return manager, cohort


def runnable(manager, requested, pending, args):
    """Inspect readiness without claiming tasks or admitting GPUs."""
    from srgc_rebuttal import cluster
    active = {key for key in phase_pending(manager, pending)
              if not cluster.stop_requested(manager.tasks[key][0])
              and manager.dataset_allowed(manager.tasks[key][0])
              and not manager.tasks[key][0].complete(manager.tasks[key][1])
              and not manager.owned(*manager.tasks[key])}
    if os.environ.get("SRGC_CACHE_ONLY") == "1":
        active = {key for key in active if manager.receipt(*manager.tasks[key]).get("status") != "failed"
                  or time.time() >= manager.receipt(*manager.tasks[key]).get("finished", 0) + args.retry_delay}
    for queue in manager.queues.values():
        if cluster.stop_requested(queue) or not manager.dataset_allowed(queue):
            continue
        for task in queue.tasks:
            if os.environ.get("SRGC_CACHE_ONLY") == "1" and task.arm != "cache":
                continue
            key = (str(queue.plan_path.resolve()), task.key)
            if queue not in requested and key not in pending:
                continue
            if (os.environ.get("SRGC_RESUME_ONLY") == "1"
                    and os.environ.get("SRGC_CACHE_ONLY") != "1" and key not in pending):
                continue
            if active and key not in active:
                continue
            if queue.complete(task) or not queue.ready(task) or manager.owned(queue, task):
                continue
            row = manager.receipt(queue, task)
            if row.get('status') == 'failed' and time.time() < row.get('finished', 0) + args.retry_delay:
                continue
            return True
    return False


def worker_loop(worker, factory, pattern, env_key, label, plans, args, group, common_root, admission):
    """Check the full resume cohort before completion, runtime binding or admission."""
    from scripts.srgc_seed_order import seed_first
    from srgc_rebuttal import cluster
    from srgc_rebuttal.cluster_queue import TaskQueue

    requested = [TaskQueue(plan) for plan in plans]
    for queue in requested:
        queue.tasks = seed_first(queue.tasks, queue.plan["seeds"])
    manager, cohort = cohort_for(requested, factory, pattern)
    if os.environ.get("SRGC_CACHE_ONLY") == "1":
        # Construct the full manager first so historical backlog entries remain
        # valid. Restrict only this worker's runnable tasks, never plan contents.
        for queue in cohort:
            queue.tasks = [task for task in queue.tasks if task.arm == "cache"]
    requested = phase_queues(cohort) if (os.environ.get("SRGC_DATASET_PHASE") is not None
                                        or active_dataset_phase() is not None) else requested
    for queue in cohort:
        queue.bind()
    worker_id = uuid.uuid4().hex
    metadata = {"started": time.time()}

    def update(state, **details):
        worker.write_worker(cohort, worker_id, state, **metadata, **details)

    update("preflight")
    try:
        stopped = {q.plan_path for q in requested if cluster.stop_requested(q)}
        if len(stopped) == len(requested):
            update("stopped")
            if stop_code() is not None:
                raise SystemExit(stop_code())
            return
        with lease(manager.marker.with_suffix(".lock"), wait=True):
            pending = {key for key in phase_pending(manager, manager.pending())
                       if not cluster.stop_requested(manager.tasks[key][0])}
        if (os.environ.get("SRGC_RESUME_ONLY") == "1"
                and os.environ.get("SRGC_CACHE_ONLY") != "1" and not pending):
            update("yielded", idle_reason="unfinished tasks recovered; fresh work deferred")
            return
        if not pending and all(q.plan_path in stopped or
                               all(q.complete(task) and not manager.owned(q, task)
                                   for task in q.tasks) for q in requested):
            for queue in requested:
                if queue.plan_path not in stopped:
                    cluster.publish_reports(queue)
            update("stopped" if stopped else "complete")
            if stopped and stop_code() is not None:
                raise SystemExit(stop_code())
            return
        if os.environ.get('SRGC_MODEL_CHAIN_CHILD') == '1':
            from .model_chain import YIELD
            with lease(manager.marker.with_suffix('.lock'), wait=True):
                ready = runnable(manager, requested, pending, args)
            if not ready:
                update('yielded', idle_reason='no claimable tasks; waiting to resume this model')
                raise SystemExit(YIELD)
        environment = cluster.child_environment()
        environment[env_key.removesuffix("S")] = str(plans[0])
        environment[env_key] = json.dumps([str(q.plan_path) for q in cohort])
        identity = cohort[0].plan.get("model_snapshot_sha256")
        if identity is not None:
            environment[env_key.removesuffix("_PLANS") + "_MODEL_SHA256"] = identity
        signature = worker.runtime_signature()
        # Refuse incompatible saved runtimes before GPU admission (including
        # Llama's allocated-process cleanup). bind_runtime rechecks under locks.
        for queue in cohort:
            path = queue.directory / "runtime.json"
            if path.is_file() and json.loads(path.read_text()) != signature:
                raise ValueError(f"node runtime differs from this experiment: {path}")
        devices, uuids = cluster.gpu_identity()
        environment["CUDA_VISIBLE_DEVICES"] = devices
        metadata.update(devices=devices, gpu_uuids=uuids)
        with ExitStack() as locks:
            roots = {Path(common_root) / "gpu-node-locks", Path(group) / ".srgc-gpu-node-locks"}
            fds = tuple(fd for root in sorted(roots)
                        for fd in locks.enter_context(cluster.device_leases(root, uuids)))
            worker.bind_runtime(cohort, signature)
            metadata["admission"] = admission(cluster.admit,
                requested[0].directory / "admission" / worker_id, environment, cluster.run_child,
                pass_fds=fds, plan=requested[0].plan,
                heartbeat=lambda pid: update("preflight", child_pid=pid),
                should_stop=lambda: all(cluster.stop_requested(q) for q in requested))
            code = drain(worker, factory, pattern, env_key, label, requested, args, environment,
                         fds, worker_id, update, manager=manager)
            if code is not None:
                raise SystemExit(code)
    except BaseException as error:
        from srgc_rebuttal.runtime import Busy

        from .model_chain import BUSY, PAUSED, YIELD, enabled
        command = getattr(error, "cmd", ())
        if (enabled() and isinstance(error, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
                and isinstance(command, (list, tuple)) and command and Path(command[0]).name == "nvidia-smi"):
            update("yielded", error=f"{type(error).__name__}: {error}",
                   idle_reason="GPU query failed; waiting for allocated GPUs to return")
            raise SystemExit(BUSY) from None
        if enabled() and isinstance(error, SystemExit) and error.code == PAUSED:
            update("stopped", idle_reason="requested dataset is paused; later datasets remain blocked")
            raise
        if enabled() and (isinstance(error, Busy) or isinstance(error, SystemExit) and error.code == YIELD):
            update('yielded', idle_reason=str(error))
            raise SystemExit(BUSY if isinstance(error, Busy) else YIELD) from None
        interrupted = (isinstance(error, KeyboardInterrupt) or
                       isinstance(error, SystemExit) and error.code in (129, 130, 137, 143))
        update("stopped" if interrupted else "failed",
               error=f"{type(error).__name__}: {error}")
        raise


def drain(worker, factory, pattern, env_key, label, requested, args, environment,
          gpu_fds, worker_id, update, *, manager=None):
    from srgc_rebuttal import cluster

    if manager is None:
        manager, cohort = cohort_for(requested, factory, pattern)
    else:
        cohort = list(manager.queues.values())
    requested = phase_queues(cohort) if (os.environ.get("SRGC_DATASET_PHASE") is not None
                                        or active_dataset_phase() is not None) else requested
    siblings = [q for q in cohort if q not in requested]
    for queue in siblings:
        queue.bind()
    runtime = requested[0].directory / "runtime.json"
    if siblings and runtime.is_file():
        worker.bind_runtime(cohort, json.loads(runtime.read_text()))
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
            return stop_code()
        if os.environ.get("SRGC_RESUME_ONLY") == "1" and os.environ.get("SRGC_CACHE_ONLY") != "1":
            with lease(manager.marker.with_suffix(".lock"), wait=True):
                remaining = phase_pending(manager, manager.pending())
            if not remaining:
                progress("yielded", idle_reason="unfinished tasks recovered; fresh work deferred")
                return
        pending = None
        claimed = False
        waiting_on_gpu = False
        for queue in cohort:
            if cluster.stop_requested(queue) or not manager.dataset_allowed(queue):
                continue
            # A dataset-only command can rescue old sibling work, but cannot
            # open a new sibling seed or continuation after the backlog clears.
            tasks = queue.tasks
            if queue in siblings:
                if pending is None:
                    with lease(manager.marker.with_suffix(".lock"), wait=True):
                        pending = phase_pending(manager, manager.pending())
                queue.tasks = [task for task in tasks
                               if (str(queue.plan_path.resolve()), task.key) in pending]
            try:
                if not queue.tasks:
                    continue
                with manager.claim(queue, retry_failed=True, retry_delay=args.retry_delay,
                                   worker_id=worker_id) as task:
                    if task is None:
                        continue
                    claimed = True
                    task_label = f"{label} {queue.plan['dataset']}:{task.key}"
                    log = queue.directory / "logs" / f"{task.key}.log"
                    print(f"RUN {task_label} log={log}", flush=True)
                    log_start = log_position(log)
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
                    except Busy as error:
                        # GPU teardown can outlive a failed launcher. No rank
                        # has started here, so this must not become a new ERROR.
                        code = 75
                        log.parent.mkdir(parents=True, exist_ok=True)
                        with log.open("a") as handle:
                            handle.write(f"GPU WAIT: {error}\n")
                    except (OSError, subprocess.SubprocessError) as error:
                        # A transient fork/exec or GPU-query failure still owns
                        # a real task receipt. Finish that attempt and let the
                        # same retry-first loop resume it without exiting.
                        code = 1
                        log.parent.mkdir(parents=True, exist_ok=True)
                        with log.open("a") as handle:
                            handle.write(f"LAUNCH ERROR: {type(error).__name__}: {error}\n")
                        print(f"LAUNCH ERROR {task_label}: {error}", flush=True)
                    except KeyboardInterrupt:
                        queue.finish(task, 130, interrupted=True)
                        progress("stopped", active=queue, task=task)
                        raise
                    except BaseException:
                        queue.finish(task, 1)
                        raise
                    signal_code = interruption_code(code, log, log_start)
                    code = signal_code if signal_code is not None else (128 - code if code < 0 else code)
                    interrupted = signal_code is not None
                    waiting_on_gpu = code == 75
                    code = queue.finish(task, code, interrupted=interrupted or waiting_on_gpu)
                    # An interrupted process can also expose corrupted inputs
                    # or evidence. finish deliberately records that as failed.
                    interrupted = interrupted and manager.receipt(queue, task).get("status") == "interrupted"
                    waiting_on_gpu = waiting_on_gpu and manager.receipt(queue, task).get("status") == "interrupted"
                    print(f"DONE {task_label} exit={code}", flush=True)
                    if waiting_on_gpu:
                        reason = f"GPU allocation is not ready; task not started; retrying saved work; log={log}"
                        progress("idle", active=queue, task=task, idle_reason=reason)
                        if os.environ.get('SRGC_MODEL_CHAIN_CHILD') == '1':
                            raise Busy(reason)
                    if interrupted:
                        progress("stopped", active=queue, task=task)
                        return code
                    if code and not waiting_on_gpu:
                        from .rank_failures import failure_details

                        print(failure_details(log, start=log_start), flush=True)
                        if os.environ.get('SRGC_MODEL_CHAIN_CHILD') == '1':
                            # Release task/device contexts before the coordinator
                            # restarts this model. Later models remain blocked.
                            progress("failed", active=queue, task=task,
                                     error=f"child exit={code}; log={log}")
                            return code
                    if not waiting_on_gpu:
                        progress("idle")
            finally:
                queue.tasks = tasks
            if claimed:
                break
        if claimed:
            if waiting_on_gpu:
                time.sleep(min(args.poll_seconds, 30))
            continue
        with lease(manager.marker.with_suffix(".lock"), wait=True):
            pending = phase_pending(manager, manager.pending())
            # Keep stopped work in the persisted backlog, but do not make a
            # different dataset wait for a job this launcher cannot run.
            pending = {key for key in pending if not cluster.stop_requested(manager.tasks[key][0])}
            owned = sum(manager.owned(*manager.tasks[key]) for key in pending)
        complete = True
        for queue in requested:
            if cluster.stop_requested(queue):
                continue
            if all(queue.complete(task) and not manager.owned(queue, task) for task in queue.tasks):
                if queue.plan_path not in published:
                    cluster.publish_reports(queue)
                    published.add(queue.plan_path)
            else:
                complete = False
        if complete and not pending:
            progress("stopped" if any(cluster.stop_requested(q) for q in requested) else "complete")
            return stop_code() if any(cluster.stop_requested(q) for q in requested) else None
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
        if os.environ.get('SRGC_MODEL_CHAIN_CHILD') == '1':
            from .model_chain import YIELD
            progress('yielded', idle_reason=reason)
            print(f'YIELD {label}: {reason}; waiting to resume this model', flush=True)
            raise SystemExit(YIELD)
        progress("idle", idle_reason=reason)
        print(f"IDLE {label}: {reason}", flush=True)
        time.sleep(min(args.poll_seconds, 30))


@contextmanager
def resume_worker(worker, factory, *, pattern, env_key, label, result_model=None):
    """Replace operational dispatch while leaving every frozen adapter intact."""
    def run(queues, args, environment, gpu_fds, worker_id, update):
        return drain(worker, factory, pattern, env_key, label, queues, copy.copy(args),
                     environment, gpu_fds, worker_id, update)

    def start(plans, args, group, common_root, admission):
        return worker_loop(worker, factory, pattern, env_key, label, plans, copy.copy(args),
                           group, common_root, admission)

    with ExitStack() as stack:
        stack.enter_context(patch.object(worker, "drain", run))
        stack.enter_context(patch.object(worker, "worker", start))
        if result_model is not None:
            from scripts import srgc_process_guard as guard
            from srgc_rebuttal import cluster

            from .model_results import publish
            from .process_recovery import recording_guard
            from .rank_runtime import diagnostic_child

            # Qwen's frozen CLI imports the guard by its historical top-level
            # name. Both names must share the same target/owner protections.
            stack.enter_context(patch.dict(sys.modules, {'srgc_process_guard': guard}))
            stack.enter_context(patch.object(guard, 'process_guard',
                recording_guard(guard.process_guard, guard, result_model)))
            stack.enter_context(patch.object(cluster, "publish_reports", lambda queue: publish(result_model, queue)))
            original_child = cluster.run_child
            stack.enter_context(patch.object(cluster, "run_child",
                diagnostic_child(original_child, result_model)))
        yield
