"""Reuse the tested shared resume policy for the isolated Llama queues."""

import copy
from contextlib import ExitStack, contextmanager
from types import MethodType
from unittest.mock import patch

from srgc_research.dispatch.qwen_resume import ResumeFirst


@contextmanager
def resume_first_worker():
    from . import worker

    original = worker.drain

    def drain(queues, args, environment, gpu_fds, worker_id, update):
        # Include the sibling backlog even on a math-only or MBPP-only node.
        cohort = list(queues)
        parent = queues[0].plan_path.parent
        for name in ("math", "mbpp"):
            path = parent / f"llama31-8b-{name}.json"
            if path.exists() and all(
                q.plan_path.resolve() != path.resolve() for q in cohort
            ):
                cohort.append(type(queues[0])(path))
        manager = ResumeFirst(cohort)
        options = copy.copy(args)
        options.retry_failed = True
        with ExitStack() as stack:
            for queue in queues:

                def claim(instance, _manager=manager, **kwargs):
                    return _manager.claim(instance, **kwargs)

                stack.enter_context(
                    patch.object(queue, "claim", MethodType(claim, queue))
                )
            return original(queues, options, environment, gpu_fds, worker_id, update)

    with patch.object(worker, "drain", drain):
        yield
