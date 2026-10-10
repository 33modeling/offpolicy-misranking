"""Recover historical failure limits while preserving bounded, shared retries."""

import json
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import pytest

from srgc_rebuttal.cluster_queue import Task, TaskQueue
from srgc_rebuttal.runtime import atomic_json
from srgc_rebuttal.tests.test_cluster import finish_fake
from srgc_research.dispatch.model_resume import LlamaResumeFirst, failure_footer
from srgc_research.tests import test_qwen_resume as resume_tests
from srgc_research.tests.test_qwen_resume import abandoned


@pytest.fixture
def queues(tmp_path):
    return resume_tests.queues.__wrapped__(tmp_path)


def test_historical_failure_limit_gets_one_new_bounded_budget(queues):
    queue = queues[0]
    old = Task(9, "sr")
    abandoned(queue, old, attempt=3, state="failed")
    checkpoint = queue.root / "seed-9/sr-latest.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint preserved")
    initial = {p: p.read_bytes() for p in queue.plan_path.parent.glob("*.json")}
    for index in range(3):
        with LlamaResumeFirst(queues).claim(queue, retry_failed=True, retry_delay=0) as task:
            assert task == old
            receipt = json.loads(queue.receipt(task).read_text())
            assert receipt["attempt"] == 4 + index
            assert receipt["retry_attempt"] == index + 1
            queue.finish(task, 1)
    with pytest.raises(RuntimeError, match="Llama-3.1.*bounded retries"), LlamaResumeFirst(queues).claim(queue, retry_failed=True, retry_delay=0):
        pytest.fail("same revision must not renew another retry budget")
    assert checkpoint.read_bytes() == b"checkpoint preserved"
    assert {p: p.read_bytes() for p in initial} == initial
    history = list((queue.directory / "attempts").glob("*.json"))
    assert len(history) == 4
    assert json.loads((queue.directory / "attempts" / f"old-{old.key}.json").read_text())["attempt"] == 3


def test_interruption_does_not_spend_new_real_failure_budget(queues):
    queue = queues[0]
    old = Task(9, "prefix")
    abandoned(queue, old, attempt=99, state="failed")
    for _ in range(5):
        with LlamaResumeFirst(queues).claim(queue, retry_failed=True, retry_delay=0) as task:
            assert task == old
            queue.finish(task, 130, interrupted=True)
    with LlamaResumeFirst(queues).claim(queue, retry_failed=True, retry_delay=0) as task:
        assert task == old
        assert json.loads(queue.receipt(task).read_text())["retry_attempt"] == 1


def test_other_dataset_backlog_still_blocks_fresh_tasks(queues):
    math, mbpp = queues
    old = Task(9, "switch")
    abandoned(mbpp, old, state="failed")
    manager = LlamaResumeFirst(queues)
    with manager.claim(math, retry_failed=True, retry_delay=0) as task:
        assert task is None
    with manager.claim(mbpp, retry_failed=True, retry_delay=0) as task:
        assert task == old
        finish_fake(mbpp, task)
        assert mbpp.finish(task, 0) == 0
    with manager.claim(math) as task:
        assert task is not None and task.seed == 5


def test_failure_tracebacks_are_printed_at_the_bottom(queues, capsys):
    queue = queues[0]
    task = Task(9, "prefix")
    abandoned(queue, task, state="failed")
    log = queue.directory / "logs" / f"{task.key}.log"
    log.parent.mkdir(parents=True)
    log.write_text("traceback\nRuntimeError: actual model failure\n")
    print("BACKUP waiting_for_checkpoint")
    failure_footer(queue.root, [queue.plan_path], "LLAMA")
    text = capsys.readouterr().out
    assert "LLAMA FAILURE DETAILS" in text
    assert text.endswith("RuntimeError: actual model failure\n")
    assert text.index("FAILURE DETAILS") > text.index("BACKUP")


def test_llama_wrapper_installs_recovery_without_changing_adapter(tmp_path):
    from srgc_research.dispatch import llama_run
    from srgc_research.dispatch.llama31 import adapter, cli, resume

    before = adapter.adapter_digest()
    with patch.object(cli, "main", side_effect=lambda args: int(resume.ResumeFirst is not LlamaResumeFirst)):
        assert llama_run.main(["all", "status"]) == 0
    assert adapter.adapter_digest() == before


def test_changed_baseline_or_lost_attempt_history_fails_closed(queues):
    queue = queues[0]
    task = Task(9, "prefix")
    abandoned(queue, task, state="failed")
    manager = LlamaResumeFirst(queues)
    with manager.claim(queue, retry_failed=True, retry_delay=0) as claimed:
        queue.finish(claimed, 1)
    for path in (queue.directory / "attempts").glob("*.json"):
        path.unlink()
    with pytest.raises(ValueError, match="history shrank"), manager.claim(queue, retry_failed=True, retry_delay=0):
        pass


def test_renewed_legacy_receipt_cools_down_without_blocking_worker(queues):
    import time

    queue = queues[0]
    old = Task(9, "prefix")
    abandoned(queue, old, attempt=3, state="failed")
    row = json.loads(queue.receipt(old).read_text())
    row["finished"] = time.time()
    atomic_json(queue.receipt(old), row)
    with LlamaResumeFirst(queues).claim(queue, retry_failed=True, retry_delay=60) as task:
        assert task is None
    status = next(row for row in queue.status(max_attempts=3) if row["task"] == old.key)
    assert status["status"] == "failed" and status["retry_attempt"] == 0


def concurrent_recovery(path, barrier, output):
    queue = TaskQueue(Path(path))
    queue.bind()
    barrier.wait(timeout=15)
    with LlamaResumeFirst([queue]).claim(queue, retry_failed=True, retry_delay=0) as task:
        output.put((task.key, json.loads(queue.receipt(task).read_text())["retry_attempt"]))
        barrier.wait(timeout=15)
        queue.finish(task, 130, interrupted=True)


def test_four_workers_share_one_recovery_budget_and_distinct_tasks(queues):
    queue = queues[0]
    expected = {Task(seed, "prefix").key for seed in queue.plan["seeds"][:4]}
    for seed in queue.plan["seeds"][:4]:
        abandoned(queue, Task(seed, "prefix"), attempt=3, state="failed")
    ctx = multiprocessing.get_context("fork")
    barrier, output = ctx.Barrier(4), ctx.Queue()
    processes = [ctx.Process(target=concurrent_recovery, args=(str(queue.plan_path), barrier, output)) for _ in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(25)
        if process.is_alive():
            process.kill()
            process.join()
        assert process.exitcode == 0
    rows = [output.get(timeout=3) for _ in processes]
    assert {key for key, _ in rows} == expected
    assert all(attempt == 1 for _, attempt in rows)
    assert len(list(queue.plan_path.parent.glob(".dispatch/llama31-resume-budget-v1.json"))) == 1
