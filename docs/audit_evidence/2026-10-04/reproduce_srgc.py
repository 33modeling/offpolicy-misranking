"""Reproduce four audit findings, not a passing runtime regression suite.

All fixtures live in TemporaryDirectory. No GPU, /dev/shm, cluster data,
production queue, or unrelated process is touched. Run with the experiment's
Python environment. Exit zero means all four reviewed defects were observed.
"""

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts import srgc_process_guard as guard
from scripts.srgc_step_checkpoints import checkpoint_engine
from srgc_rebuttal.cluster_queue import TaskQueue
from srgc_rebuttal.runtime import code_digest
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.tests.test_cluster import write_inputs
from srgc_rebuttal.toy_backend import ToyBackend, make_problem


def foreign_shared_memory():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "torch_unrelated_live"
        path.write_bytes(b"live unrelated training storage")
        with path.open("rb") as live_handle, redirect_stdout(StringIO()):
            removed = guard.clean_shm(shm_dir=path.parent, table={
                os.getpid(): (os.getppid(), os.getuid(), "python unrelated_training.py")})
            return {"removed": removed, "file_still_exists": path.exists(),
                    "open_handle_still_readable": bool(live_handle.read())}


def interrupted_retry_budget():
    with tempfile.TemporaryDirectory() as directory:
        queue = TaskQueue(write_inputs(Path(directory)))
        queue.bind()
        queue.tasks = [task for task in queue.tasks if task.arm == "prefix" and task.seed == 5]
        for _ in range(3):
            with queue.claim(max_attempts=3) as task:
                assert task is not None
                queue.finish(task, 130, interrupted=True)
        with queue.claim(max_attempts=3, retry_failed=True, retry_delay=0) as task:
            claimed = task is not None
        status = queue.status(max_attempts=3)[0]
        return {"claimed_after_three_user_interrupts": claimed,
                "status": status["status"], "attempt": status["attempt"],
                "last_exit_code": status["exit_code"]}


def attention_resume():
    features, answers, candidates, validation, _, cache = make_problem(5)
    config = Config(seed=5, projection_dim=64)

    def make(cls):
        backend = ToyBackend(features, answers, projection_dim=64, seed=5)
        return cls(backend, candidates, validation, cache, arm="on_policy", config=config)

    original = make(Engine)
    original.run_until(1)
    saved = {**original.state_dict(), "checkpoint_policy": {"attention": "sdpa"}}
    with tempfile.TemporaryDirectory() as directory:
        resumed = make(checkpoint_engine(Engine, Path(directory), "prefix",
                       {"attention": "eager", "interval_updates": 1}, total_updates=25))
        resumed.load_state_dict(saved)
        return {"saved_attention": saved["checkpoint_policy"]["attention"],
                "resumed_attention": resumed.state_dict()["checkpoint_policy"]["attention"],
                "resume_rejected": False, "step": resumed.step}


def assertion_completion_bypass():
    from srgc_rebuttal.verifiers import code_reward

    record = {"answer": "assert False"}
    response = ('import os, sys\n'
                'os.write(int(sys.argv[2]), sys.argv[3].encode("ascii"))\n'
                'os._exit(0)')
    return {"test": record["answer"], "normal_reward": code_reward(record, "pass"),
            "bypass_reward": code_reward(record, response)}


def main():
    findings = {"F1": foreign_shared_memory(), "F2": attention_resume(),
                "F3": interrupted_retry_budget(), "F4": assertion_completion_bypass()}
    print(json.dumps({"runtime_sha256": code_digest(), "findings": findings}, indent=2))
    assert findings["F1"]["removed"] == ["torch_unrelated_live"]
    assert not findings["F1"]["file_still_exists"]
    assert findings["F2"]["saved_attention"] != findings["F2"]["resumed_attention"]
    assert findings["F3"]["status"] == "attempts_exhausted"
    assert not findings["F3"]["claimed_after_three_user_interrupts"]
    assert findings["F4"]["normal_reward"] == 0.0
    assert findings["F4"]["bypass_reward"] == 1.0


if __name__ == "__main__":
    main()
