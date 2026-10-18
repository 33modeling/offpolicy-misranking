"""The original rank exception survives elastic's outer failure summary."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from srgc_research.dispatch import rank_failures, rank_runtime
from srgc_research.dispatch.reward_curves import recording_command

REPO = Path(__file__).resolve().parents[2]


def log_failure(cause):
    return (f"[rank2]: {cause}\n" + "launcher cleanup\n" * 70 +
            "torch.distributed.elastic.multiprocessing.errors.ChildFailedError:\n"
            "Root Cause (first observed failure):\n"
            "  rank      : 2 (local_rank: 2)\n"
            "  exitcode  : 1 (pid: 12345)\n"
            "  error_file: <N/A>\n")


def test_exception_before_last_25_lines_is_preferred_over_elastic_wrapper(tmp_path):
    log = tmp_path / "task.log"
    log.write_text(log_failure("RuntimeError: optimizer checkpoint cannot be restored"))
    assert rank_failures.last_error_line(log) == "[rank2]: RuntimeError: optimizer checkpoint cannot be restored"
    details = rank_failures.failure_details(log)
    assert "optimizer checkpoint cannot be restored" in details and "local_rank: 2" in details


def test_current_attempt_does_not_inherit_old_oom_or_exception(tmp_path):
    log = tmp_path / "task.log"
    log.write_text(log_failure("OutOfMemoryError: historical failure"))
    start = log.stat().st_size
    with log.open("a") as handle:
        handle.write("SRGC CHILD START model=llama31\n" + log_failure("ValueError: current failure"))
    assert "historical failure" not in rank_failures.failure_details(log, start=start)
    assert "current failure" in rank_failures.last_error_line(log)
    # An import failure before any RANK START marker is still a new attempt.
    with log.open("a") as handle:
        handle.write("SRGC CHILD START model=llama31\nModuleNotFoundError: no dependency\n")
    assert rank_failures.last_error_line(log) == "ModuleNotFoundError: no dependency"


def test_signal_only_failure_is_shown_without_inventing_python_cause(tmp_path):
    log = tmp_path / "task.log"
    log.write_text("Root Cause (first observed failure):\n"
                   "  exitcode  : -9 (pid: 12345)\n  traceback : Signal 9 (SIGKILL) received by PID 12345\n")
    assert "-9" in rank_failures.last_error_line(log)
    assert "No Python rank exception recorded" in rank_failures.failure_details(log)


@pytest.mark.parametrize("model", rank_runtime.MODELS)
@pytest.mark.parametrize("stage", ("smoke", "cache", "train"))
def test_every_model_rank_is_wrapped_without_changing_frozen_arguments(model, stage):
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "--max_restarts=0"]
    if model == "qwen35":
        command += ["/repo/scripts/srgc_qwen35_rank_runtime.py"]
    else:
        command += ["--module", f"srgc_research.dispatch.{model}.rank"]
    command += ["--stage", stage, "--plan", "/frozen/plan.json", "--seed", "5", "--task", "on_policy", "--resume"]
    measured = recording_command(command)
    actual = rank_runtime.diagnostic_command(command, model)
    entry_flag = "--entry-script" if model == "qwen35" else "--entry-module"
    entry = actual[actual.index(entry_flag) + 1]
    assert actual[actual.index("--") + 1:] == measured[measured.index(entry) + 1:]
    assert actual[:6] == command[:6]
    assert rank_runtime.diagnostic_command(actual, model) == actual
    assert "--nproc_per_node=4" in actual and "--max_restarts=0" in actual
    if stage == "train":
        assert any("reward_curve_rank" in arg for arg in actual)


def test_unknown_child_command_is_not_rewritten():
    command = [sys.executable, "-m", "selection_nccl_preflight"]
    assert rank_runtime.diagnostic_command(command, "llama31") == command


@pytest.mark.parametrize("entry_type", ("script", "module"))
def test_real_rank_writes_elastic_error_json_for_import_and_runtime_failures(tmp_path, entry_type):
    pytest.importorskip("torch")
    entry = tmp_path / "crashing_rank.py"
    entry.write_text("raise RuntimeError('actual inner rank failure seed=5 arm=on_policy')\n")
    error = tmp_path / "rank-error.json"
    env = {**os.environ, "TORCHELASTIC_ERROR_FILE": str(error),
           "PYTHONPATH": os.pathsep.join((str(REPO), str(tmp_path)))}
    target = str(entry) if entry_type == "script" else entry.stem
    result = subprocess.run([sys.executable, "-m", rank_runtime.MODULE, "--model", "llama31",
                             f"--entry-{entry_type}", target, "--", "--seed", "5", "--task", "on_policy"],
                            cwd=REPO, env=env, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode != 0
    recorded = json.loads(error.read_text())["message"]
    assert "actual inner rank failure" in recorded["message"]
    assert "crashing_rank.py" in recorded["extraInfo"]["py_callstack"]
    assert "SRGC RANK START" in result.stdout


@pytest.mark.parametrize("model", rank_runtime.MODELS)
def test_actual_four_rank_elastic_failure_then_optimizer_checkpoint_resume(tmp_path, model):
    torch = pytest.importorskip("torch")
    # Execute real torchrun, Gloo, SRGC Engine, AdamW and a rank-local failure.
    # Model loading/CUDA are deliberately outside this CPU integration test.
    entry = tmp_path / "learning_rank.py"
    entry.write_text('''import os
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from srgc_research.tests.test_node_kill_resume import make_engine, run_to
root = Path(os.environ["TEST_CHECKPOINT_ROOT"])
dist.init_process_group("gloo", timeout=timedelta(seconds=30))
rank = dist.get_rank()
arms = ("prefix", "on_policy", "switch", "sr", "random")
try:
    for arm in arms:
        engine = make_engine(root, arm)
        path = root / f"{arm}-latest.pt"
        if path.exists():
            state = torch.load(path, weights_only=False)
            engine.load_state_dict(state)
        run_to(engine, root, arm, engine.step + 1)
    dist.barrier()
    if os.environ["TEST_FAIL"] == "1" and rank == 2:
        raise RuntimeError("actual rank-2 failure after saved optimizer state")
    dist.barrier()
finally:
    if os.environ["TEST_FAIL"] != "1":
        dist.destroy_process_group()
''')
    root = tmp_path / "checkpoints"
    root.mkdir()
    environment = {**os.environ, "PYTHONPATH": str(REPO), "TEST_CHECKPOINT_ROOT": str(root),
                   "SRGC_PROGRESS_DIR": str(tmp_path / "attempt-1"), "TEST_FAIL": "1"}
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               "--max_restarts=0", "--module", rank_runtime.MODULE, "--model", model,
               "--entry-script", str(entry), "--"]
    log = tmp_path / "task.log"

    def child(command, log_path, environment, **kwargs):
        with log_path.open("a") as handle:
            return subprocess.run(command, cwd=REPO, env=environment, stdout=handle,
                                  stderr=subprocess.STDOUT, timeout=75, check=False).returncode

    run = rank_runtime.diagnostic_child(child, model)
    assert run(command, log, environment) == 1
    first = log.read_text()
    assert "ChildFailedError" in first and "actual rank-2 failure" in first
    errors = list((tmp_path / "attempt-1/elastic-errors").rglob("error.json"))
    # Peers may also record the subsequent Gloo connection-closed exception.
    original = next(path for path in errors if path.parent.name == "2")
    recorded = json.loads(original.read_text())["message"]
    assert "actual rank-2 failure" in recorded["message"]
    assert "actual rank-2 failure" in rank_failures.last_error_line(log)
    arms = ("prefix", "on_policy", "switch", "sr", "random")
    for arm in arms:
        state = torch.load(root / f"{arm}-latest.pt", weights_only=False)
        assert state["step"] == (1 if arm == "prefix" else 26)
    assert len(list(root.glob("*.pt"))) == 5
    environment.update(TEST_FAIL="0", SRGC_PROGRESS_DIR=str(tmp_path / "attempt-2"))
    assert run(command, log, environment) == 0
    from srgc_research.tests.test_node_kill_resume import make_engine, run_to
    for arm in arms:
        state = torch.load(root / f"{arm}-latest.pt", weights_only=False)
        stop = 2 if arm == "prefix" else 27
        assert state["step"] == stop and len(state["history"]) == 2
        assert all(value["step"].item() == 2 for value in state["backend"]["optimizer"]["state"].values())
        baseline_folder = tmp_path / "baseline"
        baseline = make_engine(baseline_folder, arm)
        run_to(baseline, baseline_folder, arm, stop)
        expected = baseline.state_dict()
        torch.testing.assert_close(state["backend"], expected["backend"], rtol=0, atol=0)
        for key in ("active_selection", "used_training_ids", "sampling_cycle", "rule", "switched_at"):
            assert state[key] == expected[key]


def test_dashboard_uses_original_rank_exception(tmp_path):
    from srgc_research.dispatch import qwen_status

    log = tmp_path / "task.log"
    log.write_text(log_failure("RuntimeError: restore failed inside rank"))
    assert "restore failed inside rank" in qwen_status.last_error_line(log)


def test_command_wrapper_preserves_environment_and_callback_arguments(tmp_path):
    original = lambda *a, **kw: (a, kw)
    environment = {"SRGC_PROGRESS_DIR": str(tmp_path / "attempt")}
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=4", "--module",
               "srgc_research.dispatch.llama31.rank", "--stage", "smoke"]
    callback = lambda pid: None
    with patch.dict(os.environ, {}, clear=False):
        (actual, _log, env), kwargs = rank_runtime.diagnostic_child(original, "llama31")(
            command, tmp_path / "log", environment, heartbeat=callback, pass_fds=(17,))
    assert environment == {"SRGC_PROGRESS_DIR": str(tmp_path / "attempt")}
    assert env["PYTHONFAULTHANDLER"] == "1" and kwargs == {"heartbeat": callback, "pass_fds": (17,)}
    assert actual[actual.index("--log-dir") + 1] == str(tmp_path / "attempt/elastic-errors")
