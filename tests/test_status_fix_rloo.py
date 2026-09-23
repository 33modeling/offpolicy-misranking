"""RLOO status must match the launcher's own gate and keep failures to one line."""
import json
import sys

import pytest

from test_rloo_status import status, inputs, seal, write, files, task
from test_rloo_observation_runtime import legacy_contract

NOW = 1800000000.
experiment = status.experiment


@pytest.fixture
def point(tmp_path):
    run, evaluation = inputs(tmp_path / "inputs")
    root = tmp_path / "rloo"
    out = root / "math500-d0/s0"
    experiment.prepare(run, out, evaluation)
    return root, out, run, evaluation


def drift(out, name="src/gain_vs_reliability.py", value="0" * 64):
    c = experiment.ed.read(out / "experiment.json")
    c["code_hashes"][name] = value
    experiment.ed.atomic_json(out / "experiment.json", c)


def exit_code(root, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rloo_status.py", "--root", str(root), "--json"])
    code = status.main()
    capsys.readouterr()
    return code


def test_unreviewed_src_drift_is_not_ready(point, monkeypatch, capsys):
    root, out, _, _ = point
    drift(out)
    seal(out, "before")
    with pytest.raises(ValueError, match="code changed since preparation: src/gain_vs_reliability.py"):
        experiment.validate(out)  # what run/check and every queue worker do first
    before = files(root)
    data = status.snapshot(root, now=NOW)
    assert task(data, "before")["status"] == "DONE"
    for arm in experiment.ARMS:
        row = task(data, arm)
        assert row["status"] == "WAIT"
        assert row["reason"] == "launch blocked: src/gain_vs_reliability.py changed"
    suite = data["suites"][0]
    assert status.display.counts(suite)["states"]["READY"] == 0
    line = "launch blocked (s0): code changed since preparation: src/gain_vs_reliability.py"
    assert line in suite["details"] and line in suite["error"]
    for width in (80, 100, 120):
        text = status.display.render(data, width=width)
        assert line in " ".join(text.split())
        assert all(status.display.columns(row) <= width for row in text.splitlines())
    assert exit_code(root, monkeypatch, capsys) == 1
    assert files(root) == before


def test_reviewed_drift_needs_the_current_runtime_receipt(point, monkeypatch, capsys):
    root, out, run, evaluation = point
    legacy_contract(out)
    with pytest.raises(ValueError, match="runtime receipt"):
        experiment.validate(out)
    before = files(root)
    data = status.snapshot(root, now=NOW)
    row = task(data)
    assert row["status"] == "WAIT"
    assert "run 'run_rloo.sh prepare' first" in row["reason"] and "src/rloo_experiment.py" in row["reason"]
    assert "run 'run_rloo.sh prepare' first" in data["suites"][0]["error"]
    assert files(root) == before and not (out / "queue-observation-runtime.json").exists()
    experiment.prepare(run, out, evaluation)  # the launcher's own receipt
    experiment.validate(out)
    data = status.snapshot(root, now=NOW)
    assert task(data)["status"] == "READY" and not data["suites"][0]["error"]
    assert exit_code(root, monkeypatch, capsys) == 0
    write(out / "queue-observation-runtime.json", {"tampered": True})
    with pytest.raises(ValueError, match="runtime receipt"):
        experiment.validate(out)
    assert task(status.snapshot(root, now=NOW))["status"] == "WAIT"
    (out / "queue-observation-runtime.json").write_text("{")
    assert "run 'run_rloo.sh prepare' first" in task(status.snapshot(root, now=NOW))["reason"]


@pytest.mark.parametrize("kind", ["missing-file", "not-a-mapping", "absent"])
def test_unreadable_gate_input_blocks_without_crashing(point, kind):
    root, out, _, _ = point
    c = experiment.ed.read(out / "experiment.json")
    if kind == "missing-file":
        c["code_hashes"]["src/removed_module.py"] = "0" * 64
    elif kind == "not-a-mapping":
        c["code_hashes"] = []
    else:
        del c["code_hashes"]
    experiment.ed.atomic_json(out / "experiment.json", c)
    with pytest.raises((OSError, ValueError, KeyError, AttributeError)):
        experiment.validate(out)
    data = status.snapshot(root, now=NOW)
    assert task(data)["status"] == "WAIT" and task(data)["reason"].startswith("launch blocked: ")
    assert data["suites"][0]["error"]


def test_gate_runs_once_per_distinct_contract(tmp_path, monkeypatch):
    root = tmp_path / "rloo"
    for seed in range(3):
        run, evaluation = inputs(tmp_path / f"inputs-{seed}")
        write(run / "run_config.json", {**status.read(run / "run_config.json"), "seed": seed})
        experiment.prepare(run, root / f"math500-d0/s{seed}", evaluation)
    original, calls = experiment.reviewed_code_changes, []
    monkeypatch.setattr(experiment, "reviewed_code_changes", lambda hashes: calls.append(1) or original(hashes))
    data = status.snapshot(root, now=NOW)
    assert len(calls) == 1
    assert status.display.counts(data["suites"][0])["states"]["READY"] == 9


def failed_arm(out, directory):
    """The real failure path: selection_gate_gpu.meter, then queue_rloo.record."""
    import selection_gate_gpu
    import queue_rloo
    child = out.parent / "child.py"
    child.write_text("import sys\n"
                     "for i in range(100): print(f'[rank0] step {i} loss=0.{i:03d} reward=0.5 kl=0.01')\n"
                     "print('Traceback (most recent call last):')\n"
                     "print('  File \"train_policy_grpo.py\", line 900, in <module>')\n"
                     "print('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB')\n"
                     "sys.exit(1)\n")
    with pytest.raises(RuntimeError) as failure:
        selection_gate_gpu.meter(directory, "train", "test", commands=[([sys.executable, str(child)], "0")],
                                 env={}, timeout=60, devices=4)
    queue_rloo.record(directory, "FAILED", f"{type(failure.value).__name__}: {failure.value}")


@pytest.mark.parametrize("meter", ["failed", "absent"])
def test_failed_arm_shows_the_exception_line_not_the_worker_log(point, meter):
    root, out, _, _ = point
    directory = out / "random"
    clean = len(status.display.render(status.snapshot(root, now=NOW), width=100).splitlines())
    failed_arm(out, directory)
    assert status.read(directory / "queue-attempt.json")["error"].count("\n") > 100
    if meter == "absent":
        (directory / "progress.json").unlink()
    data = status.snapshot(root, now=NOW)
    assert task(data)["status"] == "WAIT"
    assert task(data)["reason"] == "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
    for all_tasks in (False, True):
        text = status.display.render(data, width=100, all_tasks=all_tasks)
        assert "[worker-log]" not in text and "step 0 loss" not in text
    assert len(status.display.render(data, width=100).splitlines()) <= clean + 2


def test_one_line_failure_is_capped(point):
    root, out, _, _ = point
    cause = "torch.OutOfMemoryError: CUDA out of memory. " + "Tried to allocate 2.00 GiB; " * 30
    write(out / "random/queue-attempt.json", dict(state="FAILED", error=f"RuntimeError: train worker failed: [1]\n{cause}"))
    reason = task(status.snapshot(root, now=NOW))["reason"]
    assert reason.startswith("torch.OutOfMemoryError: CUDA out of memory.") and len(reason) <= 160
