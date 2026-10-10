"""Real cost receipts and frozen-rank dispatch for the diagnostic-phase fix."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

from srgc_rebuttal import cost_ledger
from srgc_rebuttal.timing import CostMeter
from srgc_research.dispatch.information_meter import (
    InformationPhaseLedger,
    diagnostic_costs,
)
from srgc_research.dispatch.information_run import RANK_WRAPPER, rank_dispatch
from srgc_research.storage import freeze


def test_real_ledger_accepts_diagnostic_and_reconciles_allocated_costs(tmp_path):
    ledger = InformationPhaseLedger(tmp_path)
    clock = iter((0., 0., 2., 3.))
    meter = CostMeter(local_gpu_count=1, record=ledger.record, clock=lambda: next(clock))
    with meter.phase("diagnostic", checkpoint=0, gpu_count=1), meter.stage("probe_gradient"):
        meter.count("responses", 8)
    totals = ledger.totals()
    assert totals["complete"] and totals["total_gpu_seconds"] == 3.
    assert totals["known_gpu_seconds"]["diagnostic_gpu_seconds"] == 3.
    assert totals["exclusive_stages"]["diagnostic.probe_gradient"]["gpu_seconds"] == 2.
    assert totals["counts"]["diagnostic.responses"] == 8


@pytest.mark.parametrize("fault", ["negative", "nan", "allocation", "stages", "checkpoint", "repeat-start"])
def test_diagnostic_receipts_retain_the_original_cost_validation(tmp_path, fault):
    ledger = InformationPhaseLedger(tmp_path)
    start = {"id": "probe", "phase": "diagnostic", "checkpoint": 0, "gpu_count": 4, "state": "started"}
    ledger.record(start)
    before = (tmp_path / "probe.json").read_bytes()
    finish = {**start, "state": "finished", "gpu_seconds": 8., "wall_seconds": 2.}
    if fault == "negative":
        finish["gpu_seconds"] = -1.
    elif fault == "nan":
        finish["gpu_seconds"] = float("nan")
    elif fault == "allocation":
        finish["wall_seconds"] = 3.
    elif fault == "stages":
        finish["stages"] = {"probe": {"gpu_seconds": 3.}}
    elif fault == "checkpoint":
        finish["checkpoint"] = 1
    else:
        finish = start
    with pytest.raises(ValueError):
        ledger.record(finish)
    assert (tmp_path / "probe.json").read_bytes() == before
    assert not ledger.totals()["complete"]


def test_registration_is_scoped_and_other_unknown_phases_still_fail(tmp_path):
    original = cost_ledger.PhaseLedger
    with diagnostic_costs():
        assert cost_ledger.PhaseLedger is InformationPhaseLedger
        with pytest.raises(ValueError, match="unknown metered phase"):
            cost_ledger.PhaseLedger(tmp_path).record({"phase": "misspelled"})
    assert cost_ledger.PhaseLedger is original


def test_dispatch_wraps_only_information_ranks_and_preserves_runner_arguments():
    calls = []
    def runner(command, *args, **kwargs):
        calls.append((command, args, kwargs))
        return 9
    dispatch = rank_dispatch(runner)
    original = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=4",
                "/frozen/srgc_research/information_rank.py", "--output", "/output"]
    assert dispatch(original, "log", {"PYTHONPATH": "/frozen"}, pass_fds=(2, 3)) == 9
    command, args, kwargs = calls[-1]
    assert command[4:7] == [str(RANK_WRAPPER), "--rank-script", original[4]]
    assert args == ("log", {"PYTHONPATH": "/frozen"}) and kwargs == {"pass_fds": (2, 3)}
    assert original[4] == "/frozen/srgc_research/information_rank.py"
    unrelated = [sys.executable, "scripts/selection_nccl_preflight.py", "--world-size", "4"]
    dispatch(unrelated)
    assert calls[-1][0] == unrelated


@pytest.fixture
def frozen_rank(tmp_path):
    runtime, _ = freeze(tmp_path / "measurement-code")
    target = runtime / "srgc_research/information_rank.py"
    target.write_text('''import argparse
from pathlib import Path
from srgc_rebuttal import cost_ledger
from srgc_rebuttal.cost_ledger import PhaseLedger
from srgc_rebuttal.timing import CostMeter
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
ledger = PhaseLedger(args.output / "cost-receipts")
with CostMeter(record=ledger.record).phase("diagnostic"):
    pass
print("FROZEN_LEDGER:", cost_ledger.__file__)
''')
    # A test-only frozen program, using the actual historical ledger and timer.
    record = json.loads((runtime / "runtime.json").read_text())
    record["files"]["srgc_research/information_rank.py"] = hashlib.sha256(target.read_bytes()).hexdigest()
    sha = hashlib.sha256(json.dumps(record["files"], sort_keys=True).encode()).hexdigest()
    (runtime / "runtime.json").write_text(json.dumps({**record, "sha256": sha}))
    (tmp_path / "manifest.json").write_text(json.dumps({"runtime": str(runtime),
                                                        "identity": {"measurement_sha256": sha}}))
    return tmp_path, runtime, target


@pytest.mark.parametrize("fault", [None, "different-rank", "tampered-code"])
def test_real_subprocess_registers_frozen_ledger_and_verifies_code_before_execution(frozen_rank, fault):
    if fault is None:
        pytest.importorskip("torch")
    output, runtime, target = frozen_rank
    ledger_source = runtime / "srgc_rebuttal/cost_ledger.py"
    before = ledger_source.read_bytes()
    if fault == "different-rank":
        target = output / "unrelated.py"
    elif fault == "tampered-code":
        target.write_text("raise AssertionError('must not execute changed code')")
    result = subprocess.run([sys.executable, str(RANK_WRAPPER), "--rank-script", str(target),
                             "--output", str(output)], capture_output=True, text=True, timeout=20, check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(runtime), str(runtime / "src"),
                                                        os.environ.get("PYTHONPATH", ""))),
             "PYTHONDONTWRITEBYTECODE": "1"})
    if fault:
        assert result.returncode != 0
        assert not (output / "cost-receipts").exists()
        assert "differs" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert str(ledger_source) in result.stdout
        totals = InformationPhaseLedger(output / "cost-receipts").totals()
        assert totals["complete"] and totals["known_gpu_seconds"]["diagnostic_gpu_seconds"] == 0.
    assert ledger_source.read_bytes() == before


def test_actual_frozen_subprocess_collects_gradients_from_the_real_backward(frozen_rank):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    output, runtime, target = frozen_rank
    repo = RANK_WRAPPER.parents[2]
    target.write_text(f'''
from pathlib import Path
import srgc_research, srgc_rebuttal
srgc_research.__path__.append({str(repo / "srgc_research")!r})
srgc_rebuttal.__path__.append({str(repo / "srgc_rebuttal")!r})
from srgc_research import information
from srgc_research.tests.test_information_gradients import precision_backend, batch
from srgc_research.tests.test_research import equal_tree
backend = precision_backend()
backend.logit_chunk_tokens = 8
records, probe, gradient = batch(backend, 4)
initial = backend.state_dict()
result, tensors = information.inspect_update(backend, records, seed=29,
    probe=probe, probe_loss_gradient=gradient)
equal_tree(initial, backend.state_dict())
assert result["metrics"]["problem_gradient_definition"] == "actual-GRPO-backward-contributions"
assert Path(information.__file__).is_relative_to({str(runtime)!r})
assert "collector.gradients()" in tensors["problem_gradient_adapter"]["measured_inspect_source"]
print("PASS: actual frozen rank wrapper, BF16/FP32 LoRA, multi-chunk backward")
''')
    record = json.loads((runtime / "runtime.json").read_text())
    record["files"]["srgc_research/information_rank.py"] = hashlib.sha256(target.read_bytes()).hexdigest()
    sha = hashlib.sha256(json.dumps(record["files"], sort_keys=True).encode()).hexdigest()
    (runtime / "runtime.json").write_text(json.dumps({**record, "sha256": sha}))
    (output / "manifest.json").write_text(json.dumps({"runtime": str(runtime),
        "identity": {"measurement_sha256": sha}}))
    result = subprocess.run([sys.executable, str(RANK_WRAPPER), "--rank-script", str(target),
        "--output", str(output)], capture_output=True, text=True, timeout=30, check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(runtime), str(runtime / "src"),
            os.environ.get("PYTHONPATH", ""))), "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: actual frozen rank wrapper" in result.stdout
