"""Any failed SR-GC measurement blocks only its state; fixed Pair controls keep running."""
import pytest

import selection_gate as core
import selector_pair_gpu as gpu
import selector_pair_srgc as srgc
import selector_pair_parallel as parallel
from test_selector_pair_gpu import fake_study  # noqa: F401
from test_selector_pair_parallel import study  # noqa: F401
from test_selector_pair_srgc import current_policy, srgc_study  # noqa: F401

DEVICES = ["0", "1", "2", "3"]


@pytest.mark.parametrize("failure", [KeyError, TypeError, IndexError, AttributeError])
def test_malformed_measurement_blocks_only_its_state_and_is_retried(tmp_path, srgc_study, monkeypatch, failure):
    p, calls, _, _ = srgc_study
    measure = srgc.measure

    def malformed(directory, *args):
        if directory.name == "s3-t25":
            raise failure("malformed SR-GC measurement artifact")
        return measure(directory, *args)

    monkeypatch.setattr(srgc, "measure", malformed)
    with srgc.activated(tmp_path, p, DEVICES):
        with pytest.raises(gpu.IncompletePairRun, match="s3-t25: .*malformed SR-GC measurement artifact"):
            parallel.run_distributed(tmp_path, p, DEVICES, "run", srgc.run_stages)
    assert len(calls) == 36 and not any(name.startswith("adaptive-") for name, *_ in calls)
    status = {path.parent.name: core.read(path) for path in (tmp_path / "sr-gc").glob("*/measurement-status.json")}
    assert status.pop("s3-t25")["state"] == "BLOCKED"
    assert len(status) == 5 and all(value["state"] == "DONE" for value in status.values())
    assert not (tmp_path / "test-decisions.json").exists()
    before = {path: path.read_bytes() for path in (tmp_path / "sr-gc").glob("*/decision.json")}
    # A restarted controller measures the failed state only, then runs Adaptive.
    monkeypatch.setattr(srgc, "measure", measure)
    with srgc.activated(tmp_path, p, DEVICES):
        parallel.run_distributed(tmp_path, p, DEVICES, "run", srgc.run_stages)
    assert all(path.read_bytes() == value for path, value in before.items())
    assert sum(name.startswith("adaptive-") for name, *_ in calls) == 6


@pytest.mark.parametrize("failure", [KeyError, TypeError, RuntimeError])
def test_decision_failure_outside_a_measurement_keeps_fixed_controls(tmp_path, srgc_study, monkeypatch, failure):
    p, calls, _, _ = srgc_study

    def corrupt(*args):
        raise failure("corrupt frozen SR-GC decision record")

    monkeypatch.setattr(srgc, "freeze", corrupt)
    with srgc.activated(tmp_path, p, DEVICES):
        with pytest.raises(failure, match="corrupt frozen SR-GC decision record"):
            parallel.run_distributed(tmp_path, p, DEVICES, "run", srgc.run_stages)
    assert len(calls) == 36
    assert not any(name.startswith("adaptive-") for name, *_ in calls)
