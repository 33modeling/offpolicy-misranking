from pathlib import Path
import re

import pytest

import net_gain_gate_recovery as recovery
from test_net_gain_gate_gpu import completed, protocol, source

base, core, gpu = recovery.base, recovery.core, recovery.gpu
ERROR = "[abort] finite-difference calibration failed at step=0.1: relative_l2_error=0.9233"


def failed_score(out):
    directory = out / "selection_reduced"
    def fail():
        (directory / "score-3.log").write_text(ERROR + "\n")
        raise RuntimeError("score worker failed: [None, None, None, 2]")
    with pytest.raises(RuntimeError):
        base.meter(directory, "score", "H100", action=fail, ledger="deployment")
    core.atomic_json(directory / "failure.json", {"error": "score worker failed"})
    return directory


def mock_selection(out, c, arm, cap, env, devices):
    private = recovery.private_dir(out, arm)
    point = private / "scoring/points/p0"
    core.atomic_json(point / "experiment.json", {"derivative": "autograd"})
    core.atomic_json(point / "selection.json", {"selected": {"low_order": [0, 1, 2, 3]}})
    path = private / "selected.json"
    base.bind(path, {"indices": [0, 1, 2, 3], "point": "scoring/points/p0",
                     "scoring_sha256": base.digest(point / "selection.json"),
                     "experiment_sha256": base.digest(point / "experiment.json")})
    base.bind(path.with_suffix(".sha256.json"), {"sha256": base.digest(path)})
    base.meter(out / arm, "autograd-score", "H100", action=lambda: None, ledger="deployment")
    completed(out, arm)
    return [0, 1, 2, 3]


def test_failed_finite_recovers_and_preserves_costs_and_completed_result(tmp_path, monkeypatch):
    out, c = source(tmp_path)
    p = protocol()
    directory = failed_score(out)
    prefix = (directory / "cost.jsonl").read_bytes()
    paid = base.spent(directory)
    profile = out / "measurement/measurement.json"
    core.atomic_json(profile, {"features": {}})
    base.meter(profile.parent, "diagnose", "H100", action=lambda: None)
    first = {"status": "complete", "gpu_seconds": base.spent(profile.parent), "report_sha256": base.digest(profile)}
    core.atomic_json(profile.parent / "initial.json", first)
    monkeypatch.setattr(gpu, "measure_once", lambda *a: first)
    monkeypatch.setattr(base, "verify", lambda _: c)
    monkeypatch.setattr(base, "policy", lambda *a: Path(c["source_run"]))
    monkeypatch.setattr(base, "rewards", lambda *a: {"q0": .5})
    monkeypatch.setattr(gpu, "select_once", recovery.select_once)
    monkeypatch.setattr(gpu, "validate_result", recovery.validate_result)
    monkeypatch.setattr(recovery, "_select_once", lambda *a: pytest.fail("finite calibration was repeated"))
    monkeypatch.setattr(recovery, "exact_selection", mock_selection)

    completed(out, "random_full")
    recovery.run_arm(out, {}, p, "random_full", [], {})
    baseline = {f.name: f.read_bytes() for f in (out / "random_full").iterdir() if f.is_file()}
    recovery.run_arm(out, {"eval_timeout": 5}, p, "selection_reduced", list("0123"), {})
    result = recovery.validate_result(out, p, "selection_reduced")
    assert result["numerical_recovery"]["to"] == "autograd"
    assert result["used_gpu_seconds"] > paid
    assert (directory / "cost.jsonl").read_bytes().startswith(prefix)
    assert result["budget_gpu_seconds"] == c["budget_gpu_seconds"] - first["gpu_seconds"]
    assert (directory / "score-3.log").read_text().strip() == ERROR
    assert not (directory / "failure.json").exists()
    after = base.spent(directory)
    monkeypatch.setattr(base, "meter", lambda *a, **kw: pytest.fail("completed work repeated"))
    recovery.run_arm(out, {}, p, "selection_reduced", [], {})
    recovery.run_arm(out, {}, p, "random_full", [], {})
    assert base.spent(directory) == after
    assert baseline == {f.name: f.read_bytes() for f in (out / "random_full").iterdir() if f.is_file()}

    (directory / "autograd-recovery-result.json").unlink()
    with pytest.raises(OSError):
        recovery.validate_result(out, p, "selection_reduced")
    recovery.run_arm(out, {}, p, "selection_reduced", [], {})
    assert recovery.validate_result(out, p, "selection_reduced")["complete"]
    assert base.spent(directory) == after
    (directory / "autograd-recovery.json").unlink()
    with pytest.raises(ValueError, match="missing autograd recovery"):
        recovery.validate_result(out, p, "selection_reduced")


@pytest.mark.parametrize("message", ["[abort] CUDA out of memory", "[abort] invalid contract", "worker timed out"])
def test_unrelated_failures_do_not_enable_autograd(tmp_path, monkeypatch, message):
    out, c = source(tmp_path)
    directory = failed_score(out)
    (directory / "score-3.log").write_text(message + "\n")
    assert recovery.calibration_failure(directory) is None
    def fail(*a):
        raise RuntimeError(message)
    monkeypatch.setattr(recovery, "_select_once", fail)
    monkeypatch.setattr(recovery, "exact_selection", lambda *a: pytest.fail("unrelated failure recovered"))
    with pytest.raises(RuntimeError, match=re.escape(message)):
        recovery.select_once(out, c, protocol(), "selection_reduced", {}, {}, [])
    assert not (directory / "autograd-recovery.json").exists()


def test_new_calibration_failure_recovers_in_same_invocation(tmp_path, monkeypatch):
    out, c = source(tmp_path)
    def fail(*a):
        failed_score(out)
        raise RuntimeError("score worker failed")
    monkeypatch.setattr(recovery, "_select_once", fail)
    monkeypatch.setattr(recovery, "exact_selection", lambda *a: [1, 2])
    assert recovery.select_once(out, c, protocol(), "selection_reduced",
                                {"budget_gpu_seconds": 999}, {}, []) == [1, 2]
    assert core.read(out / "selection_reduced/autograd-recovery.json")["to"] == "autograd"


def test_recovery_rejects_deleted_costs_and_changed_binding(tmp_path):
    out, c = source(tmp_path)
    directory = failed_score(out)
    p = protocol()
    recovery.begin_recovery(out, c, p, "selection_reduced", recovery.calibration_failure(directory))
    recovery.validate_recovery(out, p, "selection_reduced")
    with pytest.raises(ValueError, match="binding"):
        recovery.validate_recovery(out, {**p, "recent_window": 21}, "selection_reduced")
    (directory / "cost.jsonl").write_text("")
    with pytest.raises(ValueError, match="ledger"):
        recovery.validate_recovery(out, p, "selection_reduced")


@pytest.mark.parametrize("artifact", ["execution.json", "result.json", "policy"])
def test_cannot_recover_after_selection_is_committed(tmp_path, artifact):
    out, c = source(tmp_path)
    directory = failed_score(out)
    (directory / artifact).touch()
    with pytest.raises(ValueError, match="subset is frozen"):
        recovery.begin_recovery(out, c, protocol(), "selection_reduced", recovery.calibration_failure(directory))


def test_held_out_test_protocol_is_not_amended(tmp_path):
    out, c = source(tmp_path)
    directory = failed_score(out)
    with pytest.raises(ValueError, match="development"):
        recovery.begin_recovery(out, c, protocol("test"), "selection_full", recovery.calibration_failure(directory))


def test_historical_abort_is_not_a_current_calibration_failure(tmp_path):
    out, _ = source(tmp_path)
    directory = failed_score(out)
    (directory / "score-3.log").write_text(ERROR + "\n[abort] CUDA out of memory\n")
    assert recovery.calibration_failure(directory) is None


def test_closed_cost_is_required_before_backend_switch(tmp_path):
    out, c = source(tmp_path)
    directory = failed_score(out)
    base.journal(directory / "cost.jsonl", {"event_id": "open", "state": "started", "phase": "score",
        "ledger": "deployment", "gpus": 4, "gpu_type": "H100", "time": 0})
    with pytest.raises(ValueError, match="unclosed"):
        recovery.begin_recovery(out, c, protocol(), "selection_reduced", {})


def test_summary_distinguishes_repaired_selector_costs(tmp_path, monkeypatch):
    out, _ = source(tmp_path)
    root = tmp_path / "summary"
    core.atomic_json(root / "points/p0/selection_reduced/autograd-recovery.json", {})
    original = {"points": [{"scope": {"selector": "low_order"}, "branches": {}}]}
    monkeypatch.setattr(recovery, "_summarize", lambda _: original)
    result = recovery.summarize(root)
    assert result["points"][0]["scope"]["selector"] == "low_order:autograd-recovery-v1"
    assert result["numerical_protocol"] == "finite_with_audited_autograd_recovery/v1"
    assert core.read(root / "study.json") == result
