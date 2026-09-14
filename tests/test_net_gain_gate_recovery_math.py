"""Run the real four-shard scoring pipeline on a small CPU model."""

import pytest

torch = pytest.importorskip("torch")

import low_order_experiment as low
import net_gain_gate_recovery as recovery
from test_low_order_backend import TinyLoRA
from test_low_order_experiment import fixture

base, core = recovery.base, recovery.core


def test_exact_recovery_scores_all_shards_without_finite_probes(tmp_path, monkeypatch):
    run, evaluation, _ = fixture(tmp_path, monkeypatch)
    model = TinyLoRA()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.save(optimizer.state_dict(), run / "policy_step_100/optimizer.pt")
    parent = core.read(run / "policy_step_100/policy_train.json")
    parent["optimizer_sha256"] = base.digest(run / "policy_step_100/optimizer.pt")
    core.atomic_json(run / "policy_step_100/policy_train.json", parent)
    before = {str(p): base.digest(p) for p in run.rglob("*") if p.is_file()}
    c = {"source_run": str(run), "config": core.read(run / "run_config.json"),
         "scope": {"gpu_type": "H100", "selector": "low_order"}, "eval_k": 8,
         "evaluation": {"val": core.read(evaluation)["test"], "provenance": core.read(evaluation)["provenance"]}}
    out, arm = tmp_path / "gate-point", "selection_reduced"
    core.atomic_json(out / arm / "autograd-recovery.json", {"test": "isolated scoring fixture"})
    private = recovery.private_dir(out, arm)
    old = out / "selector-work" / arm / "scores/p0.json"
    core.atomic_json(old, {"derivative": "finite", "must_not_reuse": True})
    monkeypatch.setattr(low.backend, "load_current", lambda *a: (TinyLoRA(), None))
    monkeypatch.setattr(low.backend, "calibrate", lambda *a, **k: pytest.fail("finite calibration repeated"))
    monkeypatch.setattr(low.backend, "finite_directional", lambda *a, **k: pytest.fail("finite scores used"))
    real_meter, stages = base.meter, []
    def cpu_meter(directory, name, gpu_type, **kwargs):
        if kwargs.get("commands"):
            assert kwargs["timeout"] <= (10000 - base.spent(directory)) / 4
            assert len(kwargs["commands"]) == 1
            commands = kwargs.pop("commands")
            assert commands[0][0][1] == str(recovery.MEMORY_WORKER)
            assert commands[0][0][2] == "supervise" and commands[0][1] == "0,1,2,3"
            point = next(low.entries(private / "scoring"))
            worker = low.validation_worker if name == "autograd-validation" else low.score_worker
            assert all(command[command.index("--stage") + 1] in {"score", "validation"} for command, _ in commands)
            kwargs.pop("env")
            kwargs.pop("timeout")
            kwargs["action"] = lambda: [worker(point, shard) for shard in range(4)]
            stages.append(name)
        return real_meter(directory, name, gpu_type, **kwargs)
    monkeypatch.setattr(base, "meter", cpu_meter)
    indices = recovery.exact_selection(out, c, arm, 10000, {}, list("0123"))
    assert len(indices) == 4
    point = next(low.entries(private / "scoring"))
    assert core.read(point / "experiment.json")["derivative"] == "autograd"
    scores = [core.read(path) for path in (point / "scores").glob("p*.json")]
    assert len(scores) == 40 and all(row["derivative"] == "autograd" for row in scores)
    assert all(row["cost"]["candidate_backward_passes"] == 8 for row in scores)
    assert stages == ["autograd-validation", "autograd-score"]
    assert core.read(old) == {"derivative": "finite", "must_not_reuse": True}
    assert before == {str(p): base.digest(p) for p in run.rglob("*") if p.is_file()}
    spent = base.spent(out / arm)
    assert 0 < spent < 10000
    monkeypatch.setattr(base, "meter", lambda *a, **k: pytest.fail("finished scoring repeated"))
    assert recovery.exact_selection(out, c, arm, 10000, {}, []) == indices
    assert base.spent(out / arm) == spent


def test_exhausted_cap_does_not_start_autograd(tmp_path, monkeypatch):
    out, arm = tmp_path / "point", "selection_reduced"
    base.meter(out / arm, "failed-finite-work", "H100", action=lambda: None)
    paid = base.spent(out / arm)
    monkeypatch.setattr(low, "prepare", lambda *a, **k: pytest.fail("over-budget scoring started"))
    with pytest.raises(ValueError, match="does not reset its budget"):
        recovery.exact_selection(out, {}, arm, paid, {}, [])
    assert base.spent(out / arm) == paid
