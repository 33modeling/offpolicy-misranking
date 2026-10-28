"""Exercise the public recorder with real CPU optimizer state and eval failures.

These are execution tests with a tiny network, not Gemma accuracy measurements.
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from scripts import srgc_step_checkpoints as checkpoints
from srgc_rebuttal.runtime import atomic_json
from srgc_rebuttal.srgc import Config, Engine, stream_seed
from srgc_research.dispatch import reward_curve_rank
from srgc_research.tests.test_node_kill_resume import OptimizingBackend, run_to


class EvaluatingBackend(OptimizingBackend):
    def __init__(self):
        super().__init__()
        self.evaluations = []

    def evaluate(self, ids, *, seed, responses):
        self.evaluations.append((seed, self.optimizer_steps))
        return {rid: .5 for rid in ids}


def make_engine(folder, task, *, data=None):
    folder.mkdir(parents=True, exist_ok=True)
    ids = [f"s{i}" for i in range(40)] + [f"o{i}" for i in range(360)]
    cache = {rid: [0, 1] * 4 for rid in ids}
    plan = {"shared_prefix_updates": 25, "total_updates": 275}
    data = data or {"evaluation_ids": ["held-a", "held-b"], "cached_rewards": cache}
    identity = {"seed": 5, "plan_sha256": "a" * 64,
                "input_sha256": "b" * 64, "implementation_sha256": "c" * 64}
    if not (folder / "run.json").exists():
        atomic_json(folder / "run.json", identity)
    bundle = folder / "bundle.json"
    atomic_json(bundle, data)
    classes = []

    def frozen_entry():
        classes.append(checkpoints.checkpoint_engine(
            Engine, folder, task, {"attention": "eager"},
            total_updates=25 if task == "prefix" else 275))

    # Invoke the actual public recorder entry; replace only model/rank startup.
    with patch("srgc_rebuttal.plan.load_plan", return_value=plan), \
            patch("srgc_rebuttal.plan.input_path", return_value=bundle), \
            patch.object(reward_curve_rank.importlib, "import_module",
                         return_value=SimpleNamespace(main=frozen_entry)), \
            patch.object(sys, "argv", ["rank", "--model", "gemma4_it", "--stage", "train",
                                      "--plan", str(folder / "plan.json"), "--seed", "5", "--task", task]):
        reward_curve_rank.main()
    backend = EvaluatingBackend()
    engine = classes[0](backend, ids, ["reference"], cache,
                        arm="on_policy" if task == "prefix" else task,
                        config=Config(seed=5, projection_dim=2),
                        step=0 if task == "prefix" else 25)
    backend.strengths = {rid: float(len(ids) - n) for n, rid in enumerate(engine.sr_ranked_ids)}
    return engine


@pytest.mark.parametrize("task", ["prefix", "random", "sr", "on_policy", "switch"])
def test_eval_failure_preserves_completed_update_and_restore_finishes_eval_without_training(tmp_path, task):
    folder = tmp_path / "interrupted"
    engine = make_engine(folder, task)
    boundary = 25 if task == "prefix" else 50
    evaluation_seed = stream_seed(5, boundary, "reporting-evaluation")
    original = engine.backend.evaluate

    def fail_boundary(ids, **kwargs):
        if kwargs["seed"] == evaluation_seed:
            raise RuntimeError("interrupted boundary evaluation")
        return original(ids, **kwargs)

    with patch.object(engine.backend, "evaluate", side_effect=fail_boundary), \
            pytest.raises(RuntimeError, match="interrupted boundary evaluation"):
        run_to(engine, folder, task, boundary)
    checkpoint = folder / f"{task}-latest.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["step"] == boundary
    assert state["backend"]["optimizer"]["state"]
    assert json.loads((folder / f"reward-curves/{task}/training.json").read_text())["step"] == boundary
    before = checkpoint.read_bytes()

    restored = make_engine(folder, task)
    restored.load_state_dict(state)
    # Prefix's training loop is already exhausted at 25. The missing eval must
    # finish in restore, or the unmodified runner would publish prefix-ready
    # without ever revisiting update(). Continuations likewise must not redo 50.
    assert restored.step == boundary
    assert restored.backend.optimizer_steps == state["backend"]["optimizer_steps"]
    assert restored.backend.evaluations == [(evaluation_seed, restored.backend.optimizer_steps)]
    assert (folder / f"reward-curves/{task}/step-{boundary}.json").exists()
    assert checkpoint.read_bytes() == before

    stop = boundary if task == "prefix" else boundary + 1
    run_to(restored, folder, task, stop)
    baseline_folder = tmp_path / "baseline"
    baseline = make_engine(baseline_folder, task)
    run_to(baseline, baseline_folder, task, stop)
    torch.testing.assert_close(restored.backend.state_dict(), baseline.backend.state_dict(), rtol=0, atol=0)
    assert [r["completed_updates"] for r in restored.history] == list(
        range(1 if task == "prefix" else 26, stop + 1))


@pytest.mark.parametrize("task", ["random", "sr", "on_policy", "switch"])
def test_all_periodic_evals_reach_275_and_final_resume_performs_no_optimizer_update(tmp_path, task):
    engine = make_engine(tmp_path, task)
    initial = {k: v.clone() for k, v in engine.backend.model.state_dict().items()}
    run_to(engine, tmp_path, task, 275)
    assert engine.step == 275 and engine.backend.optimizer_steps == 250
    assert any(not torch.equal(initial[k], v) for k, v in engine.backend.model.state_dict().items())
    # 25..250 are checkpoint evaluations; 275 is reserved for the runner's
    # final evaluation, which the recorder must never duplicate.
    assert [seed for seed, _ in engine.backend.evaluations] == [
        stream_seed(5, step, "reporting-evaluation") for step in range(25, 275, 25)]
    state = torch.load(tmp_path / f"{task}-latest.pt", map_location="cpu", weights_only=False)
    restored = make_engine(tmp_path, task)
    restored.load_state_dict(state)
    run_to(restored, tmp_path, task, 275)
    assert restored.backend.evaluations == []
    assert restored.backend.optimizer_steps == 250
    torch.testing.assert_close(restored.backend.state_dict(), engine.backend.state_dict(), rtol=0, atol=0)
    assert (tmp_path / f"reward-curves/{task}/step-275.pt").exists()


def test_failed_training_scalar_export_cannot_advance_boundary_checkpoint(tmp_path):
    engine = make_engine(tmp_path, "random")
    run_to(engine, tmp_path, "random", 49)
    checkpoint = tmp_path / "random-latest.pt"
    before = checkpoint.read_bytes()
    with patch.object(engine, "save_training_rewards", side_effect=OSError("disk full")), \
            pytest.raises(OSError, match="disk full"):
        engine.update()
    assert checkpoint.read_bytes() == before
    assert torch.load(checkpoint, weights_only=False)["step"] == 49


@pytest.mark.parametrize("periodic", [False, True])
def test_original_runner_publishes_done_for_all_arms_and_recovers_final_endpoint_failure(tmp_path, monkeypatch, periodic):
    monkeypatch.setenv("SRGC_INTERMEDIATE_EVAL", "1" if periodic else "0")
    from srgc_rebuttal import reports, run_experiment as runner, torch_backend
    from srgc_rebuttal.plan import DEFAULT_PLAN
    from srgc_research.dispatch.qwen_status import cell

    plan = json.loads(DEFAULT_PLAN.read_text())
    plan.update(seeds=[5], model="cpu-optimizer-fixture", model_revision="a" * 40,
                input_pattern="../runs/seed-{seed}/bundle.json", output_root="../runs")
    path = tmp_path / "experiments/plan.json"
    atomic_json(path, plan)
    candidates = [f"c{i}" for i in range(400)]
    validation = [f"v{i}" for i in range(100)]
    evaluation = [f"e{i}" for i in range(300)]
    data = {"schema": "srgc-inputs-v1", "candidate_ids": candidates,
            "validation_pool_ids": validation, "ranking_validation_ids": validation[:50],
            "evaluation_ids": evaluation, "cached_rewards": {rid: [0, 1] * 4 for rid in candidates},
            "records": {rid: {"question": rid, "prompt": f"CPU fixture {rid}", "answer": "1"}
                        for rid in candidates + validation + evaluation},
            "provenance": {"scope": "synthetic CPU execution test; no Gemma measurement"}}
    folder = tmp_path / "runs/seed-5"
    backends = []

    def backend(*args, **kwargs):
        result = EvaluatingBackend()
        result.cost_meter = kwargs["cost_meter"]
        backends.append(result)
        return result

    peft = SimpleNamespace(LoraConfig=lambda **kwargs: kwargs,
                           get_peft_model=lambda model, config: model)
    original_json = runner.atomic_json

    with patch.dict(sys.modules, {"peft": peft}), \
            patch.object(runner, "load_plan", return_value={**plan, "world_size": 0, "projection_dim": 2}), \
            patch.object(runner, "initialize", return_value=(0, 0)), \
            patch.object(torch.distributed, "broadcast_object_list"), \
            patch.object(torch.distributed, "destroy_process_group"), \
            patch.object(runner.importlib.metadata, "version", return_value="cpu-test"), \
            patch.object(runner, "load_model", return_value=(torch.nn.Linear(1, 1), None)), \
            patch.object(torch_backend, "TorchBackend", side_effect=backend):
        for task in ["prefix", *plan["arms"]]:
            cls = type(make_engine(folder, task, data=data))
            if task == "prefix":
                (folder / "run.json").unlink()  # Let the real runner write its actual identity.
            with patch.object(runner, "Engine", cls), \
                    patch.object(sys, "argv", ["run", "--plan", str(path), "--seed", "5",
                                              "--task", task, "--resume"]):
                if task == plan["arms"][-1]:
                    def fail_endpoint(destination, value):
                        if destination.name == f"{task}-endpoint.json":
                            raise OSError("interrupted endpoint publication")
                        return original_json(destination, value)
                    with patch.object(runner, "atomic_json", side_effect=fail_endpoint), \
                            pytest.raises(OSError, match="interrupted endpoint publication"):
                        runner.main()
                    assert torch.load(folder / f"{task}-latest.pt", weights_only=False)["step"] == 275
                    assert not (folder / f"{task}-endpoint.json").exists()
                runner.main()
                if task != "prefix":
                    assert backends[-1].optimizer_steps == 275
                    count = len(backends)
                    runner.main()  # A validated endpoint skips model/optimizer creation entirely.
                    assert len(backends) == count
        assert len(backends[-1].training) == 0  # Last attempt only restored 275 and evaluated.
        assert len(backends[-1].evaluations) == 1
        assert json.loads((folder / "run.json").read_text())["status"] == "complete"
    if not periodic:
        assert all(seed == stream_seed(5, 275, "reporting-evaluation")
                   for backend in backends for seed, _ in backend.evaluations)
        assert not list((folder / "reward-curves").glob("*/step-25.json"))
        report = reports.snapshot(path)
        assert report["complete"] and report["errors"] == []
        assert all(cell(row).startswith("DONE") for row in report["tasks"])


@pytest.fixture(autouse=True)
def periodic_evaluation_compatibility(monkeypatch):
    """These legacy measurements explicitly exercise the opt-in evaluation path."""
    monkeypatch.setenv("SRGC_INTERMEDIATE_EVAL", "1")
