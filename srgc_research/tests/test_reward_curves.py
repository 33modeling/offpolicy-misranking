"""Intermediate measurements must survive export/resume without changing learning."""
import json
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from srgc_rebuttal.runtime import atomic_json
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.tests.test_reference import ScriptedBackend
from srgc_research.dispatch import model_results
from srgc_research.dispatch import reward_curves as curves
from srgc_research.dispatch.reward_curve_results import collect
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 - pytest fixture registration
)
from srgc_research.tests.test_model_results import finish, read_result


def history(start=26, stop=50):
    return [{"checkpoint": step-1, "completed_updates": step, "selector": "sr",
             "metrics": {"sample_reward": .5}, "raw_responses": ["never export"]}
            for step in range(start, stop+1)]


@pytest.mark.parametrize("bad", [[], history(27), history(stop=49), [None]])
def test_missing_or_noncontiguous_training_history_is_rejected(bad):
    with pytest.raises(ValueError):
        curves.training_points(bad, start=25, stop=50)


@pytest.mark.parametrize("model", ["gemma4", "gemma4_text", "llama31"])
def test_only_training_rank_is_wrapped(model):
    cmd = ["python", "runner.py", "--module", f"srgc_research.dispatch.{model}.rank", "--stage", "train", "--task", "sr"]
    changed = curves.recording_command(cmd)
    assert changed[3:6] == ["srgc_research.dispatch.reward_curve_rank", "--model", model]
    assert changed[6:] == cmd[4:]
    assert curves.recording_command([x if x != "train" else "cache" for x in cmd]) == [x if x != "train" else "cache" for x in cmd]


def test_existing_history_is_exported_for_all_models_without_gpu(cohort):
    folder = finish(cohort, "math")
    atomic_json(folder / "on_policy-progress.json", {"seed": 5, "arm": "on_policy", "step": 50,
        "sampling_protocol": Engine.SAMPLING_PROTOCOL, "history": history()})
    assert model_results.export(cohort.model, "math", environment=cohort.env) == 0
    row = next(x for x in read_result(cohort)["datasets"]["math"]["reward_curves"] if x["seed"] == 5 and x["arm"] == "on_policy")
    assert len(row["training_rewards"]) == 25
    assert row["training_windows_25"][0]["reward_percent"] == 50
    assert [p["step"] for p in row["held_out_rewards"]] == [275]
    assert row["missing_held_out_steps"] == list(range(25, 275, 25))
    assert "raw_responses" not in json.dumps(row)


def make_engine(folder, measured):
    torch = pytest.importorskip("torch")
    from srgc_rebuttal.timing import torch_meter

    class Backend(ScriptedBackend):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(1, 1)
            self.device = torch.device("cpu")
            self.cost_meter = torch_meter(cuda=False)
            self.evaluations = []
        def train(self, ids, **kwargs):
            self.parameter += random.random() + np.random.random() + torch.rand(()).item()
            self.optimizer_steps += 1
            return {"sample_reward": .5}
        def evaluate(self, ids, *, seed, responses):
            self.evaluations.append((list(ids), seed, responses, self.model.training))
            random.random(); np.random.random(); torch.rand(10)
            return {rid: .5 for rid in ids}
        def state_dict(self):
            return {**super().state_dict(), "optimizer": {"steps": self.optimizer_steps}}

    backend = Backend()
    ids = [f"s{i}" for i in range(40)] + [f"o{i}" for i in range(360)]
    cache = {i: [0, 1]*4 for i in ids}
    plan = {"shared_prefix_updates": 25, "total_updates": 275}
    data = {"evaluation_ids": ["held-a", "held-b"]}
    identity = {"seed": 3, "plan_sha256": "a"*64, "input_sha256": "b"*64, "implementation_sha256": "c"*64}
    atomic_json(folder / "run.json", identity)
    base = curves.recording_engine(Engine, folder, "sr", folder / "plan.json", plan, data) if measured else Engine
    engine = base(backend, ids, ["v"], cache, arm="sr", config=Config(projection_dim=2), step=25)
    return engine, backend, plan, data, identity


def seed_rng():
    import torch
    random.seed(98); np.random.seed(98); torch.manual_seed(98)


def test_actual_boundaries_are_durable_and_do_not_change_training(tmp_path):
    torch = pytest.importorskip("torch")
    seed_rng(); plain, baseline, *_ = make_engine(tmp_path / "plain", False); plain.run_until(50)
    seed_rng(); engine, backend, plan, data, identity = make_engine(tmp_path / "measured", True); engine.run_until(50)
    assert backend.parameter == baseline.parameter
    assert backend.optimizer_steps == baseline.optimizer_steps == 25
    assert backend.model.training
    assert len(backend.evaluations) == 2
    assert all(ids == data["evaluation_ids"] and n == 8 and mode is False for ids, _, n, mode in backend.evaluations)
    folder = tmp_path / "measured"
    state = torch.load(folder / "reward-curves/sr/step-50.pt", weights_only=False)
    assert state["step"] == 50 and state["evaluation_only"]
    assert "optimizer" not in state["backend"]
    assert state["backend"]["parameter"] == backend.parameter
    row = collect(folder, "sr", plan, identity, data["evaluation_ids"], lambda p: json.loads(p.read_text()) if p.exists() else None)
    assert [p["step"] for p in row["held_out_rewards"]] == [25, 50]
    assert len(row["training_rewards"]) == 25
    assert all(c["scope"] == "all_attempts_at_this_step" for c in row["diagnostic_costs"])
    engine.capture_boundary()
    assert len(backend.evaluations) == 2
    checkpoint = folder / "reward-curves/sr/step-50.pt"
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="binding"):
        engine.capture_boundary()


def test_final_boundary_is_archived_without_repeating_final_eval(tmp_path):
    pytest.importorskip("torch")
    engine, backend, *_ = make_engine(tmp_path, True)
    engine.step = 275
    engine.capture_boundary()
    assert backend.evaluations == []
    assert (tmp_path / "reward-curves/sr/step-275.pt").exists()


def test_cpu_recording_uses_no_gpu_allocation_even_on_a_cuda_host(tmp_path):
    torch = pytest.importorskip("torch")
    engine, backend, *_ = make_engine(tmp_path, True)
    assert backend.gpu_count == 0
    with patch.object(torch.cuda, "is_available", return_value=True), \
            patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CPU recorder touched CUDA")):
        engine.capture_boundary()
    directory = tmp_path / "reward-curves/prefix"
    point = json.loads((directory / "step-25.json").read_text())
    assert point["reward"] == .5
    receipts = list((directory / "cost-receipts/step-25").glob("*.json"))
    assert receipts
    for path in receipts:
        event = json.loads(path.read_text())
        assert event["gpu_count"] == 0
        if event["state"] == "finished":
            assert event["gpu_seconds"] == 0


def test_interrupted_checkpoint_is_replaced_from_actual_weights(tmp_path):
    pytest.importorskip("torch")
    engine, backend, *_ = make_engine(tmp_path, True)
    point = tmp_path / "reward-curves/prefix/step-25.pt"
    point.parent.mkdir(parents=True)
    point.write_bytes(b"unfinished archive")
    engine.capture_boundary()
    assert len(backend.evaluations) == 1
    assert (point.with_suffix(".json")).exists()

@pytest.mark.parametrize("model", ["gemma4", "gemma4_text", "llama31", "qwen35"])
def test_public_rank_installs_recording_on_the_frozen_checkpoint_factory(tmp_path, model):
    import sys
    from types import SimpleNamespace

    from scripts import srgc_step_checkpoints as checkpoints
    from srgc_research.dispatch import reward_curve_rank
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"evaluation_ids": ["held"]}))
    observed = []
    def original(base, *args, **kwargs):
        return base
    def frozen_main():
        import srgc_step_checkpoints as legacy
        cls = legacy.checkpoint_engine(Engine, tmp_path, "sr", {}, total_updates=275)
        observed.append(cls)
        assert "--model" not in sys.argv
    with patch.object(checkpoints, "checkpoint_engine", original), \
            patch("srgc_rebuttal.plan.load_plan", return_value={"shared_prefix_updates": 25, "total_updates": 275}), \
            patch("srgc_rebuttal.plan.input_path", return_value=bundle), \
            patch.object(reward_curve_rank.importlib, "import_module", return_value=SimpleNamespace(main=frozen_main)) as entry, \
            patch.object(sys, "argv", ["rank", "--model", model, "--plan", str(tmp_path / "plan.json"), "--seed", "3", "--task", "sr", "--stage", "train"]):
        reward_curve_rank.main()
        assert checkpoints.checkpoint_engine is original
        entry.assert_called_once_with("scripts.srgc_qwen35_rank_runtime" if model == "qwen35"
                                      else f"srgc_research.dispatch.{model}.rank")
    assert len(observed) == 1 and issubclass(observed[0], Engine)
    assert hasattr(observed[0], "capture_boundary")


@pytest.mark.parametrize("entry", ["srgc_qwen35_rank.py", "srgc_qwen35_rank_runtime.py"])
def test_qwen_training_arguments_and_nontraining_entries_are_preserved(entry):
    command = ["python", "-m", "torch.distributed.run", "--nproc_per_node", "4",
               f"/repo/scripts/{entry}", "--stage", "train", "--plan", "/frozen/plan.json", "--seed", "5", "--task", "switch"]
    wrapped = curves.recording_command(command)
    assert Path(wrapped[5]).name == "srgc_qwen35_reward_curve_rank.py"
    assert wrapped[:5] == command[:5] and wrapped[6:] == command[6:]
    for stage in ("cache", "smoke"):
        other = [stage if arg == "train" else arg for arg in command]
        assert curves.recording_command(other) == other


def test_saved_measurement_cannot_be_reused_for_different_actual_weights(tmp_path):
    pytest.importorskip("torch")
    engine, backend, *_ = make_engine(tmp_path, True)
    engine.capture_boundary()
    backend.parameter += 1
    with pytest.raises(ValueError, match="resumed learner"):
        engine.capture_boundary()


def test_shared_prefix_training_reward_is_included_at_step_25(cohort):
    from srgc_rebuttal.plan import input_path, load_plan
    folder = finish(cohort, "math")
    plan = load_plan(cohort.plans["math"])
    data = json.loads(input_path(cohort.plans["math"], plan, 5).read_text())
    identity = json.loads((folder / "run.json").read_text())
    identity.pop("status")
    protocol = {**identity, "schema": curves.SCHEMA, "arm": "prefix", "extension_sha256": "e"*64,
                "interval_updates": 25, "responses_per_prompt": 8, "question_count": len(data["evaluation_ids"]),
                "evaluation_ids_sha256": curves.fingerprint(data["evaluation_ids"]), "stream": "reporting-evaluation",
                "training_batch_reward_is_held_out": False}
    rows = history(1, 25)
    for row in rows: row["selector"] = "on_policy"
    points = curves.training_points(rows, start=0, stop=25)
    atomic_json(folder / "reward-curves/prefix/protocol.json", protocol)
    atomic_json(folder / "reward-curves/prefix/training.json", {**protocol, "step": 25, "points": points})
    assert model_results.export(cohort.model, "math", environment=cohort.env) == 0
    row = next(x for x in read_result(cohort)["datasets"]["math"]["reward_curves"] if x["seed"] == 5 and x["arm"] == "on_policy")
    assert row["shared_prefix_training_recorded_updates"] == 25
    assert [p["step"] for p in row["training_rewards"]] == list(range(1, 26))
    assert all(p["shared_prefix"] for p in row["training_rewards"])
    assert row["training_windows_25"][0]["end_step"] == 25


@pytest.fixture(autouse=True)
def periodic_evaluation_compatibility(monkeypatch):
    """These legacy measurements explicitly exercise the opt-in evaluation path."""
    monkeypatch.setenv("SRGC_INTERMEDIATE_EVAL", "1")
