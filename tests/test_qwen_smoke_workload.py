import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import srgc_qwen35_smoke as smoke


class Tokenizer:
    def encode(self, text, **kwargs):
        return [int(text)] if text in (" 2", " 3") else list(range(len(text)))


def test_prompt_selection_uses_only_nonempty_candidates():
    data = {"candidate_ids": ["long", "empty", "short"], "records": {
        "long": {"prompt": "long candidate"}, "empty": {"prompt": "  "},
        "short": {"prompt": "short"}, "evaluation": {"prompt": "e"},
        "validation": {"prompt": "v"}, "long_evaluation": {"prompt": "e" * 5000}}}
    assert smoke.candidate_prompt(data, Tokenizer()) == ("short", "short", 5)


def test_empty_candidates_fail_closed():
    with pytest.raises(ValueError, match="nonempty candidate"):
        smoke.candidate_prompt({"candidate_ids": [], "records": {}}, Tokenizer())


@pytest.mark.parametrize("stage", ["train", "cache"])
def test_entry_rejects_experiment_stages(monkeypatch, stage):
    monkeypatch.setattr(sys, "argv", ["smoke", "--stage", stage, "--plan", "/unused"])
    with pytest.raises(SystemExit) as caught:
        smoke.main()
    assert caught.value.code == 2


def test_entry_patches_exact_rank_module_and_restores(monkeypatch):
    original = object()
    qwen = SimpleNamespace(smoke=original)
    calls = []

    def lightweight(path):
        calls.append(path)
        return "rank result"

    def rank_main():
        assert sys.modules["srgc_qwen35"].smoke is smoke.lightweight_smoke
        return sys.modules["srgc_qwen35"].smoke(Path("/unused"))

    monkeypatch.setattr(smoke, "lightweight_smoke", lightweight)
    monkeypatch.setitem(sys.modules, "srgc_qwen35", qwen)
    monkeypatch.setitem(sys.modules, "srgc_qwen35_rank", SimpleNamespace(main=rank_main))
    monkeypatch.setattr(sys, "argv", ["smoke", "--stage", "smoke", "--plan", "/unused"])
    assert smoke.main() == "rank result"
    assert calls == [Path("/unused")]
    assert qwen.smoke is original


class Tensor:
    device = "cpu"

    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, key):
        return Tensor(self.value[key])

    def __len__(self):
        return len(self.value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return Tensor(self.value.copy())


@pytest.mark.parametrize("fail_scoring", [False, True])
def test_smoke_runs_short_workload_and_preserves_failures(tmp_path, monkeypatch, capsys, fail_scoring):
    events = []
    plan_path = tmp_path / "plan.json"
    bundle_path = tmp_path / "first.json"
    bundle_path.write_text(json.dumps({"candidate_ids": ["candidate"], "records": {
        "candidate": {"prompt": "short"}, "evaluation": {"prompt": "e"}}}))
    plan = {"seeds": [5, 6, 7, 8, 9], "projection_dim": 4, "projection_seed": 0, "max_new_tokens": 2048}
    parameter = Tensor([1.0])
    failure = RuntimeError("scoring failed")

    def input_path(path, value, seed):
        assert (path, value, seed) == (plan_path, plan, 5)
        return bundle_path

    def primary(call):
        events.append("startup_collective")
        return call()

    def load_model(*args):
        events.append("model_load")
        return object(), Tokenizer()

    class Backend:
        def __init__(self, model, tokenizer, records, verifier, **kwargs):
            assert kwargs["max_new_tokens"] == 32
            assert records["p0"]["prompt"] == "short"
            self.train_parameters = [("adapter", parameter)]

        def _rollout(self, prompt, responses, seed):
            events.append("rollout")
            assert responses == 8
            return [Tensor([1, 2, 3])] * 8, None, 2

        def score_gradients(self, ids, **kwargs):
            events.append("scoring")
            sequences, rewards, start = self._rollout("p0", 8, 17)
            assert len(sequences) == kwargs["responses"] == 8
            assert all(len(sequence) - start == 32 for sequence in sequences)
            assert np.array_equal(rewards, [0., 1.] * 4)
            assert not np.array_equal(sequences[0].value, sequences[1].value)
            if fail_scoring:
                raise failure
            return {key: np.ones(4) for key in ids}

        def train(self, ids, **kwargs):
            events.append("update")
            assert kwargs["responses"] == 8 and kwargs["objective"] == "grpo"
            parameter.value += 1

    dist = SimpleNamespace(barrier=lambda: events.append("final_barrier"),
                           destroy_process_group=lambda: events.append("destroy"))
    torch = SimpleNamespace(distributed=dist, manual_seed=lambda seed: None,
        device=lambda kind, rank: (kind, rank), long="long",
        tensor=lambda value, **kwargs: Tensor(value),
        cat=lambda values: Tensor(np.concatenate([value.value for value in values])),
        isfinite=lambda value: np.isfinite(value.value),
        equal=lambda left, right: np.array_equal(left.value, right.value),
        cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0,
                             max_memory_reserved=lambda: 0))
    modules = {
        "torch": torch, "torch.distributed": dist,
        "peft": SimpleNamespace(LoraConfig=lambda **kwargs: kwargs, get_peft_model=object()),
        "srgc_rebuttal.distributed": SimpleNamespace(initialize=lambda world: (0, 0), primary=primary),
        "srgc_rebuttal.plan": SimpleNamespace(input_path=input_path),
        "srgc_qwen35": SimpleNamespace(MODEL="model", REVISION="revision", load_model=load_model,
            validate_extension=lambda path: plan, attach_adapter=lambda model, *args: model),
        "srgc_qwen35_memory": SimpleNamespace(QwenBackend=Backend),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("SRGC_QWEN_PLANS", json.dumps(["/must/not/be/read"]))
    if fail_scoring:
        with pytest.raises(RuntimeError) as caught:
            smoke.lightweight_smoke(plan_path)
        assert caught.value is failure
        assert events == ["startup_collective", "model_load", "rollout", "scoring"]
        assert "PASS:" not in capsys.readouterr().out
    else:
        smoke.lightweight_smoke(plan_path)
        assert events == ["startup_collective", "model_load", "rollout", "scoring", "update", "final_barrier", "destroy"]
        output = capsys.readouterr().out
        assert '"protocol": "candidate-readiness-v1"' in output
        for stage in ("init", "startup_collective", "model_load", "rollout", "scoring", "update", "final_barrier"):
            assert f'"stage": "{stage}"' in output
        assert "32-token synthetic smoke" in output
    assert plan["max_new_tokens"] == 2048
    assert list(tmp_path.iterdir()) == [bundle_path]
