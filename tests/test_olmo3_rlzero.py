from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import _extract_code, extract_answer, reward
from model_matrix import RUNTIME_DEFAULTS, _load_config, _load_specs
from qualify_rlzero_signal import read_rewards
from rollout import chat_ids, render_prompt


class RecordingTokenizer:
    eos_token_id = 0

    def __init__(self) -> None:
        self.text: str | None = None

    def __call__(self, text: str, **kwargs) -> SimpleNamespace:
        self.text = text
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("raw OLMo RL-Zero must not use a missing chat template")


class CompletionTokenizer(RecordingTokenizer):
    def __call__(self, text: str, **kwargs) -> SimpleNamespace:
        self.text = text
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": True}
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))


def test_olmo3_contract_is_raw_base_grpo_with_two_verifier_domains() -> None:
    config_path = ROOT / "configs/olmo3_rlzero.json"
    config = _load_config(config_path)
    specs = _load_specs(config_path)
    model = specs["olmo3-7b-base"]
    experiment = config["experiment"]
    assert model["repository"] == "allenai/Olmo-3-1025-7B"
    assert model["revision"] == "a81bae42db3975be1671e27b9c9a56da1a9f980f"
    assert model["model_type"] == "olmo3"
    assert model["prompt_format"] == "olmo_rlzero"
    assert experiment["policy_method"] == "grpo"
    assert experiment["datasets"] == ["math500", "mbpp"]
    assert experiment["seeds"] == [0, 1, 2, 3, 4]
    assert experiment["drifts"] == [0, 25, 100, 400]
    assert experiment["max_new_tokens"] == 2048
    assert experiment["temperature"] == 1.0
    assert experiment["top_p"] == 1.0
    assert experiment["grpo"] == {
        "world_size": 4,
        "group_size": 8,
        "clip_epsilon": 0.2,
        "learning_rate": 1e-5,
        "reference_kl_beta": 0.0,
        "epochs_per_batch": 1,
        "max_grad_norm": 1.0,
        "advantage_epsilon": 1e-4,
        "lora_rank": 16,
        "lora_alpha": 32,
    }
    assert experiment.get("runtime", RUNTIME_DEFAULTS) == RUNTIME_DEFAULTS


def test_h100_profile_changes_runtime_only_and_uses_larger_batches() -> None:
    baseline = _load_config(ROOT / "configs/olmo3_rlzero.json")
    h100 = _load_config(ROOT / "configs/olmo3_rlzero_h100.json")
    baseline_experiment = dict(baseline["experiment"])
    h100_experiment = dict(h100["experiment"])
    runtime = h100_experiment.pop("runtime")
    assert baseline_experiment == h100_experiment
    assert runtime == {
        "generation_batch": 8,
        "logprob_micro_batch": 4,
        "gradient_checkpointing": True,
    }


def test_olmo3_documentation_matches_single_epoch_contract() -> None:
    runbook = (ROOT / "docs/OLMO3_RLZERO_RUNBOOK.md").read_text()
    experiment_doc = (ROOT / "docs/EXPERIMENT.md").read_text()
    readme = (ROOT / "README.md").read_text()

    for text in (runbook, experiment_doc, readme):
        assert "one optimizer epoch" in text or "single optimizer epoch" in text
        assert "clip_epsilon=0.2" in text
        assert "not claimed" in text or "does not" in text
    assert "maximum `2048` new tokens" in runbook
    assert "learning rate `1e-5`" in runbook
    assert "rank `16`, alpha `32`, dropout `0`" in runbook


@pytest.mark.parametrize(
    ("mode", "lead", "tail"),
    [
        (
            "olmo_rlzero_math",
            "Solve the following problem step by step.",
            'Remember to put your answer on its own line after "Answer:"',
        ),
        (
            "olmo_rlzero_code",
            "Solve the following code problem step by step.",
            "Remember to put your solution inside the ```\npython\nCODE\n``` tags",
        ),
    ],
)
def test_official_rlzero_prompts_bypass_chat_template(
    monkeypatch: pytest.MonkeyPatch, mode: str, lead: str, tail: str
) -> None:
    tokenizer = RecordingTokenizer()
    monkeypatch.setenv("OM_PROMPT_FORMAT", mode)
    ids = chat_ids(tokenizer, "QUESTION")
    assert ids.tolist() == [1, 2, 3]
    assert tokenizer.text == render_prompt("QUESTION", mode)
    assert tokenizer.text.startswith(lead)
    assert "\n\nQUESTION\n\n" in tokenizer.text
    assert tokenizer.text.endswith(tail)


def test_base_model_completion_prompt_uses_tokenizer_special_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = CompletionTokenizer()
    monkeypatch.setenv("OM_PROMPT_FORMAT", "verifiable_completion")
    ids = chat_ids(tokenizer, "Question with an explicit #### answer contract.")
    assert ids.tolist() == [1, 2, 3]
    assert tokenizer.text == (
        "Question with an explicit #### answer contract.\n\nResponse:\n"
    )


def test_rlzero_answer_and_code_parsers_follow_the_released_output_contract() -> None:
    assert extract_answer("Answer: mentioned early\nwork\nAnswer: $1,024$.") == "1024"
    assert reward("work\nAnswer: $0.5$", r"\frac{1}{2}") == 0.0
    assert _extract_code("```\npython\ndef f():\n    return 1\n```").startswith("def f")


def test_signal_gate_requires_exact_group_coverage_and_finds_mixed_groups(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollouts.jsonl"
    rows = [
        {"prompt_idx": prompt, "reward": reward_value}
        for prompt, rewards in enumerate(([0, 1], [0, 0]))
        for reward_value in rewards
    ]
    rollout.write_text("".join(json.dumps(row) + "\n" for row in rows))
    stats = read_rewards(rollout, prompt_count=2, group_size=2)
    assert stats["correct"] == 1
    assert stats["incorrect"] == 3
    assert stats["mixed_prompt_groups"] == 1

    rollout.write_text("".join(json.dumps(row) + "\n" for row in rows[:-1]))
    with pytest.raises(ValueError, match="coverage mismatch"):
        read_rewards(rollout, prompt_count=2, group_size=2)
