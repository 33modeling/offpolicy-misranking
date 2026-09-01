"""Protocol regressions that do not require a model or GPU.

    PYTHONPATH=src .work/.venv-cu126/bin/python tests/test_protocol.py
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import _maybe_json_list
from experiment import (
    score_oracle_microgroups,
    split_validation_directions,
)
from grads import (
    ProjectionSpec,
    prompt_gradient,
    sequence_logprobs,
    sequence_logprobs_batch,
)
from hybrid import (
    _cut_prefixes,
    continue_rollouts_batch,
    validate_hybrid_cells,
)
from kcurve_floor import find_fresh_k
from select_rules import overlap_under_independent_ties, topk_count


def test_all_tie_overlap_is_chance():
    n = 256
    k = topk_count(n, 0.10)
    tied = {idx: 0.0 for idx in range(n)}
    summary = overlap_under_independent_ties(tied, tied, k, pairs=1_000)
    assert abs(summary.mean - k / n) < 0.01, (summary.mean, k / n)
    assert summary.low < summary.high


def test_oracle_uses_disjoint_ranking_and_reference_partitions():
    stack = torch.tensor(
        [[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 2 + [[1.0, 0.0]] * 2
    )
    val_groups = torch.tensor(
        [[1.0, 0.0]] * 8
    )
    val_r, val_a, val_b = split_validation_directions(val_groups)
    oracle, halves = score_oracle_microgroups(
        stack, val_r, val_a, val_b
    )
    assert halves == {"r": 1.0, "a": 0.0, "b": 1.0}
    assert oracle["score"] == 0.5
    with pytest.raises(ValueError, match="at least eight"):
        score_oracle_microgroups(
            stack[:6], val_r, val_a, val_b
        )


def test_hybrid_preserves_total_response_horizon():
    class Tok:
        pad_token_id = 0
        eos_token_id = 99

    class Model:
        device = "cpu"

        def __init__(self):
            self.budgets = []

        def generate(self, batch, attention_mask, **kwargs):
            del attention_mask
            budget = kwargs["max_new_tokens"]
            self.budgets.append(budget)
            suffix = torch.full((batch.shape[0], budget), 7, dtype=torch.long)
            return torch.cat([batch, suffix], dim=1)

    model = Model()
    prefixes = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([1, 2, 3, 4, 5, 6])]
    out = continue_rollouts_batch(
        model, Tok(), prefixes, max_new_tokens=5, temperature=1.0,
        resp_starts=[3, 3],
    )
    assert [seq.numel() - 3 for seq in out] == [5, 5]
    assert sorted(model.budgets) == [2, 3]


def test_hybrid_prefix_excludes_terminal_source_token():
    rows = [
        {"input_ids": torch.tensor([1, 2, 99]), "resp_start": 2},
        {"input_ids": torch.tensor([1, 2, 7, 8, 99]), "resp_start": 2},
    ]
    prefixes = _cut_prefixes(rows, 0.75)
    assert prefixes[0].tolist() == [1, 2]
    assert prefixes[1].tolist() == [1, 2, 7, 8]


def test_hybrid_cells_require_common_prompts_and_exact_k():
    cells = {
        cell: {
            prompt_idx: [{"cell": cell}, {"cell": cell}]
            for prompt_idx in (3, 5)
        }
        for cell in ("bb", "bp", "pb", "pp")
    }
    validate_hybrid_cells(cells, {3, 5}, 2)
    cells["pb"][5].pop()
    with pytest.raises(ValueError, match="expected K=2"):
        validate_hybrid_cells(cells, {3, 5}, 2)


def test_gradient_rejects_response_weight_truncation():
    sequences = [{"input_ids": torch.tensor([1, 2, 3, 4]), "resp_start": 2}]
    with pytest.raises(ValueError, match="response/weight length mismatch"):
        prompt_gradient(
            None, [], sequences, [torch.ones(1)], ProjectionSpec(dim=4)
        )


class ToyGradientModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 19):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, vocab_size)

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, input_ids, attention_mask=None):
        assert attention_mask is not None
        return SimpleNamespace(logits=self.embedding(input_ids))


def test_batched_gradient_and_logprob_paths_match_single_sequence_execution():
    torch.manual_seed(17)
    single_model = ToyGradientModel()
    batch_model = copy.deepcopy(single_model)
    sequences = [
        {"input_ids": torch.tensor([1, 2, 3, 4, 5]), "resp_start": 2},
        {"input_ids": torch.tensor([1, 6, 7, 8, 9, 10]), "resp_start": 3},
        {"input_ids": torch.tensor([1, 11, 12, 13]), "resp_start": 2},
    ]
    weights = [
        torch.linspace(0.2, 0.6, sequence["input_ids"].numel() - sequence["resp_start"])
        for sequence in sequences
    ]
    spec = ProjectionSpec(dim=64, seed=101, chunk=128)
    single = prompt_gradient(
        single_model,
        list(single_model.parameters()),
        sequences,
        weights,
        spec,
        micro_batch=1,
    )
    batched = prompt_gradient(
        batch_model,
        list(batch_model.parameters()),
        sequences,
        weights,
        spec,
        micro_batch=3,
    )
    torch.testing.assert_close(batched, single, rtol=1e-5, atol=1e-6)

    expected = [
        sequence_logprobs(single_model, row["input_ids"], row["resp_start"])
        for row in sequences
    ]
    actual = sequence_logprobs_batch(single_model, sequences, micro_batch=3)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)


def test_artifact_metadata_parsing():
    assert _maybe_json_list('["a", "b"]') == ["a", "b"]
    root = Path(tempfile.mkdtemp())
    (root / "manifest.json").write_text(json.dumps({"fresh_k": "16"}))
    assert find_fresh_k(root, 4) == 16


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL PASS ({len(tests)} tests)")
