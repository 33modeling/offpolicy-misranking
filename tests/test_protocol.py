"""Protocol regressions that do not require a model or GPU.

    PYTHONPATH=src .work/.venv-cu126/bin/python tests/test_protocol.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import _maybe_json_list  # noqa: E402
from experiment import score_oracle_microgroups  # noqa: E402
from hybrid import continue_rollouts_batch  # noqa: E402
from kcurve_floor import find_fresh_k  # noqa: E402
from select_rules import overlap_under_independent_ties, topk_count  # noqa: E402


def test_all_tie_overlap_is_chance():
    n = 256
    k = topk_count(n, 0.10)
    tied = {idx: 0.0 for idx in range(n)}
    summary = overlap_under_independent_ties(tied, tied, k, pairs=1_000)
    assert abs(summary.mean - k / n) < 0.01, (summary.mean, k / n)
    assert summary.low < summary.high


def test_oracle_uses_equal_odd_even_halves():
    stack = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, -1.0],
    ])
    oracle, halves = score_oracle_microgroups(stack, torch.tensor([1.0, 0.0]))
    assert halves == {"a": 1.0, "b": 0.0}
    assert oracle["score"] == 1.0
    try:
        score_oracle_microgroups(stack[:3], torch.tensor([1.0, 0.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("odd micro-group counts must be rejected")


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
