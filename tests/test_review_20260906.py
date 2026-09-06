"""Regression tests for the 2026-09-06 full-code review (BACKLOG OM-2026-09-06-*)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from code_sandbox import _validated_candidate
from data import extract_answer, normalize_math_answer, reward
from rollout_contract import gen_kwargs


def _gate(source: str) -> str | None:
    try:
        _validated_candidate(source)
    except ValueError as exc:
        return str(exc)
    return None


def test_sandbox_allows_ordinary_mbpp_idioms():
    assert _gate("from typing import List\ndef f(x: List[int]):\n    return x") is None
    assert _gate("def f():\n    pass\nif __name__ == '__main__':\n    f()") is None
    assert _gate("class A:\n    def __init__(self):\n        self._n = 1\n") is None
    assert _gate("from dataclasses import dataclass\n@dataclass\nclass P:\n    x: int\n") is None


def test_sandbox_still_blocks_escape_hatches():
    assert "blocked attribute" in _gate("x = (1).__class__")
    assert "blocked name" in _gate("getattr(1, 'real')")
    assert "allowlist" in _gate("import os")
    assert "private" in _gate("__import__('os')") or "blocked" in _gate("__import__('os')")


@pytest.mark.parametrize(
    "text, gold, expected",
    [
        ("#### (3, 1)", "(3,1)", 1.0),
        ("#### 12", "1,2", 0.0),
        ("#### 1,234", "1234", 1.0),
        ("Answer: \\dfrac{1}{2}", "\\frac{1}{2}", 1.0),
        ("Answer: \\text{Evelyn}", "Evelyn", 1.0),
        ("therefore \\boxed{\\frac{1}{2}} holds", "\\frac{1}{2}", 1.0),
        ("#### 3.0", "3", 1.0),
    ],
)
def test_math_reward_keeps_structural_commas_and_nested_boxed(text, gold, expected):
    assert reward(text, gold) == expected


def test_extract_answer_boxed_is_brace_aware():
    assert extract_answer("so \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert normalize_math_answer("1,234,567") == "1234567"
    assert normalize_math_answer("(3, 1)") == "(3,1)"


def test_gen_kwargs_passes_explicit_eos_set():
    kwargs = gen_kwargs(1.0, 1.0, 16, pad_token_id=5, eos_token_id={7, 5, 7})
    assert kwargs["eos_token_id"] == [5, 7]
    assert "eos_token_id" not in gen_kwargs(1.0, 1.0, 16, pad_token_id=5)
