"""Whole-expression reward regressions found auditing math500/s0."""

import pytest

from data import reward


@pytest.mark.parametrize(
    "prediction,gold,expected",
    [
        ("2x", "2y", 0.0),
        ("2x", "2", 0.0),
        (r"\sqrt{4}", "2", 1.0),
        (r"\frac{1}{\sqrt{2}}", r"\frac{\sqrt{2}}{2}", 1.0),
        ("x+x", "2x", 1.0),
        (r"\frac{2}{4}", r"\frac{1}{2}", 1.0),
        ("0.5", r"\frac{1}{2}", 1.0),
        ("(3, 1)", "(3, 2)", 0.0),
        ("(3, 1)", "(3,1)", 1.0),
        ("garbage", "2", 0.0),
    ],
)
def test_verifier_compares_whole_math_expression(monkeypatch, prediction, gold, expected):
    monkeypatch.setenv("OM_MATH_VERIFIER", "math_verify")
    assert reward("Answer: " + prediction, gold) == expected
