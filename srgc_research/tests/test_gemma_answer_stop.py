"""PT answer stop rule: end at the requested answer form, never rewrite tokens."""

from pathlib import Path

import pytest

from srgc_research.dispatch.gemma4_text.stopping import (AnswerStop, code_end, finished, math_end, trim_padding)

TOKENIZER = Path("/home/kms/reports/gemma4-tokenizer-audit-2026-10-21/tokenizer")


def test_math_stops_after_answer_line_or_new_question():
    solution = "J = 18/3 = 6.\nAnswer: 6\n"
    assert math_end(solution + "\nQuestion: Tom has 3 apples. Answer: 7") == len(solution)
    assert math_end(" Jerry is 4 years old.\nQuestion: The sum") == len(" Jerry is 4 years old.")
    assert math_end("x\n\nProblem 2: next") == 2
    assert not finished("Let x = 2. Then Answer:", code=False)  # line not complete
    assert not finished("We have 3 + 4 = 7.\nSo the", code=False)
    assert finished("**Answer:** $\\frac{1}{2}$\n", code=False)


def test_code_stops_after_first_closed_fence():
    text = "```python\ndef f(x):\n    return x\n```\n\nHow can I compute a mode?\n```python\nprint(1)\n```"
    assert text[:code_end(text)].endswith("return x\n```")
    assert not finished("```python\ndef f(x):\n    return x\n", code=True)


def test_padding_trim_keeps_sampled_tokens():
    import torch
    seq = torch.tensor([2, 5, 6, 7, 8, 0, 0])
    assert trim_padding(seq, 3, 0).tolist() == [2, 5, 6, 7, 8]
    assert trim_padding(torch.tensor([2, 5, 6, 7]), 3, 0).tolist() == [2, 5, 6, 7]


@pytest.mark.skipif(not TOKENIZER.is_dir(), reason="pinned Gemma tokenizer not available")
def test_stopping_criterion_marks_rows_independently_with_real_tokenizer():
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    prompt = tok.encode("Solve.\n\nSolution:\n", add_special_tokens=False)
    done = tok.encode("so 6.\nAnswer: 6\n", add_special_tokens=False)
    open_ = tok.encode("so we compute 6 and", add_special_tokens=False)
    width = max(len(done), len(open_))
    rows = [prompt + done + [0] * (width - len(done)), prompt + open_ + [0] * (width - len(open_))]
    stop = AnswerStop(tok, len(prompt), code=False)
    # Feed growing prefixes as generate does; row 0 must finish when its answer line closes.
    result = None
    for t in range(len(prompt) + 1, len(prompt) + len(done) + 1):
        result = stop(torch.tensor([r[:t] for r in rows]), None)
    assert result.tolist() == [True, False]
    # A new micro-batch reuses the object and starts fresh.
    assert stop(torch.tensor([r[:len(prompt) + 1] for r in rows]), None).tolist() == [False, False]
