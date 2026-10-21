"""End a PT response where the task's own answer ends.

The PT checkpoint does not reliably emit EOS after the requested final line. It
continues the document with a new self-written question and answer, and both
verifiers read the last answer or code block, so a correct solution is scored
on the invented follow-up instead. Generation therefore stops (without adding
any token) once the requested answer form is complete:

- MATH: a completed line that starts with ``Answer:``, or a new question/problem
  header at the start of a line.
- MBPP: the first closed fenced code block.

The same sampled tokens are scored by GRPO; nothing is rewritten after sampling.
"""

import re

STOP_RULE = "math-answer-line-or-new-question_mbpp-first-closed-fence-v1"
ANSWER_LINE = re.compile(r"(?:\A|\n)[ \t]*\**Answer\**[ \t]*:[^\n]*\S[^\n]*\n")
NEW_QUESTION = re.compile(
    r"\n[ \t]*(?:#+[ \t]*)?\**(?:Question|Problem|Q|Exercise)[ \t]*\d*[ \t]*\**[ \t]*[:.]"
    r"|\nSolve the following problem"
)
CLOSED_FENCE = re.compile(r"```[^\n]*\n.*?\n[ \t]*```", re.DOTALL)


def math_end(text):
    """Character offset after which a MATH response is complete, or None."""
    ends = [m.end() for m in (ANSWER_LINE.search(text),) if m]
    question = NEW_QUESTION.search(text)
    if question:
        ends.append(question.start())
    return min(ends) if ends else None


def code_end(text):
    match = CLOSED_FENCE.search(text)
    return match.end() if match else None


def finished(text, code):
    return (code_end if code else math_end)(text) is not None


class AnswerStop:
    """Per-row stopping criterion; decodes a row only when its newest token can end a form."""

    def __init__(self, tokenizer, start, code):
        self.tokenizer, self.start, self.code = tokenizer, start, code
        self.done = None

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        rows = input_ids.shape[0]
        # bounded_generate reuses this object for each response micro-batch.
        if self.done is None or self.done.shape[0] != rows or input_ids.shape[1] <= self.start + 1:
            self.done = torch.zeros(rows, dtype=torch.bool, device=input_ids.device)
        for row in range(rows):
            if self.done[row] or input_ids.shape[1] <= self.start:
                continue
            piece = self.tokenizer.decode(input_ids[row, -1:].tolist(), skip_special_tokens=False)
            if "\n" not in piece and "`" not in piece and ":" not in piece and "." not in piece:
                continue
            text = self.tokenizer.decode(input_ids[row, self.start:].tolist(), skip_special_tokens=True)
            if finished(text, self.code):
                self.done[row] = True
        return self.done.clone()


def with_answer_stop(generate, tokenizer, code_prefix):
    """Wrap model.generate so every Gemma draw (cache, preflight, GRPO) uses the stop rule."""

    def wrapped(*args, **kwargs):
        from transformers import StoppingCriteriaList

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None or input_ids.shape[0] != 1:
            raise ValueError("Gemma generation expects one prompt per call")
        start = input_ids.shape[1]
        prompt = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        from ..gemma4.memory import DecodeProgress

        criteria = StoppingCriteriaList(list(kwargs.get("stopping_criteria") or [DecodeProgress()]))
        criteria.append(AnswerStop(tokenizer, start, prompt.startswith(code_prefix)))
        return generate(*args, **{**kwargs, "stopping_criteria": criteria})

    return wrapped


def trim_padding(sequence, start, pad):
    """Drop the padding HF appends after a row that stopped before the longest row."""
    end = len(sequence)
    while end > start + 1 and int(sequence[end - 1]) == pad:
        end -= 1
    return sequence[:end]
