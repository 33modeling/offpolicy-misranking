"""Accept the literal fence format requested by MBPP, preserving parent-side tests."""

import re

from srgc_rebuttal import verifiers

# A language name on its own first line is fence metadata, not Python source.
# Only strip it inside a fenced block; never rewrite unfenced executable code.
CODE_BLOCK = re.compile(r"```(?:python3?|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
LANGUAGE_LINE = re.compile(r"\A[ \t]*(?:python3?|py)[ \t]*\r?\n", re.IGNORECASE)


def extract_code(response):
    blocks = CODE_BLOCK.findall(response)
    if not blocks:
        return response.strip()
    return LANGUAGE_LINE.sub("", blocks[-1], count=1).strip()


def code_reward(record, response):
    # Retain the existing resource limits, child isolation and parent-side
    # assertion checks. No supplied tests or expected values are changed.
    normalized = CODE_BLOCK.sub(
        lambda match: "```python\n" + LANGUAGE_LINE.sub("", match.group(1), count=1) + "```",
        response,
    )
    return verifiers.code_reward(record, normalized)
