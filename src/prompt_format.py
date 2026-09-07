#!/usr/bin/env python3
"""Prompt-format families: one config value, one concrete template per dataset.

A base model without a chat template is prompted with the released OLMo RL-Zero
templates, which differ between math and code. The model spec therefore names
the family (``"olmo_rlzero"``) and each dataset resolves to the concrete
template that ``rollout.PROMPT_FORMATS`` knows:

    resolve_prompt_format("olmo_rlzero", "math500") == "olmo_rlzero_math"
    resolve_prompt_format("olmo_rlzero", "mbpp") == "olmo_rlzero_code"

Concrete values pass through unchanged, so specs that already name one template
for every dataset (``tokenizer_chat``, ``verifiable_completion``) are unaffected.

    python src/prompt_format.py olmo_rlzero mbpp   # prints olmo_rlzero_code
"""

from __future__ import annotations

import sys

PROMPT_FORMAT_FAMILIES: dict[str, dict[str, str]] = {
    "olmo_rlzero": {
        "math500": "olmo_rlzero_math",
        "gsm8k": "olmo_rlzero_math",
        "mbpp": "olmo_rlzero_code",
    },
}

CONCRETE_PROMPT_FORMATS = {
    "tokenizer_chat",
    "olmo_rlzero_math",
    "olmo_rlzero_code",
    "verifiable_completion",
}


def resolve_prompt_format(value: str, dataset: str) -> str:
    """Concrete template for ``dataset`` given a spec value (family or concrete)."""
    family = PROMPT_FORMAT_FAMILIES.get(value)
    if family is not None:
        try:
            return family[dataset]
        except KeyError as exc:
            raise ValueError(
                f"prompt_format family {value!r} has no template for dataset {dataset!r}; "
                f"known: {sorted(family)}"
            ) from exc
    if value not in CONCRETE_PROMPT_FORMATS:
        raise ValueError(
            f"unsupported prompt_format={value!r}; expected one of "
            f"{sorted(CONCRETE_PROMPT_FORMATS | set(PROMPT_FORMAT_FAMILIES))}"
        )
    return value


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: prompt_format.py <spec value> <dataset>", file=sys.stderr)
        return 2
    try:
        print(resolve_prompt_format(argv[0], argv[1]))
    except ValueError as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
