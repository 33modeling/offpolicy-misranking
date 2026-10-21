"""Preserve the verified PT loader and restore its native modality-token suppression."""

import json
from pathlib import Path

from ..gemma4.model import PromptTokenizer, find_snapshot, resolve_snapshot
from ..gemma4.model import load_text_model as original_load
from .protocol import CODE_FORMAT_EXAMPLE, SUPPRESS_TOKENS
from .stopping import with_answer_stop

__all__ = ["PromptTokenizer", "find_snapshot", "load_text_model", "resolve_snapshot"]


def load_text_model(source, device, *, attention="eager"):
    native = json.loads((Path(source) / "generation_config.json").read_text())
    if native.get("suppress_tokens") != list(SUPPRESS_TOKENS):
        raise ValueError("Gemma PT snapshot has different official modality suppression")
    model, tokenizer = original_load(source, device, attention=attention)
    model.generation_config.suppress_tokens = list(SUPPRESS_TOKENS)
    model.generate = with_answer_stop(model.generate, tokenizer.tokenizer, CODE_FORMAT_EXAMPLE)
    return model, tokenizer
