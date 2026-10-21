"""The same restricted text distribution is used for sampling and GRPO scoring."""

EXTENSION = "gemma4-12b-text-v3"
PREVIOUS_EXTENSION = "gemma4-12b-text-v2"
COMPLETION_CUE = "\n\nSolution:\n"
CODE_INDENTATION_INSTRUCTION = "\n\nUse valid Python indentation: indent each function body with four spaces."
CODE_FORMAT_EXAMPLE = ("Python function formatting example:\n```python\n"
                       "def identity(value):\n    return value\n```\n\n")
PROMPT_FORMAT = "gemma4_pt_solution_single_bos"
SUPPRESS_TOKENS = (258883, 258882)  # Official PT <audio|> and <image|>.
CODE_VERIFIER = "srgc_research.dispatch.gemma4_text.verifier:code_reward"
CODE_VERSION = "parent-checked-values-v3-fence-language-line-v1"


def completion_cue(dataset):
    return (CODE_INDENTATION_INSTRUCTION if dataset == "mbpp" else "") + COMPLETION_CUE


def prompt_prefix(dataset):
    return CODE_FORMAT_EXAMPLE if dataset == "mbpp" else ""


def sampling():
    return {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "eos_token_id": 1, "pad_token_id": 0,
            "suppress_tokens": list(SUPPRESS_TOKENS)}


def mask_logits(logits, suppressed=SUPPRESS_TOKENS):
    """Mask after native soft-capping, before FP32 normalization, preserving autograd."""
    import torch

    indices = torch.tensor(suppressed, device=logits.device, dtype=torch.long)
    if indices.numel() == 0 or bool(((indices < 0) | (indices >= logits.shape[-1])).any()):
        raise ValueError("Gemma text suppression differs from the model vocabulary")
    return logits.float().index_fill(-1, indices, -torch.inf)
