"""The same restricted text distribution is used for sampling and GRPO scoring."""

EXTENSION = "gemma4-12b-it-v1"
COMPLETION_CUE = "<|turn>model\n<|channel>thought\n<channel|>"
PROMPT_FORMAT = "gemma4_it_tokenizer_chat_thinking_off"
EOS_TOKENS = (1, 106, 50)
SUPPRESS_TOKENS = (258883, 258882)  # Official IT <audio|> and <image|>.
CODE_VERIFIER = "srgc_research.dispatch.gemma4_it.verifier:code_reward"
CODE_VERSION = "parent-checked-values-v3-fence-language-line-v1"


def completion_cue(dataset):
    return COMPLETION_CUE


def prompt_prefix(dataset):
    return "<bos><|turn>user\n"


def sampling():
    return {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "eos_token_id": list(EOS_TOKENS), "pad_token_id": 0,
            "suppress_tokens": list(SUPPRESS_TOKENS)}


def mask_logits(logits, suppressed=SUPPRESS_TOKENS):
    """Mask after native soft-capping, before FP32 normalization, preserving autograd."""
    import torch

    indices = torch.tensor(suppressed, device=logits.device, dtype=torch.long)
    if indices.numel() == 0 or bool(((indices < 0) | (indices >= logits.shape[-1])).any()):
        raise ValueError("Gemma text suppression differs from the model vocabulary")
    return logits.float().index_fill(-1, indices, -torch.inf)
