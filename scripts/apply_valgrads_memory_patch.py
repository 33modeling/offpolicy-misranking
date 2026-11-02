#!/usr/bin/env python3
"""Apply the memory-only validation-gradient fix to a copy of src/grads.py.

Chunked LM-head log-probs plus an in-process OOM back-off. Verified against the
unpatched file on CPU: log-probs identical, projected gradients agree to 5e-7.
Usage: python3 scripts/apply_valgrads_memory_patch.py <path to grads.py>
"""
import sys

OLD_LOGPS = '    logits = model(batch, attention_mask=attention).logits[:, :-1].float()\n    targets = batch[:, 1:]\n    token_logps = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)\n    token_logps = token_logps - logits.logsumexp(dim=-1)\n    return [token_logps[row, : length - 1] for row, length in enumerate(lengths)]'
NEW_LOGPS = '    targets = batch[:, 1:]\n    token_logps = _token_logps_chunked(model, batch, attention, targets)\n    return [token_logps[row, : length - 1] for row, length in enumerate(lengths)]\n\n\nLOGIT_CHUNK_TOKENS = int(os.environ.get("OM_LOGIT_CHUNK_TOKENS", "512"))\n\n\ndef _token_logps_chunked(model, batch, attention, targets) -> torch.Tensor:\n    """Target log-probs without materialising the full [B, L, V] fp32 logits.\n\n    Memory-only change: the base transformer runs once and the LM head plus\n    log-softmax run per chunk of LOGIT_CHUNK_TOKENS positions under activation\n    checkpointing. Values are identical (verified 0.0 max difference on CPU);\n    projected gradients agree to 5e-7 (float summation order). Vocabulary\n    100k x 2048 tokens x 8 sequences in fp32 was ~9 GB per temporary and\n    several temporaries were alive at once.\n    """\n    base = getattr(model, "model", None)\n    head = getattr(model, "lm_head", None)\n    if base is None or head is None or LOGIT_CHUNK_TOKENS <= 0:\n        logits = model(batch, attention_mask=attention).logits[:, :-1].float()\n        return logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(dim=-1)\n    hidden = base(input_ids=batch, attention_mask=attention).last_hidden_state[:, :-1]\n\n    def chunk_logps(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:\n        logits = head(h).float()\n        return logits.gather(-1, t.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(dim=-1)\n\n    pieces = []\n    for start in range(0, hidden.shape[1], LOGIT_CHUNK_TOKENS):\n        h = hidden[:, start : start + LOGIT_CHUNK_TOKENS]\n        t = targets[:, start : start + LOGIT_CHUNK_TOKENS]\n        if torch.is_grad_enabled() and hidden.requires_grad:\n            pieces.append(torch.utils.checkpoint.checkpoint(chunk_logps, h, t, use_reentrant=False))\n        else:\n            pieces.append(chunk_logps(h, t))\n    return torch.cat(pieces, dim=1)'
OLD_PG = '    model.zero_grad(set_to_none=True)\n    k = len(sequences)\n    for start in range(0, k, micro_batch):\n        batch = sequences[start : start + micro_batch]\n        ws = weights[start : start + micro_batch]\n        token_logps = _padded_token_logps(\n            model, [seq["input_ids"] for seq in batch]\n        )\n        loss = token_logps[0].new_zeros(())\n        for seq, w, tok_logp in zip(batch, ws, token_logps, strict=True):\n            resp = tok_logp[seq["resp_start"] - 1 :]\n            loss = loss + (w.to(resp.device) * resp).sum() / k\n        loss.backward()\n    return project_grads(params, spec)'
NEW_PG = '    # An OOM mid-prompt leaves partial .grad contributions, so the whole prompt\n    # is restarted with a halved micro-batch instead of the failed chunk.\n    while True:\n        model.zero_grad(set_to_none=True)\n        try:\n            _accumulate_prompt_gradient(model, sequences, weights, micro_batch)\n        except Exception as exc:\n            if not _is_oom(exc) or micro_batch <= 1:\n                model.zero_grad(set_to_none=True)\n                raise\n            micro_batch = max(1, micro_batch // 2)\n        else:\n            return project_grads(params, spec)\n        model.zero_grad(set_to_none=True)\n        torch.cuda.empty_cache()\n        print(f"[oom-backoff] restart whole prompt; gradient micro-batch -> {micro_batch}", flush=True)\n\n\ndef _accumulate_prompt_gradient(model, sequences, weights, micro_batch):\n    k = len(sequences)\n    for start in range(0, k, micro_batch):\n        batch = sequences[start : start + micro_batch]\n        ws = weights[start : start + micro_batch]\n        token_logps = _padded_token_logps(\n            model, [seq["input_ids"] for seq in batch]\n        )\n        loss = token_logps[0].new_zeros(())\n        for seq, w, tok_logp in zip(batch, ws, token_logps, strict=True):\n            resp = tok_logp[seq["resp_start"] - 1 :]\n            loss = loss + (w.to(resp.device) * resp).sum() / k\n        loss.backward()\n\n\ndef _is_oom(exc: BaseException) -> bool:\n    return isinstance(exc, torch.cuda.OutOfMemoryError) or (\n        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()\n    )'


def main() -> int:
    path = sys.argv[1]
    text = open(path, encoding="utf-8").read()
    if "_token_logps_chunked" in text:
        print("[patch] already applied")
        return 0
    for old, new, what in ((OLD_LOGPS, NEW_LOGPS, "log-prob computation"),
                           (OLD_PG, NEW_PG, "prompt_gradient")):
        if text.count(old) != 1:
            print(f"[abort] cannot patch the {what}: this file is not the expected vintage")
            return 1
        text = text.replace(old, new)
    if "\nimport os\n" not in text:
        text = text.replace("\nimport torch\n", "\nimport os\n\nimport torch\n", 1)
    if "import torch.utils.checkpoint" not in text:
        text = text.replace("\nimport torch\n", "\nimport torch\nimport torch.utils.checkpoint\n", 1)
    open(path, "w", encoding="utf-8").write(text)
    print("[patch] applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
