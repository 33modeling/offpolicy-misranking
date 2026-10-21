"""Memory-bounded Gemma backend with normalized text-only log probabilities."""

import torch
from torch.utils.checkpoint import checkpoint

from srgc_rebuttal.timing import timed

from ..gemma4.memory import GemmaBackend as OriginalBackend
from ..gemma4.memory import bounded_generate, durable_checkpoints
from .protocol import SUPPRESS_TOKENS, mask_logits
from .stopping import trim_padding

__all__ = ["GemmaBackend", "bounded_generate", "durable_checkpoints"]


class GemmaBackend(OriginalBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model.generation_config.suppress_tokens != list(SUPPRESS_TOKENS):
            raise ValueError("Gemma generation and training must use the same text distribution")

    def _rollout(self, prompt_id, responses, seed):
        # Rows ended by the answer stop rule carry the batch padding; the
        # sampled response is the part before it (see stopping.py).
        sequences, rewards, start = super()._rollout(prompt_id, responses, seed)
        pad = self.tokenizer.pad_token_id
        return [trim_padding(sequence, start, pad) for sequence in sequences], rewards, start

    @timed("forward")
    def _logps_batch(self, sequences, start):
        previous = self.model.training
        self.model.train(True)
        try:
            self.cost_meter.count("forward_responses", len(sequences))
            lengths = [len(sequence) for sequence in sequences]
            if not lengths or any(start < 1 or start >= length for length in lengths):
                raise ValueError("expected nonempty Gemma response suffixes")
            for sequence in sequences:
                if any(bool((sequence[start:] == token).any()) for token in SUPPRESS_TOKENS):
                    raise ValueError("suppressed modality token in Gemma response; do not reuse v1 rollouts")
            ids = torch.full((len(sequences), max(lengths)), self.tokenizer.pad_token_id,
                             dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            for row, sequence in enumerate(sequences):
                ids[row, :len(sequence)] = sequence
                mask[row, :len(sequence)] = 1
            base = self.model.get_base_model()
            hidden = base.model(input_ids=ids, attention_mask=mask,
                                use_cache=False).last_hidden_state[:, :-1]

            def token_logps(states, target):
                logits = base.lm_head(states)
                cap = base.config.final_logit_softcapping
                if cap is not None:
                    logits = torch.tanh(logits / cap) * cap
                logits = mask_logits(logits)
                # Prompt tokens are not optimized; avoid gathering -inf for
                # any masked token in the prompt or the padded batch region.
                safe_target = target.clone()
                for token in SUPPRESS_TOKENS:
                    safe_target = safe_target.masked_fill(target == token, 0)
                return logits.gather(-1, safe_target[..., None]).squeeze(-1) - logits.logsumexp(-1)

            pieces = []
            for offset in range(0, hidden.shape[1], self.logit_chunk_tokens):
                piece = hidden[:, offset:offset + self.logit_chunk_tokens]
                target = ids[:, offset + 1:offset + 1 + piece.shape[1]]
                if torch.is_grad_enabled() and piece.requires_grad:
                    pieces.append(checkpoint(token_logps, piece, target, use_reentrant=False))
                else:
                    pieces.append(token_logps(piece, target))
            logps = torch.cat(pieces, dim=1)
            return [logps[row, start - 1:length - 1] for row, length in enumerate(lengths)]
        finally:
            self.model.train(previous)
