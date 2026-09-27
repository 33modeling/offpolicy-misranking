"""Optional causal-LM adapter: fresh generation, dense scoring, LoRA training.

Works on one device or torchrun replicas. Production runs use four GPUs;
the one-device path is also used by tiny CPU tests. No model is loaded merely
by importing this module. Prompt strings and the verifier are explicit inputs.
"""

from __future__ import annotations

import copy
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from .objectives import grpo_advantages, loo_advantages
from .srgc import stream_seed
from .timing import timed, torch_meter
from .progress import record as progress


class TorchBackend:
    def __init__(self, model, tokenizer, records: Mapping[str, dict],
                 verifier: Callable[[dict, str], float], *, projection_dim: int = 4096,
                 projection_seed: int = 0, max_new_tokens: int = 2048,
                 logprob_micro_batch: int = 2, logit_chunk_tokens: int = 512,
                 score_parameters: Sequence[tuple[str, torch.nn.Parameter]] | None = None,
                 cost_meter=None):
        self.model, self.tokenizer, self.records, self.verifier = model, tokenizer, records, verifier
        self.device = next(model.parameters()).device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.gpu_count = self.world if self.device.type == "cuda" else 0
        self.cost_meter = cost_meter or torch_meter(cuda=self.device.type == "cuda")
        self.projection_dim, self.projection_seed = projection_dim, projection_seed
        self.max_new_tokens = max_new_tokens
        if logprob_micro_batch < 1 or logit_chunk_tokens < 1:
            raise ValueError("micro-batch and token chunk sizes must be positive")
        self.logprob_micro_batch, self.logit_chunk_tokens = logprob_micro_batch, logit_chunk_tokens
        self.train_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not self.train_parameters:
            raise ValueError("model must expose trainable adapter parameters")
        if score_parameters is None:
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            layers = base.model.layers
            modules = [(f"layer_{i}", layers[i]) for i in range(max(0, len(layers) - 4), len(layers))]
            modules.append(("final_norm", base.model.norm))
            score_parameters = [(f"{prefix}.{name}", p) for prefix, module in modules
                                for name, p in module.named_parameters() if "lora_" not in name]
        self.score_parameters = list(score_parameters)
        if not self.score_parameters or len({id(p) for _, p in self.score_parameters}) != len(self.score_parameters):
            raise ValueError("scoring parameters must be a nonempty, unique dense parameter list")
        self.optimizer = torch.optim.AdamW([p for _, p in self.train_parameters], lr=1e-5,
                                          betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        self.model.eval()  # All configured dropout is zero; reproducible generation/logps.

    @timed("communication")
    def _gather(self, local: dict) -> dict:
        if self.world == 1:
            return local
        parts = [None] * self.world
        dist.all_gather_object(parts, local)
        result = {}
        for part in parts:
            if set(result) & set(part):
                raise ValueError("duplicate prompt across distributed shards")
            result.update(part)
        return result

    @timed("rollout_overhead")
    def _rollout(self, prompt_id: str, responses: int, seed: int):
        record = self.records[prompt_id]
        self.cost_meter.count("prompts")
        self.cost_meter.count("responses", responses)
        with self.cost_meter.stage("tokenization"):
            tokens = self.tokenizer(record["prompt"], return_tensors="pt", add_special_tokens=True)
            inputs = {k: v.to(self.device) for k, v in tokens.items()}
        start = inputs["input_ids"].shape[1]
        if not start:
            raise ValueError("empty tokenized prompt")
        devices = [self.device.index] if self.device.type == "cuda" else []
        draw_seed = stream_seed(seed, self.rank, prompt_id) % (2**63 - 1)
        with self.cost_meter.stage("generation"), torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(draw_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(draw_seed)
            generated = self.model.generate(**inputs, do_sample=True, temperature=1.0,
                top_p=1.0, top_k=0, num_return_sequences=responses,
                max_new_tokens=self.max_new_tokens, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        sequences, rewards = [], []
        eos = self.tokenizer.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        for sequence in generated:
            completion = sequence[start:]
            stop = len(completion)
            for position, token in enumerate(completion.tolist()):
                if token in eos_ids:
                    stop = position + 1
                    break
            if stop == 0:
                raise ValueError("model produced no response tokens")
            sequences.append(sequence[:start + stop].detach())
            self.cost_meter.count("generated_tokens", stop)
            with self.cost_meter.stage("decode"):
                text = self.tokenizer.decode(completion[:stop], skip_special_tokens=True)
            with self.cost_meter.stage("reward_verification"):
                reward = float(self.verifier(record, text))
            if reward not in (0.0, 1.0):
                raise ValueError("verifier must return a binary reward")
            rewards.append(reward)
        if len(sequences) != responses:
            raise ValueError("generation returned the wrong response count")
        progress("rollout", prompt=prompt_id, responses=responses)
        return sequences, np.asarray(rewards), start

    def _logps(self, sequence: torch.Tensor, start: int) -> torch.Tensor:
        return self._logps_batch([sequence], start)[0]

    @timed("forward")
    def _logps_batch(self, sequences: Sequence[torch.Tensor], start: int) -> list[torch.Tensor]:
        self.cost_meter.count("forward_responses", len(sequences))
        lengths = [len(s) for s in sequences]
        if not lengths or any(start < 1 or start >= n for n in lengths):
            raise ValueError("expected nonempty response suffixes")
        ids = torch.full((len(sequences), max(lengths)), self.tokenizer.pad_token_id,
                         dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for row, sequence in enumerate(sequences):
            ids[row, :len(sequence)] = sequence
            mask[row, :len(sequence)] = 1
        # This backend uses plain PEFT replicas and explicit reductions, not DDP
        # forward hooks. Calling the decoder keeps its active LoRA layers intact.
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        hidden = base.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, :-1]
        def token_logps(h, target):
            logits = base.lm_head(h).float()
            return logits.gather(-1, target[..., None]).squeeze(-1) - logits.logsumexp(-1)
        pieces = []
        # Recompute the small head chunks in backward instead of retaining a
        # vocabulary-sized logits/softmax tensor for every token in the batch.
        for offset in range(0, hidden.shape[1], self.logit_chunk_tokens):
            piece = hidden[:, offset:offset + self.logit_chunk_tokens]
            target = ids[:, offset + 1:offset + 1 + piece.shape[1]]
            if torch.is_grad_enabled() and piece.requires_grad:
                pieces.append(checkpoint(token_logps, piece, target, use_reentrant=False))
            else:
                pieces.append(token_logps(piece, target))
        logps = torch.cat(pieces, dim=1)
        return [logps[row, start - 1:n - 1] for row, n in enumerate(lengths)]

    @timed("gradient_projection")
    def _project(self, name: str, gradient: torch.Tensor) -> np.ndarray:
        # Map derived only from parameter name, fixed projection seed and index.
        # Generate in bounded chunks; never allocate a dense projection matrix.
        rng = np.random.default_rng(stream_seed(self.projection_seed, 0, name))
        flat = gradient.detach().reshape(-1)
        output = np.zeros(self.projection_dim, dtype=np.float64)
        for start in range(0, flat.numel(), 262144):
            chunk = flat[start:start + 262144].float().cpu().numpy()
            buckets = rng.integers(self.projection_dim, size=len(chunk))
            signs = rng.choice([-1.0, 1.0], size=len(chunk))
            output += np.bincount(buckets, weights=chunk * signs, minlength=self.projection_dim)
        return output

    @timed("gradient_scoring_overhead")
    def score_gradients(self, ids: Sequence[str], *, responses: int,
                        group_size: int, seed: int) -> dict[str, np.ndarray]:
        params = [p for _, p in self.score_parameters]
        all_params = list(self.model.parameters())
        prior = [p.requires_grad for p in all_params]
        result = {}
        try:
            # Only dense scoring derivatives are needed here; retain the current
            # LoRA values but avoid recording a graph through earlier adapters.
            for p in all_params:
                p.requires_grad_(False)
            for p in params:
                p.requires_grad_(True)
            for prompt_id in ids[self.rank::self.world]:
                sequences, rewards, start = self._rollout(prompt_id, responses, seed)
                advantages = loo_advantages(rewards, group_size)
                projected = np.zeros(self.projection_dim, dtype=np.float64)
                accumulated = [None] * len(params)
                active = [i for i, a in enumerate(advantages) if a != 0]
                self.cost_meter.count("zero_advantage_responses", responses - len(active))
                for offset in range(0, len(active), self.logprob_micro_batch):
                    indices = active[offset:offset + self.logprob_micro_batch]
                    values = self._logps_batch([sequences[i] for i in indices], start)
                    objective = sum(lp.sum() * (float(advantages[i]) / responses)
                                    for i, lp in zip(indices, values))
                    with self.cost_meter.stage("backward"):
                        gradients = torch.autograd.grad(objective, params, allow_unused=True)
                    self.cost_meter.count("backward_responses", len(indices))
                    for j, gradient in enumerate(gradients):
                        if gradient is not None:
                            value = gradient.detach().float()
                            if accumulated[j] is None:
                                accumulated[j] = value.clone()
                            else:
                                accumulated[j].add_(value)
                    del gradients, values, objective
                # Linearity permits one projection per prompt instead of eight full scans.
                for (name, _), gradient in zip(self.score_parameters, accumulated):
                    if gradient is not None:
                        projected += self._project(name, gradient)
                result[prompt_id] = projected
                if not np.isfinite(projected).all():
                    raise FloatingPointError(f"nonfinite scoring gradient for {prompt_id}")
                progress("gradient_scoring", prompt=prompt_id)
        finally:
            for p, required in zip(all_params, prior):
                p.requires_grad_(required)
        return self._gather(result)

    @timed("training_overhead")
    def train(self, ids: Sequence[str], *, responses: int,
              objective: str, seed: int) -> dict[str, float]:
        if objective not in {"grpo", "rloo"}:
            raise ValueError("unknown objective")
        self.optimizer.zero_grad(set_to_none=True)
        summaries = {}
        for prompt_id in ids[self.rank::self.world]:
            sequences, rewards, start = self._rollout(prompt_id, responses, seed)
            advantages = (grpo_advantages(rewards) if objective == "grpo" else
                          loo_advantages(rewards, responses))
            active = [i for i, a in enumerate(advantages) if a != 0]
            self.cost_meter.count("zero_advantage_responses", responses - len(active))
            for offset in range(0, len(active), self.logprob_micro_batch):
                indices = active[offset:offset + self.logprob_micro_batch]
                values = self._logps_batch([sequences[i] for i in indices], start)
                losses = []
                for i, logps in zip(indices, values):
                    advantage = float(advantages[i])
                    if objective == "grpo":
                        ratio = torch.exp(logps - logps.detach())
                        weighted = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
                        losses.append(-weighted.mean())
                    else:
                        losses.append(-advantage * logps.sum())
                with self.cost_meter.stage("backward"):
                    (sum(losses) / (responses * len(ids))).backward()
                self.cost_meter.count("backward_responses", len(indices))
            summaries[prompt_id] = float(rewards.mean())
        for _, p in self.train_parameters:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if self.world > 1:
                # Each rank contributed its prompt(s), already divided by global prompt count.
                with self.cost_meter.stage("gradient_reduction"):
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        with self.cost_meter.stage("optimizer"):
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in self.train_parameters], 1.0,
                                                 error_if_nonfinite=True)
            self.optimizer.step()
        progress("policy_update")
        summaries = self._gather(summaries)
        return {"sample_reward": float(np.mean(list(summaries.values()))), "gradient_norm": float(norm)}

    @timed("evaluation_overhead")
    def evaluate(self, ids: Sequence[str], *, seed: int, responses: int = 8) -> dict[str, float]:
        result = {}
        for prompt_id in ids[self.rank::self.world]:
            _, rewards, _ = self._rollout(prompt_id, responses, seed)
            result[prompt_id] = float(rewards.mean())
        return self._gather(result)

    def state_dict(self) -> dict:
        return {"trainable": {n: p.detach().cpu().clone() for n, p in self.train_parameters},
                "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                "projection_dim": self.projection_dim, "projection_seed": self.projection_seed,
                "score_names": [n for n, _ in self.score_parameters]}

    def load_state_dict(self, state: dict) -> None:
        if (state["projection_dim"] != self.projection_dim or
                state["projection_seed"] != self.projection_seed or
                state["score_names"] != [n for n, _ in self.score_parameters] or
                set(state["trainable"]) != {n for n, _ in self.train_parameters}):
            raise ValueError("checkpoint parameter or projection mismatch")
        with torch.no_grad():
            for n, p in self.train_parameters:
                p.copy_(state["trainable"][n].to(p.device))
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.optimizer.zero_grad(set_to_none=True)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self.world > 1:
            dist.barrier()
