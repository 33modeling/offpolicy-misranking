"""Add readout features and isolated optimizer audits to the existing backend."""

import hashlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from scripts.srgc_resumable_rollouts import ResumableRolloutMixin
from srgc_rebuttal.objectives import grpo_advantages, loo_advantages
from srgc_rebuttal.srgc import stream_seed
from srgc_rebuttal.timing import timed
from srgc_rebuttal.torch_backend import TorchBackend


def vector_cosine(a, b):
    a, b = a.double(), b.double()
    denominator = a.norm() * b.norm()
    return float(a.dot(b) / denominator) if denominator > 0 else None


class ResearchBackend(ResumableRolloutMixin, TorchBackend):
    def __init__(self, *args, rollout_root=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_cache_root = Path(rollout_root) if rollout_root else None
        self.namespace = "initial"
        self.scorer = "dense"
        self.last_rewards = {}
        self.capture_tokens = False
        self.tokens = {}
        self.replay = None
        self._readout_projections = None

    @contextmanager
    def operation(self, name):
        previous = self.namespace
        self.namespace = name
        try:
            yield
        finally:
            self.namespace = previous

    def _rollout(self, prompt_id, responses, seed):
        if self.replay is not None:
            row = self.replay[prompt_id]
            if len(row[1]) != responses:
                raise ValueError("replayed response count differs")
            result = ([s.to(self.device) for s in row[0]], np.asarray(row[1]), row[2])
        else:
            key = hashlib.sha256(f"{self.namespace}:{seed}:rank-{self.rank}".encode()).hexdigest()
            self._resumable_dir = self.rollout_cache_root / key if self.rollout_cache_root else None
            try:
                with self.cost_meter.stage("rollout_cache_io"):
                    result = super()._rollout(prompt_id, responses, seed)
            finally:
                self._resumable_dir = None
        self.last_rewards[prompt_id] = result[1].tolist()
        if self.capture_tokens:
            self.tokens[prompt_id] = ([s.detach().cpu().clone() for s in result[0]],
                                     result[1].tolist(), result[2])
        return result

    def score_gradients(self, ids, *, responses, group_size, seed):
        # Bypass the old mixin's block deletion: retention lasts until the
        # surrounding state transition is checkpointed, including training.
        if self.scorer == "dense":
            return TorchBackend.score_gradients(self, ids, responses=responses,
                                                group_size=group_size, seed=seed)
        return self.readout_gradients(ids, responses=responses, group_size=group_size, seed=seed)

    def evaluate(self, ids, *, seed, responses=8):
        self.last_rewards = {}
        means = TorchBackend.evaluate(self, ids, seed=seed, responses=responses)
        self.evaluation_samples = {i: list(r) for i, r in self._gather(self.last_rewards).items()}
        return means

    def train(self, ids, *, responses, objective, seed):
        self.cost_meter.count("optimizer_prompt_exposures", len(ids[self.rank::self.world]))
        self.cost_meter.count("optimizer_response_exposures", len(ids[self.rank::self.world]) * responses)
        return TorchBackend.train(self, ids, responses=responses, objective=objective, seed=seed)

    def collect(self, ids, responses, seed):
        self.tokens, self.last_rewards = {}, {}
        self.capture_tokens = True
        try:
            for rid in ids[self.rank::self.world]:
                self._rollout(rid, responses, seed)
            return self._gather(self.tokens)
        finally:
            self.capture_tokens = False
            self.tokens = {}

    @contextmanager
    def replaying(self, records):
        if self.replay is not None:
            raise RuntimeError("nested rollout replay")
        self.replay = records
        try:
            yield
        finally:
            self.replay = None

    def _projections(self, head):
        if self._readout_projections is None:
            side = int(np.sqrt(self.projection_dim))
            if side * side != self.projection_dim:
                raise ValueError("LESSER-style feature dimension must be square")
            rng = np.random.default_rng(stream_seed(self.projection_seed, 0, "readout-projections"))
            matrices = [rng.choice(np.asarray([-1., 1.], dtype=np.float32), size=(n, side)) /
                        np.sqrt(side) for n in head.weight.shape]
            self._readout_projections = tuple(torch.as_tensor(m, device=self.device, dtype=torch.float32)
                                              for m in matrices)
        return self._readout_projections

    @timed("readout_scoring")
    def readout_gradients(self, ids, *, responses, group_size, seed):
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        output_projection, hidden_projection = self._projections(base.lm_head)
        side = output_projection.shape[1]
        result = {}
        with torch.no_grad():
            for rid in ids[self.rank::self.world]:
                sequences, rewards, start = self._rollout(rid, responses, seed)
                advantages = loo_advantages(rewards, group_size)
                feature = torch.zeros((side, side), dtype=torch.float64, device=self.device)
                for sequence, advantage in zip(sequences, advantages):
                    if advantage == 0:
                        continue
                    with self.cost_meter.stage("readout_forward"):
                        hidden = base.model(input_ids=sequence[None], use_cache=False).last_hidden_state[0]
                    for offset in range(start - 1, len(sequence) - 1, self.logit_chunk_tokens):
                        stop = min(offset + self.logit_chunk_tokens, len(sequence) - 1)
                        h = hidden[offset:stop]
                        target = sequence[offset + 1:stop + 1]
                        with self.cost_meter.stage("readout_projection"):
                            probabilities = base.lm_head(h).float().softmax(-1)
                            # Reward ASCENT: (one_hot - probability), not the loss gradient.
                            residual = output_projection[target] - probabilities @ output_projection
                            compressed = h.float() @ hidden_projection
                            feature.add_((residual.T @ compressed).double(), alpha=float(advantage) / responses)
                        del probabilities, residual, compressed
                    del hidden
                    self.cost_meter.count("readout_forward_responses")
                values = feature.flatten().cpu().numpy()
                if not np.isfinite(values).all():
                    raise FloatingPointError("nonfinite output-layer feature")
                result[rid] = values
        return self._gather(result)

    def parameter_vector(self):
        return torch.cat([p.detach().float().cpu().flatten() for _, p in self.train_parameters])

    def gradient_audit(self, ids, *, seed, responses=8, probe=None, evaluation_ids=None):
        """Actual backend GRPO, clipping and AdamW; restore every state on exit."""
        initial = self.state_dict()
        weights = self.parameter_vector()
        original_clip = torch.nn.utils.clip_grad_norm_
        captured = {}

        def capture(parameters, *args, **kwargs):
            parameters = list(parameters)
            captured["gradient"] = torch.cat([p.grad.detach().float().cpu().flatten()
                                                for p in parameters])
            return original_clip(parameters, *args, **kwargs)

        try:
            for _, p in self.train_parameters:
                p.grad = torch.zeros_like(p)  # None would skip Adam's history and step counter.
            self.optimizer.step()
            zero_update = self.parameter_vector() - weights
            self.load_state_dict(initial)
            torch.nn.utils.clip_grad_norm_ = capture
            try:
                metrics = self.train(ids, responses=responses, objective="grpo", seed=seed)
            finally:
                torch.nn.utils.clip_grad_norm_ = original_clip
            update = self.parameter_vector() - weights
            vectors = {**captured, "update": update, "zero_update": zero_update,
                       "incremental_update": update - zero_update}
            metrics.update({f"{k}_norm": float(v.double().norm()) for k, v in vectors.items()})
            metrics["parameter_dtypes"] = sorted({str(p.dtype) for _, p in self.train_parameters})
            if probe is not None:
                metrics["frozen_probe_loss_after"] = self.probe_loss(probe)
            if evaluation_ids is not None:
                metrics["fresh_per_question_after"] = self.evaluate(
                    evaluation_ids, seed=stream_seed(seed, 0, "post-update-evaluation"), responses=responses)
            return vectors, metrics
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
            self.load_state_dict(initial)

    def frozen_probe(self, ids, *, responses, seed):
        records = self.collect(ids, responses, seed)
        local = {}
        with torch.no_grad():
            for rid in ids[self.rank::self.world]:
                seq, rewards, start = records[rid]
                logps = [self._logps(s.to(self.device), start).cpu() for s in seq]
                local[rid] = (seq, rewards, start, logps)
        return self._gather(local)

    def probe_loss(self, probe):
        local = {}
        with torch.no_grad():
            for rid in list(probe)[self.rank::self.world]:
                sequences, rewards, start, old_logps = probe[rid]
                loss = 0.
                for seq, old, advantage in zip(sequences, old_logps, grpo_advantages(rewards)):
                    logps = self._logps(seq.to(self.device), start)
                    ratio = torch.exp(logps - old.to(self.device))
                    loss -= float(torch.minimum(ratio * float(advantage),
                        ratio.clamp(.8, 1.2) * float(advantage)).mean()) / len(rewards)
                local[rid] = loss
        return float(np.mean(list(self._gather(local).values())))
