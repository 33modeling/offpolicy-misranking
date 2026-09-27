"""Small analytic policy used only to execute and test the reference on CPU.

This is a one-token categorical model with binary verifiable rewards. It
performs genuine fresh sampling and AdamW updates; it is NOT OLMo, MATH, or
a reproduction of the paper's empirical results.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np

from .objectives import CountSketch, grpo_advantages, loo_advantages
from .srgc import stream_seed


class ToyBackend:
    gpu_count = 0

    def __init__(self, features: dict[str, np.ndarray], answers: dict[str, int],
                 *, projection_dim: int = 4096, seed: int = 0, learning_rate: float = 1e-5):
        self.features = {i: np.asarray(x, dtype=np.float64).copy() for i, x in features.items()}
        self.answers = dict(answers)
        width = len(next(iter(self.features.values())))
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(scale=0.1, size=(width, 2))
        self.m = np.zeros_like(self.weights)
        self.v = np.zeros_like(self.weights)
        self.updates = 0
        self.learning_rate = learning_rate
        self.projection = CountSketch(self.weights.size, projection_dim, seed=0)
        self.score_calls: list[tuple[str, ...]] = []
        self.train_calls: list[tuple[str, ...]] = []

    def _probabilities(self, prompt: str) -> np.ndarray:
        logits = self.features[prompt] @ self.weights
        values = np.exp(logits - logits.max())
        return values / values.sum()

    def _rollout(self, prompt: str, responses: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        probs = self._probabilities(prompt)
        rng = np.random.default_rng(stream_seed(seed, 0, prompt))
        actions = rng.choice(2, size=responses, p=probs)
        rewards = (actions == self.answers[prompt]).astype(np.float64)
        # Per-response gradient of log pi(action | prompt).
        derivative = np.eye(2)[actions] - probs
        gradients = self.features[prompt][None, :, None] * derivative[:, None, :]
        return rewards, gradients

    def score_gradients(self, ids: Sequence[str], *, responses: int,
                        group_size: int, seed: int) -> dict[str, np.ndarray]:
        self.score_calls.append(tuple(ids))
        result = {}
        for prompt in ids:
            rewards, gradients = self._rollout(prompt, responses, seed)
            advantages = loo_advantages(rewards, group_size)
            gradient = (advantages[:, None, None] * gradients).mean(axis=0)
            result[prompt] = self.projection(gradient.ravel())
        return result

    def train(self, ids: Sequence[str], *, responses: int,
              objective: str, seed: int) -> dict[str, float]:
        self.train_calls.append(tuple(ids))
        gradient = np.zeros_like(self.weights)
        means = []
        for prompt in ids:
            rewards, gradients = self._rollout(prompt, responses, seed)
            if objective == "grpo":
                advantages = grpo_advantages(rewards)
            elif objective == "rloo":
                advantages = loo_advantages(rewards, responses)
            else:
                raise ValueError("unknown training objective")
            # One-token responses: token sum and token mean coincide.
            gradient -= (advantages[:, None, None] * gradients).mean(axis=0) / len(ids)
            means.append(float(rewards.mean()))
        norm = float(np.linalg.norm(gradient))
        gradient *= min(1.0, 1.0 / max(norm, 1e-12))
        self.updates += 1
        self.m = 0.9 * self.m + 0.1 * gradient
        self.v = 0.999 * self.v + 0.001 * gradient * gradient
        m_hat = self.m / (1 - 0.9 ** self.updates)
        v_hat = self.v / (1 - 0.999 ** self.updates)
        self.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        return {"sample_reward": float(np.mean(means)), "gradient_norm": norm}

    def expected_reward(self, ids: Sequence[str]) -> float:
        return float(np.mean([self._probabilities(i)[self.answers[i]] for i in ids]))

    def state_dict(self) -> dict:
        return copy.deepcopy({"weights": self.weights, "m": self.m, "v": self.v,
                              "updates": self.updates, "learning_rate": self.learning_rate})

    def load_state_dict(self, state: dict) -> None:
        state = copy.deepcopy(state)
        if state["weights"].shape != self.weights.shape:
            raise ValueError("model shape mismatch")
        self.weights, self.m, self.v = state["weights"], state["m"], state["v"]
        self.updates, self.learning_rate = state["updates"], state["learning_rate"]

    def synchronize(self) -> None:
        pass


def make_problem(seed: int = 0, candidates: int = 400, validation: int = 50,
                 evaluation: int = 100) -> tuple[dict, dict, list, list, list, dict]:
    """Synthetic input; 50 validation prompts is a demo setting, not an online-paper claim."""
    rng = np.random.default_rng(seed)
    n = candidates + validation + evaluation
    x = rng.normal(size=(n, 8))
    labels = (x[:, 0] + x[:, 1] > 0).astype(int)
    ids = [f"p{i:04d}" for i in range(n)]
    features, answers = dict(zip(ids, x)), dict(zip(ids, map(int, labels)))
    # Existing initial-policy reward cache, frozen for every arm.
    initial = ToyBackend(features, answers, seed=seed)
    cache = {i: initial._rollout(i, 8, stream_seed(seed, 0, "cache"))[0].tolist()
             for i in ids[:candidates]}
    return (features, answers, ids[:candidates], ids[candidates:candidates + validation],
            ids[candidates + validation:], cache)
