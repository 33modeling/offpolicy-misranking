"""NumPy reference equations for a model adapter; no automatic differentiation.

Token log-probability gradients are inputs to score_gradient. A large-model
adapter should accumulate/project these gradients without materializing the
entire [responses, tokens, parameters] array used by this small reference.
"""

from __future__ import annotations

import numpy as np


def loo_advantages(rewards: np.ndarray, group_size: int) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 1 or group_size < 2 or len(r) % group_size or not np.isfinite(r).all():
        raise ValueError("rewards must split into complete finite leave-one-out groups")
    groups = r.reshape(-1, group_size)
    return ((group_size * groups - groups.sum(axis=1, keepdims=True)) /
            (group_size - 1)).reshape(-1)


def grpo_advantages(rewards: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all() or epsilon <= 0:
        raise ValueError("invalid GRPO group")
    return (r - r.mean()) / (r.std(ddof=0) + epsilon)


def _tokens(logps: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, m = np.asarray(logps, dtype=np.float64), np.asarray(mask, dtype=bool)
    if p.ndim != 2 or m.shape != p.shape or not np.isfinite(p).all() or not m.any(axis=1).all():
        raise ValueError("expected finite response-token values and nonempty response masks")
    return p, m


def grpo_loss(logps: np.ndarray, old_logps: np.ndarray, rewards: np.ndarray,
              mask: np.ndarray, clip: float = 0.2) -> float:
    p, m = _tokens(logps, mask)
    old, _ = _tokens(old_logps, mask)
    if old.shape != p.shape or not 0 < clip < 1:
        raise ValueError("invalid old policy or clip")
    a = grpo_advantages(rewards)
    if len(a) != len(p):
        raise ValueError("response count mismatch")
    ratio = np.exp(p - old)
    terms = np.minimum(ratio * a[:, None], np.clip(ratio, 1 - clip, 1 + clip) * a[:, None])
    return float(-((terms * m).sum(axis=1) / m.sum(axis=1)).mean())


def rloo_loss(logps: np.ndarray, rewards: np.ndarray, mask: np.ndarray) -> float:
    p, m = _tokens(logps, mask)
    a = loo_advantages(rewards, len(rewards))
    if len(a) != len(p):
        raise ValueError("response count mismatch")
    return float(-(a * (p * m).sum(axis=1)).mean())


def score_gradient(token_logp_gradients: np.ndarray, rewards: np.ndarray,
                   mask: np.ndarray, group_size: int = 4) -> np.ndarray:
    """Positive reward-ascent gradient: average responses, SUM their tokens."""
    g = np.asarray(token_logp_gradients, dtype=np.float64)
    if g.ndim != 3 or not np.isfinite(g).all():
        raise ValueError("expected finite [responses, tokens, parameters] gradients")
    _, m = _tokens(np.zeros(g.shape[:2]), mask)
    a = loo_advantages(rewards, group_size)
    if len(a) != len(g):
        raise ValueError("response count mismatch")
    return (a[:, None] * (g * m[..., None]).sum(axis=1)).mean(axis=0)


class CountSketch:
    """Fixed linear map, shared by candidate and validation gradients.

    Production parameter order must be fixed across checkpoints. Chunked
    accumulation can use the same bucket/sign slices instead of a dense map.
    """

    def __init__(self, input_dim: int, output_dim: int = 4096, seed: int = 0):
        if min(input_dim, output_dim) < 1:
            raise ValueError("projection dimensions must be positive")
        rng = np.random.default_rng(seed)
        self.buckets = rng.integers(output_dim, size=input_dim)
        self.signs = rng.choice([-1.0, 1.0], size=input_dim)
        self.output_dim = output_dim

    def __call__(self, vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float64)
        if v.shape != self.signs.shape or not np.isfinite(v).all():
            raise ValueError("invalid projection input")
        return np.bincount(self.buckets, weights=self.signs * v, minlength=self.output_dim)
