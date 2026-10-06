"""ARCUS equations 2-5 and Appendix D, adapted to this GRPO/LoRA backend.

Source: https://arxiv.org/html/2609.38018v1 . This is an independent
implementation, not an official-code reproduction. B=4, G=8, M=5;
the original generated groups, not fresh replacement groups, train the model.
"""

import math

import numpy as np


class Arcus:
    def __init__(self, ids, batch=4, responses=8):
        if len(set(ids)) != len(ids) or batch < 1 or responses < 4:
            raise ValueError("invalid ARCUS dimensions")
        self.ids = list(ids)
        self.batch, self.responses = batch, responses
        self.count = math.ceil(1.25 * batch)
        if self.count > len(ids):
            raise ValueError("ARCUS candidate margin exceeds pool")
        self.mu = np.full(len(ids), np.pi / 4)
        self.variance = np.full(len(ids), np.pi**2 / 48)
        self.last = np.full(len(ids), -1, dtype=int)
        self.drift, self.diffusion, self.target = 0., 1e-5, np.pi / 4
        self.step = 0

    def yield_estimate(self):
        a = 1 + 2 * self.responses * self.variance
        return np.clip(1 - (np.exp(-self.responses * self.mu**2 / a) +
            np.exp(-self.responses * (np.pi / 2 - self.mu)**2 / a)) / np.sqrt(a), 0, 1)

    def scores(self, target):
        variance = min(np.sin(target), np.cos(target))**2 / 2
        return np.sqrt(variance / (variance + self.variance)) * np.exp(
            -(self.mu - target)**2 / (2 * (variance + self.variance))) * self.yield_estimate()

    def inclusion(self, scores):
        logs = np.log(np.maximum(scores, 1e-300)) / .3
        weights = np.exp(np.maximum(logs - logs.max(), -700))
        low, high = 0., self.count / weights.min()
        # Log-space bisection remains well conditioned for concentrated beliefs.
        low, high = -750., math.log(high)
        for _ in range(60):
            middle = (low + high) / 2
            included = np.exp(np.minimum(0, np.log(weights) + middle))
            if included.sum() < self.count:
                low = middle
            else:
                high = middle
        return np.exp(np.minimum(0, np.log(weights) + (low + high) / 2))

    def sample(self, seed):
        self.mu = np.clip(self.mu + self.drift * np.sin(2 * self.mu), 0, np.pi / 2)
        self.variance += self.diffusion
        seen = self.last >= 0
        if seen.sum() >= 100 and (~seen).any():
            self.mu[~seen] = self.mu[seen].mean()
            self.variance[~seen] = self.mu[seen].var() + self.variance[seen].mean()
        if self.step >= math.ceil(len(self.ids) / self.count):
            edge = np.arcsin(1 / np.sqrt(2 * self.responses))
            grid = np.linspace(edge, np.pi / 2 - edge, 41)
            yields = np.asarray([np.dot(self.inclusion(self.scores(t)), self.yield_estimate()) /
                                 self.count for t in grid])
            target = grid[np.flatnonzero(yields >= yields.max() - .03)[0]]
            self.target += float(np.clip(target - self.target, -.005, .005))
        scores = self.scores(self.target)
        priorities = np.log(np.maximum(scores, 1e-300)) / .3 + np.random.default_rng(seed).gumbel(size=len(scores))
        indices = np.argsort(-priorities, kind="stable")[:self.count]
        return [self.ids[i] for i in indices]

    def observe(self, rewards):
        if len(rewards) != self.count or not set(rewards) <= set(self.ids):
            raise ValueError("ARCUS must observe all sampled candidate groups")
        innovations, denominators, excesses, gaps = [], [], [], []
        for rid, values in rewards.items():
            r = np.asarray(values)
            if r.shape != (self.responses,) or not np.isin(r, (0, 1)).all():
                raise ValueError("invalid binary ARCUS rollout group")
            i, successes = self.ids.index(rid), int(r.sum())
            if 0 < successes < self.responses:
                z = np.arcsin(np.sqrt((successes + 3 / 8) / (self.responses + 3 / 4)))
                noise = 1 / (4 * self.responses + 2)
            else:
                z, noise = (0. if successes == 0 else np.pi / 2), 1 / (2 * self.responses)
            innovation, total = z - self.mu[i], self.variance[i] + noise
            if self.last[i] >= 0:
                gap = self.step - self.last[i]
                innovations.append(innovation)
                denominators.append(gap * max(np.sin(2 * self.mu[i]), .05))
                excesses.append(innovation**2 - total)
                gaps.append(gap)
            gain = self.variance[i] / total
            self.mu[i] = np.clip(self.mu[i] + gain * innovation, 0, np.pi / 2)
            self.variance[i] *= 1 - gain
            self.last[i] = self.step
        if innovations:
            self.drift += .1 * sum(innovations) / sum(denominators)
            self.diffusion = max(1e-5, .9 * self.diffusion + .1 * max(
                0, self.diffusion + np.mean(excesses) / np.mean(gaps)))
        scores = dict(zip(self.ids, self.scores(self.target)))
        informative = [i for i, r in rewards.items() if 0 < sum(r) < self.responses]
        selected = sorted(informative, key=lambda i: (-scores[i], i))[:self.batch]
        self.step += 1
        return selected

    def state_dict(self):
        return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in vars(self).items()}

    def load_state_dict(self, state):
        if any(state[k] != getattr(self, k) for k in ("ids", "batch", "responses", "count")):
            raise ValueError("ARCUS checkpoint dimensions differ")
        for key, value in state.items():
            setattr(self, key, value.copy() if isinstance(value, np.ndarray) else value)
