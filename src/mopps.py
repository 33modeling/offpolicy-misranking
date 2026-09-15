"""MoPPS top-k, uniform-prior variant (Qu et al., KDD 2026).

Equations 15-17, https://arxiv.org/html/2507.04632v5 . Selection/update
checked against thu-rllab/MoPPS recipe/ours/mopps.py at 4110c5ab40fc.
This is a local GRPO integration, not a reproduction of the authors' trainer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

import selection_gate as core

ARMS = ("mopps", "random_online")
SCHEMA = "mopps-comparison/v1"
PAPER = {"title": "Can Prompt Difficulty be Online Predicted for Accelerating RL Finetuning of Reasoning Models?",
         "authors": "Yun Qu et al.", "venue": "KDD 2026", "arxiv": "2507.04632v5",
         "doi": "10.1145/3770854.3780263",
         "code": "https://github.com/thu-rllab/MoPPS",
         "commit": "4110c5ab40fc9a2b9d09c70989b87c9e74bfdd17"}


@dataclass(frozen=True)
class Config:
    prior_alpha: float = 1.
    prior_beta: float = 1.
    target: float = .5
    decay: float = 1.
    candidate_multiplier: int = 16

    def __post_init__(self):
        for key in ("prior_alpha", "prior_beta"):
            core.number(getattr(self, key), key, 1e-12)
        core.number(self.target, "target", 0., 1.)
        core.number(self.decay, "decay", 0., 1.)
        core.integer(self.candidate_multiplier, "candidate multiplier", 1)


def specification(arm, seed, start_step, pool_size, batch_size=4, responses=8):
    if arm not in ARMS:
        raise ValueError("unregistered online selector")
    for value, name, minimum in ((seed, "seed", 0), (start_step, "start step", 0),
                                  (pool_size, "pool size", batch_size),
                                  (batch_size, "batch size", 1), (responses, "responses", 2)):
        core.integer(value, name, minimum)
    return {"schema": SCHEMA, "arm": arm, "seed": seed, "start_step": start_step,
            "pool_size": pool_size, "batch_size": batch_size, "responses": responses,
            "config": asdict(Config()), "initialization": "uniform_prior_no_external_rewards",
            "candidate_sampling": "uniform_without_replacement_each_step",
            "paper": PAPER}


class Sampler:
    def __init__(self, spec):
        expected = specification(spec["arm"], spec["seed"], spec["start_step"], spec["pool_size"],
                                 spec["batch_size"], spec["responses"])
        if spec != expected:
            raise ValueError("online selection specification changed")
        self.spec = expected
        self.config = Config(**spec["config"])
        self.alpha = np.full(spec["pool_size"], self.config.prior_alpha)
        self.beta = np.full(spec["pool_size"], self.config.prior_beta)
        self.completed_step = spec["start_step"]

    def state(self):
        return {"spec": self.spec, "completed_step": self.completed_step,
                "alpha": self.alpha.tolist(), "beta": self.beta.tolist()}

    def select(self, candidates, rng):
        ids = np.asarray(candidates)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < self.spec["batch_size"]
                or len(set(ids.tolist())) != len(ids) or (ids < 0).any() or (ids >= len(self.alpha)).any()):
            raise ValueError("invalid MoPPS candidates")
        if self.spec["arm"] == "mopps":
            predicted = rng.beta(self.alpha[ids], self.beta[ids])
            positions = np.argsort((predicted-self.config.target)**2)[:self.spec["batch_size"]]
            scores = predicted.tolist()
        else:
            positions = rng.choice(len(ids), self.spec["batch_size"], replace=False)
            scores = None
        return {"candidates": ids.tolist(), "selected": ids[positions].tolist(), "predicted": scores}

    def begin(self, step):
        if step != self.completed_step + 1:
            raise ValueError("selector steps must be consecutive")
        # Separate deterministic streams: no effect on policy rollout RNGs.
        seed = (self.spec["seed"] * 1_000_003 + step * 7_919 + 20260915) & 0xFFFFFFFF
        candidates = np.random.RandomState(seed).choice(
            len(self.alpha), min(len(self.alpha), self.spec["batch_size"]*self.config.candidate_multiplier),
            replace=False)
        return {"step": step, **self.select(candidates, np.random.RandomState(seed ^ 0x5A17B3))}

    def finish(self, proposal, rewards):
        if proposal != self.begin(self.completed_step + 1):
            raise ValueError("selection was not made from the preceding posterior")
        values = np.asarray(rewards, dtype=float)
        if (values.shape != (self.spec["batch_size"], self.spec["responses"])
                or not np.isfinite(values).all() or not np.isin(values, (0., 1.)).all()):
            raise ValueError("MoPPS requires one binary reward group per selected prompt")
        if self.spec["arm"] == "mopps":
            ids = proposal["selected"]
            successes = values.sum(axis=1)
            self.alpha[ids] = self.config.decay*self.alpha[ids] + (1-self.config.decay)*self.config.prior_alpha + successes
            self.beta[ids] = self.config.decay*self.beta[ids] + (1-self.config.decay)*self.config.prior_beta + values.shape[1]-successes
        self.completed_step = proposal["step"]
        return {**proposal, "rewards": values.tolist(), "state_sha256": core.fingerprint(self.state())}

    def replay(self, rows, completed_step):
        for row in rows:
            evidence = row["online_selection"]
            proposal = self.begin(row["step"])
            if self.finish(proposal, evidence["rewards"]) != evidence:
                raise ValueError("online selection evidence changed")
            if (not math.isclose(float(np.mean(evidence["rewards"])), row["reward_mean"], abs_tol=1e-7)
                    or row["groups"] != self.spec["batch_size"]
                    or row["samples"] != self.spec["batch_size"]*self.spec["responses"]):
                raise ValueError("selector feedback differs from policy training metrics")
        if self.completed_step != completed_step:
            raise ValueError("posterior history does not match policy checkpoint")
        return self


def validate_policy_evidence(path, spec):
    manifest = core.read(path / "policy_train.json")
    if manifest.get("online_selection") != spec:
        raise ValueError("policy online-selector binding changed")
    with (path / "selector_state.json").open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if manifest.get("selector_state_sha256") != digest:
        raise ValueError("policy posterior receipt changed")
    rows = [json.loads(line) for line in (path / "grpo_stats.jsonl").read_text().splitlines() if line.strip()]
    sampler = Sampler(spec).replay(rows, manifest["completed_steps"])
    if core.read(path / "selector_state.json") != sampler.state():
        raise ValueError("published posterior differs from online reward history")
    return sampler
