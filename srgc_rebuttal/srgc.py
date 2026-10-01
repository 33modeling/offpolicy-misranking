"""Rebuttal experiment: periodic problem selection and online switching.

The model adapter owns generation, differentiation and optimizer state. The
controller owns candidate sampling, gradient reuse, selection and switching.
No historical rewards, A/B diagnostic batches or reported paper costs enter
the online decision. All gradients supplied by the adapter are reward-ascent
gradients under the current policy and a single fixed projection.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import hashlib
import math
import time
import uuid
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


def stream_seed(seed: int, step: int, purpose: str) -> int:
    """Independent, reproducible streams; unaffected by Python hash randomization."""
    payload = f"{seed}:{step}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _finite(values: Any, name: str) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def _unique(ids: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(ids)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must contain distinct IDs and be nonempty")
    return result


def cosine_scores(gradients: np.ndarray, validation: np.ndarray) -> np.ndarray:
    g, v = _finite(gradients, "gradients"), _finite(validation, "validation")
    if g.ndim != 2 or v.shape != (g.shape[1],):
        raise ValueError("expected gradients [prompts, dimensions] and validation [dimensions]")
    denominator = np.linalg.norm(g, axis=1) * np.linalg.norm(v)
    return np.divide(g @ v, denominator, out=np.zeros(g.shape[0]), where=denominator > 0)


def gradient_contrast(
    on_ids: Sequence[str], sr_ids: Sequence[str],
    gradients: Mapping[str, np.ndarray], validation: np.ndarray,
) -> float:
    """D = <Pv, mean(Pg for all On-policy candidates) - mean(Pg for SR comparison prompts)>.

    Inputs are already projected. There is no cosine normalization here and
    both sides contain 40 comparison prompts, separate from the four-prompt training batch.
    An overlapping prompt uses the same vector on both sides and cancels.
    """
    on, sr = _unique(on_ids, "On-policy set"), _unique(sr_ids, "SR set")
    if len(on) != len(sr):
        raise ValueError("SR-GC requires equal-sized scoring sets")
    v = _finite(validation, "validation")
    if v.ndim != 1 or not v.size:
        raise ValueError("validation must be a nonempty vector")
    def mean(ids: Sequence[str]) -> np.ndarray:
        rows = [_finite(gradients[i], f"gradient {i}") for i in ids]
        if any(row.shape != v.shape for row in rows):
            raise ValueError("all projected gradients must share the validation shape")
        return np.stack(rows).mean(axis=0)
    value = float(np.dot(v, mean(on) - mean(sr)))
    if not math.isfinite(value):
        raise ValueError("nonfinite gradient contrast")
    return value


def top_ids(ids: Sequence[str], scores: Sequence[float], k: int, seed: int) -> tuple[str, ...]:
    ids = _unique(ids, "ranking candidates")
    scores = _finite(scores, "scores")
    if scores.shape != (len(ids),) or not 1 <= k <= len(ids):
        raise ValueError("invalid ranking size")
    ties = np.random.default_rng(seed).random(len(ids))
    order = np.lexsort((ties, -scores))[:k]
    return tuple(ids[int(i)] for i in order)


def cached_sr_set(
    ids: Sequence[str], rewards: Mapping[str, Sequence[float]], k: int, seed: int,
    responses: int = 8,
) -> tuple[str, ...]:
    scores = []
    for i in ids:
        r = _finite(rewards[i], f"cached rewards {i}")
        if r.shape != (responses,) or not np.all((r == 0) | (r == 1)):
            raise ValueError(f"{i}: cache must contain {responses} binary rewards")
        scores.append(-abs(float(r.mean()) - 0.5))
    return top_ids(ids, scores, k, seed + 1000)


@dataclass
class TemporalRule:
    interval: int = 25
    last_step: int | None = None
    window: list[float] = field(default_factory=list)
    switched: bool = False

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError("check interval must be positive")

    def observe(self, step: int, d: float) -> bool:
        if self.switched:
            raise RuntimeError("checks stop after switching")
        if step < 0 or step % self.interval or not math.isfinite(d):
            raise ValueError("a check needs a scheduled step and a finite D")
        if self.last_step is not None:
            if step <= self.last_step:
                raise ValueError("check steps must strictly increase")
            if step != self.last_step + self.interval:
                self.window.clear()  # Missing checks cannot confirm a transition.
        self.last_step = step
        if not self.window:
            if d < 0:
                self.window = [float(d)]
        elif len(self.window) == 1:
            if d < 0:
                self.switched = True
            else:
                self.window.append(float(d))
        else:
            # Here the window is [negative, nonnegative].
            if d < 0 and math.fsum([*self.window, d]) < 0:
                self.switched = True
            else:
                self.window = [float(d)] if d < 0 else []
        return self.switched


class Backend(Protocol):
    """Adapter contract for the current policy, verifier and optimizer.

    score_gradients must generate fresh responses once per distinct prompt,
    compute leave-one-out reward-ascent gradients (sum response-token terms,
    mean responses), and return vectors in the fixed projected space.
    It must not update weights or optimizer state. For OLMo the projection
    uses the final four dense layers and final normalization, not LoRA-only
    optimizer gradients. train generates its OWN fresh responses.
    """

    gpu_count: int

    def score_gradients(self, ids: Sequence[str], *, responses: int,
                        group_size: int, seed: int) -> Mapping[str, np.ndarray]: ...

    def train(self, ids: Sequence[str], *, responses: int,
              objective: str, seed: int) -> Mapping[str, float]: ...

    def state_dict(self) -> Any: ...

    def load_state_dict(self, state: Any) -> None: ...

    def synchronize(self) -> None: ...


@dataclass(frozen=True)
class Config:
    seed: int = 3
    scoring_prompts: int = 40
    training_prompts: int = 4
    responses: int = 8
    candidate_group_size: int = 4
    projection_dim: int = 4096
    selection_interval: int = 25
    check_interval: int = 25
    first_check: int = 25
    objective: str = "grpo"

    def __post_init__(self) -> None:
        if not 1 <= self.training_prompts <= self.scoring_prompts:
            raise ValueError("invalid training/scoring prompt counts")
        if (self.responses < 2 or self.candidate_group_size < 2 or
                self.responses % self.candidate_group_size or self.projection_dim < 1):
            raise ValueError("invalid gradient response groups or projection")
        if (self.selection_interval < 1 or self.check_interval < 1 or self.first_check < 1 or
                self.first_check % self.check_interval or
                self.check_interval % self.selection_interval):
            raise ValueError("checks must coincide with scheduled selection refreshes")
        if self.objective not in {"grpo", "rloo"}:
            raise ValueError("objective must be grpo or rloo")


class Engine:
    """Online experiment; step is the number of COMPLETED optimizer updates.

    A check at t scores policy theta_t, then any transition affects update
    t+1. Every selection_interval updates, draw and score 40 candidates and
    retain their top four until the next refresh. D reuses fresh scoring
    gradients at scheduled refreshes. All calls
    are synchronous so measured totals charge generation/backward work once.
    """

    ARMS = {"on_policy", "switch", "sr", "random"}
    SAMPLING_PROTOCOL = "random-candidate40-training4-switch-only-contrast40-v3"

    def __init__(self, backend: Backend, candidate_ids: Sequence[str],
                 validation_ids: Sequence[str], cached_rewards: Mapping[str, Sequence[float]],
                 *, arm: str = "switch", config: Config = Config(), step: int = 0, cost_event=None):
        if arm not in self.ARMS or step < 0:
            raise ValueError("invalid arm or initial step")
        self.backend, self.config, self.arm, self.step = backend, config, arm, step
        self.cost_event = cost_event
        self._phase_event = None
        self.candidates = _unique(candidate_ids, "candidate pool")
        self.validation = _unique(validation_ids, "validation prompts")
        if len(self.candidates) < config.scoring_prompts:
            raise ValueError("candidate pool is too small")
        if set(self.candidates) & set(self.validation):
            raise ValueError("candidate and validation prompts must be disjoint")
        if type(backend.gpu_count) is not int or backend.gpu_count < 0:
            raise ValueError("gpu_count must describe the allocated GPUs (zero for CPU)")
        preparation_started = time.perf_counter()
        self.sr_ranked_ids = cached_sr_set(self.candidates, cached_rewards, len(self.candidates),
                                         config.seed, config.responses)
        self.sr_preparation_wall_seconds = time.perf_counter() - preparation_started
        self.random_ids = self.candidates
        self.used_training_ids: set[str] = set()
        self.sampling_cycle = 0
        self.rule = TemporalRule(config.check_interval)
        self.active_selection: dict[str, Any] | None = None
        self.switched_at: int | None = None
        self.costs = {"selection_wall_seconds": 0.0, "training_wall_seconds": 0.0,
                      "selection_gpu_seconds": 0.0, "training_gpu_seconds": 0.0,
                      "d_arithmetic_wall_seconds": 0.0,
                      "sr_preparation_wall_seconds": 0.0, "sr_preparation_gpu_seconds": 0.0}
        if arm in {"sr", "switch"}:
            self._charge_preparation()
        self.history: list[dict[str, Any]] = []

    def _charge_preparation(self) -> None:
        self.costs["sr_preparation_wall_seconds"] = self.sr_preparation_wall_seconds
        self.costs["sr_preparation_gpu_seconds"] = self.sr_preparation_wall_seconds * self.backend.gpu_count

    def _sr_comparison(self) -> tuple[str, ...]:
        """Preview the diagnostic SR set without consuming any training prompts."""
        count = self.config.scoring_prompts
        if not 1 <= count <= len(self.candidates):
            raise ValueError("invalid sampling count")
        used = set(self.used_training_ids)
        selected: list[str] = []
        while len(selected) < count:
            if len(used) == len(self.candidates):
                used.clear()
            available = [i for i in self.sr_ranked_ids if i not in used and i not in selected]
            batch = available[:count - len(selected)]
            selected.extend(batch)
            used.update(batch)
        return tuple(selected)

    def _draw_candidates(self) -> tuple[str, ...]:
        rng = np.random.default_rng(stream_seed(self.config.seed, self.step, "candidate-draw"))
        return tuple(rng.choice(self.candidates, self.config.scoring_prompts, replace=False))

    def _training_batch(self, selector: str, candidates: Sequence[str]) -> tuple[str, ...]:
        if selector == "random":
            rng = np.random.default_rng(stream_seed(self.config.seed, self.step, "random-batch-draw"))
            return tuple(rng.choice(candidates, self.config.training_prompts, replace=False))
        if selector == "sr":
            eligible = set(candidates)
            return tuple(i for i in self.sr_ranked_ids if i in eligible)[:self.config.training_prompts]
        raise ValueError("unsupported training selector")

    def _vectors(self, ids: Sequence[str], group_size: int, purpose: str) -> dict[str, np.ndarray]:
        c = self.config
        with self._timing_scope("candidate_sr_union" if purpose == "selection" else purpose, section=True):
            received = self.backend.score_gradients(
                ids, responses=c.responses, group_size=group_size,
                seed=stream_seed(c.seed, self.step, purpose))
        if set(received) != set(ids):
            raise ValueError("backend must return exactly one gradient per requested prompt")
        result = {}
        for i in ids:
            g = _finite(received[i], f"gradient {i}")
            if g.shape != (c.projection_dim,):
                raise ValueError("backend projection dimension mismatch")
            result[i] = g.copy()
        return result

    def _begin(self, phase: str) -> float:
        meter = getattr(self.backend, "cost_meter", None)
        if meter is not None:
            meter.begin_phase(phase, self.step, self.backend.gpu_count)
            return 0.0
        self.backend.synchronize()
        self._phase_event = {"id": uuid.uuid4().hex, "phase": phase, "checkpoint": self.step,
                             "gpu_count": self.backend.gpu_count, "state": "started"}
        if self.cost_event is not None:
            self.cost_event(self._phase_event)
        return time.perf_counter()

    def _end(self, phase: str, started: float) -> None:
        meter = getattr(self.backend, "cost_meter", None)
        if meter is not None:
            report = meter.end_phase()
            self.costs[f"{phase}_wall_seconds"] += report["wall_seconds"]
            self.costs[f"{phase}_gpu_seconds"] += report["gpu_seconds"]
            return
        self.backend.synchronize()
        elapsed = time.perf_counter() - started
        self.costs[f"{phase}_wall_seconds"] += elapsed
        self.costs[f"{phase}_gpu_seconds"] += elapsed * self.backend.gpu_count
        if self.cost_event is not None:
            self.cost_event({**self._phase_event, "state": "finished", "wall_seconds": elapsed,
                             "gpu_seconds": elapsed * self.backend.gpu_count})
        self._phase_event = None

    def _timing_scope(self, name, *, section=False):
        from contextlib import nullcontext
        meter = getattr(self.backend, "cost_meter", None)
        if meter is None:
            return nullcontext()
        return meter.section(name) if section else meter.stage(name)

    def update(self) -> dict[str, Any]:
        c = self.config
        use_on = self.arm in {"on_policy", "switch"} and self.switched_at is None
        before = dict(self.costs)
        refresh = use_on and self.step % c.selection_interval == 0
        record: dict[str, Any] = {"checkpoint": self.step, "d": None, "switched": False,
                                  "selection_refreshed": refresh}
        if use_on and not refresh and self.active_selection is None:
            raise ValueError("mid-block continuation requires the saved selected prompts")
        if refresh:
            started = self._begin("selection")
            on_ids = self._draw_candidates()
            check_due = (self.arm == "switch" and self.step >= c.first_check and
                         self.step % c.check_interval == 0)
            sr_ids = self._sr_comparison() if check_due else ()
            # The 40-vs-40 diagnostic is separate from the four-prompt training batch.
            union = tuple(dict.fromkeys((*on_ids, *sr_ids)))
            gradients = self._vectors(union, c.candidate_group_size, "selection")
            val = self._vectors(self.validation, c.responses, "validation")
            with self._timing_scope("cosine_ranking"):
                v = np.stack([val[i] for i in self.validation]).mean(axis=0)
                scores = cosine_scores(np.stack([gradients[i] for i in on_ids]), v)
                train_ids = top_ids(on_ids, scores, c.training_prompts,
                                    stream_seed(c.seed + 1000, self.step, "online-ties"))
            self.active_selection = {"step": self.step, "on_ids": list(on_ids),
                                     "train_ids": list(train_ids)}
            if check_due:
                d_started = time.perf_counter()
                # Pure array arithmetic. No backend call, no generation/backward pass.
                with self._timing_scope("sr_gc_check"):
                    d = gradient_contrast(on_ids, sr_ids, gradients, v)
                    on_dot = float(np.dot(v, np.stack([gradients[i] for i in on_ids]).mean(axis=0)))
                    sr_dot = float(np.dot(v, np.stack([gradients[i] for i in sr_ids]).mean(axis=0)))
                    transition = self.rule.observe(self.step, d)
                self.costs["d_arithmetic_wall_seconds"] += time.perf_counter() - d_started
                record.update(d=d, switched=transition, on_mean_validation_dot=on_dot,
                              sr_mean_validation_dot=sr_dot)
                if transition:
                    self.switched_at = self.step
            self._end("selection", started)
            record.update(on_ids=list(on_ids), sr_ids=list(sr_ids),
                          ranking_scores=scores.tolist(), selected_on_ids=list(train_ids),
                          scored_distinct_prompts=len(union),
                          validation_ids=list(self.validation), scoring_responses_per_prompt=c.responses)
        if use_on:
            train_ids = tuple(self.active_selection["train_ids"])
            record["selection_step"] = self.active_selection["step"]
        used, cycle = set(self.used_training_ids), self.sampling_cycle
        started = self._begin("training")
        if not use_on or self.switched_at is not None:
            selector = "random" if self.arm == "random" else "sr"
            with self._timing_scope("candidate_sampling_and_ranking"):
                candidates = self._draw_candidates()
                train_ids = self._training_batch(selector, candidates)
            record.update(sampling_pool_size=len(self.candidates), training_candidate_ids=list(candidates))
        used.update(train_ids)
        if len(used) == len(self.candidates):
            used.clear()
            cycle += 1
        metrics = dict(self.backend.train(train_ids, responses=c.responses, objective=c.objective,
                                          seed=stream_seed(c.seed, self.step, "training")))
        self._end("training", started)
        self.used_training_ids, self.sampling_cycle = used, cycle
        self.step += 1
        record.update(completed_updates=self.step, train_ids=list(train_ids),
                      selection_gpu_seconds=self.costs["selection_gpu_seconds"] - before["selection_gpu_seconds"],
                      training_gpu_seconds=self.costs["training_gpu_seconds"] - before["training_gpu_seconds"],
                      selector="on_policy" if use_on and self.switched_at is None else
                      ("random" if self.arm == "random" else "sr"), metrics=metrics)
        self.history.append(record)
        return record

    def run_until(self, total_updates: int) -> list[dict[str, Any]]:
        if total_updates < self.step:
            raise ValueError("cannot run backwards")
        while self.step < total_updates:
            self.update()
        return self.history

    def state_dict(self) -> dict[str, Any]:
        """Contains model AND optimizer, selector state and temporal window."""
        return copy.deepcopy({"config": asdict(self.config), "arm": self.arm, "step": self.step,
            "candidates": self.candidates, "validation": self.validation,
            "random_ids": self.random_ids,
            "sampling_protocol": self.SAMPLING_PROTOCOL, "sr_ranked_ids": self.sr_ranked_ids,
            "used_training_ids": [i for i in self.candidates if i in self.used_training_ids],
            "sampling_cycle": self.sampling_cycle,
            "active_selection": self.active_selection,
            "sr_preparation_wall_seconds": self.sr_preparation_wall_seconds,
            "rule": asdict(self.rule), "switched_at": self.switched_at,
            "costs": self.costs, "history": self.history, "backend": self.backend.state_dict()})

    def load_state_dict(self, state: Mapping[str, Any], *, fork_arm: str | None = None) -> None:
        """A fork shares theta/optimizer and sampling pools; its costs start at zero.

        Pass fork_arm at the shared prefix only. Normal resume preserves all
        timing and check history. A fork never obtains a decision from another arm.
        """
        s = copy.deepcopy(dict(state))
        if s.get("sampling_protocol") != self.SAMPLING_PROTOCOL:
            raise ValueError("checkpoint sampling protocol mismatch; changed sampling requires a new run")
        if tuple(s["random_ids"]) != self.random_ids:
            raise ValueError("checkpoint Random sampling pool mismatch; cannot resume a fixed-subset "
                             "checkpoint with full-pool Random sampling")
        if (s["config"] != asdict(self.config) or tuple(s["candidates"]) != self.candidates or
                tuple(s["validation"]) != self.validation or
                tuple(s["sr_ranked_ids"]) != self.sr_ranked_ids):
            raise ValueError("checkpoint protocol, data or cache mismatch")
        used = s["used_training_ids"]
        if (len(used) != len(set(used)) or not set(used) <= set(self.candidates) or
                type(s["sampling_cycle"]) is not int or s["sampling_cycle"] < 0):
            raise ValueError("invalid saved sampling progress")
        if fork_arm is not None and fork_arm not in self.ARMS:
            raise ValueError("invalid fork arm")
        active = s["active_selection"]
        if active is not None:
            on = _unique(active["on_ids"], "saved candidates")
            selected = _unique(active["train_ids"], "saved selected prompts")
            if (len(on) != self.config.scoring_prompts or
                    len(selected) != self.config.training_prompts or
                    not set(selected) <= set(on) <= set(self.candidates) or
                    active["step"] % self.config.selection_interval or
                    not 0 <= active["step"] < s["step"]):
                raise ValueError("invalid saved selection block")
        continuing_on = (fork_arm or s["arm"]) in {"on_policy", "switch"} and (
            fork_arm is not None or s["switched_at"] is None)
        if continuing_on and s["step"] % self.config.selection_interval and (
                active is None or s["step"] - active["step"] >= self.config.selection_interval):
            raise ValueError("missing or stale saved selection block")
        self.backend.load_state_dict(s["backend"])
        self.step = s["step"]
        self.active_selection = active
        self.used_training_ids, self.sampling_cycle = set(used), s["sampling_cycle"]
        self.sr_preparation_wall_seconds = s["sr_preparation_wall_seconds"]
        if fork_arm is None:
            self.arm, self.switched_at = s["arm"], s["switched_at"]
            self.rule = TemporalRule(**s["rule"])
            self.costs, self.history = s["costs"], s["history"]
        else:
            self.arm, self.switched_at = fork_arm, None
            self.rule = TemporalRule(self.config.check_interval)
            self.costs = dict.fromkeys(self.costs, 0.0)
            if fork_arm in {"sr", "switch"}:
                self._charge_preparation()
            self.history = []
