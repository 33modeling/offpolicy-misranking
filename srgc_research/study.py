"""Resumable trajectories and checkpoint-matched diagnostic interventions."""

import copy
import itertools
from collections import Counter
from dataclasses import asdict, replace

import numpy as np

from srgc_rebuttal.srgc import (
    Engine,
    cached_sr_set,
    cosine_scores,
    gradient_contrast,
    stream_seed,
    top_ids,
)

from .arcus import Arcus
from .design import (
    EVAL_UPDATES,
    PROTOCOL,
    candidate_draw,
    matched_control,
    reference_sets,
    score_bins,
)


class NestedEngine(Engine):
    SAMPLING_PROTOCOL = PROTOCOL + ":nested-candidate-permutation"

    def _draw_candidates(self):
        return candidate_draw(self.candidates, self.config.scoring_prompts, self.config.seed, self.step)


def config_for(config, condition):
    return replace(config, scoring_prompts=condition.candidates, training_prompts=condition.batch)


def make_engine(backend, data, config, condition):
    references = data["ranking_validation_ids"]
    if condition.reference >= 0:
        references = reference_sets(data, config.seed)[0][condition.reference]
    arm = "on_policy" if condition.arm == "lesser" else condition.arm
    return NestedEngine(backend, data["candidate_ids"], references, data["cached_rewards"],
                        config=config_for(config, condition), arm=arm)


def evaluate(backend, data, seed, step):
    means = backend.evaluate(data["evaluation_ids"], responses=8,
                             seed=stream_seed(seed, step, "research-heldout"))
    return {"update": step, "reward": float(np.mean(list(means.values()))),
            "per_question_reward": means, "binary_samples": backend.evaluation_samples,
            "sampling": {"responses": 8, "temperature": 1., "top_p": 1., "top_k": 0,
                         "max_new_tokens": backend.max_new_tokens}}


def feature_comparison(backend, data, config):
    """Score exactly the same t0 responses with dense and readout representations."""
    candidates = candidate_draw(data["candidate_ids"], 40, config.seed, 0)
    references = data["ranking_validation_ids"]
    records = backend.collect([*candidates, *references], 8, stream_seed(config.seed, 0, "feature-comparison"))
    rows = {}
    previous = backend.scorer
    try:
        for scorer in ("dense", "lesser"):
            backend.scorer = scorer
            with backend.cost_meter.section(scorer), backend.replaying(records):
                g = backend.score_gradients(candidates, responses=8, group_size=4, seed=0)
                v = backend.score_gradients(references, responses=8, group_size=8, seed=0)
                scores = cosine_scores(np.stack([g[i] for i in candidates]), np.stack(list(v.values())).mean(0))
                rows[scorer] = {"scores": dict(zip(candidates, scores.tolist())),
                    "top_ids": list(top_ids(candidates, scores, 4, config.seed))}
    finally:
        backend.scorer = previous
    return {"candidate_ids": list(candidates), "reference_ids": list(references), "representations": rows,
            "top4_overlap": len(set(rows["dense"]["top_ids"]) & set(rows["lesser"]["top_ids"])),
            "same_rollouts": True, "cost_role": "diagnostic-not-deployment"}


class Trajectory:
    def __init__(self, backend, data, config, condition):
        self.backend, self.data, self.config, self.condition = backend, data, config, condition
        self.engine = None if condition.arm == "arcus_adapted" else make_engine(backend, data, config, condition)
        self.arcus = Arcus(data["candidate_ids"], condition.batch, config.responses) if self.engine is None else None
        backend.scorer = "lesser" if condition.arm == "lesser" else "dense"
        self.step, self.curve, self.history = 0, [], []

    @property
    def done(self):
        return self.step == self.condition.updates and self.curve and self.curve[-1]["update"] == self.step

    def advance(self):
        b, c = self.backend, self.config
        if ((self.step % c.selection_interval == 0 or self.step == self.condition.updates) and
                (not self.curve or self.curve[-1]["update"] != self.step)):
            with b.operation(f"eval-{self.step}"), b.cost_meter.phase("evaluation", self.step, b.gpu_count):
                self.curve.append(evaluate(b, self.data, c.seed, self.step))
            return "evaluation"
        with b.operation(f"update-{self.step}"):
            if self.engine is not None:
                self.engine.update()
                self.step = self.engine.step
            else:
                with b.cost_meter.phase("selection", self.step, b.gpu_count), b.cost_meter.stage("arcus_belief_sampling"):
                    ids = self.arcus.sample(stream_seed(c.seed, self.step, "arcus-sampling"))
                with b.cost_meter.phase("training", self.step, b.gpu_count):
                    # ARCUS's extra margin is generated once and shared with training.
                    records = b.collect(ids, c.responses, stream_seed(c.seed, self.step, "training"))
                    with b.cost_meter.stage("arcus_belief_filter"):
                        selected = self.arcus.observe({i: r[1] for i, r in records.items()})
                    if selected:
                        with b.replaying(records):
                            metrics = b.train(selected, responses=c.responses, objective=c.objective,
                                              seed=stream_seed(c.seed, self.step, "training"))
                    else:
                        metrics = {"optimizer_update_skipped": True}
                self.history.append({"checkpoint": self.step, "candidate_ids": ids, "train_ids": selected,
                    "generated_responses": len(ids) * c.responses, "target": self.arcus.target,
                    "informative_fraction": sum(0 < sum(r[1]) < c.responses for r in records.values()) / len(ids),
                    "metrics": metrics})
                self.step += 1
        return "training"

    def state_dict(self):
        return {"protocol": PROTOCOL, "condition": self.condition.record(), "config": asdict(self.config),
                "step": self.step, "curve": self.curve, "history": self.history,
                "engine": self.engine.state_dict() if self.engine else None,
                "arcus": self.arcus.state_dict() if self.arcus else None,
                "backend": self.backend.state_dict() if self.arcus else None}

    def load_state_dict(self, state):
        if (state["protocol"] != PROTOCOL or state["condition"] != self.condition.record()
                or state["config"] != asdict(self.config)):
            raise ValueError("research trajectory identity differs")
        self.step, self.curve, self.history = state["step"], copy.deepcopy(state["curve"]), copy.deepcopy(state["history"])
        if self.engine:
            self.engine.load_state_dict(state["engine"])
        else:
            self.backend.load_state_dict(state["backend"])
            self.arcus.load_state_dict(state["arcus"])

    def result(self):
        return {"curve": self.curve, "history": self.engine.history if self.engine else self.history,
                "switched_at": self.engine.switched_at if self.engine else None,
                "optimizer_updates": self.step if self.engine else sum(
                    not r["metrics"].get("optimizer_update_skipped", False) for r in self.history),
                "sampling_rounds": self.step,
                "reference_overlap": reference_sets(self.data, self.config.seed)[1]
                    if self.condition.reference >= 0 else None,
                "scoring_representation": "readout-rademacher-64x64" if self.condition.arm == "lesser"
                    else "final-four-dense-layers-and-final-norm" if self.condition.arm in {"on_policy", "switch"}
                    else "no-gradient-scorer"}


class Diagnostic:
    def __init__(self, backend, data, config, condition, anchor):
        self.backend, self.data, self.config, self.condition = backend, data, config, condition
        self.anchor = copy.deepcopy(anchor)
        if anchor["step"] != condition.stage:
            raise ValueError("diagnostic anchor is from the wrong stage")
        if (anchor["config"] != asdict(config) or list(anchor["candidates"]) != data["candidate_ids"]
                or list(anchor["validation"]) != data["ranking_validation_ids"]
                or anchor["sampling_protocol"] != NestedEngine.SAMPLING_PROTOCOL):
            raise ValueError("diagnostic anchor configuration or inputs differ")
        self.step, self.phase, self.branch, self.local_step = 0, "acquire", 0, 0
        self.selected, self.details, self.rows, self.vectors, self.centers = {}, {}, [], {}, []
        self.baseline, self.probe, self.last_evaluated = None, None, -1
        backend.load_state_dict(anchor["backend"])

    @property
    def done(self):
        return self.phase == "done"

    def seed(self, purpose):
        return stream_seed(self.config.seed, self.condition.stage, f"draw-{self.condition.draw}:{purpose}")

    def acquire(self):
        b, c, data = self.backend, self.config, self.data
        sampling_seed = c.seed if self.condition.draw == 0 else self.seed("candidate-draw")
        ids = candidate_draw(data["candidate_ids"], 40, sampling_seed, self.condition.stage)
        sr = ()
        if self.condition.arm == "n04":
            preview = NestedEngine(b, data["candidate_ids"], data["ranking_validation_ids"],
                                   data["cached_rewards"], config=c, arm="on_policy")
            preview.load_state_dict(self.anchor)
            sr = preview._sr_comparison()
        union = list(dict.fromkeys([*ids, *sr])) if self.condition.arm == "n04" else list(ids)
        b.last_rewards = {}
        gradients = b.score_gradients(union, responses=8, group_size=4, seed=self.seed("diagnostic-candidates"))
        rewards = b._gather(b.last_rewards)
        if self.condition.arm == "n04":
            references, overlap = reference_sets(data, c.seed)
            self.details = {"reference_overlap": overlap, "candidate_ids": list(ids), "sr_ids": list(sr),
                            "references": references, "topic_matching": "not available in source bundle"}
            reference_union = [i for i in data["validation_pool_ids"] if any(i in r for r in references)]
            for repeat in (0, 1):
                vectors = b.score_gradients(reference_union, responses=8, group_size=8,
                    seed=self.seed(f"reference-repeat-{repeat}"))
                for index, reference in enumerate(references):
                    v = np.stack([vectors[i] for i in reference]).mean(0)
                    scores = cosine_scores(np.stack([gradients[i] for i in ids]), v)
                    self.rows.append({"reference": index, "repeat": repeat,
                        "d": gradient_contrast(ids, sr, gradients, v), "scores": dict(zip(ids, scores.tolist())),
                        "top_ids": list(top_ids(ids, scores, 4, self.seed("ties"))),
                        "reference_norm": float(np.linalg.norm(v))})
            self.phase = "done"
            return
        query = b.score_gradients(data["ranking_validation_ids"], responses=8, group_size=8,
                                  seed=self.seed("diagnostic-reference"))
        v = np.stack(list(query.values())).mean(0)
        scores = cosine_scores(np.stack([gradients[i] for i in ids]), v)
        top = list(top_ids(ids, scores, 4, self.seed("ties")))
        random = np.random.default_rng(self.seed("random-batch")).choice(ids, 4, replace=False).tolist()
        self.details = {"candidate_ids": list(ids), "scores": dict(zip(ids, scores.tolist())),
            "current_rewards": {i: rewards[i] for i in ids}, "reference_norm": float(np.linalg.norm(v)),
            "cached_success": {i: float(np.mean(data["cached_rewards"][i])) for i in ids}}
        if self.condition.arm == "n03":
            self.selected, self.details["bins"] = score_bins(ids, scores, 4, self.seed("ties"))
        elif self.condition.arm == "n07":
            counts = Counter(i for r in self.anchor["history"] for i in r["train_ids"])
            axes = {"success": {i: int(sum(rewards[i])) for i in ids},
                    "exposure": {i: counts[i] for i in ids}}
            if all("topic" in data["records"][i] for i in ids):
                axes["topic"] = {i: data["records"][i]["topic"] for i in ids}
            self.selected = {"top": top, "random": random}
            self.details["matching"] = {}
            self.details["prior_exposures"] = dict(counts)
            for axis, labels in axes.items():
                selected, match = matched_control(ids, top, labels, self.seed(f"match-{axis}"))
                self.selected[f"matched-{axis}"] = selected
                self.details["matching"][axis] = {**match, "labels": labels}
            self.details["prompt_characters"] = {i: len(data["records"][i]["prompt"]) for i in ids}
        else:
            self.selected = {"on_policy": top, "sr": list(cached_sr_set(ids, data["cached_rewards"], 4,
                                                                        c.seed, c.responses)), "random": random}
        self.phase = "baseline"

    def advance(self):
        b, c = self.backend, self.config
        with b.operation(f"{self.condition.key}-{self.phase}-{self.branch}-{self.local_step}-{len(self.centers)}"), \
                b.cost_meter.phase("evaluation", self.step, b.gpu_count), \
                b.cost_meter.section(f"{self.condition.arm}_{self.phase}"):
            if self.phase == "acquire":
                self.acquire()
            elif self.phase == "baseline":
                self.baseline = evaluate(b, self.data, c.seed, self.condition.stage)
                if self.condition.arm == "n02":
                    self.probe = b.frozen_probe(self.data["evaluation_ids"][:8], responses=8,
                                                seed=self.seed("frozen-probe"))
                    self.details["frozen_probe_loss_before"] = b.probe_loss(self.probe)
                    self.phase = "center"
                else:
                    self.phase = "restore"
            elif self.phase == "center":
                i = len(self.centers)
                ids = candidate_draw(self.data["candidate_ids"], 4, self.seed("independent-centering"), i)
                vectors, _ = b.gradient_audit(ids, seed=self.seed(f"center-{i}"))
                self.centers.append(vectors["gradient"])
                if len(self.centers) == 8:
                    self.phase = "audit"
            elif self.phase == "audit":
                name = list(self.selected)[self.branch]
                vectors, metrics = b.gradient_audit(self.selected[name], seed=self.seed("audit-training"),
                    probe=self.probe, evaluation_ids=self.data["evaluation_ids"])
                import torch
                vectors["centered_a"] = vectors["gradient"] - torch.stack(self.centers[:4]).mean(0)
                vectors["centered_b"] = vectors["gradient"] - torch.stack(self.centers[4:]).mean(0)
                self.vectors[name] = vectors
                self.rows.append({"selector": name, "train_ids": self.selected[name], **metrics})
                self.branch += 1
                if self.branch == len(self.selected):
                    from .backend import vector_cosine
                    self.details["alignment"] = [{"left": a, "right": d,
                        **{k: vector_cosine(self.vectors[a][k], self.vectors[d][k])
                           for k in ("gradient", "update", "incremental_update")},
                        "centered_crossfit": [vector_cosine(self.vectors[a]["centered_a"], self.vectors[d]["centered_b"]),
                            vector_cosine(self.vectors[a]["centered_b"], self.vectors[d]["centered_a"])]}
                        for a, d in itertools.combinations(self.selected, 2)]
                    self.phase = "done"
            elif self.phase == "restore":
                b.load_state_dict(self.anchor["backend"])
                self.local_step, self.last_evaluated = 0, 0
                name = list(self.selected)[self.branch]
                self.rows.append({"selector": name, "train_ids": self.selected[name], **self.baseline,
                                  "branch_updates": 0})
                self.phase = "train"
            elif self.phase == "train":
                name = list(self.selected)[self.branch]
                b.last_rewards = {}
                before = b.parameter_vector()
                metrics = b.train(self.selected[name], responses=8, objective="grpo",
                                  seed=self.seed(f"branch-training-{self.local_step}"))
                metrics["update_norm"] = float((b.parameter_vector() - before).double().norm())
                rewards = b._gather(b.last_rewards)
                self.local_step += 1
                self.step += 1
                self.details.setdefault("training", []).append({"selector": name, "update": self.local_step,
                    "mixed_group_fraction": float(np.mean([0 < sum(r) < 8 for r in rewards.values()])),
                    "rewards": rewards, **metrics})
                if self.local_step in EVAL_UPDATES:
                    self.phase = "evaluate"
            elif self.phase == "evaluate":
                name = list(self.selected)[self.branch]
                row = evaluate(b, self.data, c.seed, self.condition.stage + self.local_step)
                self.rows.append({"selector": name, "train_ids": self.selected[name], **row,
                    "branch_updates": self.local_step, "reward_gain": row["reward"] - self.baseline["reward"]})
                self.last_evaluated = self.local_step
                if self.local_step == 25:
                    self.branch += 1
                    self.phase = "done" if self.branch == len(self.selected) else "restore"
                else:
                    self.phase = "train"
            else:
                raise ValueError(f"unknown diagnostic phase: {self.phase}")
        return self.phase

    def state_dict(self):
        names = ("step", "phase", "branch", "local_step", "selected", "details", "rows", "vectors",
                 "centers", "baseline", "probe", "last_evaluated")
        return {"protocol": PROTOCOL, "condition": self.condition.record(), "config": asdict(self.config),
                "backend": self.backend.state_dict(), **{name: copy.deepcopy(getattr(self, name)) for name in names}}

    def load_state_dict(self, state):
        if (state["protocol"] != PROTOCOL or state["condition"] != self.condition.record()
                or state["config"] != asdict(self.config)):
            raise ValueError("diagnostic resume identity differs")
        self.backend.load_state_dict(state["backend"])
        for name, value in state.items():
            if name not in {"protocol", "condition", "config", "backend"}:
                setattr(self, name, copy.deepcopy(value))

    def result(self):
        return {"stage": self.condition.stage, "draw": self.condition.draw, "selected": self.selected, "details": self.details,
                "baseline": self.baseline, "rows": self.rows,
                "cost_role": "research-diagnostic-not-deployment-selector",
                "raw_vectors": "state-latest.pt:state.vectors" if self.condition.arm == "n02" else None}
