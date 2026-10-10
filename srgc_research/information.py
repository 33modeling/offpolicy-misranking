"""Observe selection, responses and real updates at one immutable model state.

This is a measurement, not another selector-performance trajectory. Both
selectors start from the same weights and AdamW state. Every temporary update
is restored, and original checkpoints are never written.
"""

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from srgc_rebuttal.distributed import primary
from srgc_rebuttal.objectives import grpo_advantages
from srgc_rebuttal.progress import record as progress
from srgc_rebuttal.runtime import atomic_json
from srgc_rebuttal.srgc import Engine, cosine_scores, stream_seed, top_ids

from .backend import vector_cosine
from .information_report import read_object, result_digest
from .storage import atomic_torch

PROTOCOL = "srgc-selection-information-v1"
METHODS = ("on_policy", "sr")
PHASES = ("score-A", "score-B", "probe", *METHODS)


def file_hash(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256")
    return digest.hexdigest()


def reward_summary(rewards):
    values = [float(x) for x in rewards]
    if not values or any(x not in (0., 1.) for x in values):
        raise ValueError("expected nonempty binary response rewards")
    count = int(sum(values))
    return {"responses": len(values), "successes": count,
            "success_rate": count / len(values), "mixed": 0 < count < len(values),
            "all_wrong": count == 0, "all_correct": count == len(values)}


def parameter_layout(backend):
    offset, rows = 0, []
    for name, parameter in backend.train_parameters:
        rows.append({"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                     "offset": offset, "elements": parameter.numel()})
        offset += parameter.numel()
    return rows


def raw_responses(backend, records):
    """Persist exact token sequences, rewards and observed termination only."""
    result = {}
    eos = backend.tokenizer.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    for rid, record in records.items():
        sequences, rewards, start = record[:3]
        rows = []
        for index, (sequence, reward) in enumerate(zip(sequences, rewards)):
            tokens = sequence.tolist() if isinstance(sequence, torch.Tensor) else list(sequence)
            if not 0 < start < len(tokens):
                raise ValueError("response must have a nonempty prompt and suffix")
            suffix = tokens[start:]
            rows.append({"response": index, "sequence_ids": tokens, "prompt_tokens": start,
                         "completion_tokens": len(suffix), "reward": float(reward),
                         "text": backend.tokenizer.decode(torch.tensor(suffix), skip_special_tokens=True),
                         "finish_reason": "eos" if suffix[-1] in eos else
                             "length" if len(suffix) >= backend.max_new_tokens else "unknown"})
        result[rid] = {**reward_summary(rewards), "samples": rows,
                      "unique_completions": len({tuple(r["sequence_ids"][start:]) for r in rows})}
    return result


def log_probabilities(backend, records):
    local = {}
    with torch.no_grad():
        for rid in list(records)[backend.rank::backend.world]:
            sequences, _, start = records[rid][:3]
            local[rid] = [backend._logps(s.to(backend.device), start).float().cpu().tolist()
                          for s in sequences]
            progress("information_logps", prompt=rid)
    return backend._gather(local)


def probe_gradient(backend, probe, *, distributed=True):
    """Actual trainable-parameter GRPO loss gradient; no optimizer step."""
    old_gradients = [None if p.grad is None else p.grad.detach().clone()
                     for _, p in backend.train_parameters]
    try:
        backend.optimizer.zero_grad(set_to_none=True)
        ids = list(probe)[backend.rank::backend.world] if distributed else list(probe)
        for rid in ids:
            sequences, rewards, start, old_logps = probe[rid]
            for sequence, old, advantage in zip(sequences, old_logps, grpo_advantages(rewards)):
                if not advantage:
                    continue
                logps = backend._logps(sequence.to(backend.device), start)
                ratio = (logps - old.to(backend.device)).exp()
                loss = -torch.minimum(ratio * float(advantage),
                    ratio.clamp(.8, 1.2) * float(advantage)).mean() / (len(probe) * len(rewards))
                loss.backward()
                progress("information_gradient_response", prompt=rid)
        for _, parameter in backend.train_parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            if distributed and backend.world > 1:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        gradient = torch.cat([p.grad.detach().float().cpu().flatten()
                              for _, p in backend.train_parameters])
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite probe gradient")
        return gradient
    finally:
        for (_, parameter), gradient in zip(backend.train_parameters, old_gradients):
            parameter.grad = gradient


def inspect_update(backend, records, *, seed, probe, probe_loss_gradient):
    """One real GRPO update, with replay of the newly sampled batch and restore."""
    initial = backend.state_dict()
    before = backend.parameter_vector()
    layout = parameter_layout(backend)
    old_logps = log_probabilities(backend, records)
    original_clip = torch.nn.utils.clip_grad_norm_
    captured = {}

    def capture(parameters, *args, **kwargs):
        parameters = list(parameters)
        if [id(p) for p in parameters] != [id(p) for _, p in backend.train_parameters]:
            raise ValueError("captured gradient layout differs")
        captured["gradient_before_clip"] = torch.cat(
            [p.grad.detach().float().cpu().flatten() for p in parameters])
        result = original_clip(parameters, *args, **kwargs)
        captured["gradient_after_clip"] = torch.cat(
            [p.grad.detach().float().cpu().flatten() for p in parameters])
        return result

    try:
        local_gradients = {}
        for rid in list(records)[backend.rank::backend.world]:
            sequences, rewards, start = records[rid]
            frozen = {rid: (sequences, rewards, start,
                           [torch.tensor(values) for values in old_logps[rid]])}
            local_gradients[rid] = probe_gradient(backend, frozen, distributed=False)
        gathered = backend._gather(local_gradients)
        problem_gradients = {rid: gathered[rid] for rid in records}
        # A real zero-gradient Adam step includes momentum and its step counter.
        # grad=None would skip those and would be the wrong comparison.
        for _, parameter in backend.train_parameters:
            parameter.grad = torch.zeros_like(parameter)
        backend.optimizer.step()
        zero_update = backend.parameter_vector() - before
        backend.load_state_dict(initial)
        torch.nn.utils.clip_grad_norm_ = capture
        try:
            with backend.replaying(records):
                metrics = backend.train(list(records), responses=len(next(iter(records.values()))[1]),
                                        objective="grpo", seed=seed)
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
        after = backend.parameter_vector()
        after_state = backend.state_dict()
        update = after - before
        vectors = {**captured, "parameter_before": before, "parameter_after": after,
                   "update": update, "zero_gradient_update": zero_update,
                   "batch_incremental_update": update - zero_update,
                   "probe_loss_gradient": probe_loss_gradient}
        reconstructed = torch.stack(list(problem_gradients.values())).mean(0)
        reconstruction_error = float((reconstructed - captured["gradient_before_clip"]).double().norm())
        # Per-problem accumulation rounds differently from the actual batch.
        # A norm bound avoids huge relative errors in coordinates that cancel.
        precision = max(torch.finfo(p.dtype).eps for _, p in backend.train_parameters)
        component_scale = np.mean([float(g.double().norm()) for g in problem_gradients.values()])
        reconstruction_tolerance = max(1e-4, 8 * precision) * component_scale + 1e-7
        if reconstruction_error > reconstruction_tolerance:
            raise FloatingPointError("per-problem gradients do not reconstruct the actual batch gradient")
        if any(not torch.isfinite(v).all() for v in vectors.values()):
            raise FloatingPointError("nonfinite gradient or parameter update")
        new_logps = log_probabilities(backend, records)
        samples = raw_responses(backend, records)
        for rid, row in samples.items():
            for sample, old, new in zip(row["samples"], old_logps[rid], new_logps[rid]):
                if len(old) != sample["completion_tokens"] or len(old) != len(new):
                    raise ValueError("response log-probability/token lengths differ")
                sample.update(logps_before=old, logps_after=new,
                    mean_logp_change=float(np.mean(new) - np.mean(old)),
                    sum_logp_change=float(sum(new) - sum(old)))
        parameters = []
        for item in layout:
            region = slice(item["offset"], item["offset"] + item["elements"])
            change = update[region].double()
            norm = float(before[region].double().norm())
            parameters.append({**item,
                "gradient_norm_before_clip": float(captured["gradient_before_clip"][region].double().norm()),
                "gradient_norm_after_clip": float(captured["gradient_after_clip"][region].double().norm()),
                "update_norm": float(change.norm()),
                "relative_update_norm": float(change.norm()) / norm if norm else None,
                "incremental_update_norm": float(vectors["batch_incremental_update"][region].double().norm()),
                "changed_elements": int((change != 0).sum())})
        metrics = {**metrics, "updates": 1,
            "problem_gradient_reconstruction_error_norm": reconstruction_error,
            "problem_gradient_reconstruction_tolerance": float(reconstruction_tolerance),
            **{f"{name}_norm": float(vector.double().norm()) for name, vector in vectors.items()
               if name not in ("parameter_before", "parameter_after")},
            "probe_loss_after": backend.probe_loss(probe),
            "probe_ascent_update_cosine": vector_cosine(-probe_loss_gradient, update),
            "probe_ascent_incremental_update_cosine": vector_cosine(-probe_loss_gradient, vectors["batch_incremental_update"]),
            "predicted_probe_loss_change": float(probe_loss_gradient.double().dot(update.double()))}
        problems = {}
        for rid, gradient in problem_gradients.items():
            q = float(np.mean(records[rid][1]))
            problems[rid] = {"loss_gradient_norm": float(gradient.double().norm()),
                "reward_mix_factor": q * (1 - q) / (np.sqrt(q * (1 - q)) + 1e-4),
                "probe_ascent_cosine": vector_cosine(gradient, probe_loss_gradient),
                "ascent_update_cosine": vector_cosine(-gradient, update),
                "predicted_probe_loss_change_if_sgd": -float(probe_loss_gradient.double().dot(gradient.double()))}
        return {"metrics": metrics, "responses": samples, "parameters": parameters,
                "problem_gradients": problems}, {
            "layout": layout, "vectors": vectors, "backend_before": initial,
            "backend_after": after_state, "problem_loss_gradients": problem_gradients}
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
        backend.load_state_dict(initial)


def checkpoint_state(saved, data, config, stage):
    """Read ordinary Engine or StageStudy anchor states, never branch weights."""
    state = saved.get("state", saved)
    if "anchor" in state:
        state = state["anchor"]
        if state is None:
            raise ValueError("mechanism checkpoint has no saved anchor yet")
    if (state.get("step") != stage or state.get("config", {}).get("seed") != config.seed
            or state.get("config", {}).get("objective") != "grpo"
            or list(state.get("candidates", [])) != data["candidate_ids"]
            or list(state.get("validation", [])) != data["ranking_validation_ids"]):
        raise ValueError("checkpoint stage, seed, objective or input IDs differ")
    for key in ("responses", "training_prompts", "scoring_prompts", "projection_dim", "candidate_group_size",
                "selection_interval", "check_interval", "first_check"):
        if state["config"].get(key) != getattr(config, key):
            raise ValueError(f"checkpoint configuration differs: {key}")
    return state["backend"]


def source_identity(saved, *, input_hash, plan_hash, seed):
    """Published source checkpoints must carry both input and plan identity."""
    if (saved.get("input_sha256") != input_hash or saved.get("plan_sha256") != plan_hash
            or saved.get("seed", saved.get("config", {}).get("seed")) != seed):
        raise ValueError("checkpoint source input/plan/seed identity differs")


class InformationStudy:
    def __init__(self, backend, data, config, stage, output, identity, *, probe_prompts=8):
        if config.objective != "grpo" or type(stage) is not int or stage < 0:
            raise ValueError("information study requires GRPO and a nonnegative policy step")
        pool = [i for i in data["validation_pool_ids"] if i not in data["ranking_validation_ids"]]
        if type(probe_prompts) is not int or not 1 <= probe_prompts <= len(pool):
            raise ValueError("not enough independent validation-pool prompts for the probe")
        if (set(pool) & (set(data["candidate_ids"]) | set(data["evaluation_ids"]))
                or set(data["ranking_validation_ids"]) & (set(data["candidate_ids"]) | set(data["evaluation_ids"]))):
            raise ValueError("candidate, reference, probe and final evaluation must be disjoint")
        engine = Engine(backend, data["candidate_ids"], data["ranking_validation_ids"],
                        data["cached_rewards"], arm="on_policy", config=config)
        engine.step = stage
        self.candidates = list(engine._draw_candidates())
        self.tie_order = {rid: j for j, rid in enumerate(engine.sr_ranked_ids)}
        self.probe_ids = pool[:probe_prompts]
        self.backend, self.data, self.config, self.stage = backend, data, config, stage
        self.output, self.identity = Path(output), {**identity, "protocol": PROTOCOL, "stage": stage}
        self.anchor = backend.state_dict()

    def seed(self, role):
        return stream_seed(self.config.seed, self.stage, f"selection-information:{role}")

    def read_phase(self, name):
        path = self.output / f"{name}.json"
        if not path.exists():
            return None
        receipt = read_object(path)
        if receipt.get("identity") != self.identity or receipt.get("phase") != name:
            raise ValueError("phase belongs to a different measurement")
        expected = result_digest(receipt["result"])
        if not receipt["artifacts"] or receipt.get("result_sha256") != expected:
            raise ValueError("phase result hash differs")
        for artifact in receipt["artifacts"]:
            target = (self.output / artifact["file"]).resolve()
            if not target.is_relative_to(self.output.resolve()) or file_hash(target) != artifact["sha256"]:
                raise ValueError(f"measurement artifact differs: {target}")
        if self.load_tensors(receipt).get("result_sha256") != expected:
            raise ValueError("phase result differs from its measured tensor artifact")
        return receipt

    def publish(self, name, result, tensors):
        def save():
            tensor_path = self.output / f"{name}.pt"
            result_sha = result_digest(result)
            atomic_torch(tensor_path, {**tensors, "result_sha256": result_sha})
            receipt = {"identity": self.identity, "phase": name, "result": result,
                       "result_sha256": result_sha,
                       "artifacts": [{"file": tensor_path.name, "sha256": file_hash(tensor_path)}]}
            atomic_json(self.output / f"{name}.json", receipt)
            return receipt
        return primary(save)

    def load_tensors(self, receipt):
        return torch.load(self.output / receipt["artifacts"][0]["file"], weights_only=False, map_location="cpu")

    def acquire(self, block):
        backend, config = self.backend, self.config
        backend.tokens, backend.last_rewards, backend.capture_tokens = {}, {}, True
        try:
            gradients = backend.score_gradients(self.candidates, responses=config.responses,
                group_size=config.candidate_group_size, seed=self.seed(f"{block}-candidate"))
            reference = backend.score_gradients(self.data["ranking_validation_ids"],
                responses=config.responses, group_size=config.responses, seed=self.seed(f"{block}-reference"))
            records = backend._gather(backend.tokens)
        finally:
            backend.capture_tokens = False
        matrix = np.stack([gradients[i] for i in self.candidates])
        query = np.stack([reference[i] for i in self.data["ranking_validation_ids"]]).mean(0)
        scores = cosine_scores(matrix, query)
        result = {"candidate_ids": self.candidates, "reference_ids": self.data["ranking_validation_ids"],
                  "scores": dict(zip(self.candidates, scores.tolist())),
                  "dots": dict(zip(self.candidates, (matrix @ query).tolist())),
                  "norms": dict(zip(self.candidates, np.linalg.norm(matrix, axis=1).tolist())),
                  "reference_norm": float(np.linalg.norm(query)), "responses": raw_responses(backend, records)}
        return result, {"candidate_vectors": matrix, "reference_vector": query,
                        "score_parameter_names": self.anchor["score_names"]}

    def selections(self, scoring):
        ids = self.candidates
        scores = scoring["scores"]
        on = list(top_ids(ids, [scores[i] for i in ids], self.config.training_prompts,
                          stream_seed(self.config.seed + 1000, self.stage, "online-ties")))
        sr = sorted(ids, key=lambda i: (abs(np.mean(self.data["cached_rewards"][i]) - .5), self.tie_order[i]))
        return {"on_policy": on, "sr": sr[:self.config.training_prompts]}

    def run(self, *, publish_endpoint=True):
        try:
            return self._run(publish_endpoint=publish_endpoint)
        finally:
            self.backend.load_state_dict(self.anchor)

    def _run(self, *, publish_endpoint):
        receipts = {}
        for phase in PHASES:
            receipt = primary(lambda phase=phase: self.read_phase(phase))
            if receipt is not None:
                receipts[phase] = receipt
                continue
            backend = self.backend
            backend.load_state_dict(self.anchor)
            with backend.operation(phase), backend.cost_meter.phase(
                    "selection" if phase.startswith("score-") else "training" if phase in METHODS else "diagnostic",
                    self.stage, backend.gpu_count):
                if phase.startswith("score-"):
                    result, tensors = self.acquire(phase[-1])
                elif phase == "probe":
                    probe = backend.frozen_probe(self.probe_ids, responses=self.config.responses,
                                                 seed=self.seed("probe"))
                    loss = backend.probe_loss(probe)
                    gradient = probe_gradient(backend, probe)
                    result = {"ids": self.probe_ids, "loss_before": loss,
                              "gradient_norm": float(gradient.double().norm()),
                              "responses": raw_responses(backend, probe)}
                    tensors = {"probe": probe, "gradient": gradient,
                               "layout": parameter_layout(backend)}
                else:
                    selected = self.selections(receipts["score-A"]["result"])
                    ids = selected[phase]
                    collected = backend.collect(ids, self.config.responses, self.seed(f"training-{phase}"))
                    records = {rid: collected[rid] for rid in ids}
                    probe = self.load_tensors(receipts["probe"])
                    result, tensors = inspect_update(backend, records, seed=self.seed(f"training-{phase}"),
                        probe=probe["probe"], probe_loss_gradient=probe["gradient"])
                    result.update(selected_ids=ids, candidate_ids=self.candidates,
                                  selection_source="score-A" if phase == "on_policy" else "original-cached-success-rate")
                    result["metrics"]["probe_loss_before"] = receipts["probe"]["result"]["loss_before"]
            receipts[phase] = self.publish(phase, result, tensors)
            primary(lambda phase=phase: print(f"INFORMATION stage={self.stage} phase={phase} saved", flush=True))
        self.backend.load_state_dict(self.anchor)
        result = {"identity": self.identity, "status": "complete", "methods": list(METHODS),
                  "configuration": asdict(self.config),
                  "candidate_ids": self.candidates, "reference_ids": self.data["ranking_validation_ids"],
                  "probe_ids": self.probe_ids, "selected": self.selections(receipts["score-A"]["result"]),
                  "questions": {i: self.data["records"][i] for i in self.candidates},
                  "phases": {name: f"{name}.json" for name in PHASES},
                  "phase_sha256": {name: file_hash(self.output / f"{name}.json") for name in PHASES},
                  "interpretation": "same-state selection and one-update observations; not a full training trajectory"}
        if publish_endpoint:
            primary(lambda: atomic_json(self.output / "endpoint.json", result))
        return result
