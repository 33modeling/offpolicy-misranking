"""Unmerged-LoRA response derivatives with a checked finite-difference backend."""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch

from grads import _padded_token_logps, loo_advantages, sequence_logprobs_batch


def trainable_parameters(model):
    named = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not named or any("lora_" not in name for name in named):
        raise ValueError("scoring requires unmerged trainable LoRA parameters only")
    if any(p.dtype not in (torch.float32, torch.float64) for p in named.values()):
        raise ValueError("LoRA parameters must use FP32 or FP64 for directional probes")
    return named


def load_current(config, run):
    from peft import PeftModel

    from rollout import load_model

    model, tokenizer = load_model(config["model"])
    model = PeftModel.from_pretrained(model, str(run / f"policy_step_{config['drift']}"),
                                     is_trainable=True, local_files_only=True)
    model.config.use_cache = False
    model.eval()
    trainable_parameters(model)
    return model, tokenizer


def response_logps(model, row):
    ids, start = row["input_ids"], row["resp_start"]
    if not 0 < start < ids.numel():
        raise ValueError("invalid response boundary")
    return _padded_token_logps(model, [ids])[0][start-1:]


def validation_gradient(model, groups, *, progress=None):
    """Sum per-prompt reward gradients; no response-length division on validation."""
    named = trainable_parameters(model)
    model.zero_grad(set_to_none=True)
    token_work = 0
    for done, (idx, rows) in enumerate(sorted(groups.items())):
        if progress:
            progress(done, len(groups), idx)
        advantages = loo_advantages(torch.tensor([r["reward"] for r in rows], dtype=torch.float64))
        for row, advantage in zip(rows, advantages, strict=True):
            if float(advantage) == 0.:
                continue
            values = response_logps(model, row)
            (values.sum()*(float(advantage)/len(rows))).backward()
            token_work += int(row["input_ids"].numel())
    sums = {name: (p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p, device="cpu"))
            for name, p in named.items()}
    model.zero_grad(set_to_none=True)
    if not all(torch.isfinite(g).all() for g in sums.values()):
        raise ValueError("nonfinite validation gradient")
    return {"sums": sums, "prompts": len(groups), "gradient_input_tokens": token_work}


def make_direction(partials, optimizer=None):
    names = list(partials[0]["sums"])
    if not names or any(list(p["sums"]) != names for p in partials):
        raise ValueError("validation parameter names/order differ between shards")
    count = sum(p["prompts"] for p in partials)
    if count < 1:
        raise ValueError("no validation prompts")
    direction = {}
    for name in names:
        values = [p["sums"][name].double() for p in partials]
        if any(x.shape != values[0].shape for x in values):
            raise ValueError("validation parameter shapes differ")
        direction[name] = sum(values)/count
    geometry = "identity"
    if optimizer is not None:
        groups = optimizer["param_groups"]
        ids = [i for group in groups for i in group["params"]]
        if len(groups) != 1 or len(ids) != len(names) or len(set(ids)) != len(ids):
            raise ValueError("optimizer must match the trainer's single ordered LoRA group")
        group = groups[0]
        beta2, epsilon = group["betas"][1], group["eps"]
        if not 0 <= beta2 < 1 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("invalid AdamW preconditioner")
        for name, idx in zip(names, ids, strict=True):
            state = optimizer["state"].get(idx)
            if not state:
                raise ValueError("optimizer lacks a trainable parameter's moments")
            moment = state["max_exp_avg_sq"] if group.get("amsgrad", False) else state["exp_avg_sq"]
            moment = moment.double()
            step = float(state["step"])
            if (moment.shape != direction[name].shape or not torch.isfinite(moment).all()
                    or (moment < 0).any() or not math.isfinite(step) or step < 1):
                raise ValueError("invalid optimizer second moment or step")
            direction[name] /= (moment/(1-beta2**step)).sqrt()+epsilon
        geometry = "frozen_adam_rms"
    norm = math.sqrt(sum(float(g.square().sum()) for g in direction.values()))
    if not math.isfinite(norm) or norm <= 1e-20:
        raise ValueError("validation direction is zero or nonfinite; no usable ranking signal")
    return {"direction": {name: (g/norm).float() for name, g in direction.items()},
            "normalizing_l2": norm, "geometry": geometry, "validation_prompts": count,
            "scope": "Unit direction from validation reward gradients; frozen RMS is not an exact AdamW update."}


def device_direction(model, values):
    named = trainable_parameters(model)
    if list(named) != list(values) or any(named[n].shape != values[n].shape for n in named):
        raise ValueError("direction does not match trainable LoRA coordinates")
    if not all(torch.isfinite(value).all() for value in values.values()):
        raise ValueError("nonfinite direction")
    return {name: values[name].to(device=p.device, dtype=p.dtype) for name, p in named.items()}


def exact_directional(model, rows, direction):
    named = trainable_parameters(model)
    derivatives = []
    for row in rows:
        values = response_logps(model, row)
        gradients = torch.autograd.grad(values.mean(), tuple(named.values()), allow_unused=True)
        derivative = sum(float((g.detach().double()*direction[name].double()).sum())
                         for name, g in zip(named, gradients, strict=True) if g is not None)
        derivatives.append(derivative)
    result = torch.tensor(derivatives, dtype=torch.float64)
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite autograd derivative")
    return result


@contextmanager
def perturbed(model, direction, step):
    if not math.isfinite(step) or step == 0:
        raise ValueError("probe step must be finite and nonzero")
    named = trainable_parameters(model)
    original = {name: p.detach().clone() for name, p in named.items()}
    try:
        with torch.no_grad():
            for name, p in named.items():
                p.copy_(original[name]+step*direction[name])
        yield
    finally:
        with torch.no_grad():
            for name, p in named.items():
                p.copy_(original[name])


def finite_directional(model, rows, direction, step, micro_batch):
    with perturbed(model, direction, step):
        positive = sequence_logprobs_batch(model, rows, micro_batch=micro_batch)
    with perturbed(model, direction, -step):
        negative = sequence_logprobs_batch(model, rows, micro_batch=micro_batch)
    result = torch.tensor([float((p.double()-n.double()).mean())/(2*step)
                           for p, n in zip(positive, negative, strict=True)], dtype=torch.float64)
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite finite-difference derivative")
    return result


def calibrate(model, rows, direction, *, step, micro_batch, rtol=.1, atol=1e-6):
    if not rows or not math.isfinite(step) or step <= 0:
        raise ValueError("positive probe step and calibration responses required")
    exact = exact_directional(model, rows, direction)
    if float(exact.norm()) < 1e-8:
        raise ValueError("calibration responses have no resolved directional signal")
    errors = []
    for scale in (step, step/2):
        estimated = finite_directional(model, rows, direction, scale, micro_batch)
        delta = (estimated-exact).abs()
        errors.append({"step": scale, "max_absolute_error": float(delta.max()),
                       "relative_l2_error": float(delta.norm()/exact.norm())})
        if not bool(torch.all(delta <= atol+rtol*exact.abs())):
            raise ValueError(f"finite-difference calibration failed at step={scale}: {errors[-1]}; "
                             "use an explicitly separate autograd run or review the probe step")
    return {"step": step/2, "responses": len(rows), "rtol": rtol, "atol": atol,
            "checks": errors, "scope": "Local numerical probe, not a global error certificate."}
