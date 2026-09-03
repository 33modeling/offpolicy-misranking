"""Hierarchical bootstrap intervals for three-way ranking/reference scores.

Candidate micro-groups are resampled independently within every prompt.  The
validation prompts assigned to R, A, and B are resampled independently as well.
Prompt identities stay fixed, so the interval is conditional on the candidate
pool and must not be described as population-over-prompts uncertainty.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import torch

from select_rules import jittered_topk, topk_count

RIGHT_TIE_OFFSET = 104_729


def percentile(values: list[float], q: float) -> float:
    if not values or not 0.0 <= q <= 1.0:
        raise ValueError("percentile needs non-empty values and q in [0, 1]")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _bootstrap_scores(
    candidate: torch.Tensor,
    validation: torch.Tensor,
    draws: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return [draws, prompts] cosine scores for one independent half."""
    n, groups, dim = candidate.shape
    if groups < 1 or validation.shape[0] < 2 or validation.shape[1] != dim:
        raise ValueError("invalid candidate or validation bootstrap shape")
    means = torch.zeros((draws, n, dim), dtype=torch.float32, device=candidate.device)
    prompt_index = torch.arange(n, device=candidate.device).unsqueeze(0)
    for _ in range(groups):
        group_index = torch.randint(
            groups, (draws, n), generator=generator, device=candidate.device
        )
        means.add_(candidate[prompt_index, group_index])
    means.div_(groups)
    val_index = torch.randint(
        validation.shape[0],
        (draws, validation.shape[0]),
        generator=generator,
        device=validation.device,
    )
    val_mean = validation[val_index].mean(dim=1)
    numerator = torch.einsum("bnd,bd->bn", means, val_mean)
    denominator = means.norm(dim=2) * val_mean.norm(dim=1).unsqueeze(1)
    return torch.where(denominator > 0, numerator / denominator, 0.0)


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"valid_samples": 0}
    return {
        "valid_samples": len(values),
        "lower_one_sided_95": percentile(values, 0.05),
        "upper_one_sided_95": percentile(values, 0.95),
        "lower_two_sided_95": percentile(values, 0.025),
        "upper_two_sided_95": percentile(values, 0.975),
        "median": percentile(values, 0.5),
    }


def bootstrap_regime_intervals(
    run: Path,
    strata: Mapping[str, list[int]],
    selectors: Mapping[str, Mapping[int, float]] | None = None,
    *,
    frac: float = 0.10,
    samples: int = 2_000,
    seed: int = 0,
    tie_seed: int | None = None,
    chunk_size: int | None = None,
    device: str | None = None,
    retention_threshold: float = 0.50,
) -> dict[str, dict]:
    if samples < 100:
        raise ValueError("FIRST bootstrap requires at least 100 replicates")
    if not 0.0 < retention_threshold <= 1.0:
        raise ValueError("retention threshold must be in (0, 1]")
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.device(resolved_device)
    raw_micro = torch.load(
        run / "oracle_micro_groups.pt", map_location="cpu", weights_only=True
    )
    micro = {int(idx): value for idx, value in raw_micro.items()}
    if len(micro) != len(raw_micro):
        raise ValueError(
            "candidate micro-group IDs are not unique after integer conversion"
        )
    val_groups = torch.load(
        run / "val_groups.pt", map_location="cpu", weights_only=True
    ).float()
    ids = sorted(micro)
    stacks = [micro[idx].float() for idx in ids]
    shapes = {tuple(stack.shape) for stack in stacks}
    if len(shapes) != 1:
        raise ValueError(f"inconsistent oracle micro-group shapes: {sorted(shapes)}")
    groups, dim = stacks[0].shape
    if groups < 8 or groups % 4:
        raise ValueError(
            "FIRST needs a multiple of four with at least eight candidate groups"
        )
    if (
        val_groups.ndim != 2
        or val_groups.shape[0] < 8
        or val_groups.shape[0] % 4
        or val_groups.shape[1] != dim
    ):
        raise ValueError(
            "validation groups are incompatible with candidate projections"
        )

    full = torch.stack(stacks).to(target)
    val_groups = val_groups.to(target)
    position = {idx: pos for pos, idx in enumerate(ids)}
    stratum_positions = {}
    for name, stratum_ids in strata.items():
        missing = set(stratum_ids) - set(ids)
        if missing:
            raise ValueError(
                f"stratum {name} has unknown prompt IDs: {sorted(missing)[:5]}"
            )
        stratum_positions[name] = [position[idx] for idx in stratum_ids]
    normalized_selectors = {
        name: {int(idx): float(value) for idx, value in scores.items()}
        for name, scores in (selectors or {}).items()
    }
    for name, scores in normalized_selectors.items():
        if set(scores) != set(ids):
            raise ValueError(f"selector {name} prompt IDs differ from oracle IDs")
    generator = torch.Generator(device=target).manual_seed(seed)
    fixed_tie_seed = seed if tie_seed is None else tie_seed
    if chunk_size is None:
        chunk_size = 16 if target.type == "cuda" else 1
    chunk_size = max(1, chunk_size)
    values = {
        name: {
            "floor": [],
            "fresh_gain": [],
            "selectors": {
                selector: {"gain": [], "retention_margin": []}
                for selector in normalized_selectors
            },
        }
        for name in strata
    }
    selected = {}
    for name, stratum_ids in strata.items():
        if len(stratum_ids) < 20:
            continue
        k = topk_count(len(stratum_ids), frac)
        selected[name] = {
            selector: jittered_topk(
                {idx: scores[idx] for idx in stratum_ids},
                k,
                fixed_tie_seed,
            )
            for selector, scores in normalized_selectors.items()
        }

    completed = 0
    candidate_half = groups // 2
    candidate_quarter = groups // 4
    val_count = val_groups.shape[0]
    val_half = val_count // 2
    val_quarter = val_count // 4
    while completed < samples:
        batch = min(chunk_size, samples - completed)
        score_r = _bootstrap_scores(
            full[:, :candidate_quarter], val_groups[:val_half], batch, generator
        ).cpu()
        score_a = _bootstrap_scores(
            full[:, candidate_half : candidate_half + candidate_quarter],
            val_groups[val_half : val_half + val_quarter],
            batch,
            generator,
        ).cpu()
        score_b = _bootstrap_scores(
            full[:, candidate_half + candidate_quarter :],
            val_groups[val_half + val_quarter :],
            batch,
            generator,
        ).cpu()
        for draw in range(batch):
            for name, stratum_ids in strata.items():
                if len(stratum_ids) < 20:
                    continue
                pos = stratum_positions[name]
                k = topk_count(len(pos), frac)
                left = {
                    idx: float(score_a[draw, p]) for idx, p in zip(stratum_ids, pos)
                }
                right = {
                    idx: float(score_b[draw, p]) for idx, p in zip(stratum_ids, pos)
                }
                ranking = {
                    idx: float(score_r[draw, p]) for idx, p in zip(stratum_ids, pos)
                }
                reference = {
                    idx: (left[idx] + right[idx]) / 2.0 for idx in stratum_ids
                }
                # Tie streams are fixed across bootstrap draws.  Varying them
                # here would incorrectly fold tie sensitivity into sampling
                # uncertainty; tie-stream sensitivity is reported separately.
                top_a = jittered_topk(left, k, fixed_tie_seed + 17)
                top_b = jittered_topk(right, k, fixed_tie_seed + 17 + RIGHT_TIE_OFFSET)
                top_r = jittered_topk(ranking, k, fixed_tie_seed + 31)
                values[name]["floor"].append(len(top_a & top_b) / k)
                random_utility = sum(reference.values()) / len(reference)
                fresh_gain = (
                    sum(reference[idx] for idx in top_r) / k - random_utility
                )
                values[name]["fresh_gain"].append(fresh_gain)
                for selector, top_stale in selected.get(name, {}).items():
                    gain = (
                        sum(reference[idx] for idx in top_stale) / k - random_utility
                    )
                    values[name]["selectors"][selector]["gain"].append(gain)
                    # Once the separate fresh-gain gate establishes a positive
                    # denominator, this paired contrast is nonnegative exactly
                    # when the requested retention fraction is reached.
                    values[name]["selectors"][selector]["retention_margin"].append(
                        gain - retention_threshold * fresh_gain
                    )
        completed += batch

    result = {}
    for name, observed in values.items():
        if not observed["floor"]:
            continue
        result[name] = {
            "schema": "first-hierarchical-bootstrap/v2",
            "samples": samples,
            "seed": seed,
            "device": str(target),
            "retention_threshold": retention_threshold,
            **_summary(observed["floor"]),
            "fresh_gain": _summary(observed["fresh_gain"]),
            "selectors": {
                selector: {
                    "gain": _summary(metrics["gain"]),
                    "retention_margin": _summary(metrics["retention_margin"]),
                }
                for selector, metrics in observed["selectors"].items()
            },
        }
    return result


def bootstrap_floor_intervals(
    run: Path,
    strata: Mapping[str, list[int]],
    *,
    frac: float = 0.10,
    samples: int = 2_000,
    seed: int = 0,
    tie_seed: int | None = None,
    chunk_size: int | None = None,
    device: str | None = None,
) -> dict[str, dict]:
    """Compatibility wrapper for callers that only need FIRST intervals."""
    return bootstrap_regime_intervals(
        run,
        strata,
        frac=frac,
        samples=samples,
        seed=seed,
        tie_seed=tie_seed,
        chunk_size=chunk_size,
        device=device,
    )
