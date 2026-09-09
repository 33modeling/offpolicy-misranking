"""Reliability-versus-budget readout for an on-policy reference (CPU, read-only).

    python src/reliability_budget.py <run_dir> [<run_dir> ...] [--out FILE]
                                     [--reps 40] [--pairs 20] [--target 0.20]

Question answered: how many on-policy responses per prompt (and how many
validation prompts) the split-half reference would need before the registered
gate ``floor >= 2k/n`` can pass, and which side of the reference (candidate
rollouts or validation direction) limits it now.

Inputs per run directory: ``oracle_micro_groups.pt`` (prompt -> [G, D]
micro-group gradients, G groups of ``micro_group`` responses each),
``val_groups.pt`` ([V, D], one gradient per validation prompt),
``run_config.json``, ``prompts.json`` when present, ``scores_splithalf.json``
when present (the stored A/B scores are recomputed from the artifacts and must
match before any derived number is trusted), and optionally
``rollouts_behavior_train.jsonl`` for the behavior reward profile.

Method. The registered reference scores a prompt with two disjoint candidate
groups (8 responses) per half and 25 validation prompts per half. Here both
halves are re-drawn from disjoint groups at every observable half size, in
three modes:

* ``both``        independent candidate groups and independent validation
                  halves (the registered construction);
* ``candidate``   independent candidate groups, one shared validation direction;
* ``validation``  one shared candidate mean, independent validation halves.

The split-half Pearson correlation on each axis is extrapolated with
Spearman-Brown, coupled by one constant fitted on the observed cells with the
largest cell held out, and converted to an expected top-k overlap through the
frozen Gaussian lookup of ``src/measurement_ceiling.py``. The prediction is
reported as supported only when the held-out cell reproduces, the registered
cell carries signal, the stored scores reproduce, and exact ties do not
dominate the selection boundary; otherwise the KEY line says so and makes no
budget claim.

This is a sizing diagnostic for a new reference budget. It reuses the locked
rollouts descriptively, defines no registered label, and changes nothing on
disk except the requested output file (plan section 7; section 9 row 1:
"increase reference reliability or redesign the pool"). Point estimates here
do not certify the paper's one-sided confidence-bound and positive-gain gates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from measurement_ceiling import CORRELATIONS, REGISTERED_CURVES, _interpolate
from select_rules import topk_count

MODES = ("both", "candidate", "validation")
CANDIDATE_HALF_RESPONSES = (8, 16, 32, 64, 128, 256)
VALIDATION_HALF_PROMPTS = (25, 50, 100)
FALLBACK_SIMULATIONS = 24
THREAD_CAP = 8
SELF_CHECK_TOLERANCE = 0.06      # predicted minus observed overlap at the held-out cell
MIN_REGISTERED_CORRELATION = 0.02
MAX_BOUNDARY_TIE_FRACTION = 0.05  # prompts tied with the k-th score, as a fraction of n
STORED_SCORE_TOLERANCE = 1e-3
MIN_AXIS_GAP = 0.05               # correlation gap needed to name a limiting side


@dataclass(frozen=True)
class FloorStat:
    mean: float
    low: float
    high: float
    correlation: float
    boundary_ties: float = 0.0    # mean number of prompts tied with the k-th score (both halves averaged)


@dataclass
class RunArtifacts:
    run: Path
    label: str
    dataset: str
    stack: torch.Tensor          # [P, G, D]
    val_groups: torch.Tensor     # [V, D]
    group_size: int              # responses per micro-group
    config: dict
    behavior_pass_rate: dict[int, float] | None
    scoring: str                 # "current" or "pinned"
    prompt_ids: list[int] = field(default_factory=list)
    stored_score_max_diff: float | None = None   # recomputed vs stored A/B scores, None if not checkable
    zero_norm_prompts: int = 0                   # prompts whose mean stored gradient has zero norm
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- io
def _load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def locate_artifacts(run: Path) -> tuple[Path, str]:
    """Return (directory holding the artifacts, scoring) preferring the run root.

    A point whose scoring was parked by scripts/rescore_math500.sh keeps the
    original artifacts under pinned-scoring/<stamp>/ until a GPU worker rebuilds
    the root copies; the geometry is identical, so either copy answers the
    budget question.
    """
    if (run / "oracle_micro_groups.pt").is_file() and (run / "val_groups.pt").is_file():
        return run, "current"
    parked = sorted(p for p in run.glob("pinned-scoring/*/") if p.is_dir())
    for stamp in reversed(parked):
        if (stamp / "oracle_micro_groups.pt").is_file() and (stamp / "val_groups.pt").is_file():
            return stamp, "pinned"
    raise FileNotFoundError(
        f"{run}: oracle_micro_groups.pt and val_groups.pt are missing (root and pinned-scoring)"
    )


def behavior_pass_rates(path: Path) -> dict[int, float]:
    """Behavior pass rate per prompt from the stored behavior rollouts.

    Every row is parsed as JSON (no field regexes): a regex could match text
    embedded in a string field if the row schema ever gains one, and a row
    with a missing or non-binary reward is an error rather than a skip.
    """
    rewards: dict[int, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                idx, value = row["prompt_idx"], float(row["reward"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{number}: invalid reward record") from exc
            if type(idx) is not int or idx < 0 or not math.isfinite(value):
                raise ValueError(f"{path}:{number}: invalid prompt identity or reward")
            rewards.setdefault(idx, []).append(value)
    return {idx: sum(values) / len(values) for idx, values in rewards.items() if values}


def _three_way(count: int) -> tuple[int, int, int]:
    """Registered R/A/B index blocks: proportions 1/2, 1/4, 1/4 in stored order."""
    if count < 8 or count % 4:
        raise ValueError(f"R/A/B partition needs a multiple of four with at least eight items, got {count}")
    return count // 2, count // 4, count // 4


def registered_half_scores(stack: torch.Tensor, val_groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A/B scores exactly as src/experiment.py stores them (fixed blocks, no resampling)."""
    groups = stack.shape[1]
    rank_g, a_g, _ = _three_way(groups)
    rank_v, a_v, _ = _three_way(val_groups.shape[0])
    half_a = stack[:, rank_g : rank_g + a_g].mean(dim=1)
    half_b = stack[:, rank_g + a_g :].mean(dim=1)
    direction_a = val_groups[rank_v : rank_v + a_v].mean(dim=0)
    direction_b = val_groups[rank_v + a_v :].mean(dim=0)
    return _cosine_rows(half_a, direction_a), _cosine_rows(half_b, direction_b)


def verify_stored_scores(folder: Path, stack: torch.Tensor, val_groups: torch.Tensor,
                         prompt_ids: list[int]) -> float | None:
    """Max |recomputed - stored| over the A/B scores in scores_splithalf.json, or None when absent."""
    path = folder / "scores_splithalf.json"
    if not path.is_file():
        return None
    stored = json.loads(path.read_text(encoding="utf-8"))
    try:
        score_a, score_b = registered_half_scores(stack, val_groups)
    except ValueError:
        return float("inf")
    worst = 0.0
    for position, prompt in enumerate(prompt_ids):
        halves = stored.get(str(prompt))
        if not isinstance(halves, dict) or "a" not in halves or "b" not in halves:
            return float("inf")
        try:
            stored_a, stored_b = float(halves["a"]), float(halves["b"])
        except (TypeError, ValueError):
            return float("inf")
        if not (math.isfinite(stored_a) and math.isfinite(stored_b)):
            return float("inf")
        worst = max(worst, abs(stored_a - float(score_a[position])), abs(stored_b - float(score_b[position])))
    return worst


def load_run(run: Path, label: str | None = None) -> RunArtifacts:
    folder, scoring = locate_artifacts(run)
    micro = _load_tensor(folder / "oracle_micro_groups.pt")
    if not isinstance(micro, dict) or not micro:
        raise ValueError(f"{folder}/oracle_micro_groups.pt: expected a non-empty prompt -> tensor mapping")
    prompt_ids = sorted(int(key) for key in micro)
    shapes = {tuple(micro[key].shape) for key in micro}
    if len(shapes) != 1:
        common = max(shapes, key=lambda s: sum(1 for key in micro if tuple(micro[key].shape) == s))
        offenders = sorted((int(key), tuple(micro[key].shape)) for key in micro if tuple(micro[key].shape) != common)[:5]
        raise ValueError(
            f"{folder}/oracle_micro_groups.pt: prompts differ in stored geometry, e.g. {offenders}; "
            "incomplete point, not analysed"
        )
    stack = torch.stack([micro[key].float() for key in sorted(micro, key=int)])
    if stack.ndim != 3:
        raise ValueError(f"{folder}/oracle_micro_groups.pt: expected [G, D] per prompt, got {tuple(stack.shape[1:])}")
    val_groups = _load_tensor(folder / "val_groups.pt").float()
    if val_groups.ndim != 2 or val_groups.shape[0] < 4:
        raise ValueError(f"{folder}/val_groups.pt: need a [V, D] tensor with at least four validation prompts")
    if not bool(torch.isfinite(stack).all()) or not bool(torch.isfinite(val_groups).all()):
        raise ValueError(f"{folder}: non-finite gradient values")
    config = {}
    config_path = run / "run_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    groups = stack.shape[1]
    fresh_k = int(config.get("fresh_k", 0) or 0)
    group_size = int(config.get("micro_group", 0) or 0)
    if fresh_k and group_size:
        expected_groups = fresh_k // group_size
        if expected_groups != groups:
            raise ValueError(
                f"{folder}: stored {groups} micro-groups per prompt but run_config says "
                f"fresh_k={fresh_k} / micro_group={group_size} = {expected_groups}"
            )
    elif group_size <= 0:
        group_size = fresh_k // groups if fresh_k and fresh_k % groups == 0 else 4
    expected_prompts = None
    prompts_path = run / "prompts.json"
    if prompts_path.is_file():
        prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
        expected_prompts = len(prompts.get("train", []))
        expected_val = len(prompts.get("val", []))
        if expected_val and expected_val != val_groups.shape[0]:
            raise ValueError(f"{folder}: {val_groups.shape[0]} validation gradients but prompts.json lists {expected_val}")
    else:
        if config.get("n_train"):
            expected_prompts = int(config["n_train"])
        if config.get("n_val") and int(config["n_val"]) != val_groups.shape[0]:
            raise ValueError(
                f"{folder}: {val_groups.shape[0]} validation gradients but run_config says n_val={int(config['n_val'])}"
            )
    if expected_prompts is not None and prompt_ids != list(range(expected_prompts)):
        missing = sorted(set(range(expected_prompts)) - set(prompt_ids))[:5]
        extra = sorted(set(prompt_ids) - set(range(expected_prompts)))[:5]
        raise ValueError(
            f"{folder}: prompt coverage mismatch, expected 0..{expected_prompts - 1}; "
            f"missing={missing} extra={extra}; incomplete point, not analysed"
        )
    dataset = str(config.get("dataset", run.name))
    pass_rates = None
    behavior = run / "rollouts_behavior_train.jsonl"
    if behavior.is_file():
        try:
            pass_rates = behavior_pass_rates(behavior)
        except (OSError, ValueError, KeyError):
            pass_rates = None
    zero_norm = int((stack.mean(dim=1).norm(dim=1) == 0).sum())
    notes = []
    stored_diff = verify_stored_scores(folder, stack, val_groups, prompt_ids)
    if stored_diff is None:
        notes.append("scores_splithalf.json absent; stored A/B scores not checked")
    elif stored_diff > STORED_SCORE_TOLERANCE:
        notes.append(f"stored A/B scores NOT reproduced (max diff {stored_diff:.4g}); geometry assumption invalid")
    return RunArtifacts(
        run=run, label=label or run.name, dataset=dataset, stack=stack, val_groups=val_groups,
        group_size=group_size, config=config, behavior_pass_rate=pass_rates, scoring=scoring,
        prompt_ids=prompt_ids, stored_score_max_diff=stored_diff, zero_norm_prompts=zero_norm, notes=notes,
    )


# ------------------------------------------------------------------- statistics
def _cosine_rows(matrix: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    direction = direction / direction.norm().clamp_min(1e-12)
    return (matrix @ direction) / matrix.norm(dim=1).clamp_min(1e-12)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    return float((a @ b) / denominator) if denominator > 0 else 0.0


def half_scores(
    stack: torch.Tensor,
    val_groups: torch.Tensor,
    *,
    groups_per_half: int,
    val_prompts_per_half: int,
    mode: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw two disjoint reference halves and return their per-prompt scores."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    prompts, groups, _ = stack.shape
    if groups_per_half < 1 or 2 * groups_per_half > groups:
        raise ValueError(f"groups_per_half={groups_per_half} needs 2*m <= {groups} stored groups")
    val_count = val_groups.shape[0]
    if val_prompts_per_half < 1 or 2 * val_prompts_per_half > val_count:
        raise ValueError(f"val_prompts_per_half={val_prompts_per_half} needs 2*m <= {val_count} validation prompts")
    if mode == "validation":
        candidate_a = candidate_b = stack.mean(dim=1)
    else:
        order = torch.argsort(torch.rand(prompts, groups, generator=generator), dim=1)
        picked = order[:, : 2 * groups_per_half]
        index = picked.unsqueeze(-1).expand(-1, -1, stack.shape[2])
        chosen = torch.gather(stack, 1, index)
        candidate_a = chosen[:, :groups_per_half].mean(dim=1)
        candidate_b = chosen[:, groups_per_half:].mean(dim=1)
    if mode == "candidate":
        direction_a = direction_b = val_groups.mean(dim=0)
    else:
        permutation = torch.randperm(val_count, generator=generator)
        direction_a = val_groups[permutation[:val_prompts_per_half]].mean(dim=0)
        direction_b = val_groups[permutation[val_prompts_per_half : 2 * val_prompts_per_half]].mean(dim=0)
    return _cosine_rows(candidate_a, direction_a), _cosine_rows(candidate_b, direction_b)


def _topk_masks(scores: torch.Tensor, k: int, pairs: int, generator: torch.Generator) -> torch.Tensor:
    """Top-k membership masks for `pairs` independent tie streams.

    Score order is primary; a random permutation is applied first and the sort
    is stable, so only exact ties are broken at random. No jitter touches the
    scores, so distinct values keep their order at any scale.
    """
    n = scores.numel()
    values = scores.to(torch.float64)
    permutations = torch.argsort(torch.rand(pairs, n, generator=generator), dim=1)
    permuted = values[permutations]
    order = torch.argsort(permuted, dim=1, descending=True, stable=True)
    top = torch.gather(permutations, 1, order[:, :k])
    masks = torch.zeros(pairs, n, dtype=torch.bool)
    masks.scatter_(1, top, True)
    return masks


def topk_overlap_batch(
    score_a: torch.Tensor,
    score_b: torch.Tensor,
    k: int,
    *,
    pairs: int,
    generator: torch.Generator,
) -> float:
    """Mean top-k overlap over `pairs` independent tie-breaking draws per side.

    Same estimand as select_rules.overlap_under_independent_ties (score first,
    random order only within exact ties, independent streams per side), batched.
    """
    if score_a.numel() != score_b.numel():
        raise ValueError("score vectors must have the same length")
    n = score_a.numel()
    if not 0 < k <= n:
        raise ValueError(f"k={k} must lie in (0, {n}]")
    if pairs < 1:
        raise ValueError(f"pairs must be positive, got {pairs}")
    mask_a = _topk_masks(score_a, k, pairs, generator)
    mask_b = _topk_masks(score_b, k, pairs, generator)
    return float((mask_a & mask_b).sum(dim=1).double().mean() / k)


def boundary_tie_count(scores: torch.Tensor, k: int) -> int:
    """Number of prompts whose score equals the k-th largest score exactly."""
    values = scores.to(torch.float64)
    kth = torch.topk(values, k).values[-1]
    return int((values == kth).sum())


def floor_statistic(
    stack: torch.Tensor,
    val_groups: torch.Tensor,
    *,
    k: int,
    groups_per_half: int,
    val_prompts_per_half: int,
    mode: str,
    reps: int,
    pairs: int,
    seed: int,
    prompt_positions: list[int] | None = None,
) -> FloorStat:
    """Split-half top-k overlap and Pearson correlation over resampled halves."""
    index = None if prompt_positions is None else torch.tensor(list(prompt_positions), dtype=torch.long)
    floors, correlations, ties = [], [], []
    for rep in range(reps):
        generator = torch.Generator().manual_seed(seed + 1_000_003 * rep)
        score_a, score_b = half_scores(
            stack, val_groups,
            groups_per_half=groups_per_half, val_prompts_per_half=val_prompts_per_half,
            mode=mode, generator=generator,
        )
        if index is not None:
            score_a, score_b = score_a[index], score_b[index]
        tie_generator = torch.Generator().manual_seed(seed + 17 + rep)
        floors.append(topk_overlap_batch(score_a, score_b, k, pairs=pairs, generator=tie_generator))
        correlations.append(_pearson(score_a, score_b))
        ties.append((boundary_tie_count(score_a, k) + boundary_tie_count(score_b, k)) / 2.0)
    return FloorStat(
        mean=sum(floors) / len(floors), low=min(floors), high=max(floors),
        correlation=sum(correlations) / len(correlations), boundary_ties=sum(ties) / len(ties),
    )


def spearman_brown(r_unit: float, factor: float) -> float:
    """Reliability of a measure `factor` times longer than the unit measure."""
    if r_unit <= 0.0:
        return 0.0
    if r_unit >= 1.0:
        return 1.0
    return factor * r_unit / (1.0 + (factor - 1.0) * r_unit)


def unit_correlation(r_observed: float, factor: float) -> float:
    """Invert Spearman-Brown: the unit-length reliability implied by r at `factor` units."""
    if r_observed <= 0.0:
        return 0.0
    if r_observed >= 1.0:
        return 1.0
    return r_observed / (factor - (factor - 1.0) * r_observed)


def _simulated_overlap(rho: float, n: int, k: int, sims: int = FALLBACK_SIMULATIONS) -> float:
    rho = max(0.0, min(0.999, rho))
    values = []
    for sim in range(sims):
        generator = torch.Generator().manual_seed(4_242 + sim)
        latent = torch.randn(n, generator=generator)
        noise_a = torch.randn(n, generator=generator)
        noise_b = torch.randn(n, generator=generator)
        weight, residual = math.sqrt(rho), math.sqrt(1.0 - rho)
        values.append(topk_overlap_batch(
            weight * latent + residual * noise_a, weight * latent + residual * noise_b, k,
            pairs=3, generator=torch.Generator().manual_seed(9_000 + sim),
        ))
    return sum(values) / len(values)


def overlap_from_correlation(rho: float, n: int, k: int) -> float:
    """Expected split-half top-k overlap of two Gaussian halves with correlation rho."""
    rho = max(0.0, min(1.0, rho))
    curve = REGISTERED_CURVES.get((n, k))
    if curve is not None:
        return _interpolate(rho, CORRELATIONS, curve)
    return _simulated_overlap(rho, n, k)


def correlation_from_overlap(overlap: float, n: int, k: int) -> float:
    """Between-half correlation implied by an observed overlap (registered designs only)."""
    curve = REGISTERED_CURVES.get((n, k))
    chance = k / n
    bounded = min(1.0, max(chance, overlap))
    if curve is not None:
        return _interpolate(bounded, curve, CORRELATIONS)
    low, high = 0.0, 0.999
    for _ in range(40):
        mid = (low + high) / 2.0
        if _simulated_overlap(mid, n, k, sims=8) < bounded:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


# -------------------------------------------------------------------- analysis
@dataclass
class RunReadout:
    artifacts: RunArtifacts
    n: int
    k: int
    chance: float
    groups: int
    val_count: int
    curves: dict[str, dict[tuple[int, int], FloorStat]]   # mode -> (groups_per_half, val_half) -> stat
    registered: FloorStat | None
    r1_candidate: float
    r1_validation: float
    budget_table: dict[tuple[int, int], float]           # (responses per half, val prompts per half) -> predicted floor
    needed_candidate: int | None
    needed_candidate_margin: int | None
    self_check: tuple[float, float] | None                # predicted vs observed at the held-out largest cell
    strata: dict[str, object] | None
    coupling: float | None = None                          # fitted coupling constant of the product model
    supported: bool = False                                # whether the budget prediction may be acted on
    unsupported_reasons: list[str] = field(default_factory=list)
    limiting_side: str | None = None                       # "candidate rollouts", "validation direction", or None


def _registered_geometry(groups: int, group_size: int, val_count: int) -> tuple[int, int] | None:
    """Registered halves: 8 responses per candidate half and V/4 validation prompts."""
    if 8 % group_size or 8 // group_size < 1:
        return None
    groups_per_half = 8 // group_size
    val_half = val_count // 4
    if 2 * groups_per_half > groups or val_half < 1:
        return None
    return groups_per_half, val_half


def analyze_run(
    artifacts: RunArtifacts,
    *,
    reps: int = 40,
    pairs: int = 20,
    target: float = 0.20,
    margin_target: float | None = 0.25,
    seed: int = 0,
    topk_frac: float = 0.10,
) -> RunReadout:
    stack, val_groups = artifacts.stack, artifacts.val_groups
    n, groups, _ = stack.shape
    val_count = val_groups.shape[0]
    k = topk_count(n, topk_frac)
    chance = k / n
    max_groups_per_half = groups // 2
    val_half_options = [v for v in sorted({val_count // 4, val_count // 2}) if v >= 2]

    curves: dict[str, dict[tuple[int, int], FloorStat]] = {mode: {} for mode in MODES}
    for mode in MODES:
        for groups_per_half in range(1, max_groups_per_half + 1):
            if mode == "validation" and groups_per_half != max_groups_per_half:
                continue
            for val_half in val_half_options:
                if mode == "candidate" and val_half != val_half_options[0]:
                    continue
                curves[mode][(groups_per_half, val_half)] = floor_statistic(
                    stack, val_groups, k=k,
                    groups_per_half=groups_per_half, val_prompts_per_half=val_half,
                    mode=mode, reps=reps, pairs=pairs,
                    seed=seed + 7_919 * groups_per_half + 104_729 * val_half + 131 * MODES.index(mode),
                )

    registered_geometry = _registered_geometry(groups, artifacts.group_size, val_count)
    registered = None
    if registered_geometry is not None:
        registered = curves["both"].get(registered_geometry)
        if registered is None:
            registered = floor_statistic(
                stack, val_groups, k=k,
                groups_per_half=registered_geometry[0], val_prompts_per_half=registered_geometry[1],
                mode="both", reps=reps, pairs=pairs, seed=seed + 31,
            )

    candidate_units = [unit_correlation(stat.correlation, g) for (g, _), stat in curves["candidate"].items()]
    r1_candidate = statistics.fmean(candidate_units) if candidate_units else 0.0
    validation_units = [unit_correlation(stat.correlation, v) for (_, v), stat in curves["validation"].items()]
    r1_validation = statistics.fmean(validation_units) if validation_units else 0.0

    # Combined prediction: coupling constant times the Spearman-Brown reliability
    # of each axis, fitted through the origin on every observed registered-
    # construction cell except the largest, which is held out as a self-check.
    budget_table: dict[tuple[int, int], float] = {}
    needed_candidate = needed_candidate_margin = None
    self_check = None
    coupling = None
    if registered is not None and registered_geometry is not None and curves["both"]:
        _, reg_val = registered_geometry

        def axis_product(groups_per_half: float, val_prompts_per_half: float) -> float:
            return spearman_brown(r1_candidate, groups_per_half) * spearman_brown(r1_validation, val_prompts_per_half)

        cells = sorted(curves["both"], key=lambda key: (key[0], key[1]))
        held_out = cells[-1] if len(cells) >= 2 else None
        fit_cells = [cell for cell in cells if cell != held_out]
        numerator = sum(axis_product(*cell) * curves["both"][cell].correlation for cell in fit_cells)
        denominator = sum(axis_product(*cell) ** 2 for cell in fit_cells)
        coupling = max(0.0, numerator / denominator) if denominator > 0 else 0.0

        def predicted_rho(responses_per_half: int, val_prompts_per_half: int) -> float:
            return min(1.0, coupling * axis_product(responses_per_half / artifacts.group_size, val_prompts_per_half))

        for responses in CANDIDATE_HALF_RESPONSES:
            for val_half in sorted(set(VALIDATION_HALF_PROMPTS) | {reg_val}):
                budget_table[(responses, val_half)] = overlap_from_correlation(predicted_rho(responses, val_half), n, k)
        for responses in CANDIDATE_HALF_RESPONSES:
            predicted = budget_table[(responses, reg_val)]
            if needed_candidate is None and predicted >= target:
                needed_candidate = responses
            if margin_target is not None and needed_candidate_margin is None and predicted >= margin_target:
                needed_candidate_margin = responses
        if held_out is not None:
            observed = curves["both"][held_out].mean
            predicted = overlap_from_correlation(predicted_rho(held_out[0] * artifacts.group_size, held_out[1]), n, k)
            self_check = (predicted, observed)

    # Adequacy: the prediction is actionable only when the model reproduces the
    # held-out cell, the registered cell carries signal, exact ties do not
    # dominate the boundary, and the stored scores reproduce from the artifacts.
    reasons = []
    if not budget_table:
        reasons.append("registered geometry not observable in the stored artifacts")
    if registered is not None and registered.correlation < MIN_REGISTERED_CORRELATION:
        reasons.append(f"registered-cell correlation {registered.correlation:.3f} below {MIN_REGISTERED_CORRELATION}")
    if self_check is not None and abs(self_check[0] - self_check[1]) > SELF_CHECK_TOLERANCE:
        reasons.append(f"held-out self-check off by {self_check[0] - self_check[1]:+.3f} (tolerance {SELF_CHECK_TOLERANCE})")
    if registered is not None and registered.boundary_ties > MAX_BOUNDARY_TIE_FRACTION * n:
        reasons.append(f"{registered.boundary_ties:.0f} prompts tied at the selection boundary (max {MAX_BOUNDARY_TIE_FRACTION * n:.0f})")
    if artifacts.stored_score_max_diff is None:
        reasons.append("stored A/B scores absent (scores_splithalf.json); recomputation not verifiable")
    elif artifacts.stored_score_max_diff > STORED_SCORE_TOLERANCE:
        reasons.append("stored A/B scores not reproduced from the artifacts")
    if artifacts.zero_norm_prompts:
        reasons.append(f"{artifacts.zero_norm_prompts} prompts with a zero-norm stored gradient")
    supported = not reasons
    if not supported:
        needed_candidate = needed_candidate_margin = None

    limiting_side = None
    if supported and curves["candidate"] and curves["validation"]:
        best_candidate = max(stat.correlation for stat in curves["candidate"].values())
        best_validation = max(stat.correlation for stat in curves["validation"].values())
        if best_validation - best_candidate >= MIN_AXIS_GAP:
            limiting_side = "candidate rollouts"
        elif best_candidate - best_validation >= MIN_AXIS_GAP:
            limiting_side = "validation direction"

    strata = None
    if artifacts.behavior_pass_rate and registered_geometry is not None:
        rates = artifacts.behavior_pass_rate
        ids = artifacts.prompt_ids or list(range(n))
        mixed_positions = [pos for pos, prompt in enumerate(ids) if 0.0 < rates.get(prompt, 0.0) < 1.0]
        all_wrong = sum(1 for prompt in ids if rates.get(prompt, 0.0) == 0.0)
        all_right = sum(1 for prompt in ids if rates.get(prompt, 0.0) == 1.0)
        strata = {"mixed": len(mixed_positions), "all_wrong": all_wrong, "all_right": all_right,
                  "mixed_floor": None, "mixed_k": None, "mixed_chance": None}
        if len(mixed_positions) >= 20:
            k_mixed = topk_count(len(mixed_positions), topk_frac)
            stat = floor_statistic(
                stack, val_groups, k=k_mixed,
                groups_per_half=registered_geometry[0], val_prompts_per_half=registered_geometry[1],
                mode="both", reps=reps, pairs=pairs, seed=seed + 977, prompt_positions=mixed_positions,
            )
            strata.update(mixed_floor=stat, mixed_k=k_mixed, mixed_chance=k_mixed / len(mixed_positions))

    return RunReadout(
        artifacts=artifacts, n=n, k=k, chance=chance, groups=groups, val_count=val_count,
        curves=curves, registered=registered, r1_candidate=r1_candidate, r1_validation=r1_validation,
        budget_table=budget_table, needed_candidate=needed_candidate,
        needed_candidate_margin=needed_candidate_margin, self_check=self_check, strata=strata,
        coupling=coupling, supported=supported, unsupported_reasons=reasons, limiting_side=limiting_side,
    )


# --------------------------------------------------------------------- report
def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def render_run(readout: RunReadout, target: float) -> list[str]:
    art = readout.artifacts
    gsize = art.group_size
    lines = [
        f"## {art.label}  dataset={art.dataset}  scoring={art.scoring}  "
        f"n={readout.n} k={readout.k} chance={readout.chance:.3f}  "
        f"stored groups={readout.groups}x{gsize} responses  validation prompts={readout.val_count}",
        "",
    ]
    if art.stored_score_max_diff is None:
        lines.append("stored A/B scores: not checkable (scores_splithalf.json absent)")
    else:
        verdict = "reproduced" if art.stored_score_max_diff <= STORED_SCORE_TOLERANCE else "NOT reproduced"
        lines.append(f"stored A/B scores: {verdict} from the artifacts (max |diff| {art.stored_score_max_diff:.2e})")
    lines.append(f"zero-norm stored gradients: {art.zero_norm_prompts} of {readout.n} prompts")
    if readout.registered is not None:
        reg = readout.registered
        lines.append(
            f"registered geometry (8 responses and {readout.val_count // 4} validation prompts per half): "
            f"floor={reg.mean:.3f} [{reg.low:.3f}~{reg.high:.3f}]  r={reg.correlation:.3f}  "
            f"boundary ties={reg.boundary_ties:.1f}  "
            f"{'GATE-OK' if reg.mean >= target else 'gate-LOW'} (point estimate vs {target:.2f}; "
            "the paper gate uses a one-sided 95% lower bound and a positive-gain condition)"
        )
    lines.append("")
    lines.append("observed floor by half size (both = registered construction; candidate = shared validation direction; validation = shared candidate mean)")
    lines.append("| responses/half | val prompts/half | both floor | both r | candidate floor | candidate r | validation floor | validation r |")
    lines.append("|---|---|---|---|---|---|---|---|")
    keys = sorted(set().union(*(set(c) for c in readout.curves.values())))
    for groups_per_half, val_half in keys:
        both = readout.curves["both"].get((groups_per_half, val_half))
        cand = readout.curves["candidate"].get((groups_per_half, val_half))
        vali = readout.curves["validation"].get((groups_per_half, val_half))
        lines.append(
            f"| {groups_per_half * gsize} | {val_half} | "
            f"{_fmt(both.mean if both else None)} | {_fmt(both.correlation if both else None)} | "
            f"{_fmt(cand.mean if cand else None)} | {_fmt(cand.correlation if cand else None)} | "
            f"{_fmt(vali.mean if vali else None)} | {_fmt(vali.correlation if vali else None)} |"
        )
    lines.append("")
    lines.append(
        f"unit reliabilities (Spearman-Brown inverted): one micro-group of {gsize} responses r1={readout.r1_candidate:.3f}; "
        f"one validation prompt r1={readout.r1_validation:.4f}"
    )
    if readout.budget_table:
        lines.append("")
        status = "supported" if readout.supported else "UNSUPPORTED, shown for inspection only"
        lines.append(
            f"predicted floor (product model, coupling={_fmt(readout.coupling)} fitted on the observed cells; "
            f"GATE mark = point estimate >= {target:.2f}; prediction {status})"
        )
        columns = sorted({val_half for _, val_half in readout.budget_table})
        lines.append("| responses/half | " + " | ".join(f"val {v}/half" for v in columns) + " |")
        lines.append("|---|" + "---|" * len(columns))
        for responses in CANDIDATE_HALF_RESPONSES:
            cells = []
            for val_half in columns:
                value = readout.budget_table[(responses, val_half)]
                cells.append(f"{value:.3f}{' GATE' if value >= target else ''}")
            lines.append(f"| {responses} | " + " | ".join(cells) + " |")
        lines.append(
            f"note: val {VALIDATION_HALF_PROMPTS[-1]}/half needs {2 * VALIDATION_HALF_PROMPTS[-1]} validation prompts "
            f"(new prompts; this run has {readout.val_count}); responses per validation prompt are not varied here"
        )
        if readout.self_check is not None:
            predicted, observed = readout.self_check
            lines.append(
                f"self-check on the held-out largest cell: predicted {predicted:.3f} vs observed {observed:.3f} "
                f"(difference {predicted - observed:+.3f}; tolerance {SELF_CHECK_TOLERANCE})"
            )
    for note in art.notes:
        lines.append(f"note: {note}")
    reg_val = readout.val_count // 4
    if not readout.supported:
        lines.append(f"KEY {art.label}: prediction unsupported, no budget claim; " + "; ".join(readout.unsupported_reasons))
    elif readout.needed_candidate is None:
        lines.append(
            f"KEY {art.label}: no candidate half size up to {CANDIDATE_HALF_RESPONSES[-1]} responses reaches "
            f"floor {target:.2f} with {reg_val} validation prompts per half"
        )
    else:
        margin = (
            f"; >= 0.25 (margin for the lower bound) at {readout.needed_candidate_margin} per half"
            if readout.needed_candidate_margin is not None
            else f"; 0.25 not reached up to {CANDIDATE_HALF_RESPONSES[-1]}"
        )
        lines.append(
            f"KEY {art.label}: floor >= {target:.2f} expected at {readout.needed_candidate} responses per half "
            f"({2 * readout.needed_candidate} per prompt for A+B, {3 * readout.needed_candidate} with an equal ranking split) "
            f"at {reg_val} validation prompts per half{margin}"
        )
    if readout.supported and readout.curves["candidate"] and readout.curves["validation"]:
        best_candidate = max(stat.correlation for stat in readout.curves["candidate"].values())
        best_validation = max(stat.correlation for stat in readout.curves["validation"].values())
        side = readout.limiting_side or "no clear limiting side"
        lines.append(
            f"KEY {art.label}: limiting side now = {side} "
            f"(best candidate-only r={best_candidate:.3f}, best validation-only r={best_validation:.3f})"
        )
    if readout.strata is not None:
        s = readout.strata
        lines.append("")
        lines.append(f"behavior reward profile: mixed={s['mixed']}  all-wrong={s['all_wrong']}  all-right={s['all_right']}")
        if s["mixed_floor"] is not None:
            stat = s["mixed_floor"]
            lines.append(
                f"mixed-reward stratum floor at the registered geometry: {stat.mean:.3f} "
                f"[{stat.low:.3f}~{stat.high:.3f}] (k={s['mixed_k']}, chance={s['mixed_chance']:.3f}, r={stat.correlation:.3f})"
            )
        else:
            lines.append("mixed-reward stratum has fewer than 20 prompts; no stratum floor")
    lines.append("")
    return lines


def render_report(readouts: list[RunReadout], target: float, reps: int, pairs: int) -> str:
    lines = [
        "# Reliability versus reference budget (descriptive sizing; defines no registered label)",
        "",
        f"resamples per cell: {reps}; tie-stream pairs: {pairs}; gate target {target:.2f} = 2k/n for the registered designs",
        "floor = split-half top-k overlap of two disjoint reference halves; r = Pearson correlation of the two halves' scores",
        "",
    ]
    for readout in readouts:
        lines.extend(render_run(readout, target))
    by_dataset: dict[str, list[RunReadout]] = {}
    for readout in readouts:
        by_dataset.setdefault(readout.artifacts.dataset, []).append(readout)
    lines.append("## summary per dataset")
    lines.append("")
    for dataset, group in sorted(by_dataset.items()):
        registered = [r.registered.mean for r in group if r.registered is not None]
        needed = [r.needed_candidate for r in group if r.supported and r.needed_candidate is not None]
        unsupported = sum(1 for r in group if not r.supported)
        not_reached = sum(1 for r in group if r.supported and r.needed_candidate is None)
        lines.append(
            f"KEY {dataset}: runs={len(group)}  registered floor mean={_fmt(statistics.fmean(registered) if registered else None)}"
            f"  responses/half needed (median of supported runs that reach {target:.2f}) = "
            f"{int(statistics.median(needed)) if needed else 'none'}"
            f"  unsupported predictions: {unsupported}  supported but not reaching within {CANDIDATE_HALF_RESPONSES[-1]}: {not_reached}"
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ cli
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="point directories with oracle_micro_groups.pt and val_groups.pt")
    parser.add_argument("--out", type=Path, default=None, help="write the report here as well as to stdout")
    parser.add_argument("--reps", type=int, default=40, help="resampled half pairs per cell")
    parser.add_argument("--pairs", type=int, default=20, help="independent tie-stream pairs per overlap")
    parser.add_argument("--target", type=float, default=0.20, help="floor the gate requires (2k/n = 0.20 for the registered designs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", action="append", default=None, help="label per run, in order (default: directory name)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reps < 1 or args.pairs < 1:
        print("[abort] --reps and --pairs must be positive", file=sys.stderr)
        return 2
    labels = args.label or []
    torch.set_num_threads(max(1, min(THREAD_CAP, os.cpu_count() or 1)))
    readouts = []
    for position, run in enumerate(args.runs):
        label = labels[position] if position < len(labels) else None
        t_load = time.time()
        try:
            artifacts = load_run(run, label)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[skip] {run}: {exc}", file=sys.stderr)
            continue
        print(f"[reliability-budget] {artifacts.label}: n={artifacts.stack.shape[0]} groups={artifacts.stack.shape[1]} "
              f"val={artifacts.val_groups.shape[0]} scoring={artifacts.scoring} loaded in {time.time() - t_load:.0f}s; analysing ...", flush=True)
        t_analyse = time.time()
        try:
            readouts.append(analyze_run(artifacts, reps=args.reps, pairs=args.pairs, target=args.target, seed=args.seed))
        except (ValueError, KeyError, RuntimeError) as exc:
            print(f"[skip] {run}: analysis failed: {exc!r}", file=sys.stderr)
            continue
        print(f"[reliability-budget] {artifacts.label}: done in {time.time() - t_analyse:.0f}s", flush=True)
    if not readouts:
        print("[abort] no run directory could be read", file=sys.stderr)
        return 1
    report = render_report(readouts, args.target, args.reps, args.pairs)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(args.out.name + ".tmp")
        temporary.write_text(report, encoding="utf-8")
        temporary.replace(args.out)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
