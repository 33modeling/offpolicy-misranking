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
``run_config.json``, and optionally ``rollouts_behavior_train.jsonl`` for the
behavior reward profile of every prompt.

Method. The registered reference scores a prompt with two disjoint candidate
groups (8 responses) per half and 25 validation prompts per half. Here both
halves are re-drawn from disjoint groups at every observable half size, in
three modes:

* ``both``        independent candidate groups and independent validation
                  halves (the registered construction);
* ``candidate``   independent candidate groups, one shared validation direction;
* ``validation``  one shared candidate mean, independent validation halves.

The split-half Pearson correlation on each axis is extrapolated with
Spearman-Brown and converted to an expected top-k overlap through the frozen
Gaussian lookup of ``src/measurement_ceiling.py`` (registered (n, k) designs;
other designs fall back to a small bivariate-normal simulation).

This is a sizing diagnostic for a new reference budget. It reuses the locked
rollouts descriptively, defines no registered label, and changes nothing on
disk except the requested output file (plan section 7; section 9 row 1:
"increase reference reliability or redesign the pool").
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from measurement_ceiling import CORRELATIONS, REGISTERED_CURVES, _interpolate
from select_rules import overlap_under_independent_ties, topk_count

MODES = ("both", "candidate", "validation")
CANDIDATE_HALF_RESPONSES = (8, 16, 32, 64, 128, 256)
VALIDATION_HALF_PROMPTS = (25, 50, 100)
FALLBACK_SIMULATIONS = 24


@dataclass(frozen=True)
class FloorStat:
    mean: float
    low: float
    high: float
    correlation: float


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


# --------------------------------------------------------------------------- io
def _load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def locate_artifacts(run: Path) -> tuple[Path, Path, str]:
    """Return (micro_groups, val_groups, scoring) preferring the run root.

    A point whose scoring was parked by scripts/rescore_math500.sh keeps the
    original artifacts under pinned-scoring/<stamp>/ until a GPU worker rebuilds
    the root copies; the geometry is identical, so either copy answers the
    budget question.
    """
    root_micro = run / "oracle_micro_groups.pt"
    root_val = run / "val_groups.pt"
    if root_micro.is_file() and root_val.is_file():
        return root_micro, root_val, "current"
    parked = sorted(p for p in run.glob("pinned-scoring/*/") if p.is_dir())
    for stamp in reversed(parked):
        micro = stamp / "oracle_micro_groups.pt"
        val = stamp / "val_groups.pt"
        if micro.is_file() and val.is_file():
            return micro, val, "pinned"
    raise FileNotFoundError(
        f"{run}: oracle_micro_groups.pt and val_groups.pt are missing (root and pinned-scoring)"
    )


def behavior_pass_rates(path: Path) -> dict[int, float]:
    """Behavior pass rate per prompt from the stored behavior rollouts."""
    rewards: dict[int, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rewards.setdefault(int(row["prompt_idx"]), []).append(float(row["reward"]))
    return {idx: sum(values) / len(values) for idx, values in rewards.items() if values}


def load_run(run: Path, label: str | None = None) -> RunArtifacts:
    micro_path, val_path, scoring = locate_artifacts(run)
    micro = _load_tensor(micro_path)
    if not isinstance(micro, dict) or not micro:
        raise ValueError(f"{micro_path}: expected a non-empty prompt -> tensor mapping")
    groups = min(int(tensor.shape[0]) for tensor in micro.values())
    stack = torch.stack([micro[idx][:groups].float() for idx in sorted(micro)])
    val_groups = _load_tensor(val_path).float()
    if val_groups.ndim != 2 or val_groups.shape[0] < 4:
        raise ValueError(f"{val_path}: need a [V, D] tensor with at least four validation prompts")
    config = {}
    config_path = run / "run_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    fresh_k = int(config.get("fresh_k", 0) or 0)
    group_size = int(config.get("micro_group", 0) or 0)
    if group_size <= 0:
        group_size = fresh_k // groups if fresh_k and fresh_k % groups == 0 else 4
    dataset = str(config.get("dataset", run.name))
    pass_rates = None
    behavior = run / "rollouts_behavior_train.jsonl"
    if behavior.is_file():
        try:
            pass_rates = behavior_pass_rates(behavior)
        except (OSError, ValueError, KeyError):
            pass_rates = None
    return RunArtifacts(
        run=run,
        label=label or run.name,
        dataset=dataset,
        stack=stack,
        val_groups=val_groups,
        group_size=group_size,
        config=config,
        behavior_pass_rate=pass_rates,
        scoring=scoring,
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
        raise ValueError(
            f"groups_per_half={groups_per_half} needs 2*m <= {groups} stored groups"
        )
    val_count = val_groups.shape[0]
    if val_prompts_per_half < 1 or 2 * val_prompts_per_half > val_count:
        raise ValueError(
            f"val_prompts_per_half={val_prompts_per_half} needs 2*m <= {val_count} validation prompts"
        )
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
        direction_b = val_groups[
            permutation[val_prompts_per_half : 2 * val_prompts_per_half]
        ].mean(dim=0)
    return _cosine_rows(candidate_a, direction_a), _cosine_rows(candidate_b, direction_b)


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
    prompt_ids: list[int] | None = None,
) -> FloorStat:
    """Split-half top-k overlap and Pearson correlation over resampled halves."""
    ids = list(range(stack.shape[0])) if prompt_ids is None else list(prompt_ids)
    floors, correlations = [], []
    for rep in range(reps):
        generator = torch.Generator().manual_seed(seed + 1_000_003 * rep)
        score_a, score_b = half_scores(
            stack, val_groups,
            groups_per_half=groups_per_half,
            val_prompts_per_half=val_prompts_per_half,
            mode=mode,
            generator=generator,
        )
        left = {idx: float(score_a[idx]) for idx in ids}
        right = {idx: float(score_b[idx]) for idx in ids}
        floors.append(
            overlap_under_independent_ties(left, right, k, seed=seed + 17 + rep, pairs=pairs).mean
        )
        correlations.append(_pearson(score_a[ids], score_b[ids]))
    return FloorStat(
        mean=sum(floors) / len(floors),
        low=min(floors),
        high=max(floors),
        correlation=sum(correlations) / len(correlations),
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
        weight = math.sqrt(rho)
        residual = math.sqrt(1.0 - rho)
        a = {i: float(v) for i, v in enumerate(weight * latent + residual * noise_a)}
        b = {i: float(v) for i, v in enumerate(weight * latent + residual * noise_b)}
        values.append(overlap_under_independent_ties(a, b, k, seed=sim, pairs=3).mean)
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

    # Candidate axis: unit = one micro-group; average the implied unit reliability
    # over every observed half size (each inversion is exact under Spearman-Brown).
    candidate_units = [
        unit_correlation(stat.correlation, groups_per_half)
        for (groups_per_half, _), stat in curves["candidate"].items()
    ]
    r1_candidate = statistics.fmean(candidate_units) if candidate_units else 0.0
    # Validation axis: unit = one validation prompt.
    validation_units = [
        unit_correlation(stat.correlation, val_half)
        for (_, val_half), stat in curves["validation"].items()
    ]
    r1_validation = statistics.fmean(validation_units) if validation_units else 0.0

    # Combined prediction. Each half's score is a cosine between a candidate
    # estimate and a validation direction, so the between-half correlation is
    # modelled as a coupling constant times the Spearman-Brown reliability of
    # each axis. The constant is fitted through the origin on every observed
    # registered-construction cell except the largest, which is held out as a
    # self-check; predictions convert correlation to overlap with the frozen
    # Gaussian lookup.
    budget_table: dict[tuple[int, int], float] = {}
    needed_candidate = needed_candidate_margin = None
    self_check = None
    coupling = None
    if registered is not None and registered_geometry is not None and curves["both"]:
        _, reg_val = registered_geometry

        def axis_product(groups_per_half: int, val_prompts_per_half: int) -> float:
            return spearman_brown(r1_candidate, groups_per_half) * spearman_brown(
                r1_validation, val_prompts_per_half
            )

        cells = sorted(curves["both"], key=lambda key: (key[0], key[1]))
        held_out = cells[-1] if len(cells) >= 2 else None
        fit_cells = [cell for cell in cells if cell != held_out]
        numerator = sum(axis_product(*cell) * curves["both"][cell].correlation for cell in fit_cells)
        denominator = sum(axis_product(*cell) ** 2 for cell in fit_cells)
        coupling = numerator / denominator if denominator > 0 else 0.0
        coupling = max(0.0, coupling)

        def predicted_rho(responses_per_half: int, val_prompts_per_half: int) -> float:
            return min(1.0, coupling * axis_product(responses_per_half / artifacts.group_size, val_prompts_per_half))

        for responses in CANDIDATE_HALF_RESPONSES:
            for val_half in VALIDATION_HALF_PROMPTS:
                budget_table[(responses, val_half)] = overlap_from_correlation(
                    predicted_rho(responses, val_half), n, k
                )
        for responses in CANDIDATE_HALF_RESPONSES:
            predicted = budget_table[(responses, reg_val)]
            if needed_candidate is None and predicted >= target:
                needed_candidate = responses
            if margin_target is not None and needed_candidate_margin is None and predicted >= margin_target:
                needed_candidate_margin = responses
        if held_out is not None:
            observed = curves["both"][held_out].mean
            predicted = overlap_from_correlation(
                predicted_rho(held_out[0] * artifacts.group_size, held_out[1]), n, k
            )
            self_check = (predicted, observed)

    strata = None
    if artifacts.behavior_pass_rate and registered_geometry is not None:
        rates = artifacts.behavior_pass_rate
        mixed = [idx for idx in range(n) if 0.0 < rates.get(idx, 0.0) < 1.0]
        all_wrong = sum(1 for idx in range(n) if rates.get(idx, 0.0) == 0.0)
        all_right = sum(1 for idx in range(n) if rates.get(idx, 0.0) == 1.0)
        strata = {
            "mixed": len(mixed),
            "all_wrong": all_wrong,
            "all_right": all_right,
            "mixed_floor": None,
            "mixed_k": None,
            "mixed_chance": None,
        }
        if len(mixed) >= 20:
            k_mixed = topk_count(len(mixed), topk_frac)
            stat = floor_statistic(
                stack, val_groups, k=k_mixed,
                groups_per_half=registered_geometry[0], val_prompts_per_half=registered_geometry[1],
                mode="both", reps=reps, pairs=pairs, seed=seed + 977, prompt_ids=mixed,
            )
            strata["mixed_floor"] = stat
            strata["mixed_k"] = k_mixed
            strata["mixed_chance"] = k_mixed / len(mixed)

    return RunReadout(
        artifacts=artifacts, n=n, k=k, chance=chance, groups=groups, val_count=val_count,
        curves=curves, registered=registered, r1_candidate=r1_candidate,
        r1_validation=r1_validation, budget_table=budget_table,
        needed_candidate=needed_candidate, needed_candidate_margin=needed_candidate_margin,
        self_check=self_check, strata=strata, coupling=coupling,
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
    if readout.registered is not None:
        reg = readout.registered
        lines.append(
            f"registered geometry (8 responses and {readout.val_count // 4} validation prompts per half): "
            f"floor={reg.mean:.3f} [{reg.low:.3f}~{reg.high:.3f}]  r={reg.correlation:.3f}  "
            f"{'GATE-OK' if reg.mean >= target else 'gate-LOW'} (point estimate vs {target:.2f})"
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
        lines.append(
            f"predicted floor (product model, coupling={_fmt(readout.coupling)} fitted on the observed cells; "
            f"GATE mark = point estimate >= {target:.2f})"
        )
        header = "| responses/half | " + " | ".join(f"val {v}/half" for v in VALIDATION_HALF_PROMPTS) + " |"
        lines.append(header)
        lines.append("|---|" + "---|" * len(VALIDATION_HALF_PROMPTS))
        for responses in CANDIDATE_HALF_RESPONSES:
            cells = []
            for val_half in VALIDATION_HALF_PROMPTS:
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
                f"(difference {predicted - observed:+.3f}; a large gap means the extrapolation is unreliable)"
            )
        reg_val = readout.val_count // 4
        if readout.needed_candidate is None:
            lines.append(
                f"KEY {art.label}: no candidate half size up to {CANDIDATE_HALF_RESPONSES[-1]} responses reaches "
                f"floor {target:.2f} with {reg_val} validation prompts per half"
            )
        else:
            lines.append(
                f"KEY {art.label}: floor >= {target:.2f} expected at {readout.needed_candidate} responses per half "
                f"({2 * readout.needed_candidate} per prompt for A+B, {3 * readout.needed_candidate} with an equal ranking split) "
                f"at {reg_val} validation prompts per half"
                + (
                    f"; >= {0.25:.2f} (margin for the lower bound) at {readout.needed_candidate_margin} per half"
                    if readout.needed_candidate_margin is not None else
                    f"; 0.25 not reached up to {CANDIDATE_HALF_RESPONSES[-1]}"
                )
            )
        if readout.curves["candidate"] and readout.curves["validation"]:
            cand_reg = max(readout.curves["candidate"].values(), key=lambda s: s.correlation)
            vali_reg = max(readout.curves["validation"].values(), key=lambda s: s.correlation)
            limiting = "candidate rollouts" if cand_reg.correlation < vali_reg.correlation else "validation direction"
            lines.append(
                f"KEY {art.label}: limiting side now = {limiting} "
                f"(best candidate-only r={cand_reg.correlation:.3f}, best validation-only r={vali_reg.correlation:.3f})"
            )
    if readout.strata is not None:
        s = readout.strata
        lines.append("")
        lines.append(
            f"behavior reward profile: mixed={s['mixed']}  all-wrong={s['all_wrong']}  all-right={s['all_right']}"
        )
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
        needed = [r.needed_candidate for r in group if r.needed_candidate is not None]
        missing = sum(1 for r in group if r.budget_table and r.needed_candidate is None)
        lines.append(
            f"KEY {dataset}: runs={len(group)}  registered floor mean={_fmt(statistics.fmean(registered) if registered else None)}"
            f"  responses/half needed (median of runs that reach {target:.2f}) = "
            f"{int(statistics.median(needed)) if needed else 'none'}"
            f"  runs not reaching within {CANDIDATE_HALF_RESPONSES[-1]}: {missing}"
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
    readouts = []
    for position, run in enumerate(args.runs):
        label = labels[position] if position < len(labels) else None
        try:
            artifacts = load_run(run, label)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[skip] {run}: {exc}", file=sys.stderr)
            continue
        print(f"[reliability-budget] {artifacts.label}: n={artifacts.stack.shape[0]} groups={artifacts.stack.shape[1]} "
              f"val={artifacts.val_groups.shape[0]} scoring={artifacts.scoring}", flush=True)
        readouts.append(analyze_run(artifacts, reps=args.reps, pairs=args.pairs, target=args.target, seed=args.seed))
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
