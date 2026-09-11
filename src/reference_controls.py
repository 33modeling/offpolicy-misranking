"""Signal-resolution control for a d=0 reference point (CPU, read-only).

    python src/reference_controls.py <point dir> [<point dir> ...] [--out FILE]
    bash scripts/reference_controls.sh            # every h100 d0 point

The registered reference ranks prompts by projected gradient alignment and its
A/B split-half top-k overlap sits near chance at eight responses per half.
This control asks whether *coarser* signals computed from the very same
responses are reproducible at that budget. From `rollouts_fresh_train.jsonl`
(K current-policy responses per prompt, binary reward) it forms the registered
halves A and B (the last two quarters of the response block, as in the paper's
R / R+ / A / B allocation) and ranks prompts three ways on each half:

- pass-rate      : highest empirical pass rate first (easiest prompts);
- learnability   : -|p - 1/2| first (mixed-difficulty prompts, the ordering
                   used by variance-based rollout allocation and GRPO-style
                   prompt filters);
- hardest        : lowest pass rate first (the asymmetric-weighting choice).

For each ranking it reports the A/B top-k overlap with independent tie
streams (the paper's estimand), both on the registered halves and averaged
over random re-splits of the K responses, next to the gradient-alignment
overlap stored in the point's report.json (current and, if parked, pinned
scoring). It also counts prompts whose responses are all wrong or all right
and, when the projected gradients are present, the fraction of zero-norm
micro-groups, i.e. the part of the nominal budget that carries no ranking
signal for the gradient reference.

The result is descriptive: it compares the reproducibility of signals of
different resolution at one budget. It defines no registered label, does not
certify the implementation, and writes only the requested report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from reliability_budget import THREAD_CAP, load_run, topk_overlap_batch
from select_rules import topk_count

RANKINGS = ("pass-rate", "learnability", "hardest")


def ranking_scores(pass_rate: torch.Tensor, ranking: str) -> torch.Tensor:
    if ranking == "pass-rate":
        return pass_rate
    if ranking == "learnability":
        return -(pass_rate - 0.5).abs()
    if ranking == "hardest":
        return -pass_rate
    raise ValueError(f"unknown ranking: {ranking}")


def read_rewards(path: Path) -> torch.Tensor:
    """[prompts, responses] binary reward matrix from structured JSONL records.

    Identities come from the JSON fields, never from text search, so a response
    body that happens to contain '"reward": 1' cannot change the count.
    """
    rewards: dict[int, dict[int, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                idx, ridx, value = row["prompt_idx"], row["rollout_idx"], float(row["reward"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{number}: invalid reward record") from exc
            if type(idx) is not int or type(ridx) is not int or min(idx, ridx) < 0:
                raise ValueError(f"{path}:{number}: invalid prompt/rollout identity")
            if not math.isfinite(value) or value not in (0.0, 1.0):
                raise ValueError(f"{path}:{number}: expected a finite binary reward")
            if ridx in rewards.get(idx, {}):
                raise ValueError(f"{path}:{number}: duplicate prompt/rollout identity")
            rewards.setdefault(idx, {})[ridx] = value
    prompts = sorted(rewards)
    if len(prompts) < 2 or prompts != list(range(len(prompts))):
        raise ValueError(f"{path}: need complete contiguous prompt identities and at least two prompts")
    counts = {len(rewards[p]) for p in prompts}
    if len(counts) != 1:
        raise ValueError(f"{path}: prompts differ in stored response counts: {sorted(counts)[:5]}")
    responses = counts.pop()
    if responses < 4 or responses % 4:
        raise ValueError(f"{path}: responses must be a positive multiple of four, got {responses}")
    if any(sorted(rewards[p]) != list(range(responses)) for p in prompts):
        raise ValueError(f"{path}: missing or noncontiguous rollout identities")
    return torch.tensor([[rewards[p][j] for j in range(responses)] for p in prompts], dtype=torch.float64)


def registered_halves(responses: int) -> tuple[slice, slice]:
    """A and B are the last two quarters of the response block (paper: R = first
    two micro-groups, R+ = first four, A/B = the final two groups each)."""
    quarter = responses // 4
    return slice(2 * quarter, 3 * quarter), slice(3 * quarter, 4 * quarter)


@dataclass
class OverlapStat:
    registered: float       # A/B = registered halves, mean over tie streams
    resampled_mean: float   # random re-splits of the K responses into two halves
    resampled_sd: float
    boundary_ties: int      # prompts tied with the k-th score on registered half A


def overlap_by_ranking(matrix: torch.Tensor, *, k_frac: float, reps: int, pairs: int, seed: int) -> dict[str, OverlapStat]:
    n, responses = matrix.shape
    if reps < 1 or pairs < 1 or not 0 < k_frac < 1:
        raise ValueError("positive resample counts and 0 < topk fraction < 1 required")
    k = topk_count(n, k_frac)
    a_slice, b_slice = registered_halves(responses)
    half = responses // 2
    out: dict[str, OverlapStat] = {}
    for ranking in RANKINGS:
        pa = ranking_scores(matrix[:, a_slice].mean(dim=1), ranking)
        pb = ranking_scores(matrix[:, b_slice].mean(dim=1), ranking)
        registered = topk_overlap_batch(pa, pb, k, pairs=pairs, generator=torch.Generator().manual_seed(seed + 11))
        kth = torch.topk(pa, k).values[-1]
        ties = int((pa == kth).sum())
        draws = []
        for rep in range(reps):
            generator = torch.Generator().manual_seed(seed + 1_000_003 * rep)
            order = torch.argsort(torch.rand(n, responses, generator=generator), dim=1)
            shuffled = torch.gather(matrix, 1, order)
            sa = ranking_scores(shuffled[:, :half].mean(dim=1), ranking)
            sb = ranking_scores(shuffled[:, half:2 * half].mean(dim=1), ranking)
            draws.append(topk_overlap_batch(sa, sb, k, pairs=pairs, generator=torch.Generator().manual_seed(seed + 17 + rep)))
        out[ranking] = OverlapStat(
            registered=registered,
            resampled_mean=statistics.fmean(draws),
            resampled_sd=statistics.pstdev(draws) if len(draws) > 1 else 0.0,
            boundary_ties=ties,
        )
    return out


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def gradient_floors(run: Path) -> dict[str, float]:
    """Gradient-alignment A/B overlap of this point: report.json (current) and
    the newest parked pinned scoring, when present."""
    floors: dict[str, float] = {}
    report = _load_json(run / "report.json")
    if isinstance(report, dict) and isinstance(report.get("noise_floor"), (int, float)):
        floors["current"] = float(report["noise_floor"])
    parked = sorted(p for p in run.glob("pinned-scoring/*/") if p.is_dir())
    if parked:
        report = _load_json(parked[-1] / "report.json")
        if isinstance(report, dict) and isinstance(report.get("noise_floor"), (int, float)):
            floors["pinned"] = float(report["noise_floor"])
    return floors


def zero_group_fraction(run: Path) -> float | None:
    """Fraction of zero-norm oracle micro-groups, or None when gradients are absent."""
    try:
        stack = load_run(run).stack
    except (FileNotFoundError, ValueError, KeyError, RuntimeError):
        return None
    norms = stack.norm(dim=2)
    return 1.0 - float((norms > 0).double().mean())


@dataclass
class PointRow:
    label: str
    prompts: int
    responses: int
    k: int
    chance: float
    overlaps: dict[str, OverlapStat]
    floors: dict[str, float]
    all_wrong: int
    all_right: int
    mean_pass_rate: float
    zero_groups: float | None


def audit_point(run: Path, label: str, *, k_frac: float, reps: int, pairs: int, seed: int) -> PointRow:
    fresh = run / "rollouts_fresh_train.jsonl"
    if not fresh.is_file():
        raise FileNotFoundError(f"{run}: rollouts_fresh_train.jsonl absent")
    matrix = read_rewards(fresh)
    n, responses = matrix.shape
    means = matrix.mean(dim=1)
    return PointRow(
        label=label, prompts=n, responses=responses, k=topk_count(n, k_frac), chance=topk_count(n, k_frac) / n,
        overlaps=overlap_by_ranking(matrix, k_frac=k_frac, reps=reps, pairs=pairs, seed=seed),
        floors=gradient_floors(run),
        all_wrong=int((means == 0).sum()), all_right=int((means == 1).sum()),
        mean_pass_rate=float(means.mean()), zero_groups=zero_group_fraction(run),
    )


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def render(rows: list[PointRow], *, reps: int, pairs: int, k_frac: float) -> str:
    lines = [
        "# Signal-resolution control at the registered d=0 budget (descriptive; defines no registered label)",
        "",
        f"A/B = registered halves of the stored current-policy responses; top-{k_frac:.0%} overlap averaged over {pairs} independent tie-stream pairs;",
        f"'resampled' = mean +- sd over {reps} random re-splits of the same responses into two halves. Chance = k/n.",
        "gradient = A/B overlap of the registered gradient-alignment ranking from report.json (current scoring; pinned in brackets when parked).",
        "",
        " point              n    K   chance  gradient        pass-rate reg/resampled   learnability reg/resampled   hardest reg/resampled    all-wrong all-right zero-groups",
    ]
    for r in rows:
        grad = _fmt(r.floors.get("current"))
        if "pinned" in r.floors:
            grad += f" [{_fmt(r.floors['pinned'])}]"
        cells = []
        for ranking in RANKINGS:
            o = r.overlaps[ranking]
            cells.append(f"{_fmt(o.registered)} / {_fmt(o.resampled_mean)}+-{_fmt(o.resampled_sd, 2)}")
        lines.append(
            f" {r.label:<18} {r.prompts:<4} {r.responses:<3} {r.chance:<7.3f} {grad:<15} {cells[0]:<25} {cells[1]:<28} {cells[2]:<24} "
            f"{r.all_wrong:<9} {r.all_right:<9} {_fmt(r.zero_groups)}"
        )
    lines.append("")
    for r in rows:
        ties = ", ".join(f"{k} {r.overlaps[k].boundary_ties}" for k in RANKINGS)
        lines.append(f" {r.label}: mean pass rate {r.mean_pass_rate:.3f}; prompts tied with the k-th score on half A: {ties}")
    lines.append("")
    if rows:
        def mean_of(get):
            values = [get(r) for r in rows]
            values = [v for v in values if v is not None and not math.isnan(v)]
            return statistics.fmean(values) if values else float("nan")
        lines.append(
            f"KEY means over {len(rows)} point(s): gradient {_fmt(mean_of(lambda r: r.floors.get('current')))}; "
            + "; ".join(f"{k} {_fmt(mean_of(lambda r, k=k: r.overlaps[k].registered))}" for k in RANKINGS)
            + f"; chance {_fmt(mean_of(lambda r: r.chance))}"
        )
        lines.append(
            "KEY reading: a ranking whose overlap sits well above chance while the gradient overlap does not is reproducible at this budget;"
            " none of these numbers is a selection-gain or downstream result."
        )
    return "\n".join(lines) + "\n"


def refuse_output_inside_inputs(output: Path, inputs: list[Path]) -> None:
    destination = output.resolve()
    for run in inputs:
        root = run.resolve()
        if destination == root or root in destination.parents:
            raise ValueError(f"refusing to write the report inside an experiment point: {destination}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="completed d=0 point directories")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk-frac", type=float, default=0.10)
    parser.add_argument("--label", action="append", default=None, help="label per run, in order")
    args = parser.parse_args(argv)
    if args.reps < 1 or args.pairs < 1 or not 0 < args.topk_frac < 1:
        parser.error("positive --reps/--pairs and 0 < --topk-frac < 1 required")
    if args.out is not None:
        refuse_output_inside_inputs(args.out, args.runs)
    torch.set_num_threads(max(1, min(THREAD_CAP, os.cpu_count() or 1)))
    labels = args.label or []
    rows: list[PointRow] = []
    for position, run in enumerate(args.runs):
        label = labels[position] if position < len(labels) else run.name
        started = time.time()
        try:
            rows.append(audit_point(run, label, k_frac=args.topk_frac, reps=args.reps, pairs=args.pairs, seed=args.seed))
        except (OSError, ValueError) as exc:
            print(f"[skip] {run}: {exc}", file=sys.stderr)
            continue
        print(f"[reference-controls] {label}: done in {time.time() - started:.0f}s", flush=True)
    if not rows:
        print("[abort] no point directory could be analysed", file=sys.stderr)
        return 1
    report = render(rows, reps=args.reps, pairs=args.pairs, k_frac=args.topk_frac)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(args.out.name + ".tmp")
        temporary.write_text(report, encoding="utf-8")
        temporary.replace(args.out)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
