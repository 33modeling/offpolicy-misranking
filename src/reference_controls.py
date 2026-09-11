"""Signal-resolution control for a d=0 reference point (CPU, read-only).

    python src/reference_controls.py <point dir> [<point dir> ...] [--out FILE]
    bash scripts/reference_controls.sh            # every h100 d0 point

The registered reference ranks prompts by projected gradient alignment, and
its A/B split-half top-k overlap sits near chance at eight responses per
half. A near-chance overlap says the fine *ranking* is not reproducible at
that budget; it does not say that no signal picks useful data. This control
separates the two questions on the stored artifacts of one point:

1. What is reproducible at this budget?  From the K current-policy responses
   per prompt (`rollouts_fresh_train.jsonl`) it forms the registered halves
   A and B (the last two quarters of the response block) and measures
   (a) the agreement of the coarse difficulty partition
       {all-wrong, mixed, all-right} between the halves (per-band Jaccard
       against its independence expectation, and Cohen's kappa), and
   (b) the top-k overlap of three rankings on the same rewards (pass rate,
       learnability -|p-1/2|, hardest-first) with independent tie streams,
       next to the gradient-alignment overlap stored in report.json.
2. Does a coarse, reproducible signal pick data as useful as the fine
   ranking?  On the paper's own utility (the averaged A/B alignment score of
   the selected set, minus the uniform-selection mean) it compares the fresh
   top-k from split R, each stale estimator's top-k, a uniform random draw
   inside the mixed band of the fresh R responses, a uniform random draw
   inside the mixed band of the stored behavior responses (no fresh rollout
   at all), and uniform selection. It also reports what fraction of the
   fresh top-k lies inside those bands.

Everything is descriptive and paired within a point: it defines no
registered label, does not certify the implementation, and writes only the
requested report. Utility here is gradient alignment on a noisy reference,
not downstream reward; the reduced E5 measures the latter.
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

from reliability_budget import THREAD_CAP, load_run, topk_overlap_batch
from select_rules import jittered_topk, topk_count

RANKINGS = ("pass-rate", "learnability", "hardest")
BANDS = ("all-wrong", "mixed", "all-right")
ESTIMATORS = ("g00", "g10", "g01", "g11")
TIE_PAIRS = 20


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


def primary_r(responses: int) -> slice:
    """The primary ranking split R: the first quarter of the response block."""
    return slice(0, responses // 4)


# ---------------------------------------------------------------- 1a. partition agreement

def bands(pass_rate: torch.Tensor) -> torch.Tensor:
    """0 = all-wrong, 1 = mixed, 2 = all-right."""
    return torch.where(pass_rate <= 0, 0, torch.where(pass_rate >= 1, 2, 1))


@dataclass
class PartitionAgreement:
    kappa: float
    jaccard: dict[str, float]
    jaccard_chance: dict[str, float]   # expectation for independent sets with the observed sizes
    size_a: dict[str, int]
    size_b: dict[str, int]


def partition_agreement(pa: torch.Tensor, pb: torch.Tensor) -> PartitionAgreement:
    ba, bb = bands(pa), bands(pb)
    n = ba.numel()
    observed = float((ba == bb).double().mean())
    expected = sum(float((ba == c).double().mean()) * float((bb == c).double().mean()) for c in range(3))
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    jaccard, chance, size_a, size_b = {}, {}, {}, {}
    for c, name in enumerate(BANDS):
        a, b = ba == c, bb == c
        na, nb, both = int(a.sum()), int(b.sum()), int((a & b).sum())
        union = na + nb - both
        jaccard[name] = both / union if union else float("nan")
        expected_both = na * nb / n
        expected_union = na + nb - expected_both
        chance[name] = expected_both / expected_union if expected_union else float("nan")
        size_a[name], size_b[name] = na, nb
    return PartitionAgreement(kappa=kappa, jaccard=jaccard, jaccard_chance=chance, size_a=size_a, size_b=size_b)


# ---------------------------------------------------------------- 1b. ranking overlap

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


# ---------------------------------------------------------------- 2. utility of coarse selections

def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_scores(run: Path, n: int) -> dict[str, dict[int, float]] | None:
    """Per-prompt scores: 'truth' = (a+b)/2 from scores_splithalf.json, 'fresh' =
    the R split ('r', else scores_oracle.json), and the four stale estimators.
    None when the point has no complete split-half scores."""
    split = _load_json(run / "scores_splithalf.json")
    if not isinstance(split, dict) or len(split) != n:
        return None
    try:
        halves = {int(k): v for k, v in split.items()}
        truth = {i: (float(halves[i]["a"]) + float(halves[i]["b"])) / 2.0 for i in range(n)}
    except (KeyError, TypeError, ValueError):
        return None
    if all("r" in halves[i] for i in range(n)):
        fresh = {i: float(halves[i]["r"]) for i in range(n)}
    else:
        oracle = _load_json(run / "scores_oracle.json")
        if not isinstance(oracle, dict) or len(oracle) != n:
            return None
        try:
            fresh = {int(k): float(v["score"]) for k, v in oracle.items()}
        except (KeyError, TypeError, ValueError):
            return None
    scores = {"truth": truth, "fresh": fresh}
    stale = _load_json(run / "scores_offpolicy.json")
    if isinstance(stale, dict):
        for name in ESTIMATORS:
            table = stale.get(name)
            if isinstance(table, dict) and len(table) == n:
                try:
                    scores[name] = {int(k): float(v["score"]) for k, v in table.items()}
                except (KeyError, TypeError, ValueError):
                    continue
    if any(not math.isfinite(v) for table in scores.values() for v in table.values()):
        return None
    return scores


@dataclass
class UtilityComparison:
    k: int
    random_utility: float
    gain: dict[str, float]                 # selection -> U(S) - U_rand (band-random: mean over draws)
    gain_sd: dict[str, float]              # sd over random draws for the random selections
    fresh_topk_in_band: dict[str, float]   # fraction of fresh top-k prompts inside each mixed band
    band_size: dict[str, int]
    draws: int


def utility_comparison(
    scores: dict[str, dict[int, float]],
    mixed_bands: dict[str, torch.Tensor],
    *,
    k_frac: float,
    draws: int,
    seed: int,
) -> UtilityComparison:
    truth = scores["truth"]
    ids = sorted(truth)
    n = len(ids)
    k = topk_count(n, k_frac)
    random_utility = statistics.fmean(truth[i] for i in ids)

    def selected(score_table: dict[int, float], offset: int) -> tuple[float, list[set]]:
        sets = [jittered_topk(score_table, k, seed + offset + pair * 7_919) for pair in range(TIE_PAIRS)]
        return statistics.fmean(statistics.fmean(truth[i] for i in s) for s in sets), sets

    gain: dict[str, float] = {}
    gain_sd: dict[str, float] = {}
    fresh_utility, fresh_sets = selected(scores["fresh"], 17)
    gain["fresh-topk"] = fresh_utility - random_utility
    for name in ESTIMATORS:
        if name in scores:
            gain[f"stale-{name}"] = selected(scores[name], 0)[0] - random_utility
    generator = torch.Generator().manual_seed(seed + 101)
    in_band: dict[str, float] = {}
    band_size: dict[str, int] = {}
    for label, mask in mixed_bands.items():
        members = [ids[j] for j in torch.nonzero(mask).flatten().tolist()]
        band_size[label] = len(members)
        in_band[label] = statistics.fmean(sum(1 for i in s if mask[i]) / k for s in fresh_sets)
        if len(members) == 0:
            gain[f"{label}-random"] = float("nan")
            gain_sd[f"{label}-random"] = float("nan")
            continue
        values = []
        for _ in range(draws):
            if len(members) <= k:
                chosen = members
            else:
                pick = torch.randperm(len(members), generator=generator)[:k].tolist()
                chosen = [members[j] for j in pick]
            values.append(statistics.fmean(truth[i] for i in chosen) - random_utility)
        gain[f"{label}-random"] = statistics.fmean(values)
        gain_sd[f"{label}-random"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return UtilityComparison(k=k, random_utility=random_utility, gain=gain, gain_sd=gain_sd,
                             fresh_topk_in_band=in_band, band_size=band_size, draws=draws)


# ---------------------------------------------------------------- point audit

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
    partition: PartitionAgreement
    overlaps: dict[str, OverlapStat]
    floors: dict[str, float]
    all_wrong: int
    all_right: int
    mean_pass_rate: float
    zero_groups: float | None
    utility: UtilityComparison | None
    notes: list[str] = field(default_factory=list)


def audit_point(run: Path, label: str, *, k_frac: float, reps: int, pairs: int, seed: int, draws: int = 200) -> PointRow:
    fresh = run / "rollouts_fresh_train.jsonl"
    if not fresh.is_file():
        raise FileNotFoundError(f"{run}: rollouts_fresh_train.jsonl absent")
    matrix = read_rewards(fresh)
    n, responses = matrix.shape
    means = matrix.mean(dim=1)
    a_slice, b_slice = registered_halves(responses)
    partition = partition_agreement(matrix[:, a_slice].mean(dim=1), matrix[:, b_slice].mean(dim=1))
    notes: list[str] = []
    mixed_bands = {"fresh-band": bands(matrix[:, primary_r(responses)].mean(dim=1)) == 1}
    behavior = run / "rollouts_behavior_train.jsonl"
    if behavior.is_file():
        try:
            behavior_matrix = read_rewards(behavior)
            if behavior_matrix.shape[0] == n:
                mixed_bands["behavior-band"] = bands(behavior_matrix.mean(dim=1)) == 1
            else:
                notes.append("behavior rollouts cover a different prompt count; behavior band skipped")
        except ValueError as exc:
            notes.append(f"behavior rollouts unreadable ({exc}); behavior band skipped")
    else:
        notes.append("no stored behavior rollouts; behavior band skipped")
    scores = load_scores(run, n)
    utility = None
    if scores is None:
        notes.append("split-half scores absent or incomplete; utility comparison skipped")
    else:
        utility = utility_comparison(scores, mixed_bands, k_frac=k_frac, draws=draws, seed=seed)
    return PointRow(
        label=label, prompts=n, responses=responses, k=topk_count(n, k_frac), chance=topk_count(n, k_frac) / n,
        partition=partition,
        overlaps=overlap_by_ranking(matrix, k_frac=k_frac, reps=reps, pairs=pairs, seed=seed),
        floors=gradient_floors(run),
        all_wrong=int((means == 0).sum()), all_right=int((means == 1).sum()),
        mean_pass_rate=float(means.mean()), zero_groups=zero_group_fraction(run),
        utility=utility, notes=notes,
    )


# ---------------------------------------------------------------- report

def _fmt(value: float | None, digits: int = 3, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _mean_of(rows: list[PointRow], get) -> float:
    values = []
    for r in rows:
        try:
            v = get(r)
        except (KeyError, TypeError, AttributeError):
            continue
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            values.append(v)
    return statistics.fmean(values) if values else float("nan")


def render(rows: list[PointRow], *, reps: int, pairs: int, k_frac: float) -> str:
    lines = [
        "# Signal-resolution control at the registered d=0 budget (descriptive; defines no registered label)",
        "",
        "A/B = registered halves of the stored current-policy responses (8 each at K=32). Chance = k/n for overlaps;",
        "for partition Jaccard, chance = the expectation for independent sets of the observed sizes.",
        "",
        "## 1a. Is the coarse difficulty partition reproducible between A and B?",
        "",
        " point              n    K   kappa   Jaccard all-wrong (chance)  Jaccard mixed (chance)  Jaccard all-right (chance)  |A|,|B| mixed",
    ]
    for r in rows:
        p = r.partition
        cells = [f"{_fmt(p.jaccard[b])} ({_fmt(p.jaccard_chance[b])})" for b in BANDS]
        lines.append(f" {r.label:<18} {r.prompts:<4} {r.responses:<3} {_fmt(p.kappa):<7} {cells[0]:<27} {cells[1]:<23} {cells[2]:<27} {p.size_a['mixed']},{p.size_b['mixed']}")
    lines += [
        "",
        f"## 1b. Is a coarse *ranking* reproducible where the gradient ranking is not?  (top-{k_frac:.0%} overlap, {pairs} tie-stream pairs;",
        f"    'resampled' = mean +- sd over {reps} random re-splits; gradient = A/B overlap of the registered alignment ranking from report.json, pinned in brackets)",
        "",
        " point              chance  gradient        pass-rate reg/resampled   learnability reg/resampled   hardest reg/resampled    all-wrong all-right zero-groups",
    ]
    for r in rows:
        grad = _fmt(r.floors.get("current"))
        if "pinned" in r.floors:
            grad += f" [{_fmt(r.floors['pinned'])}]"
        cells = [f"{_fmt(r.overlaps[k].registered)} / {_fmt(r.overlaps[k].resampled_mean)}+-{_fmt(r.overlaps[k].resampled_sd, 2)}" for k in RANKINGS]
        lines.append(f" {r.label:<18} {r.chance:<7.3f} {grad:<15} {cells[0]:<25} {cells[1]:<28} {cells[2]:<24} {r.all_wrong:<9} {r.all_right:<9} {_fmt(r.zero_groups)}")
    lines += [
        "",
        "## 2. Does a coarse, reproducible signal pick data as useful as the fine ranking?  (paper utility: mean A/B alignment of the",
        "    selected set minus the uniform mean; band-random = uniform draw inside the mixed band, mean over draws; fresh-band uses",
        "    the R split's responses, behavior-band uses the stored behavior responses, i.e. no fresh rollout)",
        "",
        " point              k    U_rand    fresh-topk  stale-g11   stale-g00   fresh-band-random    behavior-band-random   fresh-topk in fresh-band / behavior-band   band sizes",
    ]
    for r in rows:
        u = r.utility
        if u is None:
            lines.append(f" {r.label:<18} {r.k:<4} utility comparison skipped: {'; '.join(r.notes) or 'scores absent'}")
            continue
        fb = f"{_fmt(u.gain.get('fresh-band-random'), signed=True)}+-{_fmt(u.gain_sd.get('fresh-band-random'), 3)}"
        bb = f"{_fmt(u.gain.get('behavior-band-random'), signed=True)}+-{_fmt(u.gain_sd.get('behavior-band-random'), 3)}" if "behavior-band-random" in u.gain else "-"
        inb = f"{_fmt(u.fresh_topk_in_band.get('fresh-band'))} / {_fmt(u.fresh_topk_in_band.get('behavior-band'))}"
        sizes = ", ".join(f"{k} {v}" for k, v in u.band_size.items())
        lines.append(
            f" {r.label:<18} {u.k:<4} {_fmt(u.random_utility):<9} {_fmt(u.gain.get('fresh-topk'), signed=True):<11} "
            f"{_fmt(u.gain.get('stale-g11'), signed=True):<11} {_fmt(u.gain.get('stale-g00'), signed=True):<11} {fb:<20} {bb:<22} {inb:<42} {sizes}"
        )
    lines.append("")
    for r in rows:
        ties = ", ".join(f"{k} {r.overlaps[k].boundary_ties}" for k in RANKINGS)
        extra = f"; {'; '.join(r.notes)}" if r.notes else ""
        lines.append(f" {r.label}: mean pass rate {r.mean_pass_rate:.3f}; prompts tied with the k-th score on half A: {ties}{extra}")
    lines.append("")
    if rows:
        lines.append(
            f"KEY means over {len(rows)} point(s): partition kappa {_fmt(_mean_of(rows, lambda r: r.partition.kappa))}, "
            f"Jaccard mixed {_fmt(_mean_of(rows, lambda r: r.partition.jaccard['mixed']))} (chance {_fmt(_mean_of(rows, lambda r: r.partition.jaccard_chance['mixed']))}); "
            f"overlap gradient {_fmt(_mean_of(rows, lambda r: r.floors.get('current')))}, "
            + ", ".join(f"{k} {_fmt(_mean_of(rows, lambda r, k=k: r.overlaps[k].registered))}" for k in RANKINGS)
            + f", chance {_fmt(_mean_of(rows, lambda r: r.chance))}"
        )
        lines.append(
            "KEY utility gain over uniform (mean over points): "
            + ", ".join(
                f"{name} {_fmt(_mean_of(rows, lambda r, name=name: r.utility.gain.get(name)), signed=True)}"
                for name in ("fresh-topk", "stale-g11", "stale-g00", "fresh-band-random", "behavior-band-random")
            )
            + f"; fresh top-k inside fresh-band {_fmt(_mean_of(rows, lambda r: r.utility.fresh_topk_in_band.get('fresh-band')))}"
        )
        lines.append(
            "KEY reading: 1a says which partition survives resampling at this budget; 1b says whether any fine ranking does; 2 says whether"
            " the reproducible coarse signal already captures the alignment utility of the fine ranking. Utility is reference alignment,"
            " not training reward (reduced E5); no registered gate or label is defined here."
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
    parser.add_argument("--draws", type=int, default=200, help="random draws for the band-random selections")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk-frac", type=float, default=0.10)
    parser.add_argument("--label", action="append", default=None, help="label per run, in order")
    args = parser.parse_args(argv)
    if args.reps < 1 or args.pairs < 1 or args.draws < 1 or not 0 < args.topk_frac < 1:
        parser.error("positive --reps/--pairs/--draws and 0 < --topk-frac < 1 required")
    if args.out is not None:
        refuse_output_inside_inputs(args.out, args.runs)
    torch.set_num_threads(max(1, min(THREAD_CAP, os.cpu_count() or 1)))
    labels = args.label or []
    rows: list[PointRow] = []
    for position, run in enumerate(args.runs):
        label = labels[position] if position < len(labels) else run.name
        started = time.time()
        try:
            rows.append(audit_point(run, label, k_frac=args.topk_frac, reps=args.reps, pairs=args.pairs, seed=args.seed, draws=args.draws))
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
