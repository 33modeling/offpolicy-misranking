"""Fixed, inspectable N01-N08 designs and deterministic task dependencies."""

import itertools
import re
from dataclasses import asdict, dataclass

import numpy as np

from srgc_rebuttal.srgc import stream_seed

PROTOCOL = "srgc-literature-studies-v1"
STAGES = (0, 100, 400)
EVAL_UPDATES = (0, 1, 5, 25)
SCOPES = tuple(f"n{i:02d}" for i in range(1, 9))


@dataclass(frozen=True)
class Condition:
    key: str
    kind: str
    arm: str = "on_policy"
    candidates: int = 40
    batch: int = 4
    reference: int = -1
    stage: int = -1
    updates: int = 275
    draw: int = 0

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.key):
            raise ValueError("condition key must be a safe path component")
        if self.kind not in {"trajectory", "diagnostic", "anchors", "cache", "features"}:
            raise ValueError("unknown research condition kind")
        if any(type(v) is not int for v in (self.candidates, self.batch, self.reference, self.stage, self.updates, self.draw)):
            raise ValueError("research dimensions must be integers")
        if not 1 <= self.batch <= self.candidates <= 400 or self.updates < 0 or self.draw not in range(3):
            raise ValueError("invalid research dimensions")
        if self.reference not in (-1, 0, 1, 2) or self.stage not in (-1, *STAGES):
            raise ValueError("invalid reference or diagnostic stage")
        if self.kind == "trajectory" and self.arm not in {"random", "sr", "on_policy", "switch", "lesser", "arcus_adapted"}:
            raise ValueError("unknown trajectory selector")
        if self.kind == "diagnostic" and (self.arm not in {"n02", "n03", "n04", "n07"} or self.stage < 0):
            raise ValueError("diagnostic requires a named study and anchor stage")

    def record(self):
        return asdict(self)

    @property
    def needs_cache(self):
        return self.kind in {"anchors", "diagnostic"} or self.arm in {"sr", "switch"}


def conditions(scope):
    if scope not in SCOPES:
        raise ValueError(f"unknown research scope: {scope}")
    if scope == "n08":
        return []  # Paired endpoint analysis, not another training run.
    if scope == "n03":
        return [Condition(f"n03-stage-{s}-draw-{d}", "diagnostic", arm=scope, stage=s, updates=25, draw=d)
                for s in STAGES for d in range(3)]
    if scope in {"n02", "n07"}:
        return [Condition(f"{scope}-stage-{s}", "diagnostic", arm=scope, stage=s, updates=25)
                for s in STAGES]
    if scope == "n04":
        return [Condition(f"n04-ref-{r}", "trajectory", arm="switch", reference=r) for r in range(3)] + [
            Condition(f"n04-stage-{s}", "diagnostic", arm="n04", stage=s, updates=0) for s in STAGES]
    if scope == "n05":
        sizes = ((40, 4), (80, 4), (160, 4), (40, 8), (40, 16))
        return [Condition(f"baseline-{a}" if (c, b) == (40, 4) else f"n05-{a}-c{c}-b{b}",
                          "trajectory", arm=a, candidates=c, batch=b)
                for c, b in sizes for a in ("on_policy", "sr", "random")]
    arms = ("on_policy", "switch", "lesser") if scope == "n01" else ("arcus_adapted", "switch")
    result = [Condition(f"baseline-{a}" if a in {"on_policy", "switch"} else f"{scope}-{a}", "trajectory", arm=a) for a in arms]
    return result + ([Condition("n01-features", "features", updates=0)] if scope == "n01" else [])


def tasks(scope):
    targets = conditions(scope)
    prerequisites = []
    if any(c.needs_cache for c in targets):
        prerequisites.append(Condition("cache", "cache", updates=0))
    if any(c.kind == "diagnostic" for c in targets):
        prerequisites.append(Condition("anchors", "anchors", updates=STAGES[-1]))
    return [*prerequisites, *targets]


def candidate_draw(ids, count, seed, step):
    if not 0 < count <= len(ids) or len(set(ids)) != len(ids):
        raise ValueError("candidate draw requires enough distinct prompts")
    order = np.random.default_rng(stream_seed(seed, step, "nested-candidates")).permutation(len(ids))
    return tuple(ids[int(i)] for i in order[:count])


def reference_sets(data, seed):
    pool, size = list(data["validation_pool_ids"]), len(data["ranking_validation_ids"])
    if size < 1 or len(pool) <= size:
        raise ValueError("N04 needs more validation prompts than one reference set")
    if set(pool) & (set(data["candidate_ids"]) | set(data["evaluation_ids"])):
        raise ValueError("reference pool overlaps training or evaluation")
    order = np.random.default_rng(stream_seed(seed, 0, "reference-subsets")).permutation(pool).tolist()
    offsets = (0, (len(pool) - size) // 2, len(pool) - size)
    groups = [order[o:o + size] for o in offsets]
    if len({frozenset(g) for g in groups}) != 3:
        raise ValueError("reference pool cannot provide three different equal-sized subsets")
    overlap = {f"{a}-{b}": len(set(groups[a]) & set(groups[b])) / size
               for a, b in itertools.combinations(range(3), 2)}
    return groups, overlap


def score_bins(ids, scores, k, seed):
    from srgc_rebuttal.srgc import top_ids
    if len(ids) % 4 or k > len(ids) // 4:
        raise ValueError("four equal score bins must each contain the training batch")
    ranked = top_ids(ids, scores, len(ids), seed)
    rng = np.random.default_rng(stream_seed(seed, 0, "bin-samples"))
    bins = np.array_split(np.asarray(ranked), 4)
    selected = {f"bin-{i}": rng.choice(b, k, replace=False).tolist() for i, b in enumerate(bins)}
    selected.update(top=list(ranked[:k]), random=rng.choice(ids, k, replace=False).tolist())
    by_id = dict(zip(ids, map(float, scores)))
    return selected, {"constant_scores": bool(np.ptp(scores) == 0),
        "tied_fraction": 1 - len(set(map(float, scores))) / len(scores),
        "bins": [{"ids": b.tolist(), "min": min(by_id[i] for i in b),
                  "max": max(by_id[i] for i in b)} for b in bins]}


def matched_control(ids, selected, labels, seed):
    """Match one observed categorical axis; overlap is allowed and reported."""
    if set(ids) != set(labels) or not set(selected) <= set(ids):
        raise ValueError("matching labels must cover every candidate")
    rng = np.random.default_rng(seed)
    output, quotas = [], {}
    for label in sorted({labels[i] for i in selected}, key=str):
        count = sum(labels[i] == label for i in selected)
        eligible = [i for i in ids if labels[i] == label]
        output.extend(rng.choice(eligible, count, replace=False).tolist())
        quotas[str(label)] = count
    return output, {"quotas": quotas, "matched": len(output), "unmatched": 0,
                    "overlap_with_top": len(set(output) & set(selected))}
