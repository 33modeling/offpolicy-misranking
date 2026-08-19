"""top-k 크기 규칙 단일화 — P0-5 (docs/PAPER_REVIEW_2026-08-19.md).

int()/round() 혼재로 같은 n=256, frac=0.1에서 스크립트마다 k=25/26이 갈려
본문 표(precision·chance)와 floor·통계가 다른 k를 섞어 쓰고 있었다.
규칙은 k = max(1, floor(frac·n)) 하나로 고정한다 — 본문 표 계열(int)과 일치.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt
from typing import Hashable


_RIGHT_STREAM_OFFSET = 104_729
_PAIR_STRIDE = 7_919


@dataclass(frozen=True)
class OverlapSummary:
    mean: float
    low: float
    high: float
    sd: float
    values: tuple[float, ...]


def topk_count(n: int, frac: float = 0.10) -> int:
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    return min(n, max(1, int(n * frac)))


def jittered_topk(scores: Mapping[Hashable, float], k: int, seed: int) -> set:
    """Return a deterministic draw from the random tie-breaking top-k rule."""
    if not scores:
        raise ValueError("scores must not be empty")
    if not 1 <= k <= len(scores):
        raise ValueError(f"k must be in [1, {len(scores)}], got {k}")
    rng = random.Random(seed)
    ids = sorted(scores)
    jitter = {idx: rng.random() for idx in ids}
    return set(sorted(ids, key=lambda idx: (-float(scores[idx]), jitter[idx]))[:k])


def overlap_under_independent_ties(
    left: Mapping[Hashable, float],
    right: Mapping[Hashable, float],
    k: int,
    *,
    seed: int = 0,
    pairs: int = 20,
) -> OverlapSummary:
    """Top-k overlap with independent tie streams, averaged over seeded pairs."""
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))[:5]
        missing_right = sorted(set(left) - set(right))[:5]
        raise ValueError(
            "score ID sets differ: "
            f"missing_left={missing_left}, missing_right={missing_right}"
        )
    if pairs < 1:
        raise ValueError(f"pairs must be positive, got {pairs}")
    values = []
    for pair in range(pairs):
        left_top = jittered_topk(left, k, seed + _PAIR_STRIDE * pair)
        right_top = jittered_topk(
            right, k, seed + _RIGHT_STREAM_OFFSET + _PAIR_STRIDE * pair
        )
        values.append(len(left_top & right_top) / k)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    return OverlapSummary(mean, min(values), max(values), sqrt(variance), tuple(values))


def fixed_selection_overlap(
    selected: set,
    scores: Mapping[Hashable, float],
    k: int,
    *,
    seed: int = 0,
    pairs: int = 20,
) -> OverlapSummary:
    """Overlap of a fixed selection with a tie-randomized score ranking."""
    if len(selected) != k:
        raise ValueError(f"selected must contain exactly k={k} IDs, got {len(selected)}")
    if not selected <= set(scores):
        raise ValueError("selected contains IDs absent from scores")
    values = []
    for pair in range(pairs):
        truth = jittered_topk(scores, k, seed + _RIGHT_STREAM_OFFSET + _PAIR_STRIDE * pair)
        values.append(len(selected & truth) / k)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    return OverlapSummary(mean, min(values), max(values), sqrt(variance), tuple(values))
