"""Strict loading for score artifacts consumed by CPU analysis tools."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


ESTIMATORS = ("g00", "g10", "g01", "g11")


class ScoreArtifactError(ValueError):
    """A score artifact is missing, malformed, or has inconsistent coverage."""


@dataclass(frozen=True)
class ScoreArtifacts:
    oracle: dict[int, float]
    offpolicy: dict[str, dict[int, float]]
    splithalf: dict[int, dict[str, float]]


def _load_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoreArtifactError(f"{label} read failed: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ScoreArtifactError(f"{label} must be a non-empty JSON object")
    return value


def _score_map(value: dict, label: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    try:
        for raw_idx, row in value.items():
            idx = int(raw_idx)
            if idx in scores:
                raise ScoreArtifactError(f"{label} has duplicate normalized ID {idx}")
            score = float(row["score"])
            if not math.isfinite(score):
                raise ScoreArtifactError(f"{label}[{idx}].score is not finite")
            scores[idx] = score
    except ScoreArtifactError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoreArtifactError(f"{label} has invalid score schema: {exc}") from exc
    if not scores:
        raise ScoreArtifactError(f"{label} has no scores")
    return scores


def load_complete_score_artifacts(run: Path) -> ScoreArtifacts:
    oracle = _score_map(
        _load_object(run / "scores_oracle.json", "scores_oracle.json"),
        "scores_oracle.json",
    )
    expected_ids = set(oracle)

    raw_offpolicy = _load_object(
        run / "scores_offpolicy.json", "scores_offpolicy.json"
    )
    missing = sorted(set(ESTIMATORS) - set(raw_offpolicy))
    if missing:
        raise ScoreArtifactError(f"scores_offpolicy.json is missing {missing}")
    offpolicy: dict[str, dict[int, float]] = {}
    for estimator in ESTIMATORS:
        raw_scores = raw_offpolicy[estimator]
        if not isinstance(raw_scores, dict):
            raise ScoreArtifactError(f"scores_offpolicy[{estimator}] must be an object")
        scores = _score_map(raw_scores, f"scores_offpolicy[{estimator}]")
        if set(scores) != expected_ids:
            raise ScoreArtifactError(
                f"scores_offpolicy[{estimator}] ID coverage differs: "
                f"expected={len(expected_ids)} actual={len(scores)}"
            )
        offpolicy[estimator] = scores

    raw_halves = _load_object(
        run / "scores_splithalf.json", "scores_splithalf.json"
    )
    splithalf: dict[int, dict[str, float]] = {}
    try:
        for raw_idx, row in raw_halves.items():
            idx = int(raw_idx)
            if idx in splithalf:
                raise ScoreArtifactError(
                    f"scores_splithalf.json has duplicate normalized ID {idx}"
                )
            halves = {half: float(row[half]) for half in ("a", "b")}
            if not all(math.isfinite(score) for score in halves.values()):
                raise ScoreArtifactError(
                    f"scores_splithalf.json[{idx}] contains a non-finite score"
                )
            splithalf[idx] = halves
    except ScoreArtifactError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ScoreArtifactError(
            f"scores_splithalf.json has invalid schema: {exc}"
        ) from exc
    if set(splithalf) != expected_ids:
        raise ScoreArtifactError(
            "scores_splithalf.json ID coverage differs: "
            f"expected={len(expected_ids)} actual={len(splithalf)}"
        )

    return ScoreArtifacts(oracle, offpolicy, splithalf)
