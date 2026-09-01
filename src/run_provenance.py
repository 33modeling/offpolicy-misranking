"""Read and validate immutable generation provenance for experiment runs."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")


class RunProvenanceError(ValueError):
    """A run cannot be attributed to one immutable generation revision."""


def validate_commit(value: object, source: Path | str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise RunProvenanceError(f"invalid generation commit in {source}: {value!r}")
    return value


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunProvenanceError(f"unreadable provenance document: {path}") from exc
    if not isinstance(value, dict):
        raise RunProvenanceError(f"provenance document is not an object: {path}")
    return value


def generation_commit(run: Path, *, verify_linked: bool = True) -> str:
    """Return the run's generation commit after checking linked artifacts."""
    config_path = run / "run_config.json"
    config = _read_object(config_path)
    commit = validate_commit(config.get("git"), config_path)
    if not verify_linked:
        return commit

    manifest_path = run / "manifest.json"
    if manifest_path.exists():
        manifest_commit = validate_commit(
            _read_object(manifest_path).get("git"), manifest_path
        )
        if manifest_commit != commit:
            raise RunProvenanceError(
                f"generation commit mismatch in {run}: "
                f"run_config={commit}, manifest={manifest_commit}"
            )

    score_path = run / "score_protocol.json"
    if score_path.exists():
        source_commit = validate_commit(
            _read_object(score_path).get("source_run_git"), score_path
        )
        if source_commit != commit:
            raise RunProvenanceError(
                f"generation commit mismatch in {run}: "
                f"run_config={commit}, score_protocol={source_commit}"
            )
    return commit


def partition_by_generation(
    runs: Iterable[Path], *, verify_linked: bool = True
) -> dict[str, list[Path]]:
    partitions: dict[str, list[Path]] = defaultdict(list)
    for run in runs:
        partitions[generation_commit(run, verify_linked=verify_linked)].append(run)
    return {commit: sorted(paths) for commit, paths in sorted(partitions.items())}


def require_single_generation(
    runs: Iterable[Path], *, verify_linked: bool = True
) -> str:
    partitions = partition_by_generation(runs, verify_linked=verify_linked)
    if not partitions:
        raise RunProvenanceError("no runs have generation provenance")
    if len(partitions) != 1:
        details = ", ".join(
            f"{commit}=[{', '.join(path.name for path in paths)}]"
            for commit, paths in partitions.items()
        )
        raise RunProvenanceError(f"mixed generation commits: {details}")
    return next(iter(partitions))
