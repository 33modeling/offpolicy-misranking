"""Shared run discovery for harvest and CPU analysis tools."""

from __future__ import annotations

import re
from pathlib import Path


GEN_RE = re.compile(r"^v\d+-")
LEGACY_PREFIXES = ("gate-", "drift")


def is_generation_run(name: str) -> bool:
    return bool(GEN_RE.match(name))


def has_protocol_pair(run: Path) -> bool:
    """Return whether both marker files exist; schema validation belongs to gate_rules."""
    return (run / "score_protocol.json").is_file() and (
        run / "oracle_protocol.json"
    ).is_file()


def _is_direct_run(root: Path) -> bool:
    return bool(
        (root / "DONE").exists()
        or has_protocol_pair(root)
        or (root / "scores_oracle.json").exists()
    )


def _candidates(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    if _is_direct_run(root):
        return [root]
    return sorted(path for path in root.iterdir() if path.is_dir())


def iter_runs(
    root: Path,
    *,
    need: tuple[str, ...] = (),
    require_done: bool = True,
    include_legacy: bool = False,
    skip_smoke: bool = True,
) -> list[Path]:
    """Return run directories satisfying the same conditions used by diagnostics.

    Generation runs require `DONE` unless a corrected score/oracle protocol pair exists.
    Legacy gate/drift runs are included only when requested. Unknown naming schemes are
    accepted only when they carry the corrected protocol pair.
    """
    selected = []
    for run in _candidates(root):
        name = run.name
        if skip_smoke and "smoke" in name:
            continue
        generation = is_generation_run(name)
        legacy = name.startswith(LEGACY_PREFIXES)
        protocols = has_protocol_pair(run)
        if legacy and not include_legacy:
            continue
        if not generation and not legacy and not protocols:
            continue
        if generation and require_done and not (run / "DONE").exists() and not protocols:
            continue
        if any(not (run / artifact).exists() for artifact in need):
            continue
        selected.append(run)
    return selected


def describe_skips(
    root: Path,
    chosen: list[Path],
    *,
    need: tuple[str, ...] = (),
    require_done: bool = True,
    include_legacy: bool = False,
    skip_smoke: bool = True,
) -> list[str]:
    """Explain exclusions using the exact selection conditions supplied by the caller."""
    if not root.is_dir():
        return [f"{root}: 디렉터리 없음"]
    picked = {path.resolve() for path in chosen}
    lines = []
    for run in _candidates(root):
        if run.resolve() in picked:
            continue
        reasons = []
        name = run.name
        generation = is_generation_run(name)
        legacy = name.startswith(LEGACY_PREFIXES)
        protocols = has_protocol_pair(run)
        if skip_smoke and "smoke" in name:
            reasons.append("smoke 제외")
        if legacy and not include_legacy:
            reasons.append("legacy 제외")
        elif not generation and not legacy and not protocols:
            reasons.append("세대 접두사·corrected protocol 없음")
        if generation and require_done and not (run / "DONE").exists() and not protocols:
            reasons.append("DONE·corrected protocol 없음")
        for artifact in need:
            if not (run / artifact).exists():
                reasons.append(f"{artifact} 없음")
        lines.append(f"{name}: {', '.join(reasons) or '조건 미상'}")
    return lines
