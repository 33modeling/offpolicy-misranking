#!/usr/bin/env python3
"""Find a hand-uploaded model snapshot by content and expose it at the pinned path.

    python3 src/locate_uploaded_snapshot.py --config CONFIG --model-key KEY --models-dir DIR

People upload weights by scp into whatever directory name they like, sometimes
with shard names that differ from the Hub index. The contracts, however, expect
$MODELS_DIR/<local_directory> with the official file names. This tool:

1. uses $MODELS_DIR/<local_directory> if it already holds config.json;
2. otherwise scans $MODELS_DIR (two levels) for a config.json whose model_type
   matches the spec and whose size equals the pinned official config.json, and
   links the pinned directory name to it (symlink, nothing is moved);
3. inside the chosen directory, links every official shard name that is missing
   to an existing *.safetensors file of exactly the official size (hash is still
   verified by `model_matrix seal`).

Prints the pinned path on success. Exit 1 with the scanned locations otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from model_matrix import PINNED_OFFICIAL_FILES, _load_specs


def _official(spec: dict) -> dict:
    return spec.get("official_files") or PINNED_OFFICIAL_FILES.get(
        (spec["repository"], spec["revision"]), {}
    )


def _config_matches(config_path: Path, spec: dict, official: dict) -> bool:
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if document.get("model_type") != spec.get("model_type"):
        return False
    expected = official.get("config.json", {}).get("size")
    return expected is None or config_path.stat().st_size == expected


def discover(models_dir: Path, spec: dict) -> tuple[Path | None, list[Path]]:
    official = _official(spec)
    standard = models_dir / spec["local_directory"]
    if (standard / "config.json").is_file():
        return standard, [standard]
    scanned: list[Path] = []
    matches: list[Path] = []
    for pattern in ("*/config.json", "*/*/config.json"):
        for config_path in sorted(models_dir.glob(pattern)):
            directory = config_path.parent
            scanned.append(directory)
            if _config_matches(config_path, spec, official):
                matches.append(directory.resolve())
    unique = sorted(set(matches))
    if len(unique) != 1:
        return None, scanned
    return unique[0], scanned


def link_missing_shards(directory: Path, official: dict) -> list[str]:
    """Link official shard names to same-size uploaded files; return actions."""
    actions: list[str] = []
    by_size: dict[int, list[Path]] = {}
    for candidate in directory.glob("*.safetensors"):
        if candidate.is_file():
            by_size.setdefault(candidate.stat().st_size, []).append(candidate)
    for name, record in official.items():
        if not name.endswith(".safetensors"):
            continue
        target = directory / name
        if target.exists():
            continue
        same_size = [
            path for path in by_size.get(int(record["size"]), [])
            if path.name not in official
        ]
        if len(same_size) != 1:
            actions.append(f"missing {name} (no unique {record['size']}-byte file)")
            continue
        os.symlink(same_size[0].name, target)
        actions.append(f"linked {name} -> {same_size[0].name}")
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = _load_specs(args.config)[args.model_key]
    official = _official(spec)
    standard = args.models_dir / spec["local_directory"]
    found, scanned = discover(args.models_dir, spec)
    if found is None:
        print(
            f"[locate-abort] {spec['repository']} 스냅샷을 {args.models_dir} 아래에서 못 찾음 "
            f"(config.json의 model_type={spec.get('model_type')!r}로 식별). 살펴본 폴더: "
            + (", ".join(str(p) for p in scanned) or "없음"),
            file=sys.stderr,
        )
        return 1
    if found != standard.resolve() and not standard.exists():
        os.symlink(found, standard)
        print(f"[locate] {standard} -> {found} (symlink)", file=sys.stderr)
    for action in link_missing_shards(standard.resolve(), official):
        print(f"[locate] {action}", file=sys.stderr)
    print(standard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
