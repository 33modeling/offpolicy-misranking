#!/usr/bin/env python3
"""Read-only discovery of a hand-uploaded pinned model snapshot.

Folder names are arbitrary. Candidates must match the pinned config content
(or exact Hub config metadata for models without embedded records). Full weight
and tokenizer validation is performed by model_matrix check/seal afterwards.
No directory rename, symlink creation, or index repair is performed by the CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from model_matrix import PINNED_OFFICIAL_FILES, _load_specs, _git_blob_sha1, _sha256


def _official(spec: dict) -> dict:
    return spec.get("official_files") or PINNED_OFFICIAL_FILES.get(
        (spec["repository"], spec["revision"]), {}
    )


def _config_matches(config_path: Path, spec: dict, official: dict) -> bool:
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(document, dict) or document.get("model_type") != spec.get("model_type"):
        return False
    expected = official.get("config.json")
    if expected:
        if config_path.stat().st_size != expected["size"]:
            return False
        return (_sha256(config_path) == expected["sha256"] if "sha256" in expected
                else _git_blob_sha1(config_path) == expected["git_blob_sha1"])
    metadata = config_path.parent / ".cache/huggingface/download/config.json.metadata"
    try:
        revision, etag, *_ = metadata.read_text().splitlines()
        return revision == spec["revision"] and etag in {
            _sha256(config_path), _git_blob_sha1(config_path)}
    except (OSError, ValueError):
        return False


def search_roots(models_dir: Path) -> list[Path]:
    """Every place people actually drop model folders on the shared volume."""
    roots = [models_dir]
    work = os.environ.get("OM_WORK")
    volume = os.environ.get("GROUP_VOLUME")
    user = os.environ.get("OM_USER")
    if work:
        roots.append(Path(work) / "models")
    if volume:
        roots.append(Path(volume) / "models")
        if user:
            roots.append(Path(volume) / user / "models")
    unique: list[Path] = []
    for root in roots:
        if root.is_dir() and root.resolve() not in [u.resolve() for u in unique]:
            unique.append(root)
    return unique


def _has_weights(directory: Path) -> bool:
    return any(
        path.stat().st_size > 1_000_000
        for path in directory.glob("*.safetensors")
        if path.is_file()
    )


SKIP_DIRS = {".downloads", ".quarantine", ".cache", ".git", "quarantine"}


def _iter_files(root: Path, name_or_suffix: str, limit: int = 40000):
    """Walk like the dataset loader does: any depth, skipping bookkeeping dirs."""
    import os as _os

    seen = 0
    visited = set()
    for current, directories, files in _os.walk(root, followlinks=True):
        identity = Path(current).resolve()
        if identity in visited:
            directories[:] = []
            continue
        visited.add(identity)
        directories[:] = [
            d for d in directories
            if d not in SKIP_DIRS and not d.startswith(".stale-")
        ]
        for filename in files:
            seen += 1
            if seen > limit:
                return
            if filename == name_or_suffix or (
                name_or_suffix.startswith(".") and filename.endswith(name_or_suffix)
            ):
                yield Path(current) / filename


def _weights_dir(directory: Path) -> Path:
    """Where the shards really are: this directory, or anywhere below it."""
    if _has_weights(directory):
        return directory
    for path in _iter_files(directory, ".safetensors"):
        try:
            if path.is_file() and path.stat().st_size > 1_000_000:
                return path.parent
        except OSError:
            continue
    return directory


HEAVY_DIRS = {"runs", "results", "quarantine", "console-logs", "cache", "datasets",
              "runtime-deps", "tmp", "pools", "contracts", "readouts", "locks"}


def _directories_named(root: Path, needle: str, max_depth: int = 4) -> list[Path]:
    """Directory-name search across the volume (cheap: never lists files)."""
    import os as _os

    hits: list[Path] = []
    root = root.resolve()
    base_depth = len(root.parts)
    for current, directories, _ in _os.walk(root, followlinks=False):
        depth = len(Path(current).parts) - base_depth
        if depth >= max_depth:
            directories[:] = []
            continue
        directories[:] = [
            d for d in directories
            if d not in HEAVY_DIRS and d not in SKIP_DIRS and not d.startswith(".")
        ]
        for name in list(directories):
            if needle in name.lower():
                hits.append(Path(current) / name)
    return hits


def _type_matches(config_path: Path, spec: dict) -> bool:
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return document.get("model_type") == spec.get("model_type")


def discover(models_dir: Path, spec: dict) -> tuple[Path | None, list[Path]]:
    """Return (directory, scanned). Explicit OM_SNAPSHOT_PATH wins; then the pinned
    directory; then any config.json with the right model_type under the search
    roots (three levels), disambiguated by official config/index sizes."""
    official = _official(spec)
    explicit = os.environ.get("OM_SNAPSHOT_PATH")
    if explicit:
        path = Path(explicit)
        valid = _config_matches(path / "config.json", spec, official) and _has_weights(path)
        return (path.resolve() if valid else None), [path]
    standard = models_dir / spec["local_directory"]
    if (_config_matches(standard / "config.json", spec, official)
            and _has_weights(standard)):
        return standard, [standard]
    scanned = []
    matches = []
    roots = search_roots(models_dir)
    volume = os.environ.get("GROUP_VOLUME")
    if volume and Path(volume).is_dir():
        roots += _directories_named(Path(volume), spec["repository"].split("/")[-1].lower())
    for root in roots:
        for config_path in _iter_files(root, "config.json"):
            directory = config_path.parent.resolve()
            if directory in scanned:
                continue
            scanned.append(directory)
            if _config_matches(config_path, spec, official) and _has_weights(directory):
                matches.append(directory)
    # Never choose by model_type, folder name, file size alone, or traversal order.
    return (matches[0] if len(matches) == 1 else None), scanned


def describe(directory: Path, official: dict) -> list[str]:
    """Human-readable comparison of what is on disk vs the pinned official files."""
    lines = [f"folder {directory}:"]
    present = {}
    for entry in directory.iterdir():
        try:
            if entry.is_file() or entry.is_symlink():
                present[entry.name] = entry.stat().st_size
        except OSError:
            continue
    shards = sorted(name for name in present if name.endswith(".safetensors"))
    lines.append(
        f"  {len(shards)} safetensors file(s), "
        f"{sum(present[n] for n in shards) / 1e9:.1f} GB"
    )
    for name, record in sorted(official.items()):
        size = present.get(name)
        if size is None:
            same = [n for n, s in present.items() if s == record["size"] and n.endswith(".safetensors")]
            hint = f" (same-size file present: {same[0]})" if same else ""
            lines.append(f"  MISSING  {name}  expected {record['size']:,} B{hint}")
        elif size != record["size"]:
            lines.append(f"  SIZE?    {name}  {size:,} B != expected {record['size']:,} B  (other revision or truncated)")
    extras = sorted(n for n in present if n.endswith(".safetensors") and n not in official)
    if extras:
        lines.append("  unregistered safetensors: " + ", ".join(f"{n} ({present[n]:,} B)" for n in extras))
    if len(lines) == 1:
        lines.append("  all official files present with matching sizes")
    return lines


def _safetensors_tensor_names(path: Path) -> list[str] | None:
    """Tensor names from a safetensors header (8-byte LE length + JSON)."""
    import struct
    try:
        with path.open("rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            if not 0 < length < 200_000_000:
                return None
            header = json.loads(stream.read(length).decode("utf-8"))
    except (OSError, ValueError, struct.error):
        return None
    return [name for name in header if name != "__metadata__"]


def ensure_index(directory: Path) -> list[str]:
    """Make model.safetensors.index.json point at the shard files actually present.

    Uploads often carry shards under other names than the index expects (or no
    index at all). Reading each shard's header gives its tensor names, so the
    weight_map can be rebuilt exactly; nothing is guessed from sizes. The previous
    index is kept as model.safetensors.index.json.orig.
    """
    actions: list[str] = []
    index_path = directory / "model.safetensors.index.json"
    shards = sorted(
        path for path in directory.glob("*.safetensors")
        if path.is_file() and not path.is_symlink()
    )
    if not shards:
        return ["no *.safetensors file in the directory"]
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            referenced = set(index.get("weight_map", {}).values())
        except (OSError, ValueError):
            referenced = set()
        if referenced and all((directory / name).exists() for name in referenced):
            return actions
    weight_map: dict[str, str] = {}
    for shard in shards:
        names = _safetensors_tensor_names(shard)
        if names is None:
            return [f"unreadable safetensors header: {shard.name}"]
        for name in names:
            weight_map[name] = shard.name
    if not weight_map:
        return ["shards contain no tensors"]
    if index_path.exists():
        backup = index_path.with_suffix(index_path.suffix + ".orig")
        if not backup.exists():
            index_path.replace(backup)
            actions.append(f"kept old index as {backup.name}")
    total = sum(shard.stat().st_size for shard in shards)
    index_path.write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    actions.append(
        f"rebuilt {index_path.name} from {len(shards)} shard header(s): "
        + ", ".join(shard.name for shard in shards)
    )
    return actions


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
    print(
        "[locate] roots: " + ", ".join(str(r) for r in search_roots(args.models_dir))
        + (f" (+ volume scan of {os.environ['GROUP_VOLUME']})" if os.environ.get("GROUP_VOLUME") else ""),
        file=sys.stderr,
    )
    if found is None:
        roots = ", ".join(str(r) for r in search_roots(args.models_dir))
        print(
            f"[locate-abort] no snapshot with weights for {spec['repository']} under {roots} "
            f"(config.json model_type={spec.get('model_type')!r}, 3 levels deep). "
            f"folders with config.json: " + (", ".join(str(p) for p in scanned) or "none")
            + ". Set OM_SNAPSHOT_PATH=<folder> if it lives elsewhere.",
            file=sys.stderr,
        )
        return 1
    # Discovery/doctor are read-only. No rename, symlink or index reconstruction.
    # If upload names differ, prepare the exact pinned layout separately.
    for line in describe(found, official):
        print(f"[locate] {line}", file=sys.stderr)
    print(found)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
