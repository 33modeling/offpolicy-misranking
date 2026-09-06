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
import time
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
        return (path.resolve() if (path / "config.json").is_file() else None), [path]
    standard = models_dir / spec["local_directory"]
    if (standard / "config.json").is_file():
        if _has_weights(standard):
            return standard, [standard]
        # A weightless pinned directory (left by the old prepare that downloaded
        # only config/tokenizer/index) must not shadow the real upload next to it.
        stale = standard.with_name(f".stale-{standard.name}-{int(time.time())}")
        standard.rename(stale)
        print(f"[locate] {standard} had no weights; moved aside to {stale.name}", file=sys.stderr)
    scanned: list[Path] = []
    by_type: list[Path] = []
    for root in search_roots(models_dir):
        for pattern in ("config.json", "*/config.json", "*/*/config.json", "*/*/*/config.json"):
            for config_path in sorted(root.glob(pattern)):
                directory = config_path.parent.resolve()
                if directory in scanned:
                    continue
                scanned.append(directory)
                if _type_matches(config_path, spec) and _has_weights(directory):
                    by_type.append(directory)
    if len(by_type) == 1:
        return by_type[0], scanned
    if not by_type:
        return None, scanned
    # Several Qwen3.x uploads share model_type=qwen3_5 (9B and 27B). The folder is
    # almost always named after the repository ("Qwen3.5-9B", "Qwen3.5-9B-pinned").
    base = spec["repository"].split("/")[-1].lower()
    named = [d for d in by_type if base in d.name.lower()]
    if len(named) == 1:
        return named[0], scanned
    exact = [d for d in (named or by_type) if _config_matches(d / "config.json", spec, official)]
    if len(exact) == 1:
        return exact[0], scanned
    index_size = official.get("model.safetensors.index.json", {}).get("size")
    by_index = [
        d for d in (exact or by_type)
        if index_size and (d / "model.safetensors.index.json").is_file()
        and (d / "model.safetensors.index.json").stat().st_size == index_size
    ]
    if len(by_index) == 1:
        return by_index[0], scanned
    return None, scanned


def describe(directory: Path, official: dict) -> list[str]:
    """Human-readable comparison of what is on disk vs the pinned official files."""
    lines = [f"폴더 {directory}:"]
    present = {p.name: p.stat().st_size for p in directory.iterdir() if p.is_file() or p.is_symlink()}
    for name, record in sorted(official.items()):
        size = present.get(name)
        if size is None:
            same = [n for n, s in present.items() if s == record["size"] and n.endswith(".safetensors")]
            hint = f" (같은 크기 파일: {same[0]})" if same else ""
            lines.append(f"  없음  {name}  기대 {record['size']:,} B{hint}")
        elif size != record["size"]:
            lines.append(f"  크기≠  {name}  {size:,} B ≠ 기대 {record['size']:,} B  ← 다른 revision/잘린 파일")
    extras = sorted(n for n in present if n.endswith(".safetensors") and n not in official)
    if extras:
        lines.append("  등록 안 된 safetensors: " + ", ".join(f"{n} ({present[n]:,} B)" for n in extras))
    if len(lines) == 1:
        lines.append("  공식 파일 전부 존재, 크기 일치")
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
    if found is None:
        roots = ", ".join(str(r) for r in search_roots(args.models_dir))
        print(
            f"[locate-abort] {spec['repository']} 스냅샷 없음. 찾은 곳: {roots} "
            f"(config.json의 model_type={spec.get('model_type')!r} 기준, 3단계 깊이). "
            f"config.json이 있던 폴더: " + (", ".join(str(p) for p in scanned) or "없음")
            + ". 경로가 다르면 OM_SNAPSHOT_PATH=<폴더>로 지정.",
            file=sys.stderr,
        )
        return 1
    if found != standard.resolve() and not standard.exists():
        os.symlink(found, standard)
        print(f"[locate] {standard} -> {found} (symlink)", file=sys.stderr)
    target = standard.resolve() if standard.exists() else found
    for action in link_missing_shards(target, official):
        print(f"[locate] {action}", file=sys.stderr)
    for action in ensure_index(target):
        print(f"[locate] {action}", file=sys.stderr)
    for line in describe(target, official):
        print(f"[locate] {line}", file=sys.stderr)
    print(standard if standard.exists() else found)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
