"""Choose the immutable generation commit for a partial regime matrix."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import tempfile
from pathlib import Path

COMMIT_RE = re.compile(r"[0-9a-f]{40,64}")


def validate_commit(commit: object, source: Path | str) -> str:
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise ValueError(f"invalid generation commit in {source}: {commit!r}")
    return commit


def choose_generation_commit(root: Path, current: str) -> str:
    current = validate_commit(current, "current checkout")
    commits: set[str] = set()
    for config_path in sorted(root.glob("*/run_config.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid run config: {config_path}: {exc}") from exc
        commits.add(validate_commit(config.get("git"), config_path))
    if len(commits) > 1:
        raise ValueError(
            "mixed generation commits in one regime matrix: "
            + ", ".join(sorted(commits))
        )
    return next(iter(commits), current)


def bind_generation_commit(root: Path, marker: Path, current: str) -> str:
    """Atomically bind a shared matrix to one immutable generation commit."""
    current = validate_commit(current, "current checkout")
    marker.parent.mkdir(parents=True, exist_ok=True)
    lock_path = marker.with_name(f"{marker.name}.lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        detected = choose_generation_commit(root, current)
        has_configs = any(root.glob("*/run_config.json"))

        if marker.is_symlink():
            raise ValueError(f"generation marker must not be a symlink: {marker}")
        if marker.exists():
            try:
                lines = marker.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                raise ValueError(f"cannot read generation marker: {marker}: {exc}") from exc
            if len(lines) != 1:
                raise ValueError(f"invalid generation marker: {marker}")
            recorded = validate_commit(lines[0], marker)
            if has_configs and recorded != detected:
                raise ValueError(
                    f"run configs use {detected} but generation marker uses {recorded}"
                )
            return recorded

        fd, temporary_name = tempfile.mkstemp(
            dir=marker.parent, prefix=f".{marker.name}.", text=True
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(f"{detected}\n")
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(marker)
        finally:
            temporary.unlink(missing_ok=True)
        return detected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("current")
    parser.add_argument("--marker", type=Path)
    args = parser.parse_args()
    try:
        if args.marker is None:
            commit = choose_generation_commit(args.root, args.current)
        else:
            commit = bind_generation_commit(args.root, args.marker, args.current)
        print(commit)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
