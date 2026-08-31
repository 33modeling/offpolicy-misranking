"""Choose the immutable generation commit for a partial regime matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def choose_generation_commit(root: Path, current: str) -> str:
    commits: set[str] = set()
    for config_path in sorted(root.glob("*/run_config.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid run config: {config_path}: {exc}") from exc
        commit = config.get("git")
        if not isinstance(commit, str) or not commit:
            raise ValueError(f"generation commit missing: {config_path}")
        commits.add(commit)
    if len(commits) > 1:
        raise ValueError(
            "mixed generation commits in one regime matrix: "
            + ", ".join(sorted(commits))
        )
    return next(iter(commits), current)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("current")
    args = parser.parse_args()
    try:
        print(choose_generation_commit(args.root, args.current))
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
