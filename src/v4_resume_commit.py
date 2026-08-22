"""Select the single immutable generation commit recorded by existing v4 runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def recorded_commits(runs_root: Path) -> dict[str, list[str]]:
    commits: dict[str, list[str]] = {}
    for pattern in ("v4-27b-s*/run_config.json", "v4-7b-s*/run_config.json"):
        for path in sorted(runs_root.glob(pattern)):
            if "smoke" in path.parent.name:
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                commit = str(value["git"])
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"unreadable run config: {path}: {exc}") from exc
            if not commit or commit == "?":
                raise ValueError(f"missing generation commit: {path}")
            commits.setdefault(commit, []).append(path.parent.name)
    return commits


def select_resume_commit(runs_root: Path, current: str) -> str:
    commits = recorded_commits(runs_root)
    if not commits:
        return current
    if len(commits) != 1:
        detail = "; ".join(
            f"{commit[:12]}=[{','.join(names)}]"
            for commit, names in sorted(commits.items())
        )
        raise ValueError(f"mixed v4 generation commits: {detail}")
    return next(iter(commits))


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: v4_resume_commit.py RUNS_ROOT CURRENT_GIT", file=sys.stderr)
        return 2
    try:
        print(select_resume_commit(Path(sys.argv[1]), sys.argv[2]))
    except ValueError as exc:
        print(f"[resume-v4-abort] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
