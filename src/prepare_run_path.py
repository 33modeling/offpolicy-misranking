#!/usr/bin/env python3
"""Preserve incompatible run artifacts before reusing a canonical run path."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


def prepare_run_path(
    run: Path,
    expected_git: str,
    quarantine_root: Path,
    quarantine_unconfigured: bool = False,
) -> Path | None:
    config_path = run / "run_config.json"
    if not config_path.exists():
        if not quarantine_unconfigured or not run.is_dir() or not any(run.iterdir()):
            return None
        recorded_git = "unconfigured"
    else:
        try:
            config = json.loads(config_path.read_text())
            recorded_git = str(config.get("git") or "missing")
        except (OSError, ValueError, TypeError):
            recorded_git = "unreadable"

    if recorded_git == expected_git:
        return None

    quarantine_root.mkdir(parents=True, exist_ok=True)
    git_tag = re.sub(r"[^a-zA-Z0-9_.-]", "-", recorded_git[:12]) or "missing"
    stamp = time.strftime("%Y%m%dT%H%M%S")
    destination = quarantine_root / f"{run.name}-git-{git_tag}-{stamp}"
    counter = 1
    while destination.exists():
        destination = quarantine_root / f"{run.name}-git-{git_tag}-{stamp}-{counter}"
        counter += 1
    run.rename(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--expected-git", required=True)
    parser.add_argument("--quarantine-root", required=True, type=Path)
    parser.add_argument(
        "--quarantine-unconfigured",
        action="store_true",
        help="also preserve a non-empty run directory without run_config.json",
    )
    args = parser.parse_args()

    destination = prepare_run_path(
        args.run,
        args.expected_git,
        args.quarantine_root,
        quarantine_unconfigured=args.quarantine_unconfigured,
    )
    if destination is not None:
        print(f"[run-path] previous artifacts preserved at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
