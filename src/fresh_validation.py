"""Persist validation scheduling without changing the generation contract."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path


def validation_layout(run: Path, shards: int, prompts: int) -> dict:
    """Keep legacy partials serial and resume new shards with their original layout."""
    if shards < 1 or prompts < 1:
        raise ValueError("validation requires positive shard and prompt counts")
    path = run / ".fresh-val-layout.json"
    with path.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        legacy = any((run / name).exists() for name in (
            "rollouts_fresh_val.jsonl", "rollouts_fresh_val.partial",
            "rollouts_fresh_val.tmp", "rollouts_fresh_val.manifest.json",
            "rollouts_fresh_val.manifest.json.tmp",
        ))
        sharded = bool(list(run.glob("rollouts_fresh_val.shard*")))
        if legacy and sharded:
            raise ValueError("mixed serial/sharded validation artifacts; preserving both layouts")
        if path.exists():
            layout = json.loads(path.read_text())
        else:
            layout = {
                "schema": "offpolicy-fresh-val-layout/v1",
                "mode": "serial" if legacy or shards == 1 else "sharded",
                "shards": 1 if legacy or shards == 1 else shards,
                "prompts": prompts,
            }
        if (
            not isinstance(layout, dict)
            or layout.get("schema") != "offpolicy-fresh-val-layout/v1"
            or layout.get("mode") not in {"serial", "sharded"}
            or layout.get("prompts") != prompts
            or layout.get("shards") != (1 if layout.get("mode") == "serial" else shards)
            or (layout.get("mode") == "sharded" and legacy)
            or (layout.get("mode") == "serial" and sharded)
        ):
            raise ValueError(f"validation layout changed; preserving artifacts at {run}")
        if not path.exists():
            temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
            temporary.write_text(json.dumps(layout, sort_keys=True) + "\n")
            temporary.replace(path)
        return layout
