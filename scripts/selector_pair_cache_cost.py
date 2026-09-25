"""Recover cache-generation allocation from the original, hash-matched stage log."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from cost_accounting import parse_progress


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def cache_creation_cost(contract):
    result = {"gpu_seconds": None, "status": "unknown", "source_trace": [],
              "scope": "Original behavior-rollout stage wall time times its logged GPU count; "
              "includes generation, reward verification and stage overhead, at one-second log resolution. "
              "Not the cost of reading cached rewards. No timestamp-gap or per-prompt extrapolation."}
    try:
        expected = contract["source_hashes"]["rollouts_behavior_train.jsonl"]
        source = Path(contract["source_run"]).resolve()
        visited = set()
        while source not in visited:
            visited.add(source)
            result["source_run"] = str(source)
            cache_path = source / "rollouts_behavior_train.jsonl"
            if digest(cache_path) != expected:
                raise ValueError("cache content differs from the experiment contract")
            # Selected-prefix views link the cache file, not the source directory.
            origin = cache_path.resolve(strict=True).parent
            if origin != source:
                result["source_trace"].append({"from": str(source), "to": str(origin), "via": "cache_symlink"})
                source = origin
                continue
            config = json.loads((source / "run_config.json").read_text())
            if not config.get("behavior_source"):
                break
            parent = Path(config["behavior_source"])
            if not parent.is_absolute():
                raise ValueError("relative cache source cannot be resolved unambiguously")
            result["source_trace"].append({"from": str(source), "to": str(parent.resolve()),
                                           "via": "behavior_source"})
            source = parent.resolve()
        else:
            raise ValueError("cycle in behavior-cache provenance")
        log = source / "logs/main.log"
        result["log_path"] = str(log)
        text = log.read_text()
        events = parse_progress(text)
        result["progress_records"] = [{**event, "time": event["time"].isoformat()} for event in events]
        starts = [i for i, event in enumerate(events) if event["stage"] == 2
                  and re.fullmatch(r"behavior-rollout \d+x\d+ on \d+ GPUs", event["label"])]
        if len(starts) != 1:
            raise ValueError("one unambiguous generation attempt required; missing or retried stages are unknown")
        index = starts[0]
        start = events[index]
        if index + 1 >= len(events) or events[index + 1]["stage"] != 3:
            raise ValueError("cache generation has no consecutive stage-end record")
        end = events[index + 1]
        seconds = (end["time"] - start["time"]).total_seconds()
        if seconds <= 0:
            raise ValueError("invalid cache-generation timestamps")
        match = re.fullmatch(r"behavior-rollout (\d+)x(\d+) on (\d+) GPUs", start["label"])
        prompts, responses, gpus = map(int, match.groups())
        if (prompts, responses) != (config["n_train"], config["behavior_k"]) or gpus <= 0:
            raise ValueError("cache generation budget differs from the source config")
        result.update(gpu_seconds=seconds * gpus, wall_seconds=seconds, gpus=gpus,
                      status="reconstructed_stage_allocation", source_run=str(source),
                      cache_sha256=expected, log_sha256=digest(log),
                      start=start["time"].isoformat(), end=end["time"].isoformat())
    except (OSError, KeyError, TypeError, ValueError) as exc:
        result["reason"] = str(exc)
    return result
