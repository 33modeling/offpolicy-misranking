"""Recover cache-generation allocation from the original, hash-matched stage log."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
from datetime import datetime
from pathlib import Path

from cost_accounting import parse_progress


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def legacy_generation_cost(text, source, config):
    """Account for complete GPU launch groups, retaining closed failed attempts."""
    pattern = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] GPU(\d+) "
                         r"([\u25b6\u2714\u2718]) (.*)$")
    batches, pending = [], {}
    current = None

    def finish():
        nonlocal current
        if current is None:
            return
        n = current["gpus"]
        if pending or {r["shard"] for r in current["workers"]} != set(range(n)):
            raise ValueError("legacy cache generation has missing shard starts or terminal records")
        workers = current["workers"]
        start, end = min(r["start"] for r in workers), max(r["end"] for r in workers)
        seconds = (end - start).total_seconds()
        if seconds <= 0:
            raise ValueError("invalid legacy cache-generation interval")
        batches.append({"gpus": n, "wall_seconds": seconds, "gpu_seconds": seconds * n,
                        "start": start.isoformat(), "end": end.isoformat(),
                        "successful": all(r["success"] for r in workers),
                        "workers": [{**r, "start": r["start"].isoformat(), "end": r["end"].isoformat()}
                                    for r in workers]})
        current = None

    def arguments(body):
        tokens = shlex.split(body)
        fields = {key: tokens[tokens.index(key) + 1]
                  for key in ("--run", "--n-train", "--behavior-k", "--shard")}
        shard, n = map(int, fields["--shard"].split(":"))
        if (Path(fields["--run"]).resolve() != source or
                int(fields["--n-train"]) != config["n_train"] or
                int(fields["--behavior-k"]) != config["behavior_k"] or
                n < 1 or not 0 <= shard < n):
            raise ValueError("legacy cache generation differs from the original source configuration")
        return shard, n

    for line in text.splitlines():
        if "=== RLVR point start:" in line:
            finish()
        match = pattern.fullmatch(line.strip())
        if match is None or not re.search(r"(?:^|\s)--stage rollout-behavior(?:\s|$)", match[4]):
            continue
        when, device, event, body = match.groups()
        when, device = datetime.strptime(when, "%Y-%m-%d %H:%M:%S"), int(device)
        if event == "\u25b6":
            if current and not pending and len(current["workers"]) == current["gpus"]:
                finish()
            shard, n = arguments(body)
            if current is None:
                current = {"gpus": n, "workers": []}
            if (device in pending or n != current["gpus"] or
                    any(r["shard"] == shard for r in current["workers"])):
                raise ValueError("overlapping or duplicate legacy cache shard starts")
            worker = {"device": device, "shard": shard, "start": when}
            current["workers"].append(worker)
            pending[device] = worker
        else:
            if device not in pending:
                raise ValueError("legacy cache terminal record has no matching start")
            worker = pending.pop(device)
            if when < worker["start"]:
                raise ValueError("legacy cache terminal record precedes its start")
            if event == "\u2714":
                elapsed = re.fullmatch(r"--stage rollout-behavior \((\d+)s\)", body)
                if elapsed is None:
                    raise ValueError("legacy cache completion has no elapsed timer")
                worker["elapsed_seconds"] = int(elapsed[1])
                if worker["elapsed_seconds"] + 2 < (when - worker["start"]).total_seconds():
                    raise ValueError("legacy cache timer and timestamps disagree")
            else:
                shard, n = arguments(body)
                if shard != worker["shard"] or n != current["gpus"] or not re.search(r"\brc=\d+\s*$", body):
                    raise ValueError("legacy cache failure differs from its start")
            worker.update(end=when, success=event == "\u2714")
    finish()
    if not batches or not batches[-1]["successful"]:
        raise ValueError("legacy cache log has no complete successful final generation group")
    return {"gpu_seconds": sum(b["gpu_seconds"] for b in batches), "attempts": batches,
            "status": "reconstructed_legacy_stage_allocation",
            "scope": "Sum of complete behavior-rollout launch groups: earliest logged GPU start "
            "to latest terminal record, multiplied by the group's logged shard/GPU count. "
            "Includes waiting for the slowest shard and closed failed attempts; excludes idle gaps, "
            "pre-launch verification and post-generation merge. One-second log resolution; "
            "not a file-mtime estimate or cached-subset read cost."}


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
        result.update(cache_sha256=expected, log_sha256=digest(log))
        result["raw_timing_records"] = [line for line in text.splitlines()
                                        if ("[progress]" in line or "rollout-behavior" in line
                                            or "RLVR point start:" in line)]
        if re.search(r"GPU\d+ \u25b6 --stage rollout-behavior\b", text):
            result.update(legacy_generation_cost(text, source, config))
            return result
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
    except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
        result["reason"] = str(exc)
    return result
