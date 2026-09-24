"""Find byte-identical resume artifacts without modifying source experiments."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import selection_gate as core
import selection_gate_gpu as base


class ResumeArtifactsUnavailable(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"seed={report['seed']} step={report['step']}: "
                         f"matching {', '.join(report['missing'])} not found; "
                         "no optimizer reset or future-step substitution was performed")


def artifact_bytes(descriptor):
    path = Path(descriptor["path"])
    if "through_step" not in descriptor:
        return path.read_bytes()
    retained, steps = [], []
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                retained.append(line)
                continue
            row = json.loads(line)
            if row["step"] > descriptor["through_step"]:
                break
            steps.append(row["step"])
            retained.append(line)
    if steps != list(range(descriptor["start_step"]+1, descriptor["through_step"]+1)):
        raise ValueError(f"incomplete checkpoint statistics prefix: {path}")
    return b"".join(retained)


def artifact_digest(descriptor):
    if "through_step" not in descriptor:
        return base.digest(Path(descriptor["path"]))
    return hashlib.sha256(artifact_bytes(descriptor)).hexdigest()


def search_roots(root, policy):
    # Prefer the source branch, then other saved runs/backups under the same work
    # area. An optional explicit root covers copies on another mounted volume.
    candidates = [policy, root, root.parent, root.parent.parent / "checkpoints"]
    candidates.extend(Path(value).expanduser() for value in
                      os.environ.get("PAIR_CHECKPOINT_SEARCH_ROOTS", "").split(os.pathsep) if value)
    return list(dict.fromkeys(path.resolve() for path in candidates if path.is_dir()))


def optimizer_candidates(roots, errors):
    visited = set()
    for root in roots:
        def error(exc):
            errors.append(str(exc))
        for directory, subdirs, names in os.walk(root, onerror=error, followlinks=True):
            path = Path(directory)
            resolved = path.resolve()
            if resolved in visited:
                subdirs[:] = []
                continue
            visited.add(resolved)
            subdirs[:] = [name for name in subdirs if name not in {".git", "__pycache__", "node_modules"}
                          and not name.startswith(".venv")]
            if "optimizer.pt" in names:
                yield path / "optimizer.pt"


def find_resume_artifacts(root, checkpoint, policy, state):
    artifacts = {name: {"path": str(checkpoint / name)}
                 for name in ("adapter_config.json", "adapter_model.safetensors")}
    roots = search_roots(root, policy)
    report = {"seed": state["seed"], "step": state["completed_steps"], "checkpoint": str(checkpoint),
              "optimizer_sha256": state["optimizer_sha256"], "search_roots": list(map(str, roots)),
              "candidates": [], "search_errors": [], "missing": []}
    direct = checkpoint / "optimizer.pt"
    if direct.is_file():
        if base.digest(direct) != state["optimizer_sha256"]:
            raise ValueError(f"saved optimizer hash mismatch: {direct}; preserve and inspect")
        artifacts["optimizer.pt"] = {"path": str(direct)}
    else:
        print(f"[switch] seed={state['seed']} step={state['completed_steps']} "
              "searching saved optimizer locations (no GPU work)", flush=True)
        for candidate in optimizer_candidates(roots, report["search_errors"]):
            record = {"path": str(candidate)}
            metadata = next((candidate.parent / name for name in ("checkpoint_state.json", "policy_train.json")
                             if (candidate.parent / name).is_file()), None)
            if metadata:
                try:
                    saved = core.read(metadata)
                except (OSError, ValueError) as exc:
                    report["candidates"].append({**record, "status": "invalid_metadata", "error": str(exc)})
                    continue
                record.update(seed=saved.get("seed"), step=saved.get("completed_steps"))
                fields = ("seed", "completed_steps", "start_step", "training_objective", "world_size",
                          "prompts_sha256", "adapter_sha256", "optimizer_sha256")
                mismatches = [key for key in fields if saved.get(key) != state[key]]
                if mismatches:
                    report["candidates"].append({**record, "status": "different_checkpoint", "fields": mismatches})
                    continue
            try:
                digest = base.digest(candidate)
            except OSError as exc:
                report["candidates"].append({**record, "status": "unreadable", "error": str(exc)})
                continue
            if digest != state["optimizer_sha256"]:
                report["candidates"].append({**record, "status": "hash_mismatch"})
                continue
            report["candidates"].append({**record, "status": "verified_exact_match"})
            artifacts["optimizer.pt"] = {"path": str(candidate)}
            print(f"[switch] seed={state['seed']} step={state['completed_steps']} optimizer={candidate}", flush=True)
            break
    # Archived adapters often omit the statistics too. Recover only the exact
    # bytes saved at this step, not later training rows or estimated statistics.
    stat_paths = [checkpoint / "grpo_stats.jsonl", policy / "grpo_stats.jsonl"]
    if "optimizer.pt" in artifacts:
        stat_paths.append(Path(artifacts["optimizer.pt"]["path"]).parent / "grpo_stats.jsonl")
    for path in dict.fromkeys(stat_paths):
        if not path.is_file():
            continue
        descriptor = {"path": str(path)}
        if base.digest(path) != state["grpo_stats_sha256"]:
            descriptor.update(start_step=state["start_step"], through_step=state["completed_steps"])
        try:
            if artifact_digest(descriptor) == state["grpo_stats_sha256"]:
                artifacts["grpo_stats.jsonl"] = descriptor
                break
        except (OSError, ValueError, KeyError):
            continue
    report["missing"] = [name for name in ("optimizer.pt", "grpo_stats.jsonl") if name not in artifacts]
    report["artifacts"] = artifacts
    if report["missing"]:
        raise ResumeArtifactsUnavailable(report)
    return artifacts, report
