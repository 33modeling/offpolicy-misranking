"""Versioned SR success-rate caches with staging and atomic commit (E10).

Key = source checkpoint hash + pool hash + prompt/tokenizer hash + generation
spec + verifier hash + responses/prompt + generation RNG spec + scoring version.
A file name never implies reuse; only an identical key does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from .common import JsonlLog, atomic_write_json, read_json, sha256_file, sha256_json, utc_now

SCORING_VERSION = "srgc-cached-sr-abs-p-minus-half-v1"  # code repo cached_sr_set


def cache_key(*, source_checkpoint_hash, pool_ids, prompt_hash, generation_spec, verifier_hash,
              responses, rng_spec, scoring_version=SCORING_VERSION) -> dict:
    spec = {"source_checkpoint_hash": source_checkpoint_hash, "pool_hash": sha256_json(sorted(pool_ids)),
            "prompt_hash": prompt_hash, "generation_spec": generation_spec, "verifier_hash": verifier_hash,
            "responses": int(responses), "rng_spec": rng_spec, "scoring_version": scoring_version}
    return {"key": sha256_json(spec), "spec": spec}


def rewards_from_means(means: dict, responses: int) -> dict:
    out = {}
    for pid, m in means.items():
        k = int(round(float(m) * responses))
        if not 0 <= k <= responses or abs(k / responses - float(m)) > 1e-9:
            raise ValueError(f"{pid}: success rate is not a multiple of 1/{responses}")
        out[pid] = [1.0] * k + [0.0] * (responses - k)
    return out


class CacheStore:
    def __init__(self, root):
        self.root = Path(root)
        self.versions = JsonlLog(self.root / "cache_versions.jsonl", ("cache_id", "event"))

    def path(self, key):
        return self.root / "committed" / key

    def exists(self, key) -> bool:
        return (self.path(key) / "manifest.json").is_file()

    def commit(self, keyinfo: dict, rewards: dict, expected_ids, *, source_policy_update, build_event_ids=()):
        key = keyinfo["key"]
        if self.exists(key):
            return self.load(key)["manifest"]
        staging = self.root / "staging" / f"{key}.{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        expected = sorted(expected_ids)
        missing = [i for i in expected if i not in rewards]
        bad = [i for i in expected if i in rewards and (len(rewards[i]) != keyinfo["spec"]["responses"]
               or any(r not in (0, 1, 0.0, 1.0) for r in rewards[i]))]
        if missing or bad:
            self.versions.append({"cache_id": key, "event": f"rejected:{utc_now()}", "missing": len(missing),
                                  "invalid": len(bad)})
            shutil.rmtree(staging)
            raise ValueError(f"incomplete cache: missing={len(missing)} invalid={len(bad)}")
        atomic_write_json(staging / "rewards.json", {i: rewards[i] for i in expected})
        manifest = {"cache_id": key, "spec": keyinfo["spec"], "n_expected": len(expected),
                    "n_completed": len(expected), "n_failed": 0, "rewards_sha256": sha256_file(staging / "rewards.json"),
                    "source_policy_update": source_policy_update, "build_event_ids": list(build_event_ids),
                    "committed_at": utc_now()}
        atomic_write_json(staging / "manifest.json", manifest)
        final = self.path(key)
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(staging, final)  # atomic publish of a complete directory
        except OSError:
            if not self.exists(key):
                raise
            shutil.rmtree(staging, ignore_errors=True)
        self.versions.append({"cache_id": key, "event": "committed", **{k: manifest[k] for k in
                              ("source_policy_update", "n_expected", "n_completed", "n_failed")},
                              "source_checkpoint_hash": keyinfo["spec"]["source_checkpoint_hash"]})
        return manifest

    def load(self, key):
        d = self.path(key)
        manifest = read_json(d / "manifest.json")
        if sha256_file(d / "rewards.json") != manifest["rewards_sha256"]:
            raise ValueError(f"cache {key} checksum mismatch")
        return {"manifest": manifest, "rewards": read_json(d / "rewards.json")}

    def record_failure(self, key, step, error, attempt):
        self.versions.append({"cache_id": key, "event": f"failed:{step}:{attempt}", "error": str(error)[:500]})

    def record_use(self, key, run_id, first_use_update):
        self.versions.append({"cache_id": key, "event": f"first_use:{run_id}", "first_use_update": first_use_update})


def refresh_due(step, *, entry_update, interval, origin="sr_entry_update"):
    """origin=sr_entry_update: entry 25, K 50 -> 75, 125, ...; global_clock: multiples of K."""
    if interval is None or entry_update is None or step <= entry_update:
        return False
    if origin == "sr_entry_update":
        return (step - entry_update) % interval == 0
    if origin == "global_clock":
        return step % interval == 0
    raise ValueError("unknown refresh origin")
