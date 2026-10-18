"""Source-bound inputs and immutable JSON artifacts for follow-up diagnostics."""

import csv
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

from srgc_research.rollout_distribution import integer


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    path = Path(path)
    payload = path.read_bytes()
    value = json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return value


def ids(values, name, count=None):
    if (not isinstance(values, list) or not values or
            any(not isinstance(i, str) or not i for i in values) or len(set(values)) != len(values) or
            (count is not None and len(values) != count)):
        raise ValueError(f"{name}: expected distinct nonempty string IDs, count={count}")
    return list(values)


def read_bound(path):
    before = file_hash(path)
    value = read(path)
    if file_hash(path) != before:
        raise ValueError("source JSON changed while reading")
    return value, before


def copy_bound(source, destination, expected):
    """Preserve the exact source inode; a moved latest must not change policy."""
    destination = Path(destination)
    if destination.exists():
        if file_hash(destination) != expected:
            raise ValueError("frozen diagnostic source checksum differs")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".source-", dir=destination.parent)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as handle, Path(source).open("rb") as original:
            while chunk := original.read(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if digest.hexdigest() != expected:
            raise ValueError("source changed since checkpoint preflight; use an archived checkpoint")
        Path(temporary).replace(destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def pool(data):
    """Read the actual pool size, including historical 512-prompt MBPP pools."""
    candidates = ids(data.get("candidate_ids"), "candidate pool")
    reference = ids(data.get("ranking_validation_ids"), "reference")
    evaluation = ids(data.get("evaluation_ids"), "held-out")
    validation = ids(data.get("validation_pool_ids", reference), "validation pool")
    if (set(candidates) & (set(validation) | set(evaluation)) or set(validation) & set(evaluation)
            or not set(reference) <= set(validation)):
        raise ValueError("candidate, reference and held-out splits overlap or reference is outside validation")
    cache = data.get("cached_rewards", {})
    if set(cache) != set(candidates):
        raise ValueError("complete initial candidate cache is required; selected prompts cannot define pool weights")
    counts = {}
    for prompt in candidates:
        rewards = cache[prompt]
        if (not isinstance(rewards, list) or len(rewards) != 8 or
                any(isinstance(r, bool) or not isinstance(r, (int, float)) or r not in (0, 1) for r in rewards)):
            raise ValueError("initial cache requires exactly eight binary rewards per candidate")
        counts[prompt] = int(sum(rewards))
    return candidates, reference, counts


def new_output(path, sources=()):
    path = Path(path).resolve()
    if any(Path(source).resolve().is_relative_to(path) for source in sources):
        raise ValueError("output must not contain source files")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write(path, value):
    """Atomic publication; reruns may verify an identical artifact, never replace it."""
    from srgc_rebuttal.runtime import atomic_json
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    if path.suffix == ".gz":
        from srgc_research.dispatch.rollout_records import publish
        publish(path, value)
    else:
        atomic_json(path, value)


def csv_file(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, allow_nan=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def seed_value(seed):
    return integer(seed, "seed")
