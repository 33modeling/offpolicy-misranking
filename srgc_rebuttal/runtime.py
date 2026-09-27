"""Shared-storage task identities, atomic receipts and POSIX work leases."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

from .plan import digest, input_path


class Busy(RuntimeError):
    pass


@contextmanager
def lease(path: Path, *, wait: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise Busy(str(path)) from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path: Path, value: dict):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def atomic_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def code_digest() -> str:
    value = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        value.update(path.name.encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def identity(plan_path: Path, plan: dict, seed: int) -> dict:
    return {"plan_sha256": digest(plan_path), "input_sha256": digest(input_path(plan_path, plan, seed)),
            "implementation_sha256": code_digest(), "seed": seed}


def run_root(plan_path: Path, plan: dict) -> Path:
    return (plan_path.parent / plan["output_root"]).resolve()


def matches(record: dict, expected: dict) -> bool:
    return all(record.get(k) == v for k, v in expected.items())


def prefix_ready(folder: Path, expected: dict, steps: int) -> bool:
    marker = folder / "prefix-ready.json"
    if not marker.exists():
        return False
    receipt = json.loads(marker.read_text())
    if not matches(receipt, expected) or receipt.get("completed_updates") != steps:
        raise ValueError("prefix belongs to a different experiment")
    checkpoint = folder / "prefix.pt"
    if not checkpoint.is_file() or digest(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("shared prefix checkpoint differs from its completion receipt")
    return True


def arm_complete(folder: Path, expected: dict, arm: str, updates: int) -> bool:
    path = folder / f"{arm}-endpoint.json"
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    if not matches(value, expected) or value.get("arm") != arm or value.get("total_updates") != updates:
        raise ValueError("completed arm belongs to a different experiment")
    return True


def finalize_seed(folder: Path, expected: dict, arms: list[str], updates: int) -> bool:
    """Repair a crash between the final atomic endpoint and the run receipt."""
    if not all(arm_complete(folder, expected, a, updates) for a in arms):
        return False
    with lease(folder / ".manifest.lock", wait=True):
        path = folder / "run.json"
        if not path.exists():
            raise ValueError("complete endpoints have no run manifest")
        marker = json.loads(path.read_text())
        if not matches(marker, expected):
            raise ValueError("run manifest belongs to a different experiment")
        if marker.get("status") != "complete":
            atomic_json(path, {**marker, "status": "complete"})
    return True
