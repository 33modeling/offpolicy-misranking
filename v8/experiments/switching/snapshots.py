"""Complete branch snapshots: model/adapter + optimizer + selector + controller + RNG.

Engine.state_dict() of the code repo already holds trainable weights, AdamW
moments/step, the selection block, used training IDs and the sampling cycle;
all training/selection randomness is derived from (seed, completed_update,
purpose) stream seeds, and rollout RNG is restored around every generation.
We add process RNG states (python/numpy/torch) for completeness and a
state-identity hash used to prove that sibling branches start identically.
"""

from __future__ import annotations

import hashlib
import io
import pickle
import random
from pathlib import Path

import numpy as np

from .common import atomic_write_bytes, sha256_file

try:  # optional
    import torch
except Exception:  # noqa: BLE001
    torch = None

IDENTITY_FIELDS = ("config", "step", "candidates", "validation", "random_ids", "sampling_protocol",
                   "sr_ranked_ids", "used_training_ids", "sampling_cycle", "active_selection")


def _frame(tag: bytes, payload: bytes) -> bytes:
    return tag + len(payload).to_bytes(8, "little") + payload


def canonical_bytes(value) -> bytes:
    """Typed, length-delimited serialization: distinct trees never share bytes."""
    if value is None:
        return _frame(b"n", b"")
    if isinstance(value, bool):
        return _frame(b"b", b"1" if value else b"0")
    if isinstance(value, int):
        return _frame(b"i", str(value).encode())
    if isinstance(value, float):
        return _frame(b"f", value.hex().encode())
    if isinstance(value, str):
        return _frame(b"s", value.encode())
    if isinstance(value, bytes):
        return _frame(b"y", value)
    if torch is not None and isinstance(value, torch.Tensor):
        t = value.detach().cpu().contiguous()
        meta = canonical_bytes([str(t.dtype), list(t.shape)])
        if t.dtype == torch.bfloat16:
            t = t.view(torch.int16)
        return _frame(b"T", meta + _frame(b"r", t.numpy().tobytes()))
    if isinstance(value, np.ndarray):
        meta = canonical_bytes([str(value.dtype), list(value.shape)])
        if value.dtype == object:
            return _frame(b"A", meta + canonical_bytes(value.tolist()))
        return _frame(b"a", meta + _frame(b"r", np.ascontiguousarray(value).tobytes()))
    if isinstance(value, np.generic):
        return _frame(b"g", canonical_bytes(str(value.dtype)) + canonical_bytes(value.item()))
    if isinstance(value, dict):
        items = sorted((canonical_bytes(k), canonical_bytes(v)) for k, v in value.items())
        return _frame(b"d", len(items).to_bytes(8, "little") + b"".join(k + v for k, v in items))
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = sorted(value, key=canonical_bytes) if isinstance(value, (set, frozenset)) else value
        tag = {list: b"l", tuple: b"t"}.get(type(value), b"S")
        parts = [canonical_bytes(v) for v in seq]
        return _frame(tag, len(parts).to_bytes(8, "little") + b"".join(parts))
    return _frame(b"o", canonical_bytes(type(value).__qualname__) + canonical_bytes(repr(value)))


def tree_hash(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _strip_timing(value):
    """Wall/GPU timings are measurements, not state: drop keys ending in _seconds and saved_at."""
    if isinstance(value, dict):
        return {k: _strip_timing(v) for k, v in value.items()
                if not (isinstance(k, str) and (k.endswith("_seconds") or k in ("saved_at", "costs")))}
    if isinstance(value, (list, tuple)):
        return type(value)(_strip_timing(v) for v in value)
    return value


def state_identity(state: dict) -> dict:
    """Learner + selector identity: what sibling branches (different policies) must share."""
    backend = state["backend"]
    model = {k: v for k, v in backend.items() if k not in ("optimizer", "m", "v", "updates")}
    optimizer = backend.get("optimizer", {k: backend[k] for k in ("m", "v", "updates") if k in backend})
    selector = {k: state.get(k) for k in IDENTITY_FIELDS}
    out = {"model_hash": tree_hash(model), "optimizer_hash": tree_hash(optimizer), "selector_hash": tree_hash(selector)}
    out["state_hash"] = tree_hash(out)
    return out


def branch_identity(payload: dict) -> dict:
    """Training-state equivalence for siblings: learner/selector + process RNG (controller may differ)."""
    engine = payload["engine"] if "engine" in payload else payload
    out = dict(state_identity(engine))
    out["rng_hash"] = tree_hash(payload.get("rng")) if "engine" in payload else None
    out["branch_hash"] = tree_hash([out["state_hash"], out["rng_hash"]])
    return out


def payload_identity(payload: dict) -> str:
    """Complete snapshot identity: every persisted resume-affecting field (engine incl. controller,
    v8 policy/cache state, RNG, ledger pointers); only timing measurements are excluded."""
    return tree_hash(_strip_timing(payload))


def capture_rng() -> dict:
    rng = {"python": random.getstate(), "numpy": np.random.get_state()}
    if torch is not None:
        rng["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            rng["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng


def restore_rng(rng: dict) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    if torch is not None and "torch_cpu" in rng:
        torch.set_rng_state(rng["torch_cpu"])
        if "torch_cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])


def dumps(obj) -> bytes:
    buf = io.BytesIO()
    if torch is not None:
        torch.save(obj, buf)
    else:
        pickle.dump(obj, buf, protocol=4)
    return buf.getvalue()


def loads(data: bytes):
    if torch is not None:
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    return pickle.loads(data)


def save(path, payload: dict, *, immutable: bool) -> str:
    """Write a snapshot; immutable (retained) snapshots may never change content."""
    path = Path(path)
    data = dumps(payload)
    if immutable and path.exists():
        existing = loads(path.read_bytes())
        if payload_identity(existing) != payload_identity(payload):
            raise ValueError(f"retained snapshot already exists with different persisted state: {path}")
        return sha256_file(path)
    atomic_write_bytes(path, data)
    return sha256_file(path)


def load(path, expected_sha256=None) -> dict:
    path = Path(path)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"snapshot hash mismatch: {path}")
    return loads(path.read_bytes())


def load_engine_state(path, expected_sha256=None) -> dict:
    """Accept a v8 snapshot payload or a raw code-repo checkpoint (prefix.pt)."""
    payload = load(path, expected_sha256)
    return payload["engine"] if isinstance(payload, dict) and "engine" in payload else payload
