"""Hashing, atomic files, append-only logs and locks shared by every module."""

from __future__ import annotations

from contextlib import contextmanager
import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import uuid


FSYNC = os.environ.get("V8_FSYNC", "1") != "0"  # durable by default; the CPU smoke disables it


def _sync(handle):
    handle.flush()
    if FSYNC:
        os.fsync(handle.fileno())


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_default)


def _default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value) -> str:
    return sha256_bytes(canonical_json(value).encode())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(value, n=12) -> str:
    return sha256_json(value)[:n]


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex


def atomic_write_bytes(path, data: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            _sync(handle)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_text(path, text: str):
    atomic_write_bytes(path, text.encode())


def atomic_write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_default) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_once_json(path, value):
    """Immutable artifact: identical rewrites are no-ops, different content is an error."""
    path = Path(path)
    if path.exists():
        if sha256_json(read_json(path)) != sha256_json(json.loads(canonical_json(value))):
            raise ValueError(f"immutable artifact already exists with different content: {path}")
        return False
    atomic_write_json(path, value)
    return True


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _split_lines(text):
    """(complete lines, unterminated tail or None). Appends always end in a newline, so a tail without
    one is a torn final write (crash or SIGKILL mid-append) or an append still in flight."""
    if not text or text.endswith("\n"):
        return text.splitlines(), None
    head, _, tail = text.rpartition("\n")
    return head.splitlines() if head else [], tail


def _tail_record(tail):
    """A complete JSON object whose newline was lost is still a record; a torn fragment is not."""
    if tail is None or not tail.strip():
        return None
    try:
        row = json.loads(tail)
    except json.JSONDecodeError:
        return None
    return row if isinstance(row, dict) else None


class JsonlLog:
    """Append-only JSONL with a uniqueness key.

    Re-sending a record with the same key and identical payload is a no-op;
    the same key with different content raises, because silently keeping either
    copy could hide a nondeterministic restart. A torn final line (process killed
    mid-append) is not a record: it is moved aside to <name>.torn-<id> before the
    next append, so one killed attempt never blocks every later resume. Damage
    before the last line still raises.
    """

    def __init__(self, path, key_fields):
        self.path = Path(path)
        self.key_fields = tuple(key_fields)
        self._seen = {}
        self._tail = None
        if self.path.exists():
            lines, tail = _split_lines(self.path.read_text())
            rows = [json.loads(line) for line in lines if line.strip()]
            record = _tail_record(tail)
            if record is not None:
                rows.append(record)
            if tail is not None:
                self._tail = record is not None
            for row in rows:
                self._seen[self._key(row)] = sha256_json(row)

    def _key(self, row):
        return tuple(json.dumps(row.get(k), sort_keys=True) for k in self.key_fields)

    def _repair_tail(self):
        complete, self._tail = self._tail, None
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        if complete:
            _append_bytes(self.path, b"\n")
            return
        cut = data.rfind(b"\n") + 1
        aside = self.path.with_name(f"{self.path.name}.torn-{uuid.uuid4().hex[:12]}")
        atomic_write_bytes(aside, data[cut:])
        os.truncate(self.path, cut)
        import sys
        print(f"[v8] {self.path}: torn final line ({len(data) - cut} bytes, killed mid-append) moved to {aside.name}",
              file=sys.stderr, flush=True)

    def append(self, row) -> bool:
        row = json.loads(canonical_json(row))
        key, digest = self._key(row), sha256_json(row)
        if key in self._seen:
            if self._seen[key] != digest:
                raise ValueError(f"{self.path}: conflicting record for key {key}")
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._tail is not None:
            self._repair_tail()
        _append_bytes(self.path, (canonical_json(row) + "\n").encode())
        self._seen[key] = digest
        return True

    def rows(self):
        return read_jsonl(self.path)


def _append_bytes(path, data: bytes):
    """One O_APPEND write of the complete line (no buffered partial writes), then fsync."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        if FSYNC:
            os.fsync(fd)
    finally:
        os.close(fd)


def read_jsonl(path):
    """Records of a JSONL file. A torn final line (killed mid-append, or an append in flight) is skipped;
    damage before the last line raises."""
    path = Path(path)
    if not path.exists():
        return []
    lines, tail = _split_lines(path.read_text())
    rows = [json.loads(line) for line in lines if line.strip()]
    record = _tail_record(tail)
    if record is not None:
        rows.append(record)
    return rows


class Busy(RuntimeError):
    pass


@contextmanager
def file_lock(path, wait=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise Busy(str(path)) from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def pp(a, b):
    """Paired effect in percentage points from [0,1] reward fractions."""
    return None if a is None or b is None else 100.0 * (float(a) - float(b))
