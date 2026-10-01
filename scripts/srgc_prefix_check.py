"""Bounded-memory prefix verification for extra arms; frozen core stays unchanged."""

import hashlib
import json
import os
import sys
import time

from srgc_rebuttal.runtime import matches


def verify_prefix(folder, expected, steps, *, verify_checkpoint=True):
    marker = folder / "prefix-ready.json"
    if not marker.is_file():
        raise ValueError(f"{folder}: the verified shared prefix must finish before this arm can start")
    receipt = json.loads(marker.read_text())
    if not isinstance(receipt, dict):
        raise ValueError(f"{folder}: shared prefix receipt is missing or not a JSON object")
    if not matches(receipt, expected) or receipt.get("completed_updates") != steps:
        raise ValueError(f"{folder}: prefix belongs to a different experiment")
    checkpoint = folder / "prefix.pt"
    if not checkpoint.is_file():
        raise ValueError(f"{folder}: shared prefix checkpoint differs from its completion receipt")
    # Non-primary ranks only check metadata before distributed startup. Rank 0
    # checks every byte before broadcasting permission to load the model.
    if not verify_checkpoint:
        return receipt["checkpoint_sha256"]
    started = reported = time.monotonic()
    total = checkpoint.stat().st_size
    print(f"PREFIX checking {checkpoint} bytes={total}", file=sys.stderr, flush=True)
    value = hashlib.sha256()
    read = 0
    with checkpoint.open("rb") as handle:
        before = os.fstat(handle.fileno())
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
            read += len(chunk)
            now = time.monotonic()
            if now - reported >= 5:
                print(f"PREFIX checked {read}/{total} bytes elapsed={now - started:.0f}s",
                      file=sys.stderr, flush=True)
                reported = now
        after = os.fstat(handle.fileno())
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if (signature(before) != signature(after) or signature(after) != signature(checkpoint.stat())
            or value.hexdigest() != receipt["checkpoint_sha256"]):
        raise ValueError(f"{folder}: shared prefix checkpoint differs from its completion receipt")
    print(f"PREFIX verified bytes={read} elapsed={time.monotonic() - started:.1f}s",
          file=sys.stderr, flush=True)
    return receipt["checkpoint_sha256"]
