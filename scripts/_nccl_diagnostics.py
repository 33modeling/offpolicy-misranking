"""Bounded, read-only extraction of original NCCL warnings (not torchrun tails)."""
from pathlib import Path
import re


def original_warnings(directory, world_size=4):
    directory = Path(directory).resolve()
    found = []
    seen = set()
    for rank in range(min(world_size, 4)):
        path = directory / f"nccl-check-{rank}.log"
        try:
            if not path.resolve().is_relative_to(directory):
                continue
            with path.open("rb") as handle:
                head = handle.read(128 * 1024)
                size = handle.seek(0, 2)
                handle.seek(max(len(head), size - 128 * 1024))
                tail = handle.read(128 * 1024)
        except OSError:
            continue
        for line in (head + b"\n" + tail).decode("utf-8", errors="replace").splitlines():
            if not re.search(r"\bNCCL WARN\b", line, re.IGNORECASE) or line in seen:
                continue
            seen.add(line)
            found.append(f"{path.name}: {line.encode('utf-8')[:512].decode('utf-8', errors='ignore')}")
    # Keep CUDA call sites ahead of secondary cleanup/network warnings.
    found.sort(key=lambda line: not bool(re.search(r"cuda (?:failure|error)", line, re.IGNORECASE)))
    return found[:8]
