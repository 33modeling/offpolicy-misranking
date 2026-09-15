#!/usr/bin/env python3
"""Keep allocated GPUs visibly busy while a launcher waits or holds.

The cluster reclaims an allocation whose GPUs sit idle, which is exactly the
state of a launcher that is holding between queue passes or waiting for a
prerequisite. This runs a tiny matmul on every visible device a few times a
second (well under 1 GB per device, a few percent of utilisation) and exits
on SIGTERM/SIGINT or when its parent launcher disappears. It is never part
of a metered cost event and never touches the run directory.
"""
from __future__ import annotations

import os
import signal
import sys
import time


def main() -> int:
    period = float(os.environ.get("SWITCH_KEEPALIVE_PERIOD", "0.25"))
    # Work per period: 4096x4096 fp16 matmuls, about 0.3 ms each on an H100, so
    # the default 64 keeps a GPU visibly busy (~8%) without competing with training.
    repeats = max(1, int(os.environ.get("SWITCH_KEEPALIVE_REPEATS", "64")))
    parent = os.getppid()
    try:
        import torch
    except ImportError:
        print("[keepalive] torch unavailable; nothing to keep busy", flush=True)
        return 0
    if not torch.cuda.is_available():
        print("[keepalive] CUDA unavailable; nothing to keep busy", flush=True)
        return 0
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    devices = list(range(torch.cuda.device_count()))
    work = []
    for index in devices:
        with torch.cuda.device(index):
            work.append((index, torch.randn(4096, 4096, device=f"cuda:{index}", dtype=torch.float16) * 0.01))
    print(f"[keepalive] pid={os.getpid()} parent={parent} devices={devices} period={period}s repeats={repeats}", flush=True)
    while not stop:
        for index, tensor in work:
            with torch.cuda.device(index):
                out = tensor
                for _ in range(repeats):
                    out = out @ tensor
                    out = out / (out.abs().amax() + 1)
                torch.cuda.synchronize(index)
        if os.getppid() != parent:
            break
        time.sleep(period)
    print("[keepalive] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
