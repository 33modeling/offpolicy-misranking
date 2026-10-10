"""Operational recovery shared by model launchers, outside frozen training code."""

import time
from pathlib import Path


def released_gpu_identity(identity, *, timeout=30, label="LLAMA"):
    """Allow delayed CUDA teardown between tasks without killing another job."""
    from srgc_rebuttal.runtime import Busy

    try:
        return identity()
    except Busy as error:
        reason = str(error)
    started = time.monotonic()
    print(f"{label} waiting for GPU memory release (up to {timeout}s): {reason}", flush=True)
    while time.monotonic() - started < timeout:
        time.sleep(1)
        try:
            result = identity()
        except Busy:
            continue
        print(f"{label} GPU memory release complete; continuing existing queue", flush=True)
        return result
    raise TimeoutError(f"GPU memory release did not finish within {timeout}s; allocated devices remain busy")


def admission_failure_footer(error, root, *, label="LLAMA", model_name="Llama"):
    """Print the actual admission traceback after backup shutdown messages."""
    from scripts.srgc_log_tail import tail_lines

    message = str(error)
    if message.startswith(f"{model_name} generation/backward admission failed: "):
        name = message.split(": ", 1)[1]
    elif message.startswith("four-GPU admission failed (") and "; inspect " in message:
        name = message.split("; inspect ", 1)[1]
    else:
        return
    path = Path(name)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        return
    print(f"\n{label} ADMISSION FAILURE DETAILS", flush=True)
    try:
        for line in tail_lines(path, lines=40):
            print(line, flush=True)
    except OSError as read_error:
        print(f"log unavailable: {read_error}", flush=True)
