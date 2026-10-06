"""Rank-local RNG isolation and host-owned snapshots without device copies."""

from contextlib import contextmanager
import copy

import torch


@contextmanager
def rollout_rng(device, seed):
    """Restore this rank's RNGs, preserving the first error if CUDA is unhealthy."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    failure = None
    try:
        # torch.manual_seed also changes every visible GPU's RNG, although this
        # process owns just one device and only that device's state was saved.
        torch.random.default_generator.manual_seed(seed)
        if cuda is not None:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield
        # Surface asynchronous generation failures before RNG restoration.
        if cuda is not None:
            torch.cuda.synchronize(device)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        restorers = [("CPU", lambda: torch.random.set_rng_state(cpu))]
        if cuda is not None:
            restorers.append((str(device), lambda: torch.cuda.set_rng_state(cuda, device)))
        for label, restore in restorers:
            try:
                restore()
            except BaseException as exc:
                if failure is None:
                    raise
                if hasattr(failure, "add_note"):
                    failure.add_note(f"RNG restoration also failed on {label}: {exc}")


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)
