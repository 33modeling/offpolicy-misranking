"""Shared startup decisions and bounded CUDA collective failure handling."""

from datetime import timedelta


def initialize(world):
    import os
    import torch
    import torch.distributed as dist
    if (int(os.environ.get("WORLD_SIZE", "1")) != world or
            int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != world or
            not torch.cuda.is_available() or torch.cuda.device_count() != world):
        raise ValueError(f"requires one node with exactly {world} allocated CUDA GPUs")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank), timeout=timedelta(seconds=1800))
    return dist.get_rank(), rank


def primary(call):
    """Every rank takes the same branch, including rank-zero filesystem failures."""
    import torch.distributed as dist
    if not dist.is_initialized():
        return call()
    result = [None]
    if dist.get_rank() == 0:
        try:
            result[0] = {"value": call()}
        except Exception as exc:
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise RuntimeError(result[0]["error"])
    return result[0]["value"]
