"""Coordinate filesystem decisions before CUDA/NCCL is initialized."""

import json
import os
from datetime import timedelta

_INVOCATION = 0


class Coordinator:
    def __init__(self):
        global _INVOCATION
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.store, self.operation = None, 0
        if self.world > 1:
            import torch.distributed as dist

            if not 0 <= self.rank < self.world or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != self.world:
                raise ValueError("follow-up diagnostics require one node with all ranks")
            # torchrun already owns this TCPStore. A separate namespace lets us
            # agree on completion/errors without allocating CUDA or rebuilding
            # process groups. Each invocation/restart receives distinct keys.
            store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]),
                world_size=None, is_master=False, timeout=timedelta(seconds=60))
            prefix = f"srgc-followup/{os.environ['TORCHELASTIC_RUN_ID']}/{os.environ.get('TORCHELASTIC_RESTART_COUNT', '0')}/{_INVOCATION}"
            self.store = dist.PrefixStore(prefix, store)
        _INVOCATION += 1

    @staticmethod
    def attempt(function):
        try:
            return {"value": function()}
        except Exception as error:  # noqa: BLE001 - propagate any rank-local startup error to every worker
            return {"error": f"{type(error).__name__}: {error}"}

    def root(self, function):
        if self.store is None:
            return function()
        key = f"operation-{self.operation}/root"
        self.operation += 1
        if self.rank == 0:
            self.store.set(key, json.dumps(self.attempt(function), allow_nan=False))
        result = json.loads(self.store.get(key))
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["value"]

    def validate(self, function):
        if self.store is None:
            return [function()]
        prefix = f"operation-{self.operation}/validation"
        self.operation += 1
        self.store.set(f"{prefix}/{self.rank}", json.dumps(self.attempt(function), allow_nan=False))
        result_key = f"{prefix}/result"
        if self.rank == 0:
            rows = [json.loads(self.store.get(f"{prefix}/{rank}")) for rank in range(self.world)]
            errors = [f"rank {rank}: {row['error']}" for rank, row in enumerate(rows) if "error" in row]
            result = {"error": "; ".join(errors)} if errors else {"value": [row["value"] for row in rows]}
            self.store.set(result_key, json.dumps(result, allow_nan=False))
        result = json.loads(self.store.get(result_key))
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["value"]
