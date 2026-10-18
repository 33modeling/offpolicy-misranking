"""CPU torchrun check of common startup decisions and errors, without CUDA."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from srgc_research.followup.coordination import Coordinator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("complete", "root_error", "rank_error"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coordinator = Coordinator()

    def decide():
        time.sleep(.15)  # Let other workers reach the read before rank zero.
        if args.case == "root_error":
            raise ValueError("root checkpoint corrupted")
        (args.output / "root-called.json").write_text(json.dumps({"rank": coordinator.rank}))
        return {"complete": True}

    def validate():
        if args.case == "rank_error" and coordinator.rank == 2:
            raise ValueError("rank 2 source checksum differs")
        return "same-frozen-policy"

    try:
        result = coordinator.root(decide)
        rows = coordinator.validate(validate)
        assert len(set(rows)) == 1
        record = {"result": result, "workers": len(rows)}
    except RuntimeError as error:
        record = {"error": str(error)}
    record.update(rank=coordinator.rank, cuda_initialized=torch.cuda.is_initialized(), group_initialized=dist.is_initialized())
    (args.output / f"rank-{coordinator.rank}.json").write_text(json.dumps(record))


if __name__ == "__main__":
    main()
