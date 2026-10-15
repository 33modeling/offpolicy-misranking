"""Add measurement/storage to pinned Qwen, Gemma and Llama training entries."""

import argparse
import importlib
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=("gemma4", "llama31", "qwen35"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task", required=True)
    args, _ = parser.parse_known_args()
    import json

    from scripts import srgc_step_checkpoints as checkpoints
    from srgc_rebuttal.plan import input_path, load_plan
    from srgc_research.dispatch.reward_curves import recording_engine

    plan = load_plan(args.plan)
    data = json.loads(input_path(args.plan, plan, args.seed).read_text())
    original = checkpoints.checkpoint_engine

    def factory(base, folder, task, policy, *, total_updates):
        # Publish responses/scalars before advancing the durable resume point.
        base = recording_engine(base, folder, task, args.plan, plan, data)
        return original(base, folder, task, policy, total_updates=total_updates)

    index = sys.argv.index("--model")
    del sys.argv[index:index + 2]
    # The frozen rank imports this module by its legacy top-level name.
    with patch.dict(sys.modules, {"srgc_step_checkpoints": checkpoints}), \
            patch.object(checkpoints, "checkpoint_engine", factory):
        entry = ("scripts.srgc_qwen35_rank_runtime" if args.model == "qwen35"
                 else f"srgc_research.dispatch.{args.model}.rank")
        importlib.import_module(entry).main()


if __name__ == "__main__":
    main()
