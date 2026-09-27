"""Write a frozen plan for the same four-arm switching experiment on another dataset.

The fixed schedule (prefix 25, total 275, 40 On-policy candidates, top-four
training, checks every 25, projection 4096) is copied from the additional-seed
plan; only the dataset name, verifier, seeds and paths change. The runtime
implements distinct random candidate-40 draws for SR/Random and 40-vs-40 SR-GC.

    python -m srgc_rebuttal.plan_dataset --dataset mbpp --seeds 5 6 7 8 9 \
        --verifier srgc_rebuttal.verifiers:code_reward
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .plan import DEFAULT_PLAN, load_plan


def make_plan(base: dict, *, dataset: str, seeds: list[int], verifier: str) -> dict:
    plan = dict(base)
    plan.update({"dataset": dataset, "seeds": seeds, "verifier": verifier,
                 "input_pattern": f"../inputs/{dataset}-seed-{{seed}}.json",
                 "output_root": f"../runs/{dataset}-seeds",
                 "primary_outcome": f"Mean binary reward on the {dataset} evaluation questions at total step 275, "
                                    "averaged over 300 questions and eight responses per question"})
    if dataset == "mbpp":
        from .build_inputs import MBPP_REVISION
        plan.update(dataset_revision=MBPP_REVISION, split_seed=0, ranking_validation_prompts=50)
    return load_plan_dict(plan)


def load_plan_dict(plan: dict) -> dict:
    """Round-trip through the strict loader so a generated plan obeys the frozen schema."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "plan.json"
        path.write_text(json.dumps(plan))
        return load_plan(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="bundle name prefix, e.g. mbpp or gsm8k")
    parser.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    parser.add_argument("--verifier", required=True, help="module:function returning a binary reward")
    parser.add_argument("--base", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = make_plan(load_plan(args.base), dataset=args.dataset, seeds=args.seeds, verifier=args.verifier)
    output = args.output or args.base.parent / f"{args.dataset}_seeds.json"
    output.write_text(json.dumps(plan, indent=1) + "\n")
    print(f"wrote {output}; inputs expected at {plan['input_pattern']} relative to it")


if __name__ == "__main__":
    main()
