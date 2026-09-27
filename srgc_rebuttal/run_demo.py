"""CPU execution example; never writes paper results or launches GPU jobs."""

import argparse
import json
from pathlib import Path

from .srgc import Config, Engine
from .toy_backend import ToyBackend, make_problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--updates", type=int, default=275)
    parser.add_argument("--prefix", type=int, default=25)
    parser.add_argument("--objective", choices=["grpo", "rloo"], default="grpo")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.prefix <= args.updates:
        parser.error("require 0 <= prefix <= updates")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    reports = []
    for seed in args.seeds:
        features, answers, candidates, validation, evaluation, cache = make_problem(seed)
        config = Config(seed=seed, objective=args.objective)
        initial = ToyBackend(features, answers, seed=seed)
        prefix = Engine(initial, candidates, validation, cache, arm="on_policy", config=config)
        prefix.run_until(args.prefix)
        checkpoint = prefix.state_dict()
        for arm in ["random", "sr", "on_policy", "switch"]:
            backend = ToyBackend(features, answers, seed=seed)
            engine = Engine(backend, candidates, validation, cache, arm=arm, config=config)
            engine.load_state_dict(checkpoint, fork_arm=arm)
            engine.run_until(args.updates)
            reports.append({"seed": seed, "arm": arm, "objective": args.objective,
                "shared_prefix": args.prefix, "total_updates": engine.step,
                "switched_at": engine.switched_at,
                "toy_expected_reward": backend.expected_reward(evaluation),
                "costs": engine.costs,
                "checks": [{"step": r["checkpoint"], "d": r["d"]}
                           for r in engine.history if r["d"] is not None]})
    payload = {"experiment": "synthetic CPU reference demo; not paper results", "runs": reports}
    text = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
