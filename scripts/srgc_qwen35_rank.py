#!/usr/bin/env python3
"""Rank entry: install Qwen adaptation before cache/train imports on every rank."""

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def main():
    from srgc_qwen35 import runtime_adapter, training_adapter, validate_extension, validate_bundle_model, smoke
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=("cache", "train", "smoke"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    from srgc_qwen35_storage import setup_storage
    import os
    # The worker scanned existing links before launch. Each rank checks its
    # concrete plan paths, avoiding four full scans of all historical receipts.
    setup_storage(args.plan.absolute().parent.parent, os.environ, scan_tree=False)
    plan = validate_extension(args.plan)
    # The generic child accepts SRGC_ATTENTION, but this experiment pins its
    # kernel in the plan. Do not inherit an OLMo shell's unrelated override.
    os.environ["SRGC_ATTENTION"] = plan["attention"]
    if args.stage == "cache":
        from srgc_rebuttal.plan import input_path
        cache = argparse.ArgumentParser(add_help=False)
        cache.add_argument("--bundle", type=Path, required=True)
        cache.add_argument("--cache-seed", type=int, required=True)
        options, _ = cache.parse_known_args(remaining)
        if options.cache_seed not in plan["seeds"] or options.bundle.resolve() != input_path(args.plan, plan, options.cache_seed):
            raise ValueError("cache output must be the Qwen plan's own bundle/seed")
        cache.add_argument("--max-new-tokens", type=int, default=2048)
        cache.add_argument("--attention", default="eager")
        options, _ = cache.parse_known_args(remaining)
        if options.max_new_tokens != plan["max_new_tokens"] or options.attention != plan["attention"]:
            raise ValueError("cache token/attention settings differ from the Qwen plan")
        forbidden = ("--model", "--model-revision", "--verifier", "--responses")
        if any(arg.split("=", 1)[0] in forbidden for arg in remaining):
            raise ValueError("cache protocol overrides are not allowed")
        import json
        validate_bundle_model(json.loads(options.bundle.read_text()), plan, options.cache_seed)
        remaining += ["--attention", plan["attention"], "--max-new-tokens", str(plan["max_new_tokens"])]
    elif args.stage == "train":
        import json
        from srgc_rebuttal.plan import input_path
        training = argparse.ArgumentParser(add_help=False)
        training.add_argument("--seed", type=int, required=True)
        options, _ = training.parse_known_args(remaining)
        if options.seed not in plan["seeds"]:
            raise ValueError("seed is not in the Qwen plan")
        validate_bundle_model(json.loads(input_path(args.plan, plan, options.seed).read_text()), plan, options.seed)
    sys.argv = [sys.argv[0], *remaining, "--plan", str(args.plan)]
    from srgc_verifier_fallback import install
    install()
    with runtime_adapter():
        if args.stage == "smoke":
            smoke(args.plan)
        elif args.stage == "cache":
            from srgc_rebuttal.build_cache import main as cache_main
            cache_main()
        else:
            from srgc_step_checkpoints import main as training_main
            with training_adapter():
                training_main()


if __name__ == "__main__":
    main()
