#!/usr/bin/env python3
"""Rank entry: install Qwen adaptation before cache/train imports on every rank."""

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def main():
    from srgc_qwen35 import runtime_adapter, training_adapter, validate_extension, smoke
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", choices=("cache", "train", "smoke"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    validate_extension(args.plan)
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
