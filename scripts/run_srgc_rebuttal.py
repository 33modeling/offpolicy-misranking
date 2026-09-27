#!/usr/bin/env python3
"""Additional-seed experiment entry point; run/cache automatically launch four GPU ranks."""

import os
import argparse
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    actions = {"run": "run_experiment", "cache": "build_cache", "plan": "plan",
               "summary": "summarize", "costs": "cost_report", "cluster": "cluster",
               "prepare": "prepare_inputs"}
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--dataset", choices=("math", "mbpp"))
    options.add_argument("--plan", type=Path)
    settings, args = options.parse_known_args(sys.argv[1:])
    if args in (["--help"], ["-h"]):
        parser = argparse.ArgumentParser(description=__doc__, parents=[options])
        parser.add_argument("action", nargs="?", default="run", choices=[*actions, "worker", "status", "launch", "commands", "stop", "resume"])
        parser.epilog = "Global --dataset/--plan work before or after the action. Use ACTION --help for action options."
        parser.print_help()
        return
    if args and args[0] in {"worker", "status", "launch", "commands", "stop", "resume"}:
        args.insert(0, "cluster")
    action = args.pop(0) if args and args[0] in actions else "run"
    plan = settings.plan or root / "srgc_rebuttal/experiments" / (
        "mbpp_seeds.json" if settings.dataset == "mbpp" else "additional_seeds.json")
    if settings.plan and settings.dataset:
        from srgc_rebuttal.plan import load_plan
        expected = "math_train" if settings.dataset == "math" else "mbpp"
        if load_plan(plan).get("dataset") != expected:
            options.error("--dataset and --plan refer to different datasets")
    args += ["--plan", str(plan)]
    if action in {"run", "cache"} and "WORLD_SIZE" not in os.environ and not any(a in {"-h", "--help"} for a in args):
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                   str(Path(__file__).resolve()), action, *args]
        raise SystemExit(subprocess.call(command))
    sys.argv = [f"{Path(__file__).name} {action}", *args]
    runpy.run_module(f"srgc_rebuttal.{actions[action]}", run_name="__main__")


if __name__ == "__main__":
    main()
