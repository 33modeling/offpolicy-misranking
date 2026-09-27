#!/usr/bin/env python3
"""Additional-seed experiment entry point; run/cache automatically launch four GPU ranks."""

import os
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    actions = {"run": "run_experiment", "cache": "build_cache", "plan": "plan",
               "summary": "summarize", "costs": "cost_report", "cluster": "cluster"}
    args = sys.argv[1:]
    action = args.pop(0) if args and args[0] in actions else "run"
    if action in {"run", "cache"} and "WORLD_SIZE" not in os.environ and not any(a in {"-h", "--help"} for a in args):
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                   str(Path(__file__).resolve()), action, *args]
        raise SystemExit(subprocess.call(command))
    sys.argv = [f"{Path(__file__).name} {action}", *args]
    runpy.run_module(f"srgc_rebuttal.{actions[action]}", run_name="__main__")


if __name__ == "__main__":
    main()
