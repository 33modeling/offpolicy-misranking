#!/usr/bin/env python3
"""Additional-seed experiment entry point; run/cache automatically launch four GPU ranks."""

import os
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    actions = {"run": "run_experiment", "cache": "build_cache", "plan": "plan",
               "summary": "summarize", "costs": "cost_report", "cluster": "cluster",
               "prepare": "prepare_inputs", "status": "reports", "results": "reports", "storage": None}
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--dataset", choices=("math", "mbpp"))
    options.add_argument("--plan", type=Path)
    settings, args = options.parse_known_args(sys.argv[1:])
    dataset = settings.dataset or ("mbpp" if settings.plan and
        json.loads(settings.plan.read_text()).get("dataset") == "mbpp" else "math")
    from srgc_rebuttal.existing_runtime import select_python
    select_python(dataset)
    if args in (["--help"], ["-h"]):
        parser = argparse.ArgumentParser(description=__doc__, parents=[options])
        parser.add_argument("action", nargs="?", default="run", choices=[*actions, "worker", "launch", "commands", "stop", "resume"])
        parser.epilog = "Global --dataset/--plan work before or after the action. Use ACTION --help for action options."
        parser.print_help()
        return
    if args and args[0] in {"worker", "launch", "commands", "stop", "resume"}:
        args.insert(0, "cluster")
    action = args.pop(0) if args and args[0] in actions else "run"
    plan = settings.plan or root / "srgc_rebuttal/experiments" / (
        "mbpp_seeds.json" if settings.dataset == "mbpp" else "additional_seeds.json")
    if settings.plan and settings.dataset:
        from srgc_rebuttal.plan import load_plan
        expected = "math_train" if settings.dataset == "math" else "mbpp"
        if load_plan(plan).get("dataset") != expected:
            options.error("--dataset and --plan refer to different datasets")
    writing = action in {"run", "cache", "prepare"} or (action == "cluster" and args and args[0] in {"worker", "launch", "resume"})
    if action == "storage":
        parser = argparse.ArgumentParser(description="Move stopped experiments to group storage without deleting originals")
        parser.add_argument("--migrate", action="store_true")
        storage_args = parser.parse_args(args)
        from srgc_shared_storage import route_plan
        target = route_plan(plan, writing=True, migrate=storage_args.migrate)
        print(f"Group-storage plan: {target}")
        return
    if not any(a in {"-h", "--help"} for a in args):
        from srgc_shared_storage import route_plan
        if writing:
            original_plan = plan
            plan = route_plan(plan, writing=True)
            if action == "cache":
                from srgc_rebuttal.plan import input_path, load_plan
                bundle_parser = argparse.ArgumentParser(add_help=False)
                bundle_parser.add_argument("--bundle", type=Path)
                bundle_args, remaining = bundle_parser.parse_known_args(args)
                if bundle_args.bundle:
                    spec = load_plan(plan)
                    mapping = {input_path(original_plan, spec, s): input_path(plan, spec, s) for s in spec["seeds"]}
                    target = mapping.get(bundle_args.bundle.resolve(), bundle_args.bundle.resolve())
                    if target not in mapping.values():
                        options.error("--bundle must be an input of the group-storage plan")
                    args = [*remaining, "--bundle", str(target)]
        elif not settings.plan and action != "plan":
            try:
                plan = route_plan(plan, writing=False)
            except ValueError:
                # Local CPU reports remain usable without a mounted node volume.
                pass
    args += ["--plan", str(plan)]
    if action in {"run", "cache"} and "WORLD_SIZE" not in os.environ and not any(a in {"-h", "--help"} for a in args):
        from srgc_rebuttal.cluster import direct
        raise SystemExit(direct(action, plan, args))
    if action in {"status", "results"}:
        args.insert(0, action)
    sys.argv = [f"{Path(__file__).name} {action}", *args]
    runpy.run_module(f"srgc_rebuttal.{actions[action]}", run_name="__main__")


if __name__ == "__main__":
    main()
