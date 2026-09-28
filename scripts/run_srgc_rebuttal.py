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
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.append(str(Path(__file__).resolve().parent))
    actions = {"run": "run_experiment", "cache": "build_cache", "plan": "plan",
               "summary": "summarize", "costs": "cost_report", "cluster": "cluster",
               "prepare": "prepare_inputs", "status": "reports", "results": "reports", "storage": None,
               "backup": None, "backup-watch": None}
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--dataset", choices=("math", "mbpp"))
    options.add_argument("--plan", type=Path)
    options.add_argument("--fresh", nargs="?", const="restart1", metavar="RUN",
                         help="start/join a new group-volume run, ignoring old caches (default name: restart1)")
    options.add_argument("--with-dataset", choices=("math", "mbpp"), dest="with_dataset",
                         help="worker only: also serve this dataset's queue when the main one has nothing claimable")
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
    writing = action in {"run", "cache", "prepare"} or (action == "cluster" and args and args[0] in {"worker", "launch", "resume"})
    if settings.plan:
        plan = settings.plan
    else:
        from srgc_pair_inputs import default_plan
        plan = default_plan(root, "mbpp" if settings.dataset == "mbpp" else "math", os.environ, writing=bool(writing))
    if settings.plan and settings.dataset:
        from srgc_rebuttal.plan import load_plan
        expected = "math_train" if settings.dataset == "math" else "mbpp"
        if load_plan(plan).get("dataset") != expected:
            options.error("--dataset and --plan refer to different datasets")
    if settings.fresh and not (action == "storage" or (action == "cluster" and args and args[0] in {"worker", "launch"})):
        options.error("--fresh is supported by worker, launch and storage")
    if settings.with_dataset and not (action == "cluster" and args and args[0] == "worker"):
        options.error("--with-dataset is supported by worker only")
    if settings.with_dataset and (settings.fresh or settings.plan or settings.with_dataset == dataset):
        options.error("--with-dataset needs the default plans of two different datasets and no --fresh")
    if action in {"backup", "backup-watch"}:
        from srgc_shared_storage import route_plan
        from srgc_checkpoint_backup import main as backup_main
        if not settings.plan and not any(a in {"-h", "--help"} for a in args):
            plan = route_plan(plan, writing=False)
        backup_main([*args, "--plan", str(plan), *(["--watch"] if action == "backup-watch" else [])])
        return
    if action == "storage":
        parser = argparse.ArgumentParser(description="Move stopped experiments to group storage without deleting originals")
        parser.add_argument("--migrate", action="store_true")
        storage_args = parser.parse_args(args)
        from srgc_shared_storage import route_plan
        if storage_args.migrate and settings.fresh:
            parser.error("--fresh and --migrate are mutually exclusive")
        target = route_plan(plan, writing=True, migrate=storage_args.migrate, fresh=settings.fresh)
        print(f"Group-storage plan: {target}")
        return
    if not any(a in {"-h", "--help"} for a in args):
        from srgc_shared_storage import route_plan
        if writing:
            original_plan = plan
            auto = (action == "cluster" and args[0] == "worker" and
                    settings.plan is None and settings.fresh is None)
            plan = route_plan(plan, writing=True, fresh=settings.fresh, start_or_continue=auto)
            if action == "cluster" and args[0] == "worker" and not any(
                    a == "--node-lock-root" or a.startswith("--node-lock-root=") for a in args):
                from srgc_shared_storage import storage_root
                args += ["--node-lock-root", str(storage_root(os.environ)[1] / "gpu-node-locks")]
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
    if action == "status":
        from srgc_live_status import main as status_main
        status_main()
    elif action == "cluster" and args[0] == "worker" and not any(a in {"-h", "--help"} for a in args):
        from contextlib import ExitStack
        from srgc_checkpoint_backup import automatic_backup
        from srgc_log_format import uniform_log
        from srgc_multi_queue import multi_queue
        from srgc_process_guard import process_guard
        from srgc_seed_order import seed_first_queue
        from srgc_step_checkpoints import worker_main
        extra_plans = []
        if settings.with_dataset:
            from srgc_pair_inputs import default_plan
            from srgc_shared_storage import route_plan
            extra = default_plan(root, settings.with_dataset, os.environ, writing=True)
            extra_plans.append(route_plan(extra, writing=True, start_or_continue=True))
        with ExitStack() as stack:
            stack.enter_context(uniform_log())
            for each in (plan, *extra_plans):
                stack.enter_context(automatic_backup(each))
                stack.enter_context(process_guard(each))
            stack.enter_context(seed_first_queue())
            stack.enter_context(multi_queue(extra_plans))
            worker_main()
    else:
        runpy.run_module(f"srgc_rebuttal.{actions[action]}", run_name="__main__")


if __name__ == "__main__":
    main()
