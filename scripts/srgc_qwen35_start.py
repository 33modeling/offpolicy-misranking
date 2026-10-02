#!/usr/bin/env python3
"""Prepare missing Qwen inputs, then join the existing shared training queue."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from srgc_qwen35 import (MODEL, REVISION, model_path, runtime_packages,
                        specification, validate_bundle_model, validate_extension)
from srgc_qwen35_storage import default_root, setup_storage
from srgc_rebuttal import cluster
from srgc_rebuttal.plan import input_path
from srgc_rebuttal.runtime import lease


def controller(dataset, action, root):
    return [sys.executable, str(REPO / "scripts/run_srgc_qwen35.py"),
            dataset, action, "--root", str(root)]


def validate_saved(plan):
    spec = validate_extension(plan)
    for seed in spec["seeds"]:
        validate_bundle_model(json.loads(input_path(plan, spec, seed).read_text()), spec, seed)


def run_preparation(command, root, environment, *, pass_fds=()):
    log = root / "startup-logs" / f"{uuid.uuid4().hex}.log"
    print(f"QWEN preparation log: {log}", flush=True)
    code = cluster.run_child(command, log, dict(environment), pass_fds=pass_fds)
    if code:
        raise subprocess.CalledProcessError(code, command)


def ensure_model(root, environment, *, pass_fds=()):
    models = Path(environment["MODELS_DIR"])
    destination = Path(environment.get(
        "SRGC_QWEN_MODEL_PATH", str(models / specification()["local_directory"])))
    # Different experiment roots can still share the same model download.
    key = hashlib.sha256(str(destination.resolve()).encode()).hexdigest()
    with lease(models / ".qwen-start-locks" / f"{key}.lock", wait=True) as guard:
        if not destination.exists():
            if "SRGC_QWEN_MODEL_PATH" in environment:
                raise FileNotFoundError(f"explicit Qwen model path does not exist: {destination}")
            print(f"QWEN downloading pinned model: {destination}", flush=True)
            run_preparation(controller("all", "download", root), root, environment,
                            pass_fds=(*pass_fds, guard.fileno()))
        # Never replace an existing, unverified snapshot automatically.
        model_path(MODEL, REVISION, environment)


def prepare_missing(datasets, root, environment):
    setup_storage(root, environment)
    plans = {dataset: root / "experiments" / f"qwen35-9b-{dataset}.json"
             for dataset in datasets}
    print(f"QWEN checking shared preparation: {root}", flush=True)
    with lease(root / ".start-prepare.lock", wait=True) as guard:
        for dataset, plan in plans.items():
            if plan.exists():
                validate_saved(plan)
            else:
                saved_run = root / "runs" / dataset
                if saved_run.exists() and (not saved_run.is_dir() or any(saved_run.iterdir())):
                    raise ValueError(f"missing plan {plan} with existing run {saved_run}; "
                                     "restore the original plan; refusing to prepare a replacement")
        runtime_packages()
        ensure_model(root, environment, pass_fds=(guard.fileno(),))
        for dataset, plan in plans.items():
            if not plan.exists():
                print(f"QWEN preparing {dataset} inputs", flush=True)
                run_preparation(controller(dataset, "prepare", root), root, environment,
                                pass_fds=(guard.fileno(),))
                validate_saved(plan)
            else:
                print(f"QWEN reusing saved {dataset} plan and inputs", flush=True)
    # Preparation is shared; training must not hold this lock across nodes.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("all", "math", "mbpp"), default="all", nargs="?")
    args = parser.parse_args()
    root = default_root(os.environ).resolve()
    datasets = ("math", "mbpp") if args.dataset == "all" else (args.dataset,)
    try:
        prepare_missing(datasets, root, os.environ)
    except KeyboardInterrupt:
        print("QWEN preparation interrupted; owned child stopped; training not started", file=sys.stderr)
        raise SystemExit(130) from None
    except subprocess.CalledProcessError as exc:
        print(f"QWEN preparation failed (exit {exc.returncode}); training not started", file=sys.stderr)
        raise SystemExit(exc.returncode if exc.returncode > 0 else 128 - exc.returncode)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"QWEN start refused: {exc}; saved results unchanged", file=sys.stderr)
        raise SystemExit(2)
    print("QWEN joining shared queue: cache -> prefix -> Random/SR/On-policy/Switch", flush=True)
    command = controller(args.dataset, "run", root)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
