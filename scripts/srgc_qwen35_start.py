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
from srgc_rebuttal.runtime import atomic_json, atomic_text, lease

# ded3d11 prepared Pair MATH inputs but rejected their legacy dataset label.
REJECTED_PAIR_ADAPTER = "7384c4923081f109b77cdb61d704d9b1b5284233208f440288d54966eb458073"


def controller(dataset, action, root):
    return [sys.executable, str(REPO / "scripts/run_srgc_qwen35.py"),
            dataset, action, "--root", str(root)]


def validate_saved(plan):
    spec = validate_extension(plan)
    for seed in spec["seeds"]:
        validate_bundle_model(json.loads(input_path(plan, spec, seed).read_text()), spec, seed)


def repair_rejected_pair_plan(plan):
    """Repair only the known pre-training label failure, never a started run."""
    from srgc_qwen35 import adapter_digest, engine_digest
    from srgc_rebuttal.plan import digest, validate_inputs
    try:
        saved = json.loads(plan.read_text())
    except (OSError, ValueError):
        return False
    if (saved.get("adapter_sha256") != REJECTED_PAIR_ADAPTER or
            saved.get("dataset") != "math_train" or saved.get("engine_sha256") != engine_digest()):
        return False
    root = plan.parent.parent
    with lease(root / ".math-prepare.lock", wait=True):
        if json.loads(plan.read_text()) != saved:
            raise ValueError("Qwen plan changed during preparation repair")
        spec = validate_extension(plan, read_only=True)
        run = root / "runs/math"
        if run.exists() and (not run.is_dir() or any(run.iterdir())):
            raise ValueError("Qwen run already has execution records; preserve its original checkout")
        legacy = False
        for seed in spec["seeds"]:
            bundle = input_path(plan, spec, seed)
            data = json.loads(bundle.read_text())
            cache = bundle.with_suffix(".cache")
            if data.get("cached_rewards") or (cache.exists() and (not cache.is_dir() or any(cache.iterdir()))):
                raise ValueError("Qwen cache already started; refusing to change the prepared plan")
            validate_inputs(data, require_cache=False)
            validate_bundle_model(data, spec, seed)
            legacy |= data.get("dataset") == "math500"
        if not legacy:
            return False
        backup = plan.with_name(f"{plan.stem}.before-math-label-fix-{digest(plan)[:16]}.json")
        if not backup.exists():
            atomic_text(backup, plan.read_text())
        elif json.loads(backup.read_text()) != saved:
            raise ValueError("Qwen preparation backup differs; refusing to overwrite it")
        atomic_json(plan, {**spec, "adapter_sha256": adapter_digest()})
        print(f"QWEN repaired unstarted Pair MATH preparation; inputs unchanged; original plan: {backup}", flush=True)
    return True


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
            command = [sys.executable, str(REPO / "src/model_matrix.py"),
                       "--config", str(REPO / "configs/qwen35_9b_grpo.json"),
                       "--models-dir", str(models), "download", "qwen3.5-9b-posttrained"]
            run_preparation(command, root, environment,
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
                try:
                    validate_saved(plan)
                except ValueError:
                    if not repair_rejected_pair_plan(plan):
                        raise
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
    command[1] = str(REPO / "scripts/srgc_qwen35_diagnostics.py")
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
