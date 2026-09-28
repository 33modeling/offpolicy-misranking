#!/usr/bin/env python3
"""Prepare or serve isolated Qwen3.5-9B MATH/MBPP queues (one four-GPU job/node)."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from srgc_qwen35 import (MODEL, REVISION, model_path, prepare, runtime_adapter,
                        runtime_packages, validate_extension, validate_bundle_model)
from srgc_qwen35_storage import default_root, setup_storage


def admit_with_smoke(original, root, environment, run_child, **kwargs):
    import time
    report = original(root, environment, run_child, **kwargs)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               "--max_restarts=0", str(REPO / "scripts/srgc_qwen35_rank.py"),
               "--stage", "smoke", "--plan", environment["SRGC_QWEN_PLAN"]]
    started = time.perf_counter()
    state = "failed"
    try:
        code = run_child(command, root / "qwen-smoke.log", environment,
                         pass_fds=kwargs.get("pass_fds", ()), heartbeat=kwargs.get("heartbeat", lambda pid: None),
                         should_stop=kwargs.get("should_stop", lambda: False), timeout=3600)
        if code:
            raise RuntimeError(f"Qwen generation/backward admission failed: {root / 'qwen-smoke.log'}")
        state = "passed"
    finally:
        smoke_gpu_seconds = 4 * (time.perf_counter() - started)
        result = {**report, "qwen_model_smoke": state, "qwen_smoke_log": str(root / "qwen-smoke.log"),
                  "qwen_smoke_gpu_seconds": smoke_gpu_seconds,
                  "allocated_gpu_seconds": report["allocated_gpu_seconds"] + smoke_gpu_seconds}
        from srgc_rebuttal.runtime import atomic_json
        atomic_json(root / "qwen-admission.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("math", "mbpp", "all"))
    parser.add_argument("action", choices=("prepare", "run", "status", "results", "stop", "resume", "download", "doctor"))
    parser.add_argument("--root", type=Path, default=default_root(os.environ))
    parser.add_argument("--source-plan", type=Path, help="exact OLMo cohort to match; one dataset at a time")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--now", action="store_true", help="stop: interrupt current work; default drains the task")
    parser.add_argument("--allow-tokenizer-download", action="store_true", help="prepare only: fetch pinned tokenizer files")
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    datasets = ("math", "mbpp") if args.dataset == "all" else (args.dataset,)
    if args.source_plan and len(datasets) != 1:
        parser.error("--source-plan requires math or mbpp")
    group, common_root = setup_storage(args.root, os.environ)
    args.root = args.root.resolve()
    if args.action == "download":
        models = Path(os.environ.get("MODELS_DIR", str(Path(os.environ.get("GROUP_VOLUME", "/group-volume")) / "models")))
        raise SystemExit(subprocess.call([sys.executable, str(REPO / "src/model_matrix.py"),
            "--config", str(REPO / "configs/qwen35_9b_grpo.json"), "--models-dir", str(models),
            "download", "qwen3.5-9b-posttrained"]))
    if args.action == "doctor":
        print(json.dumps({"packages": runtime_packages(), "model_path": model_path(MODEL, REVISION, os.environ),
                          "gpu_validation": "worker admission still required"}, indent=2))
        return
    if args.action == "prepare":
        from transformers import AutoTokenizer
        from srgc_pair_inputs import default_plan
        from srgc_shared_storage import route_plan
        if args.allow_tokenizer_download:
            tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path(MODEL, REVISION, os.environ), local_files_only=True)
        for dataset in datasets:
            source = args.source_plan or default_plan(REPO, dataset, os.environ, writing=False)
            if not args.source_plan:
                source = route_plan(source, writing=False)
            target = prepare(dataset, source, args.root, tokenizer)
            print(f"PREPARED/VERIFIED {target}; five seeds, four arms; Qwen-only cache/prefix; no training launched")
        return
    plans = [args.root.resolve() / "experiments" / f"qwen35-9b-{dataset}.json" for dataset in datasets]
    for plan in plans:
        spec = validate_extension(plan)
        from srgc_rebuttal.plan import input_path
        for seed in spec["seeds"]:
            validate_bundle_model(json.loads(input_path(plan, spec, seed).read_text()), spec, seed)
    with runtime_adapter():
        from srgc_rebuttal import cluster
        if args.action != "run":
            for plan in plans:
                if args.action == "results":
                    from srgc_rebuttal import reports
                    report = reports.snapshot(plan, include_costs=True)
                    report.update(model=MODEL, model_revision=REVISION)
                    destination = args.root / "reports" / f"{plan.stem}-results.txt"
                    reports.export(report, destination)
                    print(f"Model: {MODEL}\n" + reports.render(report, results=True))
                    print(f"Saved: {destination}")
                    if report["errors"]:
                        raise SystemExit(1)
                else:
                    sys.argv = [sys.argv[0], args.action, "--plan", str(plan)]
                    if args.action == "stop" and args.now:
                        sys.argv.append("--now")
                    cluster.main()
            return
        from srgc_checkpoint_backup import automatic_backup
        from srgc_log_format import uniform_log
        from srgc_process_guard import process_guard
        from srgc_qwen35_worker import worker
        args.poll_seconds, args.heartbeat_seconds = 10, 5
        args.retry_delay, args.stall_seconds = 60, 1800
        with ExitStack() as stack:
            stack.enter_context(uniform_log())
            for plan in plans:
                stack.enter_context(automatic_backup(plan))
            # The guard covers local descendants of both datasets. Nesting one
            # guard per queue duplicated cleanup and process scans.
            stack.enter_context(process_guard(plans[0]))
            worker(plans, args, group, common_root, admit_with_smoke)


if __name__ == "__main__":
    main()
