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
                        runtime_packages, validate_extension)


def default_root():
    group = Path(os.environ.get("GROUP_VOLUME", "/group-volume"))
    work = Path(os.environ.get("OM_WORK", str(group / os.environ.get("OM_USER", "minsoo3.kim") / "offpolicy-misranking")))
    return Path(os.environ.get("SRGC_QWEN_ROOT", str(work / "srgc-rebuttal/qwen35-9b")))


def admit_with_smoke(original, root, environment, run_child, **kwargs):
    report = original(root, environment, run_child, **kwargs)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               "--max_restarts=0", str(REPO / "scripts/srgc_qwen35_rank.py"),
               "--stage", "smoke", "--plan", environment["SRGC_QWEN_PLAN"]]
    code = run_child(command, root / "qwen-smoke.log", environment,
                     pass_fds=kwargs.get("pass_fds", ()), heartbeat=kwargs.get("heartbeat", lambda pid: None),
                     should_stop=kwargs.get("should_stop", lambda: False), timeout=3600)
    if code:
        raise RuntimeError(f"Qwen generation/backward admission failed: {root / 'qwen-smoke.log'}")
    return {**report, "qwen_model_smoke": "passed", "qwen_smoke_log": str(root / "qwen-smoke.log")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("math", "mbpp", "all"))
    parser.add_argument("action", choices=("prepare", "run", "status", "results", "stop", "resume", "download", "doctor"))
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--source-plan", type=Path, help="exact OLMo cohort to match; one dataset at a time")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--now", action="store_true", help="stop: interrupt current work; default drains the task")
    parser.add_argument("--allow-tokenizer-download", action="store_true", help="prepare only: fetch pinned tokenizer files")
    args = parser.parse_args()
    datasets = ("math", "mbpp") if args.dataset == "all" else (args.dataset,)
    if args.source_plan and len(datasets) != 1:
        parser.error("--source-plan requires math or mbpp")
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
            print(f"PREPARED {target}; five seeds, four arms, fresh Qwen cache/prefix; not launched")
        return
    plans = [args.root.resolve() / "experiments" / f"qwen35-9b-{dataset}.json" for dataset in datasets]
    for plan in plans:
        validate_extension(plan)
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
        from unittest.mock import patch
        from srgc_checkpoint_backup import automatic_backup
        from srgc_log_format import uniform_log
        from srgc_multi_queue import multi_queue
        from srgc_process_guard import process_guard
        from srgc_seed_order import seed_first_queue
        from srgc_shared_storage import storage_root
        from srgc_worker_status import run_with_status
        group, common_root = storage_root(os.environ)
        if not args.root.resolve().is_relative_to(group):
            parser.error("GPU artifacts must be inside group storage")
        # Same physical-GPU lock namespace as OLMo workers; never share an allocation.
        sys.argv = [sys.argv[0], "worker", "--plan", str(plans[0]),
                    "--node-lock-root", str(common_root / "gpu-node-locks")]
        if args.retry_failed:
            sys.argv.append("--retry-failed")
        os.environ["SRGC_QWEN_PLAN"] = str(plans[0])
        with ExitStack() as stack:
            original_admit = cluster.admit
            stack.enter_context(patch.object(cluster, "admit", lambda *a, **kw: admit_with_smoke(original_admit, *a, **kw)))
            stack.enter_context(uniform_log())
            for plan in plans:
                stack.enter_context(automatic_backup(plan))
                stack.enter_context(process_guard(plan))
            stack.enter_context(seed_first_queue())
            if len(plans) > 1:
                stack.enter_context(multi_queue(plans[1:]))
            original_worker = cluster.run_worker
            stack.enter_context(patch.object(cluster, "run_worker", lambda *a, **kw: run_with_status(original_worker, *a, **kw)))
            cluster.main()


if __name__ == "__main__":
    main()
