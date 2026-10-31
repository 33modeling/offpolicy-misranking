"""Display current model status, results and GPU time without saving status files."""

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from scripts import srgc_live_status as live
from srgc_rebuttal import reports
from srgc_rebuttal.plan import input_path
from srgc_rebuttal.srgc import stream_seed

CONFIGS = {
    "gemma4_it": ("srgc_research.dispatch.gemma4_it.adapter", "srgc_research.dispatch.gemma4_it.storage", "gemma4-12b-it"),
    "qwen35": ("scripts.srgc_qwen35", "scripts.srgc_qwen35_storage", "qwen35-9b"),
    "gemma4": ("srgc_research.dispatch.gemma4.adapter", "srgc_research.dispatch.gemma4.storage", "gemma4-12b-pt"),
    "gemma4_text": ("srgc_research.dispatch.gemma4_text.adapter", "srgc_research.dispatch.gemma4_text.storage", "gemma4-12b-text"),
    "llama31": ("srgc_research.dispatch.llama31.adapter", "srgc_research.dispatch.llama31.storage", "llama31-8b"),
}
LABELS = {"qwen35": "Qwen3.5-9B", "llama31": "Llama-3.1-8B-Instruct", "gemma4": "Gemma-4-12B-PT"}
LABELS["gemma4_text"] = "Gemma-4-12B-PT (text-v3)"
LABELS["gemma4_it"] = "Gemma-4-12B-IT (thinking off)"


def it_preflight_status(root, path, plan):
    """Read the fixed panel on CPU, validating its summary against raw receipts."""
    from .gemma4_it import preflight

    data = live.read_object(input_path(path, plan, plan["seeds"][0]))
    protocol = preflight.identity(path, plan, data)
    directory = preflight.folder(path, plan)
    from .gemma4_it.storage import inside

    inside(directory, root)
    rows = []
    for candidate in protocol["candidate_ids"]:
        receipt = inside(directory / f"{hashlib.sha256(candidate.encode()).hexdigest()}.json", root)
        row = preflight.read_receipt(receipt, protocol, candidate)
        if row is not None:
            rows.append(row)
    result = {"state": "PARTIAL" if rows else "NOT STARTED", "saved_prompts": len(rows),
              "total_prompts": len(protocol["candidate_ids"]), "path": str(directory)}
    summary = inside(directory / "summary.json", root)
    if len(rows) == result["total_prompts"]:
        expected = preflight.summarize(protocol, rows)
        if summary.is_file():
            if live.read_object(summary) != expected:
                raise ValueError(f"preflight summary differs from raw receipts: {summary}")
            result.update(state="PASS" if expected["passed"] else "PAUSED", **expected)
        else:
            result["state"] = "AWAITING SUMMARY"
    elif summary.is_file():
        raise ValueError(f"preflight summary has incomplete raw receipts: {summary}")
    return result


def historical_gemma_roots(root, environment):
    """Locate old PT plans without scanning or folding them into IT statistics."""
    from .gemma4_it.storage import group_work, inside

    group, work = group_work(environment)
    parent = work / "srgc-rebuttal"
    candidates = {parent / "gemma4-12b-pt-v1", parent / "gemma4-12b-text-v3"}
    candidates.update(Path(environment[key]) for key in ("SRGC_GEMMA_ROOT", "SRGC_GEMMA_TEXT_ROOT")
                      if environment.get(key))
    found = []
    for candidate in sorted(candidates):
        try:
            candidate = inside(candidate, group)
        except ValueError:
            continue
        if candidate == root.resolve():
            continue
        for model, prefix in (("gemma4_text", "gemma4-12b-text"), ("gemma4", "gemma4-12b-pt")):
            if any((candidate / "experiments" / f"{prefix}-{name}.json").is_file()
                   for name in ("math", "mbpp")):
                found.append((model, candidate))
                break
    return found


def interrupted_receipts(report):
    """Interpret old signal exits without rewriting receipts or retry history."""
    for row in report["tasks"]:
        if row["status"] != "failed":
            continue
        path = Path(report["output_root"]) / ".queue/tasks" / f"{row['task']}.json"
        try:
            receipt = live.read_object(path)
        except (OSError, ValueError, TypeError) as error:
            report["errors"].append(f"{row['task']} interruption status: {error}")
            continue
        if receipt.get("status") != "failed":
            continue
        code = receipt.get("exit_code")
        row["validation_error"] = receipt.get("validation_error")
        if type(code) is not int or row["validation_error"]:
            continue
        normalized = 128 - code if code < 0 else code
        # HUP/INT/TERM identify a terminated session. SIGKILL/137 may be OOM,
        # so an old ambiguous failure must retain its ERROR and original log.
        if normalized in (129, 130, 143):
            row.update(status="interrupted", receipt_status="failed", exit_code=code)


def initial_accuracy(data, responses):
    cache = data.get("cached_rewards", {})
    candidates = data["candidate_ids"]
    if not isinstance(cache, dict):
        raise TypeError("initial candidate rewards must be an object")
    if not cache:
        return None
    if set(cache) - set(candidates) or not candidates:
        raise ValueError("initial candidate rewards include unknown candidate prompts")
    if any(not isinstance(values, list) or len(values) != responses
           or any(reward not in (0, 1) for reward in values)
           for values in cache.values()):
        raise ValueError("invalid initial candidate rewards")
    if set(cache) != set(candidates):
        return None
    rewards = [reward for values in cache.values() for reward in values]
    return {"responses": len(rewards), "success_percent": 100 * sum(rewards) / len(rewards)}


def checkpoint_evaluation_progress(report, plan, seed):
    """Observe this task's separate checkpoint evaluator, never an older attempt."""
    folder = Path(report["output_root"]) / f"seed-{seed}"
    for row in report["tasks"]:
        if (row["seed"] != seed or row["status"] != "running" or row["arm"] == "cache"
                or row.get("endpoint_ready")):
            continue
        path = Path(report["output_root"]) / ".queue/tasks" / f"{row['task']}.json"
        if not path.is_file():
            continue
        started = live.read_object(path).get("started")
        if type(started) not in (int, float) or not math.isfinite(started):
            continue
        scope = "shared-prefix" if row["arm"] == "prefix" else row["arm"]
        primary = live.latest_phase(folder / "cost-receipts" / scope, started)
        directory = folder / "reward-curves" / row["arm"]
        candidates = [item for path in (directory / "cost-receipts").glob("step-*")
                      if (item := live.latest_phase(path, started)) is not None]
        if not candidates:
            continue
        event, modified = max(candidates, key=lambda item: item[1])
        step = event.get("checkpoint")
        if (event.get("phase") != "evaluation" or event.get("state") not in {"started", "finished"}
                or type(step) is not int or not 0 < step < plan["total_updates"]
                or step % 25 or (directory / f"step-{step}.json").is_file()):
            continue
        if primary is not None:
            current = primary[0]
            checkpoint = current.get("checkpoint")
            if type(checkpoint) is int:
                # A later optimizer update is stronger evidence than file
                # modification order, which can tie or lag on shared storage.
                completed = checkpoint + int(current.get("phase") == "training"
                                             and current.get("state") == "finished")
                next_update = (checkpoint >= step
                               and current.get("phase") in {"selection", "training"})
                if completed > step or next_update:
                    continue
        row.update(phase_stage="evaluation", step=step, completed_steps=step, current_step=None,
                   evaluation_step=step, progress_source="checkpoint_evaluation_receipt",
                   progress_age_seconds=max(0, time.time() - modified))


def finalizing_progress(report, plan_path, plan):
    """A validated endpoint with a live writer is cleanup, not another eval."""
    candidates = {}
    for row in report["tasks"]:
        if row["status"] != "running" or row["arm"] not in plan["arms"]:
            continue
        path = Path(report["output_root"]) / f"seed-{row['seed']}" / f"{row['arm']}-endpoint.json"
        try:
            st = path.stat()
            candidates[row["task"]] = path, (st.st_ino, st.st_size, st.st_mtime_ns)
        except FileNotFoundError:
            continue
    if not candidates:
        return
    # Reuse the frozen reporter's full identity/prefix/reward validation. Only
    # this read ignores leases; scheduling and the original RUN state stay intact.
    with patch.object(reports, "locked", return_value=False):
        settled = reports.snapshot(plan_path)
    verified = {row["task"]: row for row in settled["tasks"] if row["status"] == "complete"}
    for row in report["tasks"]:
        if row["task"] not in candidates or row["task"] not in verified:
            continue
        path, version = candidates[row["task"]]
        try:
            st = path.stat()
        except FileNotFoundError:
            continue
        if version != (st.st_ino, st.st_size, st.st_mtime_ns):
            continue
        row.update(endpoint_ready=True, phase_stage="finalizing", step=plan["total_updates"],
                   completed_steps=plan["total_updates"], current_step=None, last_observed_step=None,
                   saved_reward_percent=verified[row["task"]]["reward_percent"],
                   progress_source="validated_endpoint_with_active_lease")


def final_evaluation_progress(report, plan, seed, data):
    """Observe saved rollouts, without treating their presence as a valid endpoint."""
    # run_experiment.evaluate uses eight responses and this stream, independently
    # of the training/scoring caches. Read names only: token arrays can be large.
    responses = 8
    expected = {f"{hashlib.sha256(rid.encode()).hexdigest()[:24]}-{responses}.npz"
                for rid in data["evaluation_ids"]}
    for row in report["tasks"]:
        if (row["seed"] != seed or row["status"] != "running"
                or row.get("phase_stage") != "evaluation"):
            continue
        progress = {"saved_prompts": None, "total_prompts": len(data["evaluation_ids"]),
                    "responses_per_prompt": responses}
        evaluation_seed = stream_seed(seed, row.get("evaluation_step", plan["total_updates"]), "reporting-evaluation")
        directory = (Path(report["output_root"]) / f"seed-{seed}" / "rollout-cache"
                     / row["arm"] / f"eval-{evaluation_seed}")
        try:
            with os.scandir(directory) as entries:
                count = sum(entry.name in expected and entry.is_file(follow_symlinks=False)
                            for entry in entries)
            # Evaluation removes this directory before writing its endpoint.
            # Absence during that transition means unknown, not zero completed.
            if directory.is_dir():
                progress["saved_prompts"] = count
        except FileNotFoundError:
            pass
        except OSError as error:
            progress["read_error"] = str(error)
        row["final_evaluation"] = progress


def selection_progress(report, plan, seed, data):
    """Count saved scoring rollouts of a running selection block, by file name only."""
    # Engine.update scores candidates under stream (seed, step, "selection") and the
    # ranking validation set under (seed, step, "validation"), eight responses each.
    # A switch check adds up to scoring_prompts SR prompts to the candidate union.
    responses = plan["responses"]
    key = lambda rid: f"{hashlib.sha256(rid.encode()).hexdigest()[:24]}-{responses}.npz"
    for row in report["tasks"]:
        if (row["seed"] != seed or row["status"] != "running"
                or row.get("phase_stage") != "selection" or type(row.get("completed_steps")) is not int):
            continue
        step = row["completed_steps"]
        check = (row["arm"] == "switch" and step >= plan["first_check"]
                 and step % plan["check_interval"] == 0)
        progress = {"step": step, "check": check,
                    "candidate_max": plan["scoring_prompts_per_set"] * (2 if check else 1),
                    "validation_total": len(data["ranking_validation_ids"])}
        folder = Path(report["output_root"]) / f"seed-{seed}" / "rollout-cache" / row["arm"]
        for name, purpose, ids in (("candidate", "selection", data["candidate_ids"]),
                                   ("validation", "validation", data["ranking_validation_ids"])):
            expected = {key(rid) for rid in ids}
            directory = folder / f"score-{stream_seed(seed, step, purpose)}"
            progress[f"{name}_saved"] = None
            try:
                with os.scandir(directory) as entries:
                    progress[f"{name}_saved"] = sum(entry.name in expected and entry.is_file(follow_symlinks=False)
                                                    for entry in entries)
            except FileNotFoundError:
                # Not started yet, or already finished and removed.
                pass
            except OSError as error:
                progress["read_error"] = str(error)
        row["selection_progress"] = progress


def load_reports(model, dataset, root, environment):
    """Read recorded identities and isolate input failures to the affected seed."""
    if model not in CONFIGS or dataset not in {"all", "math", "mbpp"}:
        raise ValueError("unknown model or dataset")
    scripts_path = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    adapter_name, storage_name, prefix = CONFIGS[model]
    adapter, storage = importlib.import_module(adapter_name), importlib.import_module(storage_name)
    group, _ = storage.group_work(environment)
    root = storage.inside(root, group)
    # Status reads validated plans and bounded metadata paths. Walking every
    # rollout/checkpoint first can stall shared-storage status before any output.
    # validate_extension below retains each adapter's plan-path containment checks.
    storage.inside(root / "experiments", root)
    # engine_digest remains the original engine even inside runtime_adapter().
    identity = hashlib.sha256((adapter.engine_digest() + adapter.adapter_digest()).encode()).hexdigest()
    snapshots = []
    for name in ("math", "mbpp") if dataset == "all" else (dataset,):
        path = root / "experiments" / f"{prefix}-{name}.json"
        try:
            plan = adapter.validate_extension(path, read_only=True)
            with patch.object(reports, "code_digest", return_value=identity):
                report = live.snapshot(reports.snapshot, path)
                finalizing_progress(report, path, plan)
        except (OSError, ValueError, KeyError, TypeError) as error:
            report = {"dataset": name, "output_root": str(root / "runs" / name), "tasks": [],
                      "workers": [], "errors": [str(error)], "warnings": [], "complete": False,
                      "prepared": False, "stop_requested": False, "generated": time.time()}
            if (model == "gemma4_it" and isinstance(error, FileNotFoundError)
                    and error.filename == str(path) and not path.exists()):
                from .gemma_it_bootstrap import has_saved_work

                if not has_saved_work(root, name):
                    # A newly selected IT root has no experiment until prepare
                    # or run creates it. Status must identify that state rather
                    # than treat it as an unreadable existing experiment.
                    report["errors"] = []
                    report["initialization_state"] = "not_initialized"
                    report["missing_plan"] = str(path)
                else:
                    report["errors"] = [f"experiment plan is missing while saved work exists: {path}; "
                                        "restore the original plan; existing work was preserved"]
        else:
            report["prepared"] = True
            interrupted_receipts(report)
            report["initial_candidate_accuracy"] = []
            for seed in plan["seeds"]:
                try:
                    data = live.read_object(input_path(path, plan, seed))
                    adapter.validate_bundle_model(data, plan, seed)
                    accuracy = initial_accuracy(data, plan["responses"])
                    if accuracy is not None:
                        report["initial_candidate_accuracy"].append({"seed": seed, **accuracy})
                    try:
                        checkpoint_evaluation_progress(report, plan, seed)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        report["errors"].append(f"seed {seed} checkpoint eval progress: {error}")
                        report["complete"] = False
                    final_evaluation_progress(report, plan, seed, data)
                    selection_progress(report, plan, seed, data)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    report["errors"].append(f"seed {seed} {LABELS[model]} inputs: {error}")
                    for row in report["tasks"]:
                        if row["seed"] == seed:
                            row.update(status="invalid", reward_percent=None, cost_complete=False)
                            for field in reports.COST_FIELDS:
                                row[field] = None
                            from scripts.srgc_time_status import invalid
                            invalid(row, "model-specific input validation failed")
            report["complete"] = report["complete"] and not report["errors"]
            report["counts"] = dict(Counter(row["status"] for row in report["tasks"]))
            for arm, statistics in report.get("arm_statistics", {}).items():
                statistics["completed_seeds"] = sum(row["status"] == "complete" for row in report["tasks"]
                                                    if row["arm"] == arm)
                if statistics["completed_seeds"] != statistics["planned_seeds"]:
                    statistics["mean_reward_percent"] = None
                    statistics["mean_selection_training_preparation_gpu_seconds"] = None
        if model in {"gemma4_text", "gemma4_it"}:
            # Initial preparation can be cancelled before a plan/queue exists.
            # Report that stop even when the plan is still missing or invalid.
            report["stop_requested"] |= any(path.is_file() for path in (
                root / f".startup-control/{name}/stop.json",
                root / f"runs/{name}/.queue/stop.json",
            ))
        if model == "gemma4_it":
            from .gemma_it_bootstrap import preparation_lock

            report["preparing"] = (reports.locked(root / ".start-prepare.lock")
                                   or reports.locked(preparation_lock(root)))
            if report["prepared"]:
                try:
                    report["preflight"] = it_preflight_status(root, path, plan)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    report["preflight"] = {"state": "INVALID"}
                    report["errors"].append(f"preflight: {error}")
                    report["complete"] = False
        report["label"] = "MATH" if name == "math" else "MBPP"
        if report["prepared"]:
            from .submission_time import attach as attach_submission_time

            attach_submission_time(report, path, plan)
        snapshots.append(report)
    from .worker_history import prune_workers

    prune_workers(snapshots, gpu_lock_roots=(Path(group) / ".srgc-gpu-node-locks", root / "gpu-node-locks"))
    return snapshots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(CONFIGS))
    parser.add_argument("dataset", choices=("math", "mbpp", "all"), nargs="?", default="all")
    parser.add_argument("action", choices=("status",), nargs="?", default="status")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        storage = importlib.import_module(CONFIGS[args.model][1])
        root = storage.default_root(os.environ) if args.root is None else args.root
        if not args.json:
            print(f"Reading {LABELS[args.model]} status | Root: {root.resolve()}", file=sys.stderr, flush=True)
        if args.model == "gemma4_it":
            from .gemma_it_bootstrap import ensure_plans

            ensure_plans(root, args.dataset, os.environ)
        snapshots = load_reports(args.model, args.dataset, root, os.environ)
        from .model_chain import snapshot as chain_snapshot
        from .qwen_status import node_rows
        from .status_current import print_current

        coordinators = chain_snapshot()
        if args.json:
            adapter = importlib.import_module(CONFIGS[args.model][0])
            print(json.dumps({"model": adapter.MODEL, "root": str(root.resolve()),
                              "datasets": snapshots, "model_chain": coordinators,
                              "nodes": node_rows(snapshots, coordinators=coordinators,
                                                 model=args.model, root=root)}, indent=2))
        else:
            from .qwen_status import render

            text = render(snapshots, root.resolve(), model=LABELS[args.model],
                          status_model=args.model, coordinators=coordinators)
            print_current(text)
        return int(any(report["errors"] for report in snapshots))
    except KeyboardInterrupt:
        print("STATUS: Gemma initialization stopped", file=sys.stderr, flush=True)
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"STATUS ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
