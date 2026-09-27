"""CPU-only live status and partial result exports; unknown measurements stay null."""

import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time

from .cluster_queue import input_info
from .cost_report import seed_costs
from .plan import DEFAULT_PLAN, digest, input_path, load_plan
from .runtime import atomic_json, atomic_text, code_digest, lease, matches, run_root

COST_FIELDS = ("selection_gpu_seconds", "training_gpu_seconds", "preparation_gpu_seconds",
               "evaluation_gpu_seconds", "checkpoint_gpu_seconds", "startup_gpu_seconds",
               "continuation_phases_gpu_seconds", "selection_training_preparation_gpu_seconds")


def locked(path):
    try:
        with path.open("r") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(handle, fcntl.LOCK_UN)
    except FileNotFoundError:
        pass
    return False


def snapshot(plan_path, *, include_costs=False):
    plan_path = plan_path.resolve()
    plan = load_plan(plan_path)
    root = run_root(plan_path, plan)
    directory = root / ".queue"
    errors, warnings, tasks, workers, costs = [], [], [], [], []
    def read(path):
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            return value
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            return None
    protocol = read(directory / "protocol.json")
    if protocol and protocol.get("implementation_sha256") != code_digest():
        warnings.append("code changed since queue initialization; viewing recorded results only, resume is blocked")
    for seed in plan["seeds"]:
        folder = root / f"seed-{seed}"
        bundle_path = input_path(plan_path, plan, seed)
        expected = {"seed": seed, "plan_sha256": digest(plan_path)}
        try:
            info, bundle = input_info(bundle_path)
            expected["input_sha256"] = info["input_sha256"]
            cached = not info["pending_cache"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"seed {seed} inputs: {type(exc).__name__}: {exc}")
            cached, bundle = False, None
        marker = read(folder / "run.json")
        prefix = read(folder / "prefix-ready.json")
        valid_marker = (marker and bundle is not None and matches(marker, expected)
                        and isinstance(marker.get("implementation_sha256"), str))
        if marker and not valid_marker:
            errors.append(f"seed {seed}: run identity differs from plan or inputs")
        if valid_marker:
            expected["implementation_sha256"] = marker["implementation_sha256"]
        prefix_ok = bool(valid_marker and prefix and matches(prefix, expected) and
                         prefix.get("completed_updates") == plan["shared_prefix_updates"] and
                         (folder / "prefix.pt").is_file())
        seed_rows = {}
        for arm in ("cache", "prefix", *plan["arms"]):
            key = f"seed-{seed}.{arm}"
            receipt = read(directory / "tasks" / f"{key}.json") or {}
            active = locked(directory / "leases" / f"{key}.lock") or locked(
                bundle_path.with_suffix(".cache") / "execution.lock" if arm == "cache" else folder / f".{arm}.execution.lock")
            row = {"seed": seed, "arm": arm, "task": key, "status": "ready", "step": None,
                   "reward_percent": None, "switched_at": None, "attempt": receipt.get("attempt", 0),
                   "host": receipt.get("host"), "worker_id": receipt.get("worker_id"),
                   "log": str(directory / "logs" / f"{key}.log")}
            if arm == "cache" and bundle is not None:
                cache_root = bundle_path.with_suffix(".cache")
                saved, latest = 0, None
                for candidate in bundle["candidate_ids"]:
                    path = cache_root / f"{hashlib.sha256(candidate.encode()).hexdigest()}.json"
                    try:
                        modified = path.stat().st_mtime
                    except FileNotFoundError:
                        continue
                    saved += 1
                    latest = modified if latest is None else max(latest, modified)
                row.update(cache_saved_prompts=saved, cache_total_prompts=len(bundle["candidate_ids"]),
                           cache_exported_prompts=len(bundle.get("cached_rewards", {})),
                           cache_last_write_age_seconds=None if latest is None else max(0, time.time() - latest))
            progress = read(folder / f"{arm}-progress.json") if arm in plan["arms"] else None
            if progress:
                row.update(step=progress.get("step"), switched_at=progress.get("switched_at"))
            endpoint = read(folder / f"{arm}-endpoint.json") if arm in plan["arms"] else None
            if arm in plan["arms"] and endpoint is None and (folder / f"{arm}-endpoint.json").exists():
                row["status"] = "invalid"
            done = cached if arm == "cache" else prefix_ok if arm == "prefix" else False
            if endpoint:
                try:
                    if (not prefix_ok or not matches(endpoint, expected) or endpoint.get("arm") != arm or
                            endpoint.get("total_updates") != plan["total_updates"] or
                            endpoint.get("shared_prefix_updates") != plan["shared_prefix_updates"] or
                            endpoint.get("prefix_checkpoint_sha256") != prefix["checkpoint_sha256"]):
                        raise ValueError("endpoint identity, prefix or update count differs")
                    rewards = endpoint["per_question_reward"]
                    if (len(rewards) != 300 or set(rewards) != set(bundle["evaluation_ids"]) or
                            any(not math.isfinite(v) or not 0 <= v <= 1 for v in rewards.values()) or
                            not math.isfinite(endpoint["reward"]) or
                            abs(statistics.mean(rewards.values()) - endpoint["reward"]) > 1e-10):
                        raise ValueError("invalid or mismatched evaluation rewards")
                    done = True
                    if not active:
                        row.update(step=endpoint["total_updates"], reward_percent=100 * endpoint["reward"],
                                   switched_at=endpoint["switched_at"])
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append(f"{key}: {exc}")
                    row["status"] = "invalid"
            if active:
                row["status"] = "running"
            elif done:
                row["status"] = "complete"
            elif row["status"] != "invalid":
                if receipt.get("status") == "running":
                    row["status"] = "recoverable"
                elif receipt.get("status") in {"failed", "interrupted"}:
                    row["status"] = receipt["status"]
                elif arm != "cache" and not (cached if arm == "prefix" else prefix_ok):
                    row["status"] = "waiting_for_cache" if arm == "prefix" else "waiting_for_prefix"
            if receipt.get("attempt_id"):
                row["progress"] = [r for p in sorted((directory / "progress" / receipt["attempt_id"]).glob("rank-*.json"))
                                   if (r := read(p)) is not None]
            tasks.append(row)
            seed_rows[arm] = row
        if include_costs:
            try:
                if marker and not valid_marker:
                    raise ValueError("cost identity differs from current inputs/plan")
                cost = seed_costs(folder, bundle_path, plan["arms"])
                costs.append({"seed": seed, **cost})
                for arm in plan["arms"]:
                    r = seed_rows[arm]
                    r["cost_complete"] = r["status"] == "complete" and cost["arms"][arm]["complete"]
                    for field in COST_FIELDS:
                        r[field] = cost["arms"][arm][field] if r["cost_complete"] else None
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"seed {seed} costs: {type(exc).__name__}: {exc}")
    for path in sorted((directory / "workers").glob("*.json")):
        row = read(path)
        if row:
            try:
                row["heartbeat_age_seconds"] = max(0, time.time() - row["heartbeat"])
                if row["status"] in {"preflight", "running", "idle"} and row["heartbeat_age_seconds"] > 60:
                    row["status"] = "heartbeat_stale"
                workers.append(row)
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"{path}: {exc}")
    stats = {}
    for arm in plan["arms"]:
        rows = [r for r in tasks if r["arm"] == arm]
        rewards = [r["reward_percent"] for r in rows]
        times = [r.get("selection_training_preparation_gpu_seconds") for r in rows]
        stats[arm] = {"completed_seeds": sum(v is not None for v in rewards), "planned_seeds": len(plan["seeds"]),
                      "mean_reward_percent": statistics.mean(rewards) if all(v is not None for v in rewards) else None,
                      "mean_selection_training_preparation_gpu_seconds": statistics.mean(times)
                          if all(v is not None for v in times) else None}
    return {"dataset": plan["dataset"], "output_root": str(root), "generated": time.time(),
            "stop_requested": (directory / "stop.json").exists(), "counts": dict(Counter(r["status"] for r in tasks)),
            "complete": not errors and all(r["status"] == "complete" for r in tasks),
            "tasks": tasks, "workers": workers, "costs": costs, "arm_statistics": stats,
            "errors": errors, "warnings": warnings,
            "units": "reward percent; allocated GPU-seconds (not wall seconds); missing values are null"}


def render(report, *, results=False):
    lines = [f"SRGC {report['dataset']} {'RESULTS' if results else 'STATUS'}", str(report["output_root"]),
             "  ".join(f"{key}={value}" for key, value in report["counts"].items()),
             "seed arm        status                step reward(%) train+select+prep(GPU-s)"]
    def cell(value):
        return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
    for row in report["tasks"]:
        if results and row["arm"] in {"cache", "prefix"}:
            continue
        lines.append(f"{row['seed']:4} {row['arm']:10} {row['status']:21} {cell(row['step']):>4} "
                     f"{cell(row['reward_percent']):>9} {cell(row.get('selection_training_preparation_gpu_seconds')):>24}")
        if row["arm"] == "cache" and "cache_saved_prompts" in row:
            age = row["cache_last_write_age_seconds"]
            lines.append(f"     saved_prompts={row['cache_saved_prompts']}/{row['cache_total_prompts']} "
                         f"exported={row['cache_exported_prompts']} "
                         f"last_write_age={'none' if age is None else f'{age:.0f}s'}")
            for progress in row.get("progress", []):
                lines.append(f"     stage={progress.get('stage')} last_work_age="
                             f"{max(0, time.time() - progress.get('updated', time.time())):.0f}s")
            if row["status"] in {"failed", "interrupted", "recoverable"}:
                lines.append(f"     log={row['log']}")
    for worker in report["workers"]:
        lines.append(f"node={worker.get('host')} {worker['status']} task={worker.get('task')} "
                     f"heartbeat_age={worker['heartbeat_age_seconds']:.0f}s")
        if worker.get("error"):
            lines.append(f"  {worker['error']}")
    lines.extend(f"WARNING {w}" for w in report["warnings"])
    lines.extend(f"ERROR {e}" for e in report["errors"])
    return "\n".join(lines) + "\n"


def export(report, destination):
    root = Path(report["output_root"])
    with lease(root / ".queue" / "results-export.lock", wait=True):
        text = render(report, results=True)
        atomic_json(root / "results.json", report)
        atomic_text(root / "results.txt", text)
        buffer = io.StringIO()
        fields = ["seed", "arm", "status", "step", "reward_percent", "switched_at", "cost_complete", *COST_FIELDS]
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(r for r in report["tasks"] if r["arm"] not in {"cache", "prefix"})
        atomic_text(root / "results.csv", buffer.getvalue())
        atomic_text(destination, text)
        atomic_json(destination.with_suffix(".json"), report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "results"))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--watch", nargs="?", type=float, const=10)
    parser.add_argument("--output", type=Path, help="home text export path (JSON sidecar is also saved for results)")
    args = parser.parse_args()
    if args.watch is not None and (args.action != "status" or not math.isfinite(args.watch) or args.watch <= 0):
        parser.error("--watch is a positive interval for status only")
    try:
        while True:
            report = snapshot(args.plan, include_costs=args.action == "results")
            dataset = "math" if report["dataset"] == "math_train" else "mbpp"
            destination = args.output or Path.home() / f"srgc-rebuttal-{dataset}-{args.action}.txt"
            if args.action == "results":
                export(report, destination)
            else:
                atomic_text(destination, render(report))
            print(json.dumps(report, indent=2, allow_nan=False) if args.json else render(report, results=args.action == "results"), flush=True)
            if not args.json:
                print(f"Saved: {destination}", flush=True)
            if args.watch is None:
                raise SystemExit(1 if report["errors"] else 0)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
