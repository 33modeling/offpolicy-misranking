"""Read-only RLOO evidence adapted to the existing experiment dashboard."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mbpp_status as display
import rloo_experiment as experiment

LABELS = {"random": "Random", "passrate_beta": "Cached", "fresh_r": "On-policy"}


def read(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path.name}")
    return value


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def meter_progress(directory):
    progress = read(directory / 'progress.json')
    event = progress.get('event_id')
    if (progress.get('state') == 'running' and isinstance(event, str) and event
            and Path(event).name == event and event not in {'.', '..'}):
        path = directory / 'cost-events' / f'{event}.json'
        if path.is_file():
            receipt = read(path)
            fields = ('event_id', 'phase', 'ledger', 'gpus', 'gpu_type', 'host')
            if (receipt.get('state') == 'finished' and type(receipt.get('exit_code')) is int
                    and all(key in progress and key in receipt and progress[key] == receipt[key] for key in fields)
                    and all(display.switch_status.number(receipt.get(key), -1) >= 0
                            for key in ('seconds', 'allocated_gpu_seconds', 'time'))):
                progress = {**progress, 'state': 'finished' if receipt['exit_code'] == 0 else 'failed'}
    return progress


def contract(out, seed, drift):
    c = read(out / "experiment.json")
    if (c.get("schema") != experiment.SCHEMA or c.get("objective") != "rloo"
            or c.get("arms") != list(experiment.ARMS) or c.get("steps") != 100
            or c.get("eval_n") != 300 or c.get("eval_k") != 8
            or c.get("source", {}).get("seed") != seed
            or c.get("source", {}).get("drift") != drift):
        raise ValueError("RLOO contract identity mismatch")
    return c


def saved_evaluation(out, arm, c):
    """Verify reporting receipts/data without hashing model or optimizer files."""
    target = out / arm / "evaluation"
    count = 0
    for shard in range(4):
        receipt = target / f"shard-{shard}.done.json"
        if not receipt.exists():
            continue
        seal = read(receipt)
        binding = seal.get("binding", {})
        expected = {"experiment_sha256": digest(out / "experiment.json"),
                    "inputs_sha256": digest(out / "inputs.json"), "arm": arm, "shard": shard}
        if any(binding.get(key) != value for key, value in expected.items()):
            raise ValueError(f"evaluation binding mismatch: shard {shard}")
        policy = (Path(c["source"]["source_run"]) / f"policy_step_{c['source']['drift']}"
                  if arm == "before" and c["source"]["drift"] else
                  None if arm == "before" else out / arm / "policy")
        if policy is None:
            if binding.get("manifest_sha256") is not None or binding.get("adapter_sha256") is not None:
                raise ValueError("base-model evaluation binding mismatch")
        elif binding.get("manifest_sha256") != digest(policy / "policy_train.json"):
            raise ValueError("evaluation policy manifest mismatch")
        path = target / f"shard-{shard}.jsonl"
        if seal.get("rollouts_sha256") != digest(path):
            raise ValueError(f"evaluation seal mismatch: shard {shard}")
        experiment.ed.reward_rows(path, range(c["eval_n"] * shard // 4,
                                            c["eval_n"] * (shard + 1) // 4), c["eval_k"])
        count += 1
    return count


def observe(out, arm, seed, drift, c, error, *, now):
    directory = out / arm
    task = dict(kind="prefix" if arm == "before" else "branch", arm=arm,
                seed=seed, step=drift, directory=str(directory.relative_to(out.parent)),
                status="READY" if c else "WAIT", reason=error or ("" if c else "not prepared"),
                unverified=c is None)
    try:
        if c:
            if arm != "before" and (directory / "policy/policy_train.json").exists():
                manifest = read(directory / "policy/policy_train.json")
                if (manifest.get("training_objective") != "rloo" or manifest.get("start_step") != drift
                        or manifest.get("completed_steps") != drift + c["steps"]):
                    raise ValueError("RLOO policy publication mismatch")
                task.update(status="EVAL", reason="training saved; evaluation incomplete")
            elif any((directory / "policy").glob("checkpoint-*")):
                task.update(status="RESUME", reason="checkpoint available; resume validation required")
            count = saved_evaluation(out, arm, c)
            task["evaluation_shards"] = count
            if count == 4:
                task.update(status="DONE", reason="four sealed evaluation shards saved")
            elif count:
                task.update(status="EVAL", reason=f"evaluation shards {count}/4")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        task.update(status="WAIT", reason=str(exc))
    attempt = {}
    attempt_path = directory / "queue-attempt.json"
    if attempt_path.exists() and task["status"] != "DONE":
        try:
            attempt = read(attempt_path)
            if attempt.get("state") == "FAILED":
                task.update(status="WAIT", reason=str(attempt.get("error") or "queue attempt failed"))
        except (OSError, ValueError, TypeError) as exc:
            task.update(status="WAIT", reason="invalid queue receipt: " + str(exc))
    progress_path = directory / "progress.json"
    if progress_path.exists():
        try:
            progress = meter_progress(directory)
            age = now - float(progress.get("updated", 0))
            fresh = progress.get("state") == "running" and -5 <= age < 60
            event = progress.get('event_id')
            owned = (not fresh and progress.get('state') == 'running' and isinstance(event, str)
                     and bool(event) and Path(event).name == event and event not in {'.', '..'}
                     and display.switch_status.meter_lease_held(directory)
                     and meter_progress(directory) == progress)
            task.update({key: progress.get(key) for key in ("host", "phase", "seconds", "timeout")})
            task.update(heartbeat_age=age, heartbeat_fresh=fresh, owner_active=owned,
                        training_step=display.switch_status.last_training_step(directory / "policy/grpo_stats.jsonl"))
            if task["status"] != "DONE":
                if fresh or owned:
                    task["status"] = "RUNNING"
                elif progress.get("state") == "running":
                    task.update(status="STALE", reason="heartbeat expired; ownership unconfirmed")
                elif progress.get("state") == "failed":
                    task.update(status="WAIT", reason=str(attempt.get("error") or
                                f"{progress.get('phase', 'phase')} failed; see phase logs"))
        except (OSError, ValueError, TypeError) as exc:
            if task["status"] != "DONE":
                task.update(status="WAIT", reason="invalid progress: " + str(exc))
    return task


def snapshot(root, *, now=None):
    root = Path(root).resolve()
    now = time.time() if now is None else now
    suites = []
    for drift in (0, 400):
        tasks, errors, prepared = [], [], False
        for seed in range(3):
            out = root / f"math500-d{drift}" / f"s{seed}"
            c, error = None, ""
            if (out / "experiment.json").exists():
                try:
                    c = contract(out, seed, drift)
                    prepared = True
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    error = str(exc)
                    errors.append(f"s{seed}: {error}")
            tasks += [observe(out, arm, seed, drift, c, error, now=now)
                      for arm in ("before", *experiment.ARMS)]
        nodes = {}
        for task in tasks:
            if not task.get("host"):
                continue
            age = task.get("heartbeat_age", float("inf"))
            node = dict(host=task["host"], state="RUN" if display.active(task) else "STALE",
                        last_age=age)
            if node["host"] not in nodes or age < nodes[node["host"]]["last_age"]:
                nodes[node["host"]] = node
        baselines = sum(t["status"] == "DONE" for t in tasks if t["kind"] == "prefix")
        suites.append(dict(root=str(root / f"math500-d{drift}"), display_label=f"RLOO MATH d{drift}", prepared=prepared,
                           error="; ".join(errors), tasks=tasks, nodes=list(nodes.values()),
                           registered_tasks=[(s, drift, arm) for s in range(3) for arm in experiment.ARMS],
                           state_points=[(s, drift, "test") for s in range(3)],
                           shared_label="Baseline evaluation", prefix_heading="Before",
                           details=[f"100 updates per arm | baseline evaluation {baselines}/3 (not extra training)",
                                    *errors]))
    return dict(updated=now, suites=suites, subject="RLOO", arm_names=LABELS,
                legend=["18 training arms; 6 shared baseline evaluations are counted separately.",
                        "Cached / On-policy reuse the frozen GRPO selections; only continuation training uses RLOO.",
                        "DONE requires all four evaluation receipts and rollout hashes. Full model/input validation: check/report.",
                        "Node activity uses phase heartbeats; a lock file alone never proves RUN."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    data = snapshot(args.root)
    print(json.dumps(data, indent=2) if args.json else display.render(
        data, width=shutil.get_terminal_size((120, 40)).columns, all_tasks=args.all), flush=True)
    return int(any(suite["error"] for suite in data["suites"]))


if __name__ == "__main__":
    raise SystemExit(main())
