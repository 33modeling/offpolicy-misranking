"""Claim independent information measurements across nodes on shared storage."""

import argparse
import math
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from scripts.srgc_extra_plan import select_plan
from scripts.srgc_log_tail import tail_lines
from scripts.srgc_pair_inputs import PLANS
from scripts.srgc_shared_storage import route_plan, storage_root
from srgc_rebuttal.existing_runtime import python_path
from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.runtime import Busy, atomic_json, lease
from srgc_research.information_report import (
    PHASES,
    PROTOCOL,
    digest,
    read_measurement,
    read_object,
)

from .information_run import OWNERS, TARGETS

REPO = Path(__file__).resolve().parents[2]
SEEDS = (7, 5, 6, 8, 9)
# Diagnostic ledger recovery generation. Bump only for collector
# execution fixes: improving diagnostics must not relaunch exhausted GPU jobs.
DISPATCH_REVISION = "dabfb7dc5483d140a0964e6a4bb531cf648e500f025d9a70b44d71e41c67dd81"


@dataclass(frozen=True)
class Task:
    dataset: str
    seed: int
    plan: Path
    inputs: Path
    root: Path
    plan_sha256: str
    input_sha256: str

    @property
    def key(self):
        return f"{self.dataset}.seed-{self.seed}.t0"

    @property
    def output(self):
        return self.root / self.dataset / f"seed-{self.seed}" / "t0"

    @property
    def receipt(self):
        return self.root / ".queue" / f"{self.key}.json"

    @property
    def identity(self):
        return {"protocol": PROTOCOL, "dataset": self.dataset, "seed": self.seed, "stage": 0,
                "plan_sha256": self.plan_sha256, "input_sha256": self.input_sha256,
                "source_checkpoint_sha256": None, "probe_prompts": 8, "requested_attention": "sdpa"}


def saved_plan(dataset, source_root):
    """Follow the same read-only active/history routes as existing launchers."""
    templates = REPO / "srgc_rebuttal/experiments"
    names = PLANS[dataset][:2]
    for name in names:
        if (source_root / f".{Path(name).stem}-active.json").is_file():
            return route_plan(templates / name, writing=False)
    for name in names:
        candidate = source_root / "experiments" / name
        if candidate.is_file():
            return candidate
    for name in names:
        candidate = templates / name
        if candidate.is_file():
            plan = load_plan(candidate)
            if all(input_path(candidate, plan, seed).is_file() for seed in SEEDS):
                return candidate
    raise FileNotFoundError(f"no saved {dataset} plan with seed-5–9 inputs found under {source_root}")


def tasks_for(dataset):
    group, source_root = storage_root(os.environ)
    work = Path(os.environ.get("OM_WORK", str(group / os.environ.get("OM_USER", "minsoo3.kim") /
                                             "offpolicy-misranking"))).resolve()
    root = work / "selection-information"
    datasets = ("math", "mbpp") if dataset == "all" else (dataset,)
    plans = {name: saved_plan(name, source_root) for name in datasets}
    tasks = []
    # Interleave datasets so a small node pool starts both immediately.
    for seed in SEEDS:
        for name in datasets:
            if seed not in load_plan(plans[name])["seeds"]:
                raise ValueError(f"seed {seed} is absent from the frozen {name} plan")
            plan = select_plan(plans[name], seed)
            inputs = input_path(plan, load_plan(plan), seed)
            if not inputs.is_file():
                raise FileNotFoundError(f"input not found: {inputs}")
            if plan.resolve().is_relative_to(root) or inputs.resolve().is_relative_to(root):
                raise ValueError("measurement outputs must be separate from the source plan and inputs")
            tasks.append(Task(name, seed, plan.resolve(), inputs, root, digest(plan), digest(inputs)))
    for name in datasets:
        python_path(name, os.environ)
    return tasks


def command_for(task):
    return [python_path(task.dataset, os.environ), "-m", "srgc_research.dispatch.information_run",
            task.dataset, "collect", "--plan", str(task.plan), "--inputs", str(task.inputs),
            "--seed", str(task.seed), "--stage", "0", "--attention", "sdpa",
            "--output", str(task.output)]


def run_task(task, handle):
    """Persist startup errors, retain the claim, and forward owned shutdown."""
    child, stopped = None, []
    log_path = Path(read_receipt(task)["attempt_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def stop(sig, _frame):
        stopped.append(sig)
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        # Keep a real file as the child's output: it remains writable if a node
        # launcher dies, while the inherited claim still prevents a duplicate.
        with log_path.open("x") as log, log_path.open(errors="replace") as reader:
            def relay():
                while chunk := reader.read(65536):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
            child = subprocess.Popen(command_for(task), cwd=REPO, start_new_session=True,
                                     pass_fds=(handle.fileno(),), stdout=log, stderr=subprocess.STDOUT)
            if stopped:
                stop(stopped[0], None)
            while True:
                relay()
                try:
                    code = child.wait(timeout=.2)
                    relay()
                    return 128 + stopped[0] if stopped else (code if code >= 0 else 128 - code)
                except subprocess.TimeoutExpired:
                    pass
    finally:
        if child is not None and child.poll() is None:
            stop(signal.SIGTERM, None)
            child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def node_available():
    from scripts import srgc_process_guard as guard
    from srgc_rebuttal import cluster
    markers = guard.TARGET_MARKERS, guard.OWNER_MARKERS
    guard.TARGET_MARKERS, guard.OWNER_MARKERS = TARGETS, OWNERS
    try:
        guard.reap_orphans()
        _, uuids = cluster.gpu_identity()
        root = guard.canonical_lock_root()
        if root is None:
            raise ValueError("information queue requires shared group storage")
        with cluster.device_leases(root, uuids):
            pass
    except Busy:
        return False
    finally:
        guard.TARGET_MARKERS, guard.OWNER_MARKERS = markers
    return True


def matches_task(task, identity):
    if any(identity.get(k) != v for k, v in task.identity.items()):
        raise ValueError("measurement belongs to different inputs or settings")


def signature(task):
    """Cache a successful hash audit only while every evidence file is unchanged."""
    if not (task.output / "endpoint.json").is_file():
        return None
    paths = {task.output / name for name in ("endpoint.json", "manifest.json", "inputs.json", "plan.json")}
    phase_paths = read_object(task.output / "endpoint.json")["phases"]
    if set(phase_paths) != set(PHASES):
        raise ValueError("completed measurement is missing phases")
    for name in PHASES:
        path = (task.output / phase_paths[name]).resolve()
        if not path.is_relative_to(task.output.resolve()):
            raise ValueError("phase path escapes its measurement")
        paths.add(path)
        for artifact in read_object(path)["artifacts"]:
            target = (task.output / artifact["file"]).resolve()
            if not target.is_relative_to(task.output.resolve()):
                raise ValueError("measured tensor path escapes its measurement")
            paths.add(target)
    return [[str(p), s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
            for p in sorted(paths) for s in (p.stat(),)]


def read_receipt(task):
    row = read_object(task.receipt) if task.receipt.exists() else {}
    if row:
        if row.get("identity") != task.identity or row.get("task") != task.key:
            raise ValueError("queue receipt belongs to a different measurement")
        if row.get("status") not in {"running", "complete", "failed", "busy", "interrupted"}:
            raise ValueError("invalid queue status")
    if type(row.get("attempt", 0)) is not int or row.get("attempt", 0) < 0:
        raise ValueError("invalid queue attempt count")
    for field in ("started", "finished"):
        if field in row and (type(row[field]) not in (int, float) or
                             not math.isfinite(row[field]) or row[field] < 0):
            raise ValueError(f"invalid queue {field} timestamp")
    if row.get("status") == "failed" and "finished" not in row:
        raise ValueError("failed queue receipt has no finish timestamp")
    return row


def verified_signature(task):
    before = signature(task)
    endpoint, _ = read_measurement(task.output)
    matches_task(task, endpoint["identity"])
    if before is None or signature(task) != before:
        raise ValueError("measurement evidence changed during its completion audit")
    return before


def failure_detail(task, receipt):
    """Recover both new startup logs and older rank logs without GPU imports."""
    paths = ([Path(receipt["attempt_log"])] if receipt.get("attempt_log") else [])
    paths.append(task.output / "task.log")
    if receipt.get("error"):
        return receipt["error"], str(paths[0])
    for path in paths:
        # Queue metadata must never authorize reading unrelated files.
        if not path.resolve().is_relative_to(task.root.resolve()) or not path.is_file():
            continue
        try:
            lines = tail_lines(path, 80)
        except OSError:
            continue
        errors = [line.strip() for line in lines if any(marker in line for marker in (
            "ERROR:", "Error:", "Exception:", "No module named", "FAILED", "GUARD refused"))]
        specific = [line for line in errors if "ChildFailedError" not in line and "FAILED" not in line]
        if specific or errors:
            detail = (specific or errors)[-1]
        else:
            detail = next((line.strip() for line in reversed(lines) if line.strip()), "")
        if detail:
            return detail[-2000:], str(path)
    return receipt.get("error") or f"collector exited with code {receipt.get('exit_code', '?')}", (
        receipt.get("attempt_log") or str(task.output / "task.log"))


def report_failure(task, receipt):
    error, log = failure_detail(task, receipt)
    print(f"FAILED {task.key}: {error}\n  log: {log}", file=sys.stderr, flush=True)


def failure_summary(tasks, failures):
    """End stdout with the actual causes, so the visible footer can be copied."""
    rows = []
    for task in tasks:
        error = failures.get(task.key)
        if error is None:
            try:
                receipt = read_receipt(task)
                if receipt.get("status") == "failed":
                    error = failure_detail(task, receipt)[0]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                error = str(exc)
        if error is not None:
            # Keep each cause on its own line, including multi-line exceptions.
            error = " ".join(str(error).splitlines())
            rows.append(f"{task.key}: {error}")
    print("\nINFORMATION FAILURE DETAILS\n" + "\n".join(rows), flush=True)


def sweep(tasks, *, runner=None, available=None, max_attempts=3, retry_delay=120, now=time.time,
          failures=None):
    runner = run_task if runner is None else runner
    available = node_available if available is None else available
    counts = {"complete": 0, "busy": 0, "waiting": 0, "failed": 0}
    for task in tasks:
        with ExitStack() as locks:
            try:
                handle = locks.enter_context(lease(task.receipt.with_suffix(".lock")))
                # Respect manual/old launchers and ranks surviving a killed parent.
                with lease(task.output / ".dispatch.lock"), lease(task.output / ".execution.lock"):
                    pass
            except Busy:
                counts["busy"] += 1
                continue
            try:
                previous = read_receipt(task)
                if digest(task.plan) != task.plan_sha256 or digest(task.inputs) != task.input_sha256:
                    raise ValueError("source plan or cached inputs changed after queue startup")
                if (task.output / "manifest.json").exists():
                    matches_task(task, read_object(task.output / "manifest.json")["identity"])
                fingerprint = signature(task)
                if previous.get("status") == "complete" and fingerprint is None:
                    raise ValueError("previously completed measurement lost its endpoint")
                if fingerprint is not None:
                    if previous.get("status") != "complete" or previous.get("verified_files") != fingerprint:
                        fingerprint = verified_signature(task)
                        atomic_json(task.receipt, {**previous, "task": task.key, "identity": task.identity,
                            "status": "complete", "attempt": previous.get("attempt", 0),
                            "verified_files": fingerprint})
                    counts["complete"] += 1
                    if failures is not None:
                        failures.pop(task.key, None)
                    continue
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"INVALID {task.key}: {exc}", file=sys.stderr, flush=True)
                if failures is not None:
                    failures[task.key] = str(exc)
                counts["failed"] += 1
                continue
            attempts = previous.get("attempt", 0)
            if previous.get("status") == "failed" and previous.get("dispatch_revision") != DISPATCH_REVISION:
                # Archive under the same claim, only after identity/evidence
                # validation. Multiple nodes cannot replenish it repeatedly.
                history = task.root / ".queue" / "history" / task.key / f"{uuid.uuid4().hex}.json"
                atomic_json(history, previous)
                previous = {**previous, "attempt": 0, "dispatch_revision": DISPATCH_REVISION,
                            "previous_receipt": str(history), "finished": 0}
                atomic_json(task.receipt, previous)
                attempts = 0
                print(f"RETRY {task.key}: dispatcher updated; saved phases retained", flush=True)
            # Node loss does not spend a failure retry; saved phases stay intact.
            if previous.get("status") == "running":
                attempts = max(0, attempts - 1)
            if attempts >= max_attempts:
                counts["failed"] += 1
                report_failure(task, previous)
                if failures is not None:
                    failures[task.key] = failure_detail(task, previous)[0]
                continue
            if previous.get("status") == "failed" and now() < previous["finished"] + retry_delay:
                counts["waiting"] += 1
                continue
            if not available():
                return counts, 75
            record = {"task": task.key, "identity": task.identity, "host": socket.gethostname(),
                      "pid": os.getpid(), "status": "running", "attempt": attempts + 1, "started": now(),
                      "dispatch_revision": DISPATCH_REVISION,
                      "previous_receipt": previous.get("previous_receipt"),
                      "attempt_log": str(task.root / ".queue" / "logs" / task.key / f"{uuid.uuid4().hex}.log")}
            atomic_json(task.receipt, record)
            print(f"TASK {task.key} host={record['host']} attempt={record['attempt']}", flush=True)
            error = None
            try:
                code = runner(task, handle)
            except (OSError, subprocess.SubprocessError) as exc:
                error, code = f"cannot start collector: {exc}", 1
            fingerprint = None
            if code == 0:
                try:
                    fingerprint = verified_signature(task)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    print(f"INVALID {task.key} completion: {exc}", file=sys.stderr, flush=True)
                    error, code = str(exc), 1
            status = "busy" if code == 75 else "interrupted" if code in (130, 143) else (
                "complete" if code == 0 else "failed")
            record = {**record, "status": status, "exit_code": code, "finished": now(),
                "attempt": attempts if status in {"busy", "interrupted"} else attempts + 1,
                "verified_files": fingerprint}
            if status == "failed":
                record["error"] = error or failure_detail(task, record)[0]
            atomic_json(task.receipt, record)
            print(f"TASK {task.key} {status} exit={code}", flush=True)
            if status == "failed":
                report_failure(task, record)
                if failures is not None:
                    failures[task.key] = record["error"]
            elif status == "complete" and failures is not None:
                failures.pop(task.key, None)
            return counts, code
    return counts, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("math", "mbpp", "all"))
    args = parser.parse_args(argv)
    try:
        tasks = tasks_for(args.dataset)
        failures = {}
        print(f"INFORMATION {args.dataset}: {len(tasks)} shared tasks, seeds 5–9, stage 0", flush=True)
        while True:
            counts, code = sweep(tasks, failures=failures)
            if code in (130, 143):
                return code
            if code is not None and code != 75:
                continue
            if counts["complete"] == len(tasks):
                print("COMPLETE: all requested information measurements", flush=True)
                return 0
            if code is None and counts["complete"] + counts["failed"] == len(tasks):
                print(f"FAILED: {counts['failed']} measurement(s)", flush=True)
                failure_summary(tasks, failures)
                return 1
            print(f"WAIT: {counts}", flush=True)
            time.sleep(10)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
