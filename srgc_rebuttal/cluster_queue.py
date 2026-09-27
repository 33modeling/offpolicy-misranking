"""Shared cache -> prefix -> continuation DAG with immutable input handoff."""

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import uuid

from .plan import digest, input_path, load_plan, validate_inputs
from .runtime import Busy, arm_complete, atomic_json, code_digest, lease, prefix_ready, run_root


@dataclass(frozen=True)
class Task:
    seed: int
    arm: str

    @property
    def key(self):
        return f"seed-{self.seed}.{self.arm}"


def input_info(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    validate_inputs(data, require_cache=False)
    source = {k: v for k, v in data.items() if k != "cached_rewards"}
    if isinstance(source["provenance"], dict):
        source["provenance"] = {k: v for k, v in source["provenance"].items() if k != "cache"}
    complete = set(data.get("cached_rewards", {})) == set(data["candidate_ids"])
    return {"input_sha256": hashlib.sha256(raw).hexdigest(),
            "source_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
            "pending_cache": not complete}, data


class TaskQueue:
    def __init__(self, plan_path: Path):
        self.plan_path = plan_path.resolve()
        self.plan = load_plan(self.plan_path)
        self.root = run_root(self.plan_path, self.plan)
        self.directory = self.root / ".queue"
        self.protocol = {"schema": "srgc-shared-queue-v2", "plan_sha256": digest(self.plan_path),
                         "implementation_sha256": code_digest(), "inputs": {}}
        for seed in self.plan["seeds"]:
            info, _ = input_info(input_path(self.plan_path, self.plan, seed))
            self.protocol["inputs"][str(seed)] = info
        marker = self.directory / "protocol.json"
        if marker.exists():
            proposed = self.protocol
            self.protocol = json.loads(marker.read_text())
            if self.protocol.get("implementation_sha256") != proposed["implementation_sha256"]:
                self._upgrade_unstarted(proposed)
        self.identities, self.cache_ready, self._prefix_cache = {}, {}, {}
        self.claim_fds = ()
        self.tasks = [Task(s, phase) for phase in ("cache", "prefix", "on_policy", "switch", "sr", "random")
                      for s in self.plan["seeds"]]
        self.verify()

    def _upgrade_unstarted(self, proposed):
        """A rejected node admission is not an experiment that has started."""
        marker = self.directory / "protocol.json"
        with lease(self.directory / "bind.lock", wait=True):
            previous = json.loads(marker.read_text())
            self.protocol = previous
            if previous.get("implementation_sha256") == proposed["implementation_sha256"]:
                return
            if {**previous, "implementation_sha256": proposed["implementation_sha256"]} != proposed:
                return
            if any(any((self.directory / name).glob("*.json")) for name in ("tasks", "attempts")):
                return
            if any(self.root.glob("seed-*")):
                return
            for seed in self.plan["seeds"]:
                cache = input_path(self.plan_path, self.plan, seed).with_suffix(".cache")
                if any(cache.rglob("*.json")):
                    return
            for path in (self.directory / "workers").glob("*.json"):
                record = json.loads(path.read_text())
                if record.get("status") not in {"failed", "stopped"}:
                    return
            atomic_json(self.directory / "startup-history" / f"{previous['implementation_sha256']}.json", previous)
            atomic_json(marker, proposed)
            self.protocol = proposed

    def bind(self):
        with lease(self.directory / "bind.lock", wait=True):
            path = self.directory / "protocol.json"
            if path.exists():
                self.protocol = json.loads(path.read_text())
            self.verify()
            if not path.exists():
                atomic_json(path, self.protocol)

    def verify(self):
        p = self.protocol
        if (p.get("schema") != "srgc-shared-queue-v2" or
                p["plan_sha256"] != digest(self.plan_path) or p["implementation_sha256"] != code_digest()):
            raise ValueError("code or plan changed; use a new output root for a different protocol")
        for seed in self.plan["seeds"]:
            path = input_path(self.plan_path, self.plan, seed)
            info, data = input_info(path)
            original = p["inputs"][str(seed)]
            if info["source_sha256"] != original["source_sha256"]:
                raise ValueError("source inputs changed after queue initialization")
            seal_path = self.directory / "inputs" / f"seed-{seed}.json"
            if seal_path.exists():
                expected_hash = json.loads(seal_path.read_text())["input_sha256"]
            else:
                expected_hash = None if original["pending_cache"] else original["input_sha256"]
            if expected_hash is not None and info["input_sha256"] != expected_hash:
                raise ValueError("frozen inputs changed after cache completion")
            ready = not info["pending_cache"]
            if original["pending_cache"] and ready:
                cache = data.get("provenance", {}).get("cache", {})
                expected = {"model": self.plan["model"], "model_revision": self.plan["model_revision"],
                            "verifier": self.plan["verifier"], "cache_seed": seed,
                            "responses": self.plan["responses"], "max_new_tokens": self.plan["max_new_tokens"]}
                if not isinstance(cache, dict) or any(cache.get(k) != v for k, v in expected.items()):
                    raise ValueError("generated cache does not match the frozen plan and seed")
                receipt = path.with_suffix(".cache") / "cost-summary.json"
                ready = receipt.exists() and json.loads(receipt.read_text())["bundle_sha256"] == info["input_sha256"]
                if ready and (self.directory / "protocol.json").exists():
                    with lease(self.directory / "inputs" / f"seed-{seed}.lock", wait=True):
                        if seal_path.exists() and json.loads(seal_path.read_text())["input_sha256"] != info["input_sha256"]:
                            raise ValueError("concurrent cache handoff differs")
                        if not seal_path.exists():
                            atomic_json(seal_path, {"input_sha256": info["input_sha256"]})
            elif original["pending_cache"] and info["input_sha256"] != original["input_sha256"]:
                raise ValueError("pending inputs changed outside the atomic cache handoff")
            self.cache_ready[seed] = ready
            self.identities[seed] = {"seed": seed, "plan_sha256": p["plan_sha256"],
                                    "implementation_sha256": p["implementation_sha256"],
                                    "input_sha256": info["input_sha256"]}

    def complete(self, task):
        if task.arm == "cache":
            return self.cache_ready[task.seed]
        if not self.cache_ready[task.seed]:
            return False
        folder = self.root / f"seed-{task.seed}"
        if task.arm == "prefix":
            paths = [folder / "prefix.pt", folder / "prefix-ready.json"]
            if not all(p.exists() for p in paths):
                return False
            signature = tuple((p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
            signature += (self.identities[task.seed]["input_sha256"],)
            if self._prefix_cache.get(task.seed) != signature:
                if not prefix_ready(folder, self.identities[task.seed], self.plan["shared_prefix_updates"]):
                    return False
                self._prefix_cache[task.seed] = signature
            return True
        return arm_complete(folder, self.identities[task.seed], task.arm, self.plan["total_updates"])

    def dependency(self, task):
        return None if task.arm == "cache" else Task(task.seed, "cache" if task.arm == "prefix" else "prefix")

    def ready(self, task):
        parent = self.dependency(task)
        return parent is None or (self.complete(parent) and not self.locked(parent))

    def receipt(self, task):
        return self.directory / "tasks" / f"{task.key}.json"

    def locked(self, task):
        try:
            with lease(self.directory / "leases" / f"{task.key}.lock"):
                return False
        except Busy:
            return True

    @contextmanager
    def claim(self, *, retry_failed=False, max_attempts=3, retry_delay=60, worker_id=None):
        self.verify()
        for task in self.tasks:
            if self.complete(task) or not self.ready(task):
                continue
            with ExitStack() as locks:
                try:
                    handle = locks.enter_context(lease(self.directory / "leases" / f"{task.key}.lock"))
                except Busy:
                    continue
                previous = json.loads(self.receipt(task).read_text()) if self.receipt(task).exists() else {}
                attempts = previous.get("attempt", 0)
                if attempts >= max_attempts:
                    continue
                if previous.get("status") == "failed" and (not retry_failed or attempts >= max_attempts or
                        time.time() < previous.get("finished", 0) + retry_delay):
                    continue
                if self.complete(task):
                    continue
                if previous.get("status") == "running":
                    atomic_json(self.directory / "attempts" / f"{previous['attempt_id']}.json",
                                {**previous, "status": "abandoned", "recovered": time.time()})
                record = {"task": task.key, "status": "running", "host": socket.gethostname(),
                          "pid": os.getpid(), "worker_id": worker_id, "attempt": attempts + 1,
                          "attempt_id": uuid.uuid4().hex, "started": time.time()}
                atomic_json(self.receipt(task), record)
                self.claim_fds = (handle.fileno(),)
                try:
                    yield task
                finally:
                    self.claim_fds = ()
                return
        yield None

    def finish(self, task, exit_code, *, interrupted=False):
        previous = json.loads(self.receipt(task).read_text())
        error = None
        try:
            self.verify()
            if exit_code == 0 and not self.complete(task):
                exit_code = 2
        except Exception as exc:
            exit_code = exit_code or 2
            error = f"{type(exc).__name__}: {exc}"
        record = {**previous, "status": "complete" if exit_code == 0 else "interrupted" if interrupted else "failed",
                  "exit_code": exit_code, "finished": time.time(), "validation_error": error}
        atomic_json(self.directory / "attempts" / f"{previous['attempt_id']}.json", record)
        atomic_json(self.receipt(task), record)
        return exit_code

    def status(self, *, max_attempts=None):
        self.verify()
        rows = []
        for task in self.tasks:
            record = json.loads(self.receipt(task).read_text()) if self.receipt(task).exists() else {}
            if self.locked(task):
                state = "running"
            elif self.complete(task):
                state = "complete"
            elif not self.ready(task):
                state = f"waiting_for_{self.dependency(task).arm}"
            elif max_attempts is not None and record.get("attempt", 0) >= max_attempts:
                state = "attempts_exhausted"
            elif record.get("status") == "running":
                state = "recoverable"
            else:
                state = record.get("status", "ready")
            rows.append({**record, "task": task.key, "status": state})
        return rows
