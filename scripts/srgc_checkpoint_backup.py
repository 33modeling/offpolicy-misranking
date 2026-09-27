"""CPU-only rolling copies of atomically published SRGC checkpoints."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import zipfile

from srgc_rebuttal.plan import digest, input_path, load_plan
from srgc_rebuttal.runtime import Busy, atomic_json, lease, run_root
from srgc_shared_storage import storage_root


SCHEMA = "srgc-checkpoint-backup-v1"
KEEP = 2


def signature(stat):
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def inside(path, group):
    if not path.resolve().is_relative_to(group):
        raise ValueError(f"checkpoint backup path escapes group storage: {path}")
    return path


def copy_snapshot(source, archive, identity, group):
    """An open descriptor keeps the old inode if training publishes a new file."""
    inside(source, group)
    inside(archive, group)
    if source.is_symlink():
        raise ValueError(f"checkpoint must not be a symlink: {source}")
    index = inside(archive / "latest.json", group)
    with source.open("rb") as original:
        before = signature(os.fstat(original.fileno()))
        if index.is_file():
            latest = json.loads(index.read_text())
            if (latest.get("schema") != SCHEMA or latest.get("identity") != identity or
                    latest.get("source") != str(source)):
                raise ValueError(f"backup identity mismatch: {index}")
            filename = latest["sha256"]
            if len(filename) != 64 or any(c not in "0123456789abcdef" for c in filename):
                raise ValueError(f"invalid backup checksum: {index}")
            saved = inside(archive / filename / "checkpoint.pt", group)
            if (latest["source_signature"] == before and saved.is_file() and
                    signature(saved.stat()) == latest["backup_signature"]):
                return False
        archive.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=".writing-", dir=archive) as folder:
            temporary = Path(folder)
            copied = temporary / "checkpoint.pt"
            checksum = hashlib.sha256()
            with copied.open("wb") as output:
                while chunk := original.read(1024 * 1024):
                    output.write(chunk)
                    checksum.update(chunk)
                output.flush()
                os.fsync(output.fileno())
            if signature(os.fstat(original.fileno())) != before:
                raise ValueError(f"checkpoint changed in place while copying: {source}")
            value = checksum.hexdigest()
            if digest(copied) != value:
                raise ValueError(f"checkpoint backup checksum failed: {source}")
            with zipfile.ZipFile(copied) as checkpoint:
                if checkpoint.testzip() is not None:
                    raise ValueError(f"checkpoint archive integrity failed: {source}")
            receipt = {"schema": SCHEMA, "source": str(source), "identity": identity,
                       "source_signature": before, "sha256": value,
                       "backup_signature": signature(copied.stat()),
                       "saved_ns": time.time_ns(), "copy_wall_seconds": time.perf_counter() - started}
            destination = inside(archive / value, group)
            if destination.is_symlink():
                raise ValueError(f"backup generation must not be a symlink: {destination}")
            if destination.exists():
                if digest(destination / "checkpoint.pt") != value:
                    raise ValueError(f"existing backup is corrupt: {destination}")
                receipt["backup_signature"] = signature((destination / "checkpoint.pt").stat())
                atomic_json(destination / "receipt.json", receipt)
            else:
                atomic_json(temporary / "receipt.json", receipt)
                temporary.rename(destination)
            atomic_json(index, receipt)
        versions = []
        for record in archive.glob("*/receipt.json"):
            old = json.loads(record.read_text())
            if (old.get("schema") == SCHEMA and old.get("source") == str(source) and
                    old.get("identity") == identity and record.parent.name == old.get("sha256")):
                versions.append((old["saved_ns"], record.parent))
        for _, old in sorted(versions, reverse=True)[KEEP:]:
            inside(old, group)
            if old.is_symlink():
                raise ValueError(f"backup generation must not be a symlink: {old}")
            shutil.rmtree(old)
        print(f"BACKUP {source.name} -> {destination / 'checkpoint.pt'}", flush=True)
        return True


def backup_once(plan_path, *, environment=None):
    environment = os.environ if environment is None else environment
    group, _ = storage_root(environment)
    plan_path = inside(Path(plan_path).resolve(), group)
    plan = load_plan(plan_path)
    root = inside(run_root(plan_path, plan), group)
    destination = inside(root / "checkpoint-backups", group)
    result = {"found": 0, "saved": 0, "unchanged": 0, "busy": False, "errors": [],
              "directory": str(destination)}
    try:
        with lease(inside(destination / ".backup.lock", group)):
            for seed in plan["seeds"]:
                folder = inside(root / f"seed-{seed}", group)
                names = ["prefix-latest.pt", "prefix.pt", *[f"{arm}-latest.pt" for arm in plan["arms"]]]
                available = [folder / name for name in names if (folder / name).is_file()]
                result["found"] += len(available)
                if not available:
                    continue
                try:
                    marker = json.loads(inside(folder / "run.json", group).read_text())
                    identity = {key: marker[key] for key in
                                ("seed", "plan_sha256", "input_sha256", "implementation_sha256")}
                    if (identity["seed"] != seed or identity["plan_sha256"] != digest(plan_path) or
                            identity["input_sha256"] != digest(inside(input_path(plan_path, plan, seed), group))):
                        raise ValueError("checkpoint run identity differs from plan/inputs")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    result["errors"].append(f"seed {seed}: {exc}")
                    continue
                for source in available:
                    try:
                        changed = copy_snapshot(source, destination / f"seed-{seed}" / source.stem, identity, group)
                        result["saved" if changed else "unchanged"] += 1
                    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
                        result["errors"].append(f"{source}: {exc}")
    except Busy:
        result["busy"] = True
    return result


def watch(plan_path, stop, *, interval=30):
    def capture(*, final=False):
        try:
            result = backup_once(plan_path)
            state = ("error" if result["errors"] else "another_watcher_copying" if result["busy"] else
                     "copied" if result["saved"] else "no_new_checkpoint" if result["found"] else
                     "waiting_for_checkpoint")
            checked = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            following = "final=true" if final else f"next_check_in={interval}s"
            print(f"BACKUP CHECK {checked} state={state} found={result['found']} "
                  f"saved={result['saved']} unchanged={result['unchanged']} "
                  f"errors={len(result['errors'])} {following}", flush=True)
            for error in result["errors"]:
                print(f"BACKUP ERROR {error}", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"BACKUP ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    while not stop.is_set():
        capture()
        stop.wait(interval)
    capture(final=True)


@contextmanager
def automatic_backup(plan_path):
    stop = threading.Event()
    thread = threading.Thread(target=watch, args=(plan_path, stop), name="checkpoint-backup", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=30)
        if thread.is_alive():
            print("BACKUP WARNING final copy has not finished; existing backups remain intact", file=sys.stderr, flush=True)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args(args)
    if args.watch:
        group, _ = storage_root(os.environ)
        inside(args.plan.resolve(), group)
        print(f"Watching checkpoints every 30s: {args.plan}; keep={KEEP}", flush=True)
        try:
            watch(args.plan, threading.Event())
        except KeyboardInterrupt:
            pass
    else:
        report = backup_once(args.plan)
        print(json.dumps(report, indent=2), flush=True)
        if report["errors"]:
            raise SystemExit(1)
