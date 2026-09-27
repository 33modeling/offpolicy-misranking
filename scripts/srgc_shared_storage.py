"""Keep large rebuttal artifacts on group storage without changing run identities."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.cluster_queue import input_info
from srgc_rebuttal.runtime import atomic_json, lease, run_root


def storage_root(environment):
    group = Path(environment.get("GROUP_VOLUME", "/group-volume")).resolve()
    work = Path(environment.get("OM_WORK", str(group / environment.get("OM_USER", "minsoo3.kim") / "offpolicy-misranking")))
    root = Path(environment.get("SRGC_STORAGE_ROOT", str(work / "srgc-rebuttal"))).resolve()
    if not work.resolve().is_relative_to(group) or not root.is_relative_to(group) or root == group:
        raise ValueError(f"SRGC_STORAGE_ROOT must be inside group storage {group}, not a user volume: {root}")
    if not group.is_dir():
        raise ValueError(f"group volume is unavailable: {group}; refusing user-volume fallback")
    return group, root


def artifact_pairs(source, target):
    plan = load_plan(source)
    pairs = [(source, target)]
    for seed in plan["seeds"]:
        old, new = input_path(source, plan, seed), input_path(target, plan, seed)
        pairs.extend((old.with_suffix(suffix), new.with_suffix(suffix))
                     for suffix in (".json", ".cache", ".cache-responses.jsonl"))
    pairs.append((run_root(source, plan), run_root(target, plan)))
    return pairs


def has_work(source):
    plan = load_plan(source)
    root = run_root(source, plan)
    if root.exists() and any(root.iterdir()):
        return True
    for seed in plan["seeds"]:
        path = input_path(source, plan, seed)
        if path.with_suffix(".cache").exists() or path.with_suffix(".cache-responses.jsonl").exists():
            return True
        if path.exists() and json.loads(path.read_text()).get("cached_rewards"):
            return True
    return False


def publish_copy(source, target):
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.storage-copy")
    try:
        if source.is_dir():
            shutil.copytree(source, temporary)
        else:
            shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        else:
            temporary.unlink(missing_ok=True)


def stage(source, *, environment, migrate=False):
    source = source.resolve()
    group, root = storage_root(environment)
    plan = load_plan(source)
    target = root / "experiments" / source.name
    for _, path in artifact_pairs(source, target):
        if not path.resolve().is_relative_to(group):
            raise ValueError(f"plan points outside group storage: {path}")
    receipt = root / f".{source.stem}-storage.json"
    with lease(root / ".storage.lock", wait=True):
        if receipt.exists():
            if not target.is_file() or target.read_bytes() != source.read_bytes():
                raise ValueError("group-storage plan differs from the source; refusing to overwrite")
            for seed in plan["seeds"]:
                if not input_path(target, plan, seed).is_file():
                    raise ValueError("group-storage input is missing; refusing to reset existing work")
                original, _ = input_info(input_path(source, plan, seed))
                existing, _ = input_info(input_path(target, plan, seed))
                if original["source_sha256"] != existing["source_sha256"]:
                    raise ValueError("group-storage inputs differ from the source; refusing to reuse another experiment")
            return target
        work = has_work(source)
        if work and not migrate:
            raise ValueError("existing cache/results remain at " + str(source.parent.parent) +
                "; stop all dataset workers, then run: python scripts/run_srgc_rebuttal.py storage --dataset " +
                ("mbpp" if plan["dataset"] == "mbpp" else "math") + " --migrate")
        pairs = artifact_pairs(source, target)
        if any(new.exists() for _, new in pairs):
            raise ValueError(f"uncommitted storage copy exists at {root}; inspect it before retrying")
        with ExitStack() as locks:
            old_root = run_root(source, plan)
            if work:
                for path in (old_root / ".queue/workers").glob("*.json"):
                    worker = json.loads(path.read_text())
                    if worker.get("status") not in {"stopped", "failed", "finished", "complete"}:
                        raise ValueError(f"stop the source worker before migration: {path}")
                # Keep all task and execution leases throughout the copy.
                for seed in plan["seeds"]:
                    for arm in ("cache", "prefix", *plan["arms"]):
                        locks.enter_context(lease(old_root / ".queue/leases" / f"seed-{seed}.{arm}.lock"))
                    locks.enter_context(lease(input_path(source, plan, seed).with_suffix(".cache") / "execution.lock"))
                    for arm in ("prefix", "all", *plan["arms"]):
                        locks.enter_context(lease(old_root / f"seed-{seed}" / f".{arm}.execution.lock"))
            for old, new in pairs:
                publish_copy(old, new)
            for seed in plan["seeds"]:
                if not input_path(target, plan, seed).is_file():
                    raise ValueError(f"source input is missing: {input_path(source, plan, seed)}")
            atomic_json(receipt, {"source_plan": str(source), "plan": str(target),
                                 "migrated_existing_work": work, "originals_preserved": True})
        return target


def route_plan(source, *, writing, migrate=False):
    source = source.resolve()
    group, root = storage_root(os.environ)
    plan = load_plan(source)
    if source.is_relative_to(group):
        paths = [run_root(source, plan), *(input_path(source, plan, s) for s in plan["seeds"])]
        if any(not p.is_relative_to(group) for p in paths):
            raise ValueError("group-storage plan points to a user-volume artifact")
        target = source
    elif writing or migrate:
        target = stage(source, environment=os.environ, migrate=migrate)
    else:
        target = root / "experiments" / source.name
        if not (root / f".{source.stem}-storage.json").exists():
            return source
    if writing:
        os.environ.setdefault("OM_WORK", str(group / os.environ.get("OM_USER", "minsoo3.kim") / "offpolicy-misranking"))
        cache = root / "runtime-cache"
        for key, path in {"HF_HOME": cache / "huggingface", "HF_HUB_CACHE": cache / "huggingface/hub",
                          "HUGGINGFACE_HUB_CACHE": cache / "huggingface/hub",
                          "TRANSFORMERS_CACHE": cache / "huggingface/hub",
                          "HF_MODULES_CACHE": cache / "huggingface/modules",
                          "HF_DATASETS_CACHE": cache / "huggingface/datasets", "XDG_CACHE_HOME": cache,
                          "TORCH_HOME": cache / "torch"}.items():
            os.environ[key] = str(path)
        print(f"[storage] inputs/cache: {input_path(target, plan, plan['seeds'][0]).parent}; "
              f"results: {run_root(target, plan)}", file=sys.stderr, flush=True)
    return target
