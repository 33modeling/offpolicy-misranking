"""Keep large rebuttal artifacts on group storage without changing run identities."""

from contextlib import ExitStack, nullcontext
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

from srgc_rebuttal.plan import digest, input_path, load_plan, validate_inputs
from srgc_rebuttal.cluster_queue import input_info
from srgc_rebuttal.runtime import atomic_json, code_digest, lease, run_root


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


def imported_cache(data):
    """A complete cache copied from a finished source run (file + sha256) is reused, never regenerated."""
    cache = data.get("provenance", {}).get("cache")
    complete = set(data.get("cached_rewards", {})) == set(data["candidate_ids"])
    return complete and isinstance(cache, dict) and {"file", "sha256"} <= set(cache)


def fresh_plan(source, environment, name, *, _locked=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("fresh run name must contain only letters, numbers, underscores or hyphens")
    group, root = storage_root(environment)
    plan = load_plan(source)
    cohort = root / "fresh" / name / source.stem
    if not cohort.resolve().is_relative_to(group):
        raise ValueError("fresh run directory resolves outside group storage")
    target = cohort / "experiments" / source.name
    active = root / f".{source.stem}-active.json"
    with nullcontext() if _locked else lease(root / ".storage.lock", wait=True):
        if cohort.exists():
            if not target.is_file() or target.read_bytes() != source.read_bytes():
                raise ValueError("fresh run name already belongs to another plan; choose a different --fresh name")
        else:
            temporary = cohort.with_name(f".{cohort.name}.{uuid.uuid4().hex}.new")
            temporary_plan = temporary / "experiments" / source.name
            try:
                publish_copy(source, temporary_plan)
                for seed in plan["seeds"]:
                    data = json.loads(input_path(source, plan, seed).read_text())
                    if not imported_cache(data):
                        data["cached_rewards"] = {}
                        data["provenance"].pop("cache", None)
                    validate_inputs(data, require_cache=False)
                    destination = input_path(temporary_plan, plan, seed)
                    if not destination.is_relative_to(temporary.resolve()):
                        raise ValueError("fresh runs require relative input paths inside the run directory")
                    atomic_json(destination, data)
                if not run_root(temporary_plan, plan).is_relative_to(temporary.resolve()):
                    raise ValueError("fresh run outputs must be inside the run directory")
                cohort.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(cohort)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        atomic_json(active, {"plan": str(target), "source_plan_sha256": digest(source), "fresh_run": name,
                             "implementation_sha256": code_digest()})
    return target


def automatic_plan(source, environment):
    """Continue compatible work; never replace an existing run after an update."""
    group, root = storage_root(environment)
    active = root / f".{source.stem}-active.json"
    with lease(root / ".storage.lock", wait=True):
        if not active.exists():
            fresh_plan(source, environment, "candidate40-v2", _locked=True)
        pointer = json.loads(active.read_text())
        target = Path(pointer["plan"]).resolve()
        if not target.is_relative_to(group) or pointer["source_plan_sha256"] != digest(source):
            raise ValueError("active group-storage run does not match this plan")
        if target.read_bytes() != source.read_bytes():
            raise ValueError("active group-storage plan changed")
        if any(not p.resolve().is_relative_to(group) for _, p in artifact_pairs(target, target)):
            raise ValueError("active run artifacts resolve outside group storage")
        plan = load_plan(target)
        marker = run_root(target, plan) / ".queue/protocol.json"
        recorded = pointer.get("implementation_sha256")
        if marker.exists():
            protocol = json.loads(marker.read_text())
            if protocol.get("schema") != "srgc-shared-queue-v2" or protocol.get("plan_sha256") != digest(target):
                raise ValueError("saved queue plan changed; refusing automatic restart")
            recorded = protocol.get("implementation_sha256")
            if not isinstance(recorded, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded):
                raise ValueError("saved queue implementation identity is invalid")
        current = code_digest()
        if recorded is None or recorded == current:
            return target
        raise ValueError(f"code changed {recorded} -> {current}; refusing to replace the existing run "
                         f"at {run_root(target, plan)}. No active pointer or results changed. "
                         "Use the original runtime to resume; status/results remain readable.")


def route_plan(source, *, writing, migrate=False, fresh=None, start_or_continue=False):
    source = source.resolve()
    if start_or_continue and (not writing or migrate or fresh is not None):
        raise ValueError("automatic start/continuation requires writing without fresh or migration")
    if fresh is not None or start_or_continue:
        group_path = Path(os.environ.get("GROUP_VOLUME", "/group-volume")).resolve()
        if "OM_WORK" in os.environ and not Path(os.environ["OM_WORK"]).resolve().is_relative_to(group_path):
            os.environ["OM_WORK"] = str(group_path / os.environ.get("OM_USER", "minsoo3.kim") / "offpolicy-misranking")
    group, root = storage_root(os.environ)
    plan = load_plan(source)
    active = root / f".{source.stem}-active.json"
    if start_or_continue:
        target = automatic_plan(source, os.environ)
    elif fresh is not None:
        target = fresh_plan(source, os.environ, fresh)
    elif active.exists() and not migrate:
        pointer = json.loads(active.read_text())
        target = Path(pointer["plan"]).resolve()
        if not target.is_relative_to(group) or pointer["source_plan_sha256"] != digest(source):
            raise ValueError("active group-storage run does not match this plan")
        if target.read_bytes() != source.read_bytes():
            raise ValueError("active group-storage plan changed")
    elif source.is_relative_to(group):
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
        if any(not p.resolve().is_relative_to(group) for _, p in artifact_pairs(target, target)):
            raise ValueError("active run artifacts resolve outside group storage")
        os.environ.setdefault("OM_WORK", str(group / os.environ.get("OM_USER", "minsoo3.kim") / "offpolicy-misranking"))
        cache = root / "runtime-cache"
        for key, path in {"HF_HOME": cache / "huggingface", "HF_HUB_CACHE": cache / "huggingface/hub",
                          "HUGGINGFACE_HUB_CACHE": cache / "huggingface/hub",
                          "TRANSFORMERS_CACHE": cache / "huggingface/hub",
                          "HF_MODULES_CACHE": cache / "huggingface/modules",
                          "HF_DATASETS_CACHE": cache / "huggingface/datasets", "XDG_CACHE_HOME": cache,
                          "TORCH_HOME": cache / "torch", "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",
                          "TRITON_CACHE_DIR": cache / "triton", "CUDA_CACHE_PATH": cache / "cuda",
                          "TMPDIR": cache / "tmp"}.items():
            path = path.resolve()
            if not path.is_relative_to(group):
                raise ValueError(f"{key} resolves outside group storage: {path}")
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
        print(f"[storage] inputs/cache: {input_path(target, plan, plan['seeds'][0]).parent}; "
              f"results: {run_root(target, plan)}", file=sys.stderr, flush=True)
        for seed in plan["seeds"]:
            prompt_cache = input_path(target, plan, seed).with_suffix(".cache")
            print(f"[storage] seed={seed} response_cache={prompt_cache} "
                  f"costs={prompt_cache / 'cost-receipts'} live_costs={prompt_cache / 'live-costs'}",
                  file=sys.stderr, flush=True)
    return target
