"""Constrain Llama artifacts and library caches to the mounted group volume."""

import os
import socket
from pathlib import Path


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"path must be below group storage {root}: {path}")
    return path


def group_work(environment):
    group = Path(environment.get("GROUP_VOLUME", "/group-volume")).resolve()
    if not group.is_dir():
        raise ValueError(f"group volume is unavailable: {group}; no local fallback")
    default = group / environment.get("OM_USER", "minsoo3.kim") / "offpolicy-misranking"
    work = Path(environment.get("OM_WORK", str(default))).resolve()
    if work == group or not work.is_relative_to(group):
        work = default
    return group, inside(work, group)


def default_root(environment):
    group = Path(environment.get("GROUP_VOLUME", "/group-volume")).resolve()
    default = group / environment.get("OM_USER", "minsoo3.kim") / "offpolicy-misranking"
    work = Path(environment.get("OM_WORK", str(default))).resolve()
    if work == group or not work.is_relative_to(group):
        work = default
    return Path(
        environment.get("SRGC_LLAMA_ROOT", str(work / "srgc-rebuttal/llama31-8b-v1"))
    )


def validate_tree(root):
    """Check existing links once at startup; never follow directory link cycles."""
    root = Path(root).resolve()
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            path = Path(parent) / name
            if path.is_symlink():
                inside(path, root)


def setup_storage(root, environment, *, scan_tree=True):
    """Called before importing Torch/HF, also in every torchrun child."""
    group, work = group_work(environment)
    root = inside(root, group)
    if scan_tree:
        validate_tree(root)
    models = inside(environment.get("MODELS_DIR", str(group / "models")), group)
    if environment.get("SRGC_LLAMA_MODEL_PATH"):
        inside(environment["SRGC_LLAMA_MODEL_PATH"], group)
    common = inside(
        environment.get("SRGC_STORAGE_ROOT", str(work / "srgc-rebuttal")), group
    )
    cache = inside(work / "llama-runtime-cache", group)
    node = socket.gethostname().replace("/", "_")
    paths = {
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface/hub",
        "HF_DATASETS_CACHE": cache / "huggingface/datasets",
        "HF_XET_CACHE": cache / "huggingface/xet",
        "HF_ASSETS_CACHE": cache / "huggingface/assets",
        "TORCH_HOME": cache / "torch",
        "TRITON_CACHE_DIR": cache / "nodes" / node / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache / "nodes" / node / "inductor",
        "TORCH_EXTENSIONS_DIR": cache / "nodes" / node / "extensions",
        "CUDA_CACHE_PATH": cache / "nodes" / node / "cuda",
        "TMPDIR": cache / "nodes" / node / "tmp",
    }
    for path in paths.values():
        inside(path, group)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    environment.update({k: str(v) for k, v in paths.items()})
    environment.update(
        OM_WORK=str(work), MODELS_DIR=str(models), SRGC_STORAGE_ROOT=str(common)
    )
    if "LOCAL_RANK" in environment:
        rank = int(environment["LOCAL_RANK"])
        if (
            int(environment.get("WORLD_SIZE", "0")) != 4
            or int(environment.get("LOCAL_WORLD_SIZE", "0")) != 4
            or not 0 <= rank < 4
        ):
            raise ValueError("Llama ranks require four local/world ranks")
        base = cache / "nodes" / node / "rank-runtime-v1" / f"rank-{rank}"
        for key, name in (
            ("TRITON_CACHE_DIR", "triton"),
            ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
            ("TORCH_EXTENSIONS_DIR", "extensions"),
            ("CUDA_CACHE_PATH", "cuda"),
            ("TMPDIR", "tmp"),
        ):
            path = inside(base / name, group)
            path.mkdir(parents=True, exist_ok=True)
            environment[key] = str(path)
    # Prevent deprecated aliases from redirecting HF back into a home cache.
    environment["TRANSFORMERS_CACHE"] = environment["HF_HUB_CACHE"]
    environment["HUGGINGFACE_HUB_CACHE"] = environment["HF_HUB_CACHE"]
    return group, common


def validate_plan_paths(plan_path, plan):
    from srgc_rebuttal.plan import input_path
    from srgc_rebuttal.runtime import run_root

    path = Path(plan_path).absolute()
    root = path.parent.parent.resolve()
    inside(path, root)
    output = inside(run_root(path, plan), root)
    for name in (
        ".queue",
        ".queue/logs",
        ".queue/workers",
        ".queue/progress",
        "checkpoint-backups",
    ):
        inside(output / name, root)
    for seed in plan["seeds"]:
        bundle = input_path(path, plan, seed)
        for suffix in (".json", ".cache", ".cache-responses.jsonl"):
            inside(bundle.with_suffix(suffix), root)
        folder = output / f"seed-{seed}"
        for name in (
            "prefix.pt",
            "prefix-latest.pt",
            "run.json",
            "cost-receipts",
            "invocations",
            *(f"{arm}-latest.pt" for arm in plan["arms"]),
        ):
            inside(folder / name, root)
    return root
