"""Independent group-volume identities, frozen code and atomic study artifacts."""

import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from scripts.srgc_shared_storage import storage_root
from srgc_rebuttal.plan import digest, input_path, load_plan, validate_inputs
from srgc_rebuttal.runtime import atomic_json, lease

from .design import PROTOCOL, STAGES, Condition

REPO = Path(__file__).resolve().parents[1]


def root(environment=None):
    environment = os.environ if environment is None else environment
    group, base = storage_root(environment)
    target = Path(environment.get("SRGC_RESEARCH_ROOT", str(base / "literature-v1"))).resolve()
    if not target.is_relative_to(group) or target in (group, base):
        raise ValueError("research outputs must use a separate directory inside group storage")
    return target


def runtime_files(repo=REPO):
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0")
    names = {n for n in tracked if n and Path(n).parts[0] in
             {"scripts", "src", "configs", "vendor", "srgc_rebuttal"}
             and Path(n).suffix in {".py", ".json", ".whl"} and "tests" not in Path(n).parts
             and "inputs" not in Path(n).parts and "runs" not in Path(n).parts}
    names.update(str(p.relative_to(repo)) for p in (repo / "srgc_research").glob("*.py"))
    return {n: digest(repo / n) for n in sorted(names)}


def freeze(root_path, repo=REPO):
    files = runtime_files(repo)
    sha = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    out = root_path / "runtimes" / sha
    with lease(root_path / ".runtime.lock", wait=True):
        if not out.exists():
            temporary = out.with_name(f".{sha}-{uuid.uuid4().hex}")
            try:
                for name, expected in files.items():
                    target = temporary / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(repo / name, target)
                    if digest(target) != expected:
                        raise ValueError("code changed while freezing research runtime")
                atomic_json(temporary / "runtime.json", {"sha256": sha, "files": files})
                temporary.rename(out)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    verify_runtime(out, sha)
    return out, sha


def verify_runtime(path, expected):
    record = json.loads((path / "runtime.json").read_text())
    actual = hashlib.sha256(json.dumps(record["files"], sort_keys=True).encode()).hexdigest()
    if actual != expected or record["sha256"] != expected:
        raise ValueError("frozen research runtime identity differs")
    for name, sha in record["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or digest(target) != sha:
            raise ValueError(f"frozen runtime file differs: {name}")


def prepare(folder, dataset, seed, plan_path, runtime, implementation, attention):
    plan = load_plan(plan_path)
    if plan["objective"] != "grpo":
        raise ValueError("N01-N08 are GRPO experiments; refusing another training objective")
    source = input_path(plan_path, plan, seed)
    data = json.loads(source.read_text())
    validate_inputs(data, require_cache=False, recorded_rewards=True)
    # Freeze only the experiment's immutable inputs, not a live cache file.
    data["cached_rewards"] = {}
    data["provenance"].pop("cache", None)
    input_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    analysis_plan = analysis_settings(dataset)
    expected = {"protocol": PROTOCOL, "dataset": dataset, "seed": seed,
                "source_plan_sha256": digest(plan_path), "input_sha256": input_hash}
    with lease(folder / ".manifest.lock", wait=True):
        path = folder / "manifest.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError(f"research input/plan changed: {folder}")
            if any(os.environ.get(key) for key in (f"SRGC_RESEARCH_{dataset.upper()}_TARGET", "SRGC_RESEARCH_GPU_BUDGETS")) \
                    and saved.get("analysis_plan") != analysis_plan:
                raise ValueError("predeclared analysis target/budgets cannot change during a run")
            verify_inputs(folder, saved)
            return saved
        atomic_json(folder / "inputs.json", data)
        atomic_json(path, {**expected, "plan": plan, "source_plan": str(plan_path),
            "source_input": str(source), "runtime": str(runtime), "implementation_sha256": implementation,
            "attention": attention, "initial_state": "t0-fresh-seeded-base-model", "sr_cache": "fresh-once-per-seed",
            "evaluation_ids": data["evaluation_ids"], "candidate_ids": data["candidate_ids"], "analysis_plan": analysis_plan})
        return json.loads(path.read_text())


def verify_inputs(folder, manifest):
    data = json.loads((folder / "inputs.json").read_text())
    sha = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    if sha != manifest["input_sha256"]:
        raise ValueError("frozen research input differs")
    if any(manifest.get(k) != data[k] for k in ("evaluation_ids", "candidate_ids")):
        raise ValueError("manifest prompt IDs differ from frozen input")
    return data


def read_manifest(path, dataset, seed):
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("dataset") != dataset or value.get("seed") != seed:
        raise ValueError("manifest dataset/seed differs from its directory")
    return value


def analysis_settings(dataset):
    target = os.environ.get(f"SRGC_RESEARCH_{dataset.upper()}_TARGET")
    target = float(target) if target else None
    budgets = [float(v) for v in os.environ.get("SRGC_RESEARCH_GPU_BUDGETS", "").split(",") if v.strip()]
    if target is not None and (not math.isfinite(target) or not 0 <= target <= 1):
        raise ValueError("predeclared target reward must be between zero and one")
    if any(not math.isfinite(v) or v <= 0 for v in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("GPU-second budgets must be distinct positive finite values")
    return {"target_reward": target, "budgets_gpu_seconds": sorted(budgets)}


def configure_cache(environment=None):
    environment = os.environ if environment is None else environment
    group, base = storage_root(environment)
    cache = base / "runtime-cache"
    paths = {"HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
             "HUGGINGFACE_HUB_CACHE": "huggingface/hub", "TRANSFORMERS_CACHE": "huggingface/hub",
             "HF_MODULES_CACHE": "huggingface/modules", "HF_DATASETS_CACHE": "huggingface/datasets",
             "XDG_CACHE_HOME": "", "TORCH_HOME": "torch", "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
             "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda", "TMPDIR": "tmp"}
    for key, relative in paths.items():
        path = (cache / relative).resolve()
        if not path.is_relative_to(group):
            raise ValueError(f"cache path escaped group volume: {key}")
        path.mkdir(parents=True, exist_ok=True)
        environment[key] = str(path)


def atomic_torch(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def identity(manifest, condition):
    return {k: manifest[k] for k in ("protocol", "dataset", "seed", "input_sha256",
                                    "source_plan_sha256", "implementation_sha256")} | {"condition": condition.record()}


def publish_anchor(folder, manifest, state):
    step = state["step"]
    path = folder / "anchors" / f"step-{step}.pt"
    receipt = path.with_suffix(".json")
    if receipt.exists():
        verify_anchor(folder, manifest, step)
        return
    atomic_torch(path, state)
    atomic_json(receipt, {**identity(manifest, Condition("anchors", "anchors", updates=STAGES[-1])),
                         "step": step, "sha256": digest(path)})


def verify_anchor(folder, manifest, step):
    path = folder / "anchors" / f"step-{step}.pt"
    row = json.loads(path.with_suffix(".json").read_text())
    if row.get("step") != step or any(row.get(k) != manifest[k] for k in
            ("input_sha256", "implementation_sha256", "seed", "source_plan_sha256")) or digest(path) != row.get("sha256"):
        raise ValueError("diagnostic anchor identity/hash differs")
    return path


def validate_costs(costs):
    if type(costs.get("complete")) is not bool:
        raise ValueError("cost completeness must be explicit")
    known = costs["known_gpu_seconds"]
    if not {"selection_gpu_seconds", "training_gpu_seconds"} <= set(known):
        raise ValueError("missing measured phase totals")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in known.values()):
        raise ValueError("invalid phase cost")
    total = costs["total_gpu_seconds"]
    if costs["complete"]:
        if costs["unfinished_phases"] or type(total) not in (float, int) or not math.isclose(total, sum(known.values()), abs_tol=1e-6):
            raise ValueError("complete costs do not reconcile")
    elif total is not None or not costs["unfinished_phases"]:
        raise ValueError("incomplete cost must have unknown total and unfinished phases")


def validate_endpoint(value, manifest, condition):
    if any(value.get(k) != v for k, v in identity(manifest, condition).items()) or value.get("status") != "complete":
        raise ValueError("research endpoint identity/status differs")
    if not isinstance(value.get("cost_receipts"), dict):
        raise TypeError("endpoint missing cost receipts")
    validate_costs(value["cost_receipts"])
    if value.get("cost_measurement_complete") != value["cost_receipts"]["complete"]:
        raise ValueError("endpoint cost completeness differs from receipts")
    if condition.kind == "trajectory":
        curve = value["result"]["curve"]
        expected_steps = list(range(0, condition.updates + 1, 25))
        if expected_steps[-1] != condition.updates:
            expected_steps.append(condition.updates)
        if not curve or [r["update"] for r in curve] != expected_steps:
            raise ValueError("endpoint is missing the final evaluation")
        for row in curve:
            rewards = row["per_question_reward"]
            if not rewards or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in rewards.values()):
                raise ValueError("invalid endpoint evaluation")
            if (type(row["reward"]) not in (int, float) or not math.isfinite(row["reward"]) or
                    abs(sum(rewards.values()) / len(rewards) - row["reward"]) > 1e-10):
                raise ValueError("endpoint aggregate reward differs")
            if set(rewards) != set(manifest["evaluation_ids"]):
                raise ValueError("endpoint evaluation IDs differ from fixed input")
            from .report import coverage
            if coverage(row)["pass_at_k"] is None:
                raise ValueError("new endpoints must retain raw evaluation responses")
        history = value["result"]["history"]
        if len(history) != condition.updates:
            raise ValueError("missing training history")
    elif condition.kind == "diagnostic":
        result = value["result"]
        if result.get("stage") != condition.stage or not result.get("rows"):
            raise ValueError("diagnostic endpoint missing stage results")
        if condition.arm in {"n03", "n07"}:
            for name in result["selected"]:
                points = [r["branch_updates"] for r in result["rows"] if r["selector"] == name]
                if points != [0, 1, 5, 25]:
                    raise ValueError("diagnostic branch incomplete")
        elif len(result["rows"]) != (3 if condition.arm == "n02" else 6):
            raise ValueError("diagnostic measurements incomplete")
    elif condition.kind == "cache":
        result = value["result"]
        if result.get("prompts") != len(manifest["candidate_ids"]) or result.get("responses_per_prompt") != 8:
            raise ValueError("SR cache endpoint prompt/response counts differ")
        if not {"cache_generation_gpu_seconds", "cache_export_gpu_seconds"} <= set(value["cost_receipts"]["known_gpu_seconds"]):
            raise ValueError("SR cache has no generation/export timer")
    elif condition.kind == "anchors":
        if value["result"] != {"stages": list(STAGES), "carrier_updates": STAGES[-1]}:
            raise ValueError("anchor carrier incomplete")
    elif condition.kind == "features":
        result = value["result"]
        if result.get("same_rollouts") is not True or set(result.get("representations", {})) != {"dense", "lesser"}:
            raise ValueError("feature comparison incomplete")
    else:
        raise ValueError("unknown research result kind")
    return value


def complete(folder, manifest, condition):
    path = folder / condition.key / "endpoint.json"
    if not path.exists():
        return False
    value = validate_endpoint(json.loads(path.read_text()), manifest, condition)
    validate_artifacts(folder, manifest, condition, value)
    return True


def validate_artifacts(folder, manifest, condition, value):
    if condition.kind == "cache":
        cache = folder / "sr-cache.json"
        if not cache.is_file() or digest(cache) != value["result"].get("cache_sha256"):
            raise ValueError("SR cache artifact missing or changed")
    elif condition.kind == "anchors":
        for step in STAGES:
            verify_anchor(folder, manifest, step)
    elif condition.kind == "diagnostic" and condition.arm == "n02":
        if not (folder / condition.key / "state-latest.pt").is_file():
            raise ValueError("optimizer diagnostic raw vectors are missing")
