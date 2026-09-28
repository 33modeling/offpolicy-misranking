"""Qwen3.5-9B adapter for the unchanged SRGC experiment engine.

Installed only in the dedicated Qwen entry processes. The OLMo runtime and its
frozen package digest remain unchanged. Every worker and rank checks this
adapter's digest before using a plan or resuming a checkpoint.
"""

from contextlib import ExitStack, contextmanager
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "configs/qwen35_9b_grpo.json"
MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TARGETS = ("q_proj", "v_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
ADAPTER_FILES = ("scripts/srgc_qwen35.py", "scripts/run_srgc_qwen35.py",
                 "scripts/srgc_qwen35_rank.py", "scripts/srgc_verifier_fallback.py",
                 "scripts/srgc_step_checkpoints.py", "scripts/srgc_qwen35_memory.py",
                 "scripts/srgc_qwen35_storage.py", "scripts/srgc_qwen35_worker.py", "src/model_matrix.py")


def adapter_digest():
    value = hashlib.sha256()
    for name in ADAPTER_FILES:
        value.update(name.encode())
        value.update((REPO / name).read_bytes())
    return value.hexdigest()


def engine_digest():
    value = hashlib.sha256()
    for path in sorted((REPO / "srgc_rebuttal").glob("*.py")):
        value.update(path.name.encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def specification():
    spec = json.loads(SPEC_PATH.read_text())["models"][0]
    if (spec["repository"], spec["revision"], tuple(spec["lora_targets"])) != (MODEL, REVISION, TARGETS):
        raise ValueError("registered Qwen specification changed; explicitly revise the extension")
    return spec


def runtime_packages():
    from packaging.version import Version
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "peft", "numpy", "math-verify")}
    if Version(versions["transformers"]) < Version("5.14.1"):
        raise ValueError("Qwen extension needs transformers>=5.14.1; use a separate Qwen venv")
    from transformers import Qwen3_5ForCausalLM  # noqa: F401
    # Production needs the tested fast linear-attention kernels. CPU tests use
    # Transformers' native reference implementation in a separate process.
    sys.path.insert(0, str(REPO / "src"))
    from model_matrix import _require_runtime
    _require_runtime(specification())
    versions["fla-core"] = importlib.metadata.version("fla-core")
    for name, expected in {"transformers": "5.14.1", "peft": "0.20.0", "fla-core": "0.5.2"}.items():
        if versions[name] != expected:
            raise ValueError(f"Qwen runtime requires {name}=={expected}; found {versions[name]}")
    return versions


def model_path(model, revision, environment):
    if (model, revision) != (MODEL, REVISION):
        raise ValueError("Qwen extension refuses another model/revision (including 9B-Base)")
    sys.path.insert(0, str(REPO / "src"))
    from model_matrix import validate_snapshot_provenance, _weight_shards, _manifest_files
    from srgc_rebuttal.runtime import atomic_json, lease
    from srgc_qwen35_storage import group_work, inside
    import socket
    spec = specification()
    models = Path(environment.get("MODELS_DIR", str(Path(environment.get("GROUP_VOLUME", "/group-volume")) / "models")))
    path = Path(environment.get("SRGC_QWEN_MODEL_PATH", str(models / spec["local_directory"])))
    group, work = group_work(environment)
    path = inside(path, group)

    def fingerprint():
        files = [path / ".om_snapshot.json", *_manifest_files(path, _weight_shards(path))]
        return {str(f.relative_to(path)): [f.stat().st_ino, f.stat().st_size,
                f.stat().st_mtime_ns, f.stat().st_ctime_ns] for f in files}

    key = hashlib.sha256(str(path).encode()).hexdigest()
    marker = inside(work / "qwen-runtime-cache/verified-models" / socket.gethostname() / f"{key}.json", group)
    with lease(marker.with_suffix(".lock"), wait=True):
        current = {"model": model, "revision": revision, "files": fingerprint()}
        if not marker.exists() or json.loads(marker.read_text()) != current:
            validate_snapshot_provenance(spec, path)
            if current["files"] != fingerprint():
                raise ValueError("model snapshot changed during verification")
            atomic_json(marker, current)
    return str(path.resolve())


def load_text_model(source, device, *, attention="eager"):
    """Load the text weights, checking that none were silently initialized."""
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    config = Qwen3_5TextConfig.from_pretrained(source, local_files_only=True)
    policy, info = Qwen3_5ForCausalLM.from_pretrained(
        source, config=config, local_files_only=True, output_loading_info=True,
        dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
        attn_implementation=attention)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError(f"incomplete Qwen text-weight loading: {info}")
    # Vision/MTP weights are unused in a text-only causal model.
    unexpected = [k for k in info.get("unexpected_keys", [])
                  if not k.startswith(("model.visual.", "mtp."))]
    if unexpected:
        raise ValueError(f"unexpected Qwen text weights: {unexpected}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Avoid inherited generation defaults silently changing the matched sampling.
    policy.generation_config.repetition_penalty = 1.0
    policy.eval().to(device)
    from srgc_qwen35_memory import bounded_generate
    policy.generate = bounded_generate(policy.generate, torch.device(device))
    return policy, tokenizer


def load_model(model, revision, device, *, attention="eager"):
    source = model_path(model, revision, os.environ)
    return load_text_model(source, device, attention=attention)


def attach_adapter(model, config, original):
    if model.config.model_type != "qwen3_5_text":
        raise ValueError("Qwen adapter received a non-Qwen text model")
    config = copy.deepcopy(config)
    config.target_modules = set(TARGETS)
    result = original(model, config)
    # q/v alone would adapt only the eight full-attention layers. Also adapt
    # the 24 DeltaNet layers, using the registered Qwen model-matrix targets.
    for index, layer in enumerate(result.get_base_model().model.layers):
        expected = TARGETS[:2] if model.config.layer_types[index] == "full_attention" else TARGETS[2:]
        actual = {n.rsplit(".", 1)[-1] for n, m in layer.named_modules() if hasattr(m, "lora_A")}
        if not set(expected) <= actual:
            raise ValueError(f"missing hybrid-layer adapters: {set(expected) - actual}")
    return result


def make_bundle(source, tokenizer, *, source_plan, source_sha256):
    from srgc_rebuttal.plan import validate_inputs
    validate_inputs(source, require_cache=False)
    bundle = copy.deepcopy(source)
    bundle["cached_rewards"] = {}
    for record in bundle["records"].values():
        # Preserve the task instruction and all split/question IDs. Only the
        # model-specific chat wrapper changes; never wrap an already chat prompt.
        if "<|im_start|>" in record["prompt"]:
            raise ValueError("source is already chat-rendered; provide the OLMo source plan")
        record["prompt"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
    bundle["provenance"] = {
        "source_plan": str(source_plan), "source_bundle_sha256": source_sha256,
        "source_provenance": copy.deepcopy(source["provenance"]),
        "model": MODEL, "model_revision": REVISION,
        "qwen_extension": "qwen35-9b-v2", "generation_micro_batch": 2,
        "prompt_format": "qwen35_tokenizer_chat_thinking_off",
        "cache": "pending: Qwen initial-policy rewards; source rewards are NOT reused"}
    validate_inputs(bundle, require_cache=False)
    return bundle


def prepare(dataset, source_plan, destination, tokenizer):
    from srgc_rebuttal.plan import digest, input_path, load_plan, validate_inputs
    from srgc_rebuttal.cluster_queue import input_info
    from srgc_rebuttal.runtime import atomic_json, lease
    source_plan, destination = source_plan.resolve(), destination.resolve()
    original = load_plan(source_plan)
    if original["dataset"] != {"math": "math_train", "mbpp": "mbpp"}[dataset]:
        raise ValueError("dataset does not match source plan")
    if original["model"] != "allenai/Olmo-3-1025-7B":
        raise ValueError("source plan must be the OLMo comparison cohort")
    target = destination / "experiments" / f"qwen35-9b-{dataset}.json"
    plan = {**original, "model": MODEL, "model_revision": REVISION,
            "model_initialization": "posttrained", "model_loader": "Qwen3_5ForCausalLM",
            "extension": "qwen35-9b-v2", "adapter_sha256": adapter_digest(),
            "engine_sha256": engine_digest(),
            "generation_micro_batch": 2, "logprob_micro_batch": 1, "logit_chunk_tokens": 64,
            "gradient_checkpointing": "nonreentrant", "checkpoint_snapshot": "cpu",
            "lora_targets": list(TARGETS), "thinking": "off", "attention": "eager",
            "input_pattern": f"../inputs/{dataset}-seed-{{seed}}.json",
            "output_root": f"../runs/{dataset}",
            "analysis": "Report all five Qwen seeds separately by dataset. Compare the four arms within Qwen; do not pool rewards or GPU costs with OLMo.",
            "source_plan_sha256": digest(source_plan)}
    with lease(destination / f".{dataset}-prepare.lock", wait=True):
        if target.exists():
            if json.loads(target.read_text()) != plan:
                raise ValueError("prepared Qwen plan differs; use a new output directory")
            for seed in plan["seeds"]:
                data = json.loads(input_path(target, plan, seed).read_text())
                validate_inputs(data, require_cache=False)
                info, _ = input_info(input_path(source_plan, original, seed))
                if data["provenance"]["source_data_sha256"] != info["source_sha256"]:
                    raise ValueError("source cohort changed; refusing to reset the Qwen run")
            return target
        built = {}
        for seed in plan["seeds"]:
            path = input_path(source_plan, original, seed)
            built[seed] = make_bundle(json.loads(path.read_text()), tokenizer,
                                      source_plan=source_plan, source_sha256=digest(path))
            built[seed]["provenance"]["source_data_sha256"] = input_info(path)[0]["source_sha256"]
        for seed, bundle in built.items():
            path = input_path(target, plan, seed)
            if path.exists() and json.loads(path.read_text()) != bundle:
                raise ValueError(f"refusing to overwrite existing input: {path}")
            atomic_json(path, bundle)
        atomic_json(target, plan)
    return target


def validate_extension(path):
    from srgc_rebuttal.plan import load_plan
    plan = load_plan(path)
    dataset = {"math_train": "math", "mbpp": "mbpp"}.get(plan.get("dataset"))
    if dataset is None:
        raise ValueError("Qwen plan needs MATH or MBPP")
    expected = {"model": MODEL, "model_revision": REVISION, "extension": "qwen35-9b-v2",
                "adapter_sha256": adapter_digest(), "lora_targets": list(TARGETS),
                "engine_sha256": engine_digest(), "thinking": "off", "attention": "eager",
                "objective": "grpo", "max_new_tokens": 2048,
                "generation_micro_batch": 2, "logprob_micro_batch": 1, "logit_chunk_tokens": 64,
                "gradient_checkpointing": "nonreentrant", "checkpoint_snapshot": "cpu",
                "seeds": [5, 6, 7, 8, 9], "ranking_validation_prompts": 50,
                "input_pattern": f"../inputs/{dataset}-seed-{{seed}}.json", "output_root": f"../runs/{dataset}",
                "verifier": "srgc_rebuttal.run_experiment:math_reward" if dataset == "math" else "srgc_rebuttal.verifiers:code_reward"}
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError("Qwen plan/model/adapter differs; do not use the OLMo launcher or reuse a run")
    from srgc_qwen35_storage import validate_plan_paths
    validate_plan_paths(path, plan)
    return plan


def validate_bundle_model(data, plan, seed):
    provenance = data.get("provenance", {})
    if data.get("dataset") != plan["dataset"] or len(data.get("ranking_validation_ids", [])) != 50:
        raise ValueError("bundle dataset/reference size differs from the Qwen plan")
    if (provenance.get("model"), provenance.get("model_revision")) != (MODEL, REVISION):
        raise ValueError("input bundle is not prepared for the pinned Qwen model")
    if provenance.get("qwen_extension") != "qwen35-9b-v2" or provenance.get("generation_micro_batch") != 2:
        raise ValueError("bundle uses a different Qwen generation protocol")
    if data.get("cached_rewards"):
        expected = {"model": MODEL, "model_revision": REVISION, "responses": 8,
                    "max_new_tokens": plan["max_new_tokens"], "cache_seed": seed,
                    "verifier": plan["verifier"], "attention": "eager"}
        cache = provenance.get("cache")
        if not isinstance(cache, dict) or any(cache.get(k) != v for k, v in expected.items()):
            raise ValueError("cached rewards do not match the pinned Qwen protocol")


def task_command(queue, task):
    from srgc_rebuttal.plan import input_path
    validate_extension(queue.plan_path)
    print(f"QWEN dataset={queue.plan['dataset']} task={task.key} plan={queue.plan_path}", flush=True)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
               "--max_restarts=0", str(REPO / "scripts/srgc_qwen35_rank.py"),
               "--stage", "cache" if task.arm == "cache" else "train", "--plan", str(queue.plan_path)]
    if task.arm == "cache":
        return [*command, "--bundle", str(input_path(queue.plan_path, queue.plan, task.seed)),
                "--cache-seed", str(task.seed), "--max-new-tokens", str(queue.plan["max_new_tokens"]),
                "--attention", "eager"]
    return [*command, "--seed", str(task.seed), "--task", task.arm, "--resume"]


@contextmanager
def runtime_adapter():
    """Scope all substitutions to a dedicated Qwen process, including children."""
    from unittest.mock import patch
    from srgc_rebuttal import existing_runtime, runtime, admission, cluster, cluster_queue, build_cache, run_experiment, reports
    original_digest = runtime.code_digest

    def code_digest():
        return hashlib.sha256((original_digest() + adapter_digest()).encode()).hexdigest()

    class QwenCacheStore(build_cache.CacheStore):
        def __init__(self, bundle_path, bundle, protocol):
            super().__init__(bundle_path, bundle, {**protocol, "qwen_extension": "qwen35-9b-v2",
                "generation_micro_batch": 2, "adapter_sha256": adapter_digest()})

    class QwenQueue(cluster_queue.TaskQueue):
        @contextmanager
        def claim(self, **kwargs):
            with super().claim(**kwargs) as task:
                if task is not None:
                    # Another node can finish cache export after the initial
                    # verify but before this node acquires the task lease.
                    self.verify()
                    if self.complete(task):
                        self.finish(task, 0)
                        yield None
                        return
                yield task

        def verify(self):
            super().verify()
            from srgc_rebuttal.plan import input_path, digest
            for seed in self.plan["seeds"]:
                bundle = input_path(self.plan_path, self.plan, seed)
                data = json.loads(bundle.read_text())
                validate_bundle_model(data, self.plan, seed)
                if self.cache_ready[seed]:
                    receipt = bundle.with_suffix(".cache") / "cost-summary.json"
                    self.cache_ready[seed] = receipt.exists() and json.loads(receipt.read_text()).get("bundle_sha256") == digest(bundle)

    # Patch both definitions and already-bound imports; restore all of them on exit.
    with ExitStack() as stack:
        for name, value in (("load_model", load_model), ("model_path", model_path),
                            ("runtime_packages", runtime_packages)):
            stack.enter_context(patch.object(existing_runtime, name, value))
        stack.enter_context(patch.object(runtime, "code_digest", code_digest))
        for module, name, value in ((admission, "model_path", model_path),
                (admission, "runtime_packages", runtime_packages), (cluster, "runtime_packages", runtime_packages),
                (cluster_queue, "code_digest", code_digest), (build_cache, "load_model", load_model),
                (run_experiment, "load_model", load_model), (cluster, "task_command", task_command),
                (cluster_queue, "TaskQueue", QwenQueue), (cluster, "TaskQueue", QwenQueue),
                (reports, "code_digest", code_digest), (build_cache, "CacheStore", QwenCacheStore)):
            stack.enter_context(patch.object(module, name, value))
        yield


@contextmanager
def training_adapter():
    from unittest.mock import patch
    import peft
    from srgc_rebuttal import torch_backend
    from srgc_qwen35_memory import QwenBackend, durable_checkpoints
    original = peft.get_peft_model
    with patch.object(peft, "get_peft_model", lambda model, config: attach_adapter(model, config, original)), \
            patch.object(torch_backend, "TorchBackend", QwenBackend), durable_checkpoints():
        yield


def smoke(plan_path):
    """Four-rank real-weight generation + forced mixed-reward gradient admission.

    Synthetic rewards here ensure the backward path is exercised. They are
    never stored in experimental inputs, caches, endpoints or result tables.
    """
    import torch
    import torch.distributed as dist
    import numpy as np
    from peft import LoraConfig, get_peft_model
    from srgc_rebuttal.distributed import initialize
    from srgc_qwen35_memory import QwenBackend
    plan = validate_extension(plan_path)
    rank, local = initialize(4)
    try:
        torch.manual_seed(104729)
        torch.cuda.reset_peak_memory_stats()
        model, tokenizer = load_model(MODEL, REVISION, torch.device("cuda", local))
        model = attach_adapter(model, LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM"), get_peft_model)
        from srgc_rebuttal.plan import input_path
        prompt, longest = "", 0
        for entry in json.loads(os.environ.get("SRGC_QWEN_PLANS", json.dumps([str(plan_path)]))):
            source = Path(entry)
            spec = validate_extension(source)
            for seed in spec["seeds"]:
                data = json.loads(input_path(source, spec, seed).read_text())
                for record in data["records"].values():
                    length = len(tokenizer.encode(record["prompt"], add_special_tokens=True))
                    if length > longest:
                        prompt, longest = record["prompt"], length
        if not prompt:
            raise ValueError("admission needs the prepared experiment's prompts")
        records = {f"p{i}": {"prompt": prompt, "answer": "2"} for i in range(4)}
        backend = QwenBackend(model, tokenizer, records, lambda record, text: 0.0,
                               max_new_tokens=8, projection_dim=plan["projection_dim"])
        sequences, _, start = backend._rollout(f"p{rank}", 8, 17)
        # A capable model may generate the same answer eight times. Use two
        # distinct forced suffixes so admission still exercises real backward.
        prefix = sequences[0][:start]
        suffixes = [tokenizer.encode(text, add_special_tokens=False) for text in (" 2", " 3")]
        sequences = [torch.cat((prefix, torch.tensor(
            (suffixes[i % 2] * plan["max_new_tokens"])[:plan["max_new_tokens"]], device=prefix.device))) for i in range(8)]
        backend._rollout = lambda *args: (sequences, np.array([0., 1.] * 4), start)
        gradients = backend.score_gradients(list(records), responses=8, group_size=4, seed=17)
        if not all(np.isfinite(g).all() for g in gradients.values()) or not any(np.any(g) for g in gradients.values()):
            raise ValueError("Qwen scoring backward produced zero/nonfinite synthetic gradients")
        before = [p.detach().cpu().clone() for _, p in backend.train_parameters]
        backend.train(list(records), responses=8, objective="grpo", seed=17)
        if not all(torch.isfinite(p).all() for _, p in backend.train_parameters):
            raise ValueError("nonfinite Qwen adapter update")
        if not any(not torch.equal(a, p.detach().cpu()) for a, (_, p) in zip(before, backend.train_parameters)):
            raise ValueError("synthetic Qwen admission did not update adapters")
        print(json.dumps({"rank": rank, "admission_prompt_tokens": longest,
            "admission_response_tokens": plan["max_new_tokens"],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}), flush=True)
        dist.barrier()
        if rank == 0:
            print("PASS: Qwen real-weight four-rank generation/scoring/GRPO; synthetic smoke only", flush=True)
    except BaseException:
        # Let torchrun terminate peers immediately on a rank-local CUDA failure.
        # Destroying a failed NCCL group here can wait for its full timeout.
        raise
    else:
        dist.destroy_process_group()
