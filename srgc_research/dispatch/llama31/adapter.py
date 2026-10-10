"""Llama-3.1-8B-Instruct adapter for the unchanged SRGC experiment engine.

Installed only in the dedicated Llama entry processes. The OLMo runtime and its
frozen package digest remain unchanged. Every worker and rank checks this
adapter's digest before using a plan or resuming a checkpoint.
"""

import copy
import hashlib
import importlib.metadata
import json
import os
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
TARGETS = ("q_proj", "v_proj")
SHARED_FILES = (
    "scripts/srgc_verifier_fallback.py",
    "scripts/srgc_step_checkpoints.py",
    "scripts/srgc_resumable_rollouts.py",
    "scripts/srgc_child_tuning.py",
    "scripts/srgc_direction_records.py",
    "scripts/srgc_process_guard.py",
    "scripts/srgc_pair_inputs.py",
    "scripts/srgc_shared_storage.py",
    "scripts/srgc_checkpoint_backup.py",
    "scripts/srgc_log_format.py",
    "scripts/selection_nccl_preflight.py",
    "srgc_research/dispatch/qwen_resume.py",
)


def adapter_digest():
    value = hashlib.sha256()
    for name in sorted(
        (
            *SHARED_FILES,
            *(str(p.relative_to(REPO)) for p in Path(__file__).parent.glob("*.py")),
        )
    ):
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
    return {
        "repository": MODEL,
        "revision": REVISION,
        "local_directory": "Llama-3.1-8B-Instruct",
        "model_type": "llama",
        "lora_targets": list(TARGETS),
    }


def runtime_packages():
    from packaging.version import Version

    from srgc_rebuttal.existing_runtime import verifier_environment

    verifier_environment(os.environ)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "numpy", "math-verify")
    }
    if Version(versions["transformers"]) < Version("4.57.0"):
        raise ValueError("Llama SRGC memory runtime needs transformers>=4.57.0")
    from peft import LoraConfig, get_peft_model  # noqa: F401
    from transformers import AutoTokenizer, LlamaForCausalLM  # noqa: F401

    return versions


def model_path(model, revision, environment):
    if (model, revision) != (MODEL, REVISION):
        raise ValueError("Llama extension refuses another model or revision")
    from .model import resolve_snapshot

    path, identity = resolve_snapshot(environment)
    expected = environment.get("SRGC_LLAMA_MODEL_SHA256")
    if expected and expected != identity:
        raise ValueError("local Llama weights differ from the recorded experiment")
    return str(path)


def load_text_model(source, device, *, attention="eager"):
    from .model import load_text_model as load

    return load(source, device, attention=attention)


def load_model(model, revision, device, *, attention="eager"):
    source = model_path(model, revision, os.environ)
    return load_text_model(source, device, attention=attention)


def attach_adapter(model, config, original):
    if model.config.model_type != "llama":
        raise ValueError("Llama adapter received a different model family")
    config = copy.deepcopy(config)
    config.target_modules = set(TARGETS)
    result = original(model, config)
    for layer in result.get_base_model().model.layers:
        actual = {
            n.rsplit(".", 1)[-1]
            for n, m in layer.named_modules()
            if hasattr(m, "lora_A")
        }
        if not set(TARGETS) <= actual:
            raise ValueError(f"missing Llama layer adapters: {set(TARGETS) - actual}")
    return result


def make_bundle(source, tokenizer, *, source_plan, source_sha256):
    from srgc_rebuttal.plan import validate_inputs

    validate_inputs(source, require_cache=False, recorded_rewards=True)
    bundle = copy.deepcopy(source)
    bundle["cached_rewards"] = {}
    for record in bundle["records"].values():
        # Preserve the task instruction and all split/question IDs. Only the
        # model-specific chat wrapper changes; never wrap an already chat prompt.
        if any(
            marker in record["prompt"]
            for marker in ("<|im_start|>", "<|begin_of_text|>", "<|start_header_id|>")
        ):
            raise ValueError(
                "source is already chat-rendered; provide the OLMo source plan"
            )
        record["prompt"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
    bundle["provenance"] = {
        "source_plan": str(source_plan),
        "source_bundle_sha256": source_sha256,
        "source_provenance": copy.deepcopy(source["provenance"]),
        "model": MODEL,
        "model_revision": REVISION,
        "llama_extension": "llama31-8b-v1",
        "generation_micro_batch": 2,
        "prompt_format": "llama31_tokenizer_chat",
        "cache": "pending: Llama initial-policy rewards; source rewards are NOT reused",
    }
    validate_inputs(bundle, require_cache=False)
    return bundle


def prepare(dataset, source_plan, destination, tokenizer, *, snapshot_sha256):
    if (
        not isinstance(snapshot_sha256, str)
        or len(snapshot_sha256) != 64
        or any(c not in "0123456789abcdef" for c in snapshot_sha256)
    ):
        raise ValueError("Llama preparation needs the verified local model digest")
    from srgc_rebuttal.cluster_queue import input_info
    from srgc_rebuttal.plan import digest, input_path, load_plan, validate_inputs
    from srgc_rebuttal.runtime import atomic_json, lease

    source_plan, destination = source_plan.resolve(), destination.resolve()
    original = load_plan(source_plan)
    if original["dataset"] != {"math": "math_train", "mbpp": "mbpp"}[dataset]:
        raise ValueError("dataset does not match source plan")
    if original["model"] != "allenai/Olmo-3-1025-7B":
        raise ValueError("source plan must be the OLMo comparison cohort")
    target = destination / "experiments" / f"llama31-8b-{dataset}.json"
    plan = {
        **original,
        "model": MODEL,
        "model_revision": REVISION,
        "model_initialization": "posttrained",
        "model_loader": "LlamaForCausalLM",
        "model_snapshot_sha256": snapshot_sha256,
        "extension": "llama31-8b-v1",
        "adapter_sha256": adapter_digest(),
        "engine_sha256": engine_digest(),
        "generation_micro_batch": 2,
        "logprob_micro_batch": 1,
        "logit_chunk_tokens": 64,
        "gradient_checkpointing": "nonreentrant",
        "checkpoint_snapshot": "cpu",
        "lora_targets": list(TARGETS),
        "thinking": "off",
        "attention": "eager",
        "input_pattern": f"../inputs/{dataset}-seed-{{seed}}.json",
        "output_root": f"../runs/{dataset}",
        "analysis": "Report all five Llama seeds separately by dataset. Compare the four arms within Llama; do not pool rewards or GPU costs with OLMo.",
        "source_plan_sha256": digest(source_plan),
    }
    with lease(destination / f".{dataset}-prepare.lock", wait=True):
        if target.exists():
            if json.loads(target.read_text()) != plan:
                raise ValueError(
                    "prepared Llama plan differs; use a new output directory"
                )
            for seed in plan["seeds"]:
                data = json.loads(input_path(target, plan, seed).read_text())
                validate_inputs(data, require_cache=False)
                info, _ = input_info(
                    input_path(source_plan, original, seed), recorded_rewards=True
                )
                if data["provenance"]["source_data_sha256"] != info["source_sha256"]:
                    raise ValueError(
                        "source cohort changed; refusing to reset the Llama run"
                    )
            return target
        saved_run = destination / "runs" / dataset
        if saved_run.exists() and (not saved_run.is_dir() or any(saved_run.iterdir())):
            raise ValueError(
                f"missing plan {target} with existing run {saved_run}; "
                "restore the original plan; refusing to prepare a replacement"
            )
        built = {}
        for seed in plan["seeds"]:
            path = input_path(source_plan, original, seed)
            built[seed] = make_bundle(
                json.loads(path.read_text()),
                tokenizer,
                source_plan=source_plan,
                source_sha256=digest(path),
            )
            built[seed]["provenance"]["source_data_sha256"] = input_info(
                path, recorded_rewards=True
            )[0]["source_sha256"]
            validate_bundle_model(built[seed], plan, seed)
        for seed, bundle in built.items():
            path = input_path(target, plan, seed)
            if path.exists() and json.loads(path.read_text()) != bundle:
                raise ValueError(f"refusing to overwrite existing input: {path}")
            atomic_json(path, bundle)
        atomic_json(target, plan)
    return target


def validate_extension(path, *, read_only=False):
    from srgc_rebuttal.plan import load_plan

    plan = load_plan(path)
    dataset = {"math_train": "math", "mbpp": "mbpp"}.get(plan.get("dataset"))
    if dataset is None:
        raise ValueError("Llama plan needs MATH or MBPP")
    expected = {
        "model": MODEL,
        "model_revision": REVISION,
        "extension": "llama31-8b-v1",
        "adapter_sha256": adapter_digest(),
        "lora_targets": list(TARGETS),
        "engine_sha256": engine_digest(),
        "thinking": "off",
        "attention": "eager",
        "objective": "grpo",
        "max_new_tokens": 2048,
        "generation_micro_batch": 2,
        "logprob_micro_batch": 1,
        "logit_chunk_tokens": 64,
        "gradient_checkpointing": "nonreentrant",
        "checkpoint_snapshot": "cpu",
        "seeds": [5, 6, 7, 8, 9],
        "ranking_validation_prompts": 50,
        "input_pattern": f"../inputs/{dataset}-seed-{{seed}}.json",
        "output_root": f"../runs/{dataset}",
        "verifier": "srgc_rebuttal.run_experiment:math_reward"
        if dataset == "math"
        else "srgc_rebuttal.verifiers:code_reward",
    }
    if read_only:
        # Reports use the run's recorded identity; this never permits training
        # with a different adapter or rewrites a prepared experiment's hashes.
        for key in ("adapter_sha256", "engine_sha256"):
            value = plan.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"invalid recorded Llama {key}")
            expected.pop(key)
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError(
            "Llama plan/model/adapter differs; do not use the OLMo launcher or reuse a run"
        )
    value = plan.get("model_snapshot_sha256")
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("invalid recorded Llama model snapshot digest")
    from .storage import validate_plan_paths

    validate_plan_paths(path, plan)
    return plan


def validate_bundle_model(data, plan, seed):
    provenance = data.get("provenance", {})
    source = provenance.get("source_provenance", {})
    # Pair imports retain their historical math500 label under a math_train plan.
    # Their original split provenance must remain intact; GSM8K is not an alias.
    pair_math = (
        plan["dataset"] == "math_train"
        and data.get("dataset") == "math500"
        and isinstance(source, dict)
        and source.get("prompt_format") == "olmo_rlzero_math"
        and bool(source.get("source_run"))
        and type(source.get("reused_from_seed")) is int
    )
    if data.get("dataset") != plan["dataset"] and not pair_math:
        raise ValueError(
            f"bundle dataset {data.get('dataset')!r} differs from the Llama plan "
            f"{plan['dataset']!r}; only provenance-backed Pair math500 inputs are an alias"
        )
    reference_count = len(data.get("ranking_validation_ids", []))
    if reference_count != 50:
        raise ValueError(
            f"bundle ranking-validation reference size {reference_count} differs from "
            "the Llama plan (expected 50)"
        )
    if (provenance.get("model"), provenance.get("model_revision")) != (MODEL, REVISION):
        raise ValueError("input bundle is not prepared for the pinned Llama model")
    if (
        provenance.get("llama_extension") != "llama31-8b-v1"
        or provenance.get("generation_micro_batch") != 2
    ):
        raise ValueError("bundle uses a different Llama generation protocol")
    if data.get("cached_rewards"):
        expected = {
            "model": MODEL,
            "model_revision": REVISION,
            "responses": 8,
            "max_new_tokens": plan["max_new_tokens"],
            "cache_seed": seed,
            "verifier": plan["verifier"],
            "attention": "eager",
        }
        cache = provenance.get("cache")
        if not isinstance(cache, dict) or any(
            cache.get(k) != v for k, v in expected.items()
        ):
            raise ValueError("cached rewards do not match the pinned Llama protocol")


def task_command(queue, task):
    from srgc_rebuttal.plan import input_path

    validate_extension(queue.plan_path)
    print(
        f"LLAMA dataset={queue.plan['dataset']} task={task.key} plan={queue.plan_path}",
        flush=True,
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "--max_restarts=0",
        "--module",
        "srgc_research.dispatch.llama31.rank",
        "--stage",
        "cache" if task.arm == "cache" else "train",
        "--plan",
        str(queue.plan_path),
    ]
    if task.arm == "cache":
        return [
            *command,
            "--bundle",
            str(input_path(queue.plan_path, queue.plan, task.seed)),
            "--cache-seed",
            str(task.seed),
            "--max-new-tokens",
            str(queue.plan["max_new_tokens"]),
            "--attention",
            "eager",
        ]
    return [*command, "--seed", str(task.seed), "--task", task.arm, "--resume"]


@contextmanager
def runtime_adapter():
    """Scope all substitutions to a dedicated Llama process, including children."""
    from unittest.mock import patch

    from srgc_rebuttal import (
        admission,
        build_cache,
        cluster,
        cluster_queue,
        existing_runtime,
        reports,
        run_experiment,
        runtime,
    )

    original_digest = runtime.code_digest

    def code_digest():
        return hashlib.sha256(
            (original_digest() + adapter_digest()).encode()
        ).hexdigest()

    class LlamaCacheStore(build_cache.CacheStore):
        def __init__(self, bundle_path, bundle, protocol):
            super().__init__(
                bundle_path,
                bundle,
                {
                    **protocol,
                    "llama_extension": "llama31-8b-v1",
                    "generation_micro_batch": 2,
                    "adapter_sha256": adapter_digest(),
                },
            )

    class LlamaQueue(cluster_queue.TaskQueue):
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
            from srgc_rebuttal.plan import digest, input_path

            for seed in self.plan["seeds"]:
                bundle = input_path(self.plan_path, self.plan, seed)
                data = json.loads(bundle.read_text())
                validate_bundle_model(data, self.plan, seed)
                if self.cache_ready[seed]:
                    receipt = bundle.with_suffix(".cache") / "cost-summary.json"
                    self.cache_ready[seed] = receipt.exists() and json.loads(
                        receipt.read_text()
                    ).get("bundle_sha256") == digest(bundle)

    # Patch both definitions and already-bound imports; restore all of them on exit.
    with ExitStack() as stack:
        for name, value in (
            ("load_model", load_model),
            ("model_path", model_path),
            ("runtime_packages", runtime_packages),
        ):
            stack.enter_context(patch.object(existing_runtime, name, value))
        stack.enter_context(patch.object(runtime, "code_digest", code_digest))
        for module, name, value in (
            (admission, "model_path", model_path),
            (admission, "runtime_packages", runtime_packages),
            (cluster, "runtime_packages", runtime_packages),
            (cluster_queue, "code_digest", code_digest),
            (build_cache, "load_model", load_model),
            (run_experiment, "load_model", load_model),
            (cluster, "task_command", task_command),
            (cluster_queue, "TaskQueue", LlamaQueue),
            (cluster, "TaskQueue", LlamaQueue),
            (reports, "code_digest", code_digest),
            (build_cache, "CacheStore", LlamaCacheStore),
        ):
            stack.enter_context(patch.object(module, name, value))
        yield


@contextmanager
def training_adapter():
    from unittest.mock import patch

    import peft

    from srgc_rebuttal import torch_backend

    from .memory import LlamaBackend, durable_checkpoints

    original = peft.get_peft_model
    with (
        patch.object(
            peft,
            "get_peft_model",
            lambda model, config: attach_adapter(model, config, original),
        ),
        patch.object(torch_backend, "TorchBackend", LlamaBackend),
        durable_checkpoints(),
    ):
        yield


def smoke(plan_path):
    from .smoke import lightweight_smoke

    return lightweight_smoke(plan_path)
