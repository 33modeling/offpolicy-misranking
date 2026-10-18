"""New inference/scoring acquisitions from archived policies, with no updates."""

import importlib
import re
import sys
from pathlib import Path

import numpy as np

from srgc_rebuttal.srgc import Config, Engine, stream_seed
from srgc_research.dispatch.reward_curves import preserve_random_state
from srgc_research.rollout_distribution import group_report

from .checkpoints import load_source, load_weights
from .common import copy_bound, file_hash, fingerprint, pool, read, write
from .coordination import Coordinator
from .costs import DiagnosticLedger
from .n12 import panel, record_context, stratum
from .n13 import compare


def measurement_hash():
    from srgc_research.storage import runtime_files
    return fingerprint({"diagnostic_files": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                        "scientific_runtime": runtime_files()})


def captured_backend(base):
    from scripts.srgc_resumable_rollouts import ResumableRolloutMixin

    class DiagnosticBackend(ResumableRolloutMixin, base):
        namespace = "unconfigured"

        def _rollout(self, prompt_id, responses, seed):
            self._resumable_dir = self.rollout_cache_root / f"{self.namespace}-{seed}-rank-{self.rank}"
            try:
                sequences, rewards, start = super()._rollout(prompt_id, responses, seed)
            finally:
                self._resumable_dir = None
            self.captured[prompt_id] = {"prompt_tokens": int(start),
                "responses": [{"reward": float(reward), "full_token_ids": sequence.detach().cpu().tolist(),
                               "token_ids": sequence[start:].detach().cpu().tolist()}
                              for sequence, reward in zip(sequences, rewards, strict=True)]}
            return sequences, rewards, start

        def score_gradients(self, ids, *, responses, group_size, seed):
            # Retain per-prompt cache files beyond the scoring call so a killed
            # acquisition reuses completed responses, without adding generation.
            return base.score_gradients(self, ids, responses=responses, group_size=group_size, seed=seed)

        def train(self, *args, **kwargs):
            raise RuntimeError("N12/N13 diagnostic backends cannot train or resume optimizer updates")

    return DiagnosticBackend


def n12_acquire(backend, data, fixed_panel, identity):
    expected = panel(data, identity["input_sha256"], identity["seed"])
    if fixed_panel != expected:
        raise ValueError("fixed panel IDs/settings differ; do not redraw between arms or stages")
    _, _, counts = pool(data)
    backend.captured, backend.namespace = {}, "n12-panel"
    seed = stream_seed(identity["seed"], 0, f"n12-panel:{identity['policy_sha256']}")
    with preserve_random_state(backend):
        for prompt in fixed_panel["prompt_ids"][backend.rank::backend.world]:
            backend._rollout(prompt, 8, seed)
    records = backend._gather(backend.captured)
    if set(records) != set(fixed_panel["prompt_ids"]):
        raise ValueError("fixed-panel rollout coverage differs")
    rows = []
    for prompt in fixed_panel["prompt_ids"]:
        measured = group_report(records[prompt])
        if measured["responses"] != 8:
            raise ValueError("fixed-panel group requires eight responses")
        rows.append({"prompt_id": prompt, "initial_successes": counts[prompt], "initial_stratum": stratum(counts[prompt]),
                     **measured, **records[prompt]})
    return {"schema": "srgc-n12-panel-measurement-v1", "identity": identity, "panel": fixed_panel, "groups": rows,
            "kind": "training_pool_fixed_panel", "used_for_training_or_selection": False,
            # Per-prompt draws are stream_seed(generation_seed, rank, prompt) over rank::world shards.
            "generation_seed": seed, "generation_world_size": backend.world,
            "acquisition": "fresh_responses_from_archived_weights", "response_count": 160,
            "is_fixed_held_out_evaluation": False}


def n13_acquire(backend, state, data, identity, output, *, independent=True):
    from srgc_rebuttal.distributed import primary

    config = Config(**state["config"])
    preview = Engine(backend, data["candidate_ids"], data["ranking_validation_ids"], data["cached_rewards"],
                     config=config, arm="on_policy", step=state["step"])
    preview.used_training_ids = set(state["used_training_ids"])
    candidates, comparisons = list(preview._draw_candidates()), list(preview._sr_comparison())
    reference = list(data["ranking_validation_ids"])
    if len(reference) != 50:
        raise ValueError("N13 requires the original single 50-prompt reference")
    union = list(dict.fromkeys(candidates + comparisons))
    header = {"schema": "srgc-n13-vectors-v1", "identity": identity, "seed": config.seed, "stage": state["step"],
              "candidate_ids": candidates, "sr_comparison_ids": comparisons, "reference_ids": reference,
              "cached_sr_ranked_ids": list(preview.sr_ranked_ids), "used_training_ids": state["used_training_ids"],
              "projection_dim": config.projection_dim, "projection_seed": backend.projection_seed,
              "score_parameter_names": [n for n, _ in backend.score_parameters],
              "candidate_response_group_size": config.candidate_group_size, "reference_response_group_size": config.responses,
              "online_tie_seed": stream_seed(config.seed + 1000, state["step"], "online-ties"),
              "responses_per_prompt": config.responses, "sets_origin": "original_checkpoint_selector_preview",
              "acquisition": "new_scoring_rollouts_from_archived_weights", "historical_rollouts_reused": False}
    blocks = {}
    for name in (("A", "B") if independent else ("A",)):
        path = Path(output) / f"score-{name}.json.gz"
        binding = fingerprint(header)
        if path.exists():
            saved = read(path)
            if saved.get("header_sha256") != binding:
                raise ValueError("saved scoring phase belongs to a different checkpoint or protocol")
            blocks[name] = saved["block"]
        else:
            backend.captured, backend.namespace = {}, f"n13-{name}"
            candidate_seed = stream_seed(config.seed, state["step"], f"n13:{name}:candidate")
            reference_seed = stream_seed(config.seed, state["step"], f"n13:{name}:reference")
            with preserve_random_state(backend), backend.cost_meter.phase(f"n13_scoring_{name}", state["step"], backend.gpu_count):
                gradients = backend.score_gradients(union, responses=config.responses,
                    group_size=config.candidate_group_size, seed=candidate_seed)
                validation = backend.score_gradients(reference, responses=config.responses,
                    group_size=config.responses, seed=reference_seed)
                captured = backend._gather(backend.captured)
            if set(captured) != set(union + reference):
                raise ValueError("scoring raw response coverage differs from projected vector IDs")
            for group in captured.values():
                if group_report(group)["responses"] != 8:
                    raise ValueError("scoring requires eight recorded responses per prompt")
            blocks[name] = {"gradients": {i: np.asarray(g).tolist() for i, g in gradients.items()},
                            "reference_gradients": {i: np.asarray(g).tolist() for i, g in validation.items()},
                            "rollouts": captured, "candidate_seed": candidate_seed, "reference_seed": reference_seed}
            primary(lambda path=path, binding=binding, name=name: write(path, {"header_sha256": binding, "block": blocks[name]}))
    document = {**header, "blocks": blocks}
    result = compare(document)
    document.update(selected_on_ids_A=result["on4_ids"], selected_sr_ids_A=result["sr4_ids"])
    primary(lambda: write(Path(output) / "vectors.json.gz", document))
    return result


def profile_name(plan):
    extension = plan.get("extension", "")
    profiles = {"qwen35-9b-v2": "qwen35", "gemma4-12b-pt-v1": "gemma4", "llama31-8b-v1": "llama31"}
    if extension and extension not in profiles:
        raise ValueError("unsupported model extension; use the matching original diagnostic adapter")
    return profiles.get(extension, "olmo3")


def make_backend(plan_path, plan, data, source, device, meter):
    import torch

    profile = profile_name(plan)
    if profile == "gemma4":
        from srgc_research.dispatch.gemma4.runtime import disable_optional_media
        disable_optional_media()
    from peft import LoraConfig, get_peft_model

    from srgc_rebuttal import existing_runtime, torch_backend

    adapter = None
    if profile == "olmo3":
        if "olmo" not in plan["model"].lower():
            raise ValueError("unknown model profile; no fallback to an unrelated model backend")
        load_model, base = existing_runtime.load_model, torch_backend.TorchBackend
    else:
        scripts = str(Path(__file__).resolve().parents[2] / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        module = "scripts.srgc_qwen35" if profile == "qwen35" else f"srgc_research.dispatch.{profile}.adapter"
        adapter = importlib.import_module(module)
        adapter.validate_extension(Path(plan_path))
        adapter.validate_bundle_model(data, plan, source["seed"])
        load_model = adapter.load_model
        memory = importlib.import_module("scripts.srgc_qwen35_memory" if profile == "qwen35" else f"srgc_research.dispatch.{profile}.memory")
        base = getattr(memory, {"qwen35": "QwenBackend", "gemma4": "GemmaBackend", "llama31": "LlamaBackend"}[profile])
    torch.manual_seed(source["seed"])
    model, tokenizer = load_model(plan["model"], plan["model_revision"], device, attention=source["attention"])
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0., bias="none", task_type="CAUSAL_LM")
    model = adapter.attach_adapter(model, lora, get_peft_model) if adapter else get_peft_model(model, lora)
    module, function = plan["verifier"].split(":", 1)
    return captured_backend(base)(model, tokenizer, data["records"], getattr(importlib.import_module(module), function),
        projection_dim=plan["projection_dim"], projection_seed=plan["projection_seed"], max_new_tokens=plan["max_new_tokens"],
        logprob_micro_batch=plan.get("logprob_micro_batch", 2), logit_chunk_tokens=plan.get("logit_chunk_tokens", 512), cost_meter=meter)


def required_artifacts(identity):
    files = {"costs.json", "source-checkpoint.pt", "source-inputs.json", "source-plan.json", "run.json", "manifest.json"}
    if identity["experiment"] == "n12":
        files.add("panel-measurement.json.gz")
    else:
        files.update(("vectors.json.gz", "score-A.json.gz"))
        if identity["independent_B"]:
            files.add("score-B.json.gz")
    return files


def verify_endpoint(output, identity):
    saved = read(output / "endpoint.json")
    if (saved.get("schema") != "srgc-checkpoint-diagnostics-v1" or saved.get("identity") != identity or
            saved.get("optimizer_updates") != 0 or not isinstance(saved.get("artifact_sha256"), dict) or
            not isinstance(saved.get("source"), dict) or
            not required_artifacts(identity) <= set(saved["artifact_sha256"])):
        raise ValueError("completed diagnostic identity/schema or required artifacts differ")
    if saved["source"].get("identity_binding") == "reward_curve_protocol" and "protocol.json" not in saved["artifact_sha256"]:
        raise ValueError("completed evaluation archive is missing its original protocol")
    for filename, expected in saved["artifact_sha256"].items():
        target = output / filename
        if target.resolve().parent != output.resolve() or file_hash(target) != expected:
            raise ValueError("completed diagnostic evidence changed or escapes its directory")
    for name, key in (("source-inputs.json", "input_sha256"), ("source-plan.json", "plan_sha256")):
        if saved["artifact_sha256"][name] != identity[key]:
            raise ValueError("completed diagnostic source hashes differ")
    if saved["artifact_sha256"]["source-checkpoint.pt"] != saved["source"]["checkpoint_sha256"]:
        raise ValueError("completed policy checkpoint hash differs")
    if read(output / "costs.json") != saved.get("costs"):
        raise ValueError("completed diagnostic cost snapshot differs")
    if saved.get("cost_measurement_complete") != saved["costs"]["complete"]:
        raise ValueError("completed diagnostic cost coverage differs")
    if identity["experiment"] == "n12":
        result = read(output / "panel-measurement.json.gz")
    else:
        result = compare(read(output / "vectors.json.gz"))
        if result["status"] != "measured":
            raise ValueError("completed N13 diagnostic has no measured scoring A")
    if result != saved.get("result") or result.get("identity") != identity:
        raise ValueError("completed diagnostic result differs from measured evidence")
    return saved


def prepare_source(args, holder):
    """Rank zero freezes sources and decides completion while holding one lease."""
    from srgc_rebuttal.runtime import lease

    state, _data, _plan, requested = load_source(args.checkpoint, args.bundle, args.plan, args.seed)
    if state["step"] not in (25, 125, 250):
        raise ValueError("N12/N13 checkpoint stages are 25/125/250 completed updates")
    if not re.fullmatch(r"[0-9a-f]{64}", requested.get("source_implementation_sha256") or ""):
        raise ValueError("source run must record its scientific implementation SHA")
    identity = {k: requested[k] for k in ("input_sha256", "plan_sha256", "policy_sha256", "seed", "source_implementation_sha256", "attention")}
    identity.update(measurement_sha256=measurement_hash(), experiment=args.experiment, independent_B=not args.a_only)
    if args.experiment == "n13":
        if not requested["n13_rescoring_weights_and_sets_usable"]:
            raise ValueError("N13 requires the original 50-prompt reference")
        identity.update(checkpoint_sha256=requested["checkpoint_sha256"], arm=requested["arm"])
    output = Path(args.output_root).resolve() / args.experiment / fingerprint(identity)
    if any(Path(p).resolve().is_relative_to(Path(args.output_root).resolve()) and Path(p).resolve() != output / name
           for p, name in ((args.checkpoint, "source-checkpoint.pt"), (args.bundle, "source-inputs.json"), (args.plan, "source-plan.json"))):
        raise ValueError("diagnostic output root must be separate from its sources")
    lock = lease(output / ".execution.lock")
    lock.__enter__()
    holder["lock"] = lock
    metadata = {"output": str(output), "identity": identity, "request_source": requested}
    if (output / "endpoint.json").exists():
        verify_endpoint(output, identity)
        if args.experiment == "n12":
            record_context(output, identity, requested)
        return {**metadata, "complete": True}
    source = requested
    if (output / "manifest.json").exists():
        manifest = read(output / "manifest.json")
        if manifest.get("identity") != identity:
            raise ValueError("saved diagnostic manifest identity differs")
        source = manifest["source"]
    # Publish all source sidecars before the checkpoint. No rank can read an
    # incomplete frozen source until this coordinated preparation returns.
    for original, name, expected in ((args.bundle, "source-inputs.json", source["input_sha256"]),
                                     (args.plan, "source-plan.json", source["plan_sha256"])):
        copy_bound(original, output / name, expected)
    binding = {"seed": source["seed"], "input_sha256": source["input_sha256"], "plan_sha256": source["plan_sha256"],
        "implementation_sha256": source["source_implementation_sha256"], "schema": "srgc-followup-source-binding-v1",
        "source_checkpoint_sha256": source["checkpoint_sha256"], "shared_prefix": source["shared_prefix"]}
    write(output / "run.json", binding)
    protocol_path = Path(source["checkpoint"]).parent / "protocol.json"
    if protocol_path.exists() and not (output / "protocol.json").exists():
        write(output / "protocol.json", read(protocol_path))
    copy_bound(args.checkpoint, output / "source-checkpoint.pt", source["checkpoint_sha256"])
    write(output / "manifest.json", {"identity": identity, "source": source})
    return {**metadata, "complete": False, "source": source}


def collect(args):
    """torchrun entry. Existing valid phases are verified before loading the base."""
    import torch
    import torch.distributed as dist

    from scripts.srgc_verifier_fallback import install
    from srgc_rebuttal.distributed import initialize, primary
    from srgc_rebuttal.runtime import atomic_json
    from srgc_rebuttal.timing import torch_meter

    coordinator, holder, initialized = Coordinator(), {}, False
    try:
        prepared = coordinator.root(lambda: prepare_source(args, holder))
        output, identity = Path(prepared["output"]), prepared["identity"]
        endpoint = output / "endpoint.json"
        if prepared["complete"]:
            if coordinator.rank == 0:
                print(f"COMPLETE (reused, no model load): {endpoint}")
            return
        def load_frozen():
            state, data, plan, frozen = load_source(output / "source-checkpoint.pt", output / "source-inputs.json", output / "source-plan.json", args.seed)
            if any(frozen[k] != identity[k] for k in ("input_sha256", "plan_sha256", "policy_sha256", "attention")):
                raise ValueError("frozen diagnostic source differs from its measurement identity")
            if frozen["checkpoint_sha256"] != prepared["source"]["checkpoint_sha256"]:
                raise ValueError("frozen diagnostic checkpoint checksum changed")
            if measurement_hash() != identity["measurement_sha256"]:
                raise ValueError("measurement code differs between ranks")
            holder.update(state=state, data=data, plan=plan)
            return frozen["checkpoint_sha256"]
        checks = coordinator.validate(load_frozen)
        if len(set(checks)) != 1:
            raise ValueError("frozen source changed between ranks")
        state, data, plan, source = holder["state"], holder["data"], holder["plan"], prepared["source"]
        _, local = initialize(plan["world_size"])
        initialized = True
        ledger = DiagnosticLedger(output / "cost-receipts")
        meter = torch_meter(ledger.record)
        with meter.phase("source_snapshot", state["step"], plan["world_size"]):
            primary(lambda: file_hash(output / "source-checkpoint.pt"))
        install()
        with meter.phase("startup", state["step"], plan["world_size"]):
            backend = make_backend(output / "source-plan.json", plan, data, source, torch.device("cuda", local), meter)
            load_weights(backend, state["backend"])
            backend.rollout_cache_root = output / "resume-rollouts"
        if args.experiment == "n12":
            fixed_panel = panel(data, source["input_sha256"], args.seed)
            panel_path = Path(args.output_root) / "panels" / f"seed-{args.seed}-{source['input_sha256']}.json"
            primary(lambda: write(panel_path, fixed_panel))
            with meter.phase("n12_panel_inference", state["step"], plan["world_size"]):
                result = n12_acquire(backend, data, fixed_panel, identity)
            result["source"] = source
            primary(lambda: write(output / "panel-measurement.json.gz", result))
            artifacts = ["panel-measurement.json.gz"]
        else:
            result = n13_acquire(backend, state, data, identity, output, independent=not args.a_only)
            artifacts = ["vectors.json.gz", "score-A.json.gz", *([] if args.a_only else ["score-B.json.gz"])]
        costs = primary(ledger.totals)
        # Derived costs may grow across interrupted attempts until endpoint is
        # committed. Scientific artifacts remain immutable.
        primary(lambda: atomic_json(output / "costs.json", costs))
        artifacts.append("costs.json")
        artifacts.extend(("source-checkpoint.pt", "source-inputs.json", "source-plan.json", "run.json", "manifest.json"))
        if (output / "protocol.json").exists():
            artifacts.append("protocol.json")
        primary(lambda: write(output / "endpoint.json", {"schema": "srgc-checkpoint-diagnostics-v1", "identity": identity, "source": source, "result": result,
            "costs": costs, "costs_separate_from_training": True, "optimizer_updates": 0,
            "cost_measurement_complete": costs["complete"], "base_model_and_gpu_runtime_verified": True,
            "artifact_sha256": {name: file_hash(output / name) for name in artifacts}}))
        if args.experiment == "n12":
            primary(lambda: record_context(output, identity, prepared["request_source"]))
        primary(lambda: print(f"COMPLETE: {endpoint}") and None)
    finally:
        original_error = sys.exc_info()
        try:
            if initialized:
                try:
                    dist.destroy_process_group()
                except Exception as error:
                    if original_error[1] is None:
                        raise
                    original_error[1].add_note(f"Distributed cleanup also failed: {error}")
        finally:
            if coordinator.rank == 0 and "lock" in holder:
                holder["lock"].__exit__(*original_error)
