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
from .costs import DiagnosticLedger
from .n12 import panel, stratum
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
    from peft import LoraConfig, get_peft_model

    from srgc_rebuttal import existing_runtime, torch_backend

    profile = profile_name(plan)
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


def collect(args):
    """torchrun entry. Existing valid phases are verified before loading the base."""
    import torch
    import torch.distributed as dist

    from scripts.srgc_verifier_fallback import install
    from srgc_rebuttal.distributed import initialize, primary
    from srgc_rebuttal.runtime import lease
    from srgc_rebuttal.timing import torch_meter

    state, data, plan, source = load_source(args.checkpoint, args.bundle, args.plan, args.seed)
    if state["step"] not in (25, 125, 250):
        raise ValueError("N12/N13 checkpoint stages are 25/125/250 completed updates")
    if not re.fullmatch(r"[0-9a-f]{64}", source.get("source_implementation_sha256") or ""):
        raise ValueError("source run must record its scientific implementation SHA")
    identity = {k: source[k] for k in ("input_sha256", "plan_sha256", "policy_sha256", "seed", "source_implementation_sha256")}
    identity.update(measurement_sha256=measurement_hash(), experiment=args.experiment, independent_B=not args.a_only)
    # N12 shares the exact same policy once, even if filenames or arm labels
    # differ. N13 additionally binds selector progress to the source checkpoint.
    if args.experiment == "n13":
        if not source["n13_rescoring_weights_and_sets_usable"]:
            raise ValueError("N13 requires the original 50-prompt reference")
        identity["checkpoint_sha256"] = source["checkpoint_sha256"]
        identity["arm"] = source["arm"]
    output = Path(args.output_root).resolve() / args.experiment / fingerprint(identity)
    if any(Path(p).resolve().is_relative_to(Path(args.output_root).resolve()) and Path(p).resolve() != output / name
           for p, name in ((args.checkpoint, "source-checkpoint.pt"), (args.bundle, "source-inputs.json"), (args.plan, "source-plan.json"))):
        raise ValueError("diagnostic output root must be separate from its sources")
    endpoint = output / "endpoint.json"
    if endpoint.exists():
        saved = read(endpoint)
        if saved.get("identity") != identity:
            raise ValueError("completed diagnostic identity differs")
        for filename, expected in saved["artifact_sha256"].items():
            if file_hash(output / filename) != expected:
                raise ValueError("completed diagnostic evidence changed")
        print(f"COMPLETE (reused, no model load): {endpoint}")
        return
    _, local = initialize(plan["world_size"])
    lock = lease(output / ".execution.lock")
    acquired = False
    try:
        if dist.is_initialized():
            identities = [None] * dist.get_world_size()
            dist.all_gather_object(identities, fingerprint(identity))
            if len(set(identities)) != 1:
                raise ValueError("source policy changed between ranks; use an immutable archived checkpoint")
        primary(lambda: lock.__enter__() and None)
        acquired = True
        if (output / "manifest.json").exists():
            if read(output / "manifest.json").get("identity") != identity:
                raise ValueError("saved diagnostic manifest identity differs")
            source = read(output / "manifest.json")["source"]
        else:
            primary(lambda: write(output / "manifest.json", {"identity": identity, "source": source}))
        ledger = DiagnosticLedger(output / "cost-receipts")
        meter = torch_meter(ledger.record)
        with meter.phase("source_snapshot", state["step"], plan["world_size"]):
            def preserve_sources():
                for original, name, expected in ((args.checkpoint, "source-checkpoint.pt", source["checkpoint_sha256"]),
                    (args.bundle, "source-inputs.json", source["input_sha256"]), (args.plan, "source-plan.json", source["plan_sha256"])):
                    copy_bound(original, output / name, expected)
                binding = {"seed": source["seed"], "input_sha256": source["input_sha256"], "plan_sha256": source["plan_sha256"],
                    "implementation_sha256": source["source_implementation_sha256"], "schema": "srgc-followup-source-binding-v1",
                    "source_checkpoint_sha256": source["checkpoint_sha256"]}
                write(output / "run.json", binding)
                protocol_path = Path(args.checkpoint).parent / "protocol.json"
                if protocol_path.exists():
                    write(output / "protocol.json", read(protocol_path))
            primary(preserve_sources)
        install()
        with meter.phase("startup", state["step"], plan["world_size"]):
            backend = make_backend(args.plan, plan, data, source, torch.device("cuda", local), meter)
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
        primary(lambda: write(output / "costs.json", costs))
        artifacts.append("costs.json")
        artifacts.extend(("source-checkpoint.pt", "source-inputs.json", "source-plan.json", "run.json", "manifest.json"))
        if (output / "protocol.json").exists():
            artifacts.append("protocol.json")
        primary(lambda: write(output / "endpoint.json", {"identity": identity, "source": source, "result": result,
            "costs": costs, "costs_separate_from_training": True, "optimizer_updates": 0,
            "cost_measurement_complete": costs["complete"], "base_model_and_gpu_runtime_verified": True,
            "artifact_sha256": {name: file_hash(output / name) for name in artifacts}}))
        primary(lambda: print(f"COMPLETE: {endpoint}") and None)
    finally:
        if acquired:
            primary(lambda: lock.__exit__(*sys.exc_info()))
        dist.destroy_process_group()
