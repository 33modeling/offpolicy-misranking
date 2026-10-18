"""Read-only checkpoint capability checks and weights-only diagnostic loading."""

import pickle
from pathlib import Path

from .common import file_hash, fingerprint, pool, seed_value


def source_state(saved, data, plan, seed, *, input_hash, plan_hash, protocol=None, marker=None):
    """Validate Engine/anchor sources; reward-curve archives intentionally lack Adam."""
    from srgc_rebuttal.srgc import Engine, cached_sr_set

    seed_value(seed)
    if not isinstance(saved, dict):
        raise TypeError("checkpoint must contain an Engine state object")
    state = saved.get("state", saved)
    if isinstance(state, dict) and "anchor" in state:
        state = state["anchor"]
    if not isinstance(state, dict):
        raise TypeError("checkpoint has no completed Engine/anchor state")
    candidates, reference, _ = pool(data)
    config = state.get("config", {})
    if not isinstance(config, dict):
        raise TypeError("checkpoint configuration must be an object")
    stage = state.get("step")
    if type(stage) is not int or stage < 0 or config.get("seed") != seed or state.get("arm") not in Engine.ARMS:
        raise ValueError("checkpoint stage/seed is invalid")
    expected = {"responses": "responses", "training_prompts": "training_prompts", "scoring_prompts": "scoring_prompts_per_set",
                "projection_dim": "projection_dim", "selection_interval": "selection_interval", "check_interval": "check_interval",
                "first_check": "first_check", "objective": "objective"}
    for key, plan_key in expected.items():
        if plan_key not in plan or config.get(key) != plan[plan_key]:
            raise ValueError(f"checkpoint differs from the source plan: {key}")
    if (config["responses"] != 8 or config["training_prompts"] != 4 or config["scoring_prompts"] != 40 or
            config["objective"] != "grpo" or config.get("candidate_group_size") != plan.get("candidate_group_size", 4)):
        raise ValueError("N12/N13 require the eight-response, candidate40/training4 GRPO protocol")
    if list(state.get("candidates", [])) != candidates or list(state.get("validation", [])) != reference:
        raise ValueError("checkpoint candidate/reference IDs differ from the source bundle")
    expected_rank = list(cached_sr_set(candidates, data["cached_rewards"], len(candidates), seed, 8))
    if list(state.get("sr_ranked_ids", [])) != expected_rank:
        raise ValueError("checkpoint cached-SR ranking differs from the source bundle")
    if state.get("sampling_protocol") != Engine.SAMPLING_PROTOCOL:
        raise ValueError("checkpoint sampling protocol differs; historical Pair states need a separate adapter")
    used = state.get("used_training_ids")
    if not isinstance(used, list) or len(set(used)) != len(used) or not set(used) <= set(candidates):
        raise ValueError("checkpoint applied-training ID history is missing or invalid")
    expected_identity = {"seed": seed, "input_sha256": input_hash, "plan_sha256": plan_hash}
    if state.get("reward_curve_binding") is not None:
        binding = state["reward_curve_binding"]
        if (protocol is None or binding != {"protocol_sha256": fingerprint(protocol), "step": stage}
                or any(protocol.get(k) != v for k, v in expected_identity.items())):
            raise ValueError("evaluation archive requires its original matching reward-curve protocol.json")
        source = protocol
        binding_kind = "reward_curve_protocol"
    elif all(saved.get(k) == v for k, v in expected_identity.items()):
        source = saved
        binding_kind = "checkpoint_envelope"
    else:
        if marker is None or any(marker.get(k) != v for k, v in expected_identity.items()):
            raise ValueError("ordinary Engine checkpoints require their source run.json input/plan/seed binding")
        source = marker
        binding_kind = "run_marker_and_engine_contract"
    backend = state.get("backend", {})
    if not isinstance(backend, dict):
        raise TypeError("checkpoint backend must be an object")
    if (not isinstance(backend.get("trainable"), dict) or not backend["trainable"] or
            not backend.get("score_names") or backend.get("projection_dim") != config["projection_dim"] or
            backend.get("projection_seed") != plan.get("projection_seed")):
        raise ValueError("checkpoint has no compatible policy weights/projection layout")
    attention = state.get("checkpoint_policy", {}).get("attention") or plan.get("attention", "eager")
    if plan.get("attention") and attention != plan["attention"]:
        raise ValueError("saved attention differs from plan")
    return state, {"stage": stage, "arm": state.get("arm"), "seed": seed,
                   "model": plan["model"], "model_revision": plan["model_revision"], "attention": attention,
                   "source_implementation_sha256": source.get("implementation_sha256"), "identity_binding": binding_kind,
                   "optimizer_state_present": "optimizer" in backend,
                   "evaluation_only": state.get("evaluation_only", False),
                   "n12_panel_weights_usable": True, "n13_rescoring_weights_and_sets_usable": len(reference) == 50,
                   "historical_projected_vectors_present": False,
                   "training_resume_readiness_verified": False,
                   "base_model_and_gpu_runtime_verified": False,
                   "retraining_required_for_n12_n13": False}


def weights_hash(state, plan, input_hash):
    import hashlib

    import torch

    digest = hashlib.sha256(fingerprint({"model": plan["model"], "revision": plan["model_revision"], "input": input_hash}).encode())
    for name, tensor in sorted(state["backend"]["trainable"].items()):
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError("policy weights must be finite tensors")
        tensor = tensor.detach().cpu().contiguous()
        digest.update(fingerprint({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_weights(backend, saved):
    """Diagnostics only: never synthesize missing optimizer moments or resume training."""
    import torch

    parameters = dict(backend.train_parameters)
    weights = saved["trainable"]
    if (set(parameters) != set(weights) or saved["projection_dim"] != backend.projection_dim or
            saved["projection_seed"] != backend.projection_seed or
            saved["score_names"] != [n for n, _ in backend.score_parameters]):
        raise ValueError("checkpoint policy/projection parameter layout differs from the loaded model")
    for name, parameter in parameters.items():
        tensor = weights[name]
        if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype or not torch.isfinite(tensor).all():
            raise ValueError(f"checkpoint policy tensor shape/dtype/value differs: {name}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(weights[name].to(parameter.device))
    backend.optimizer.zero_grad(set_to_none=True)


def load_source(checkpoint, bundle, plan_path, seed):
    import torch

    from .common import read, read_bound
    checkpoint, bundle, plan_path = Path(checkpoint), Path(bundle), Path(plan_path)
    data, input_hash = read_bound(bundle)
    plan, plan_hash = read_bound(plan_path)
    before = file_hash(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if file_hash(checkpoint) != before:
        raise ValueError("source checkpoint changed while reading; retry after its atomic save")
    protocol_path = checkpoint.parent / "protocol.json"
    # Native prefix/latest checkpoints live beside run.json. Archived reward
    # points are two levels below it. Do not search arbitrary ancestor runs.
    marker_path = checkpoint.parent / "run.json"
    marker = read(marker_path) if marker_path.exists() else None
    receipt_path = checkpoint.parent / "receipt.json"
    backup_receipt = read(receipt_path) if checkpoint.name == "checkpoint.pt" and receipt_path.exists() else None
    if backup_receipt is not None:
        if backup_receipt.get("schema") != "srgc-checkpoint-backup-v1" or backup_receipt.get("sha256") != before:
            raise ValueError("checkpoint backup receipt/checksum differs")
        marker = backup_receipt["identity"]
    protocol = read(protocol_path) if protocol_path.exists() else None
    state, capabilities = source_state(saved, data, plan, seed, input_hash=input_hash, plan_hash=plan_hash,
                                      protocol=protocol, marker=marker)
    if backup_receipt is not None:
        capabilities["identity_binding"] = "backup_receipt_checksum_and_engine_contract"
    point = checkpoint.with_suffix(".json")
    if state.get("evaluation_only") and point.exists():
        row = read(point)
        if row.get("checkpoint_sha256") != before or row.get("step") != state["step"]:
            raise ValueError("archived point/checkpoint SHA or step binding differs")
    return state, data, plan, {**capabilities, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": before,
                              "input_sha256": input_hash, "plan_sha256": plan_hash,
                              "shared_prefix": state["step"] == 25 and ((protocol or {}).get("arm") == "prefix" or checkpoint.name in {"prefix.pt", "prefix-latest.pt"}),
                              "policy_sha256": weights_hash(state, plan, input_hash)}


def inventory(roots, bundle, plan, seed, stages=(25, 125, 250)):
    rows = []
    paths = sorted({p.resolve() for root in roots for p in ([Path(root)] if Path(root).is_file() else Path(root).rglob("*.pt"))})
    for path in paths:
        try:
            _, _, _, row = load_source(path, bundle, plan, seed)
            rows.append({"status": "usable_for_diagnostics", **row})
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
            rows.append({"checkpoint": str(path), "status": "invalid_or_other_run", "reason": str(error)})
    policies = {(r["arm"], r["stage"]) for r in rows if r["status"] == "usable_for_diagnostics"}
    shared = any(r.get("shared_prefix") for r in rows)
    return {"schema": "srgc-n12-n13-checkpoint-audit-v1", "seed": seed, "input_sha256": file_hash(bundle),
            "plan_sha256": file_hash(plan), "rows": rows,
            "requested_stage_coverage": [{"arm": arm, "stage": step, "present": (arm, step) in policies or (step == 25 and shared)}
                                         for arm in ("random", "sr", "on_policy", "switch") for step in stages],
            "issues": [] if paths else ["No .pt files found in the specified roots; server checkpoint existence is unverified."]}
