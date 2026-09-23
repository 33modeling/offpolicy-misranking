"""Independent current-policy A/B projections for the two frozen SR-GC subsets."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import selection_gate as core
import selection_gate_gpu as base

STAGES = ("validation-a", "candidate-a", "validation-b", "candidate-b")


def indices(contract, prompts, stage):
    if stage not in STAGES:
        raise ValueError("unknown SR-GC reference stage")
    n = len(prompts["val"])
    if n < 8 or n % 4:
        raise ValueError("SR-GC requires disjoint R/A/B validation partitions")
    if stage.startswith("candidate"):
        return sorted(set(contract["sets"]["on_policy"]) | set(contract["sets"]["cached"]))
    return list(range(n // 2, 3 * n // 4) if stage.endswith("a") else range(3 * n // 4, n))


def worker(root, stage, shard):
    import torch
    import evidence_downstream as ed
    from experiment import read_rollouts
    from grads import ProjectionSpec, grad_params, loo_advantages, prompt_gradient
    from net_gate_memory_worker import checkpoint_decoder_layers
    from rollout import collect_rollouts, load_policy

    c = core.read(root / "reference.json")
    cfg, parent = c["config"], Path(c["parent"])
    if (base.digest(parent / "adapter_model.safetensors") != c["adapter_sha256"]
            or base.digest(Path(c["prompts"])) != c["prompts_sha256"]):
        raise ValueError("SR-GC current-policy inputs changed")
    prompts = core.read(Path(c["prompts"]))
    ids = indices(c, prompts, stage)
    positions = range(len(ids) * shard // 4, len(ids) * (shard + 1) // 4)
    binding = {"reference_sha256": base.digest(root / "reference.json"), "stage": stage, "shard": shard}
    payload, done = root / f"{stage}-{shard}.json", root / f"{stage}-{shard}.done.json"
    if done.exists():
        if core.read(done) != {**binding, "sha256": base.digest(payload)}:
            raise ValueError("SR-GC projection shard changed")
        return
    with base.lease(root / f".{stage}-{shard}.lock"):
        base.bind(root / f"{stage}-{shard}.contract.json", binding)
        if not positions:
            base.bind(payload, {})
            base.bind(done, {**binding, "sha256": base.digest(payload)})
            return
        model, tok = load_policy(cfg["model"], parent)
        candidate = stage.startswith("candidate")
        pool = prompts["train" if candidate else "val"]
        k = 8 if candidate else cfg["val_k"]
        rows_path = root / f"{stage}-{shard}.jsonl"
        stream = (400000003 if stage.endswith("a") else 800000011) + (0 if candidate else 90000001)
        collect_rollouts(model, tok, [pool[ids[i]] for i in positions], k,
                         cfg["max_new_tokens"], cfg["temperature"], rows_path,
                         idx_offset=positions.start, sampling_seed_base=c["sampling_seed"] + stream)
        ed.reward_rows(rows_path, positions, k)
        groups = read_rollouts(rows_path)
        params = grad_params(model, cfg["grad_layers"])
        checkpoint_decoder_layers(model)
        spec = ProjectionSpec(dim=cfg["proj_dim"])
        output = {}
        for i in positions:
            saved = root / stage / f"prompt-{ids[i]}.json"
            bound = {**binding, "rollouts_sha256": base.digest(rows_path), "prompt_idx": ids[i]}
            if saved.exists():
                row = core.read(saved)
                if row["binding"] != bound:
                    raise ValueError("SR-GC partial gradient input changed")
                output[str(ids[i])] = row["value"]
                continue
            rows = sorted(groups[i], key=lambda r: r["rollout_idx"])
            vectors = []
            size = 4 if candidate else k
            for start in range(0, k, size):
                chunk = rows[start:start + size]
                advantage = loo_advantages(torch.tensor([r["reward"] for r in chunk]))
                weights = [torch.full((r["input_ids"].numel() - r["resp_start"],), float(a))
                           for r, a in zip(chunk, advantage, strict=True)]
                vectors.append(prompt_gradient(model, params, chunk, weights, spec, micro_batch=1).cpu())
            value = torch.stack(vectors).mean(0).tolist()
            base.bind(saved, {"binding": bound, "value": value})
            output[str(ids[i])] = value
            print(f"[SR-GC] {stage} shard={shard} prompt={ids[i]}", flush=True)
        base.bind(payload, output)
        base.bind(done, {**binding, "sha256": base.digest(payload)})


def projections(root, stage):
    import numpy as np
    c = core.read(root / "reference.json")
    ids = indices(c, core.read(Path(c["prompts"])), stage)
    rows = {}
    for shard in range(4):
        payload = root / f"{stage}-{shard}.json"
        expected = {"reference_sha256": base.digest(root / "reference.json"),
                    "stage": stage, "shard": shard, "sha256": base.digest(payload)}
        if core.read(root / f"{stage}-{shard}.done.json") != expected:
            raise ValueError("SR-GC shard binding changed")
        part = core.read(payload)
        if set(part) != {str(i) for i in ids[len(ids) * shard // 4:len(ids) * (shard + 1) // 4]}:
            raise ValueError("SR-GC shard coverage differs")
        rows.update(part)
    values = np.asarray([rows[str(i)] for i in ids], dtype=float)
    if values.shape != (len(ids), c["config"]["proj_dim"]) or not np.isfinite(values).all():
        raise ValueError("invalid SR-GC projections")
    return dict(zip(ids, values, strict=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--shard", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    from light_selection_gate_gpu import install_signal_handlers
    install_signal_handlers()
    worker(args.root, args.stage, args.shard)
