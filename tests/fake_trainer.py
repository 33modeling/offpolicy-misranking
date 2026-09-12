"""Stand-in for src/train_policy_grpo.py in CPU tests and the local fake cluster.

Accepts the trainer's command line and writes a policy directory that passes
``validate_policy_lineage`` (adapter files, optimizer, per-step statistics,
optional reliability log, manifest with parent and prompt hashes). No model
is loaded. The reliability log's half pass rates share a latent per-prompt
pass rate whose split-half correlation is set by ``FAKE_TRAINER_RHO``
(default 0.6) so gate decisions can be steered in tests.

    python tests/fake_trainer.py --model M --prompts P.json --output OUT --target-steps 110 \
        --start-step 100 --resume-adapter PARENT --resume-optimizer PARENT/optimizer.pt ... [--reliability-log]

The local fake cluster wraps ``python -m torch.distributed.run ... src/train_policy_grpo.py``
onto this script (see scripts/fake_venv in the simulator).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_policy_grpo import (  # noqa: E402
    POLICY_SCHEMA, GrpoConfig, advantage_normalization_for_objective,
    policy_update_for_objective, sha256_file, token_normalization_for_objective,
    validate_policy_lineage,
)


def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--objective", default="grpo")
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target-steps", type=int, required=True)
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--resume-adapter")
    p.add_argument("--resume-optimizer")
    p.add_argument("--expected-world-size", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--epochs-per-batch", type=int, default=2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--advantage-epsilon", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--logprob-micro-batch", type=int, default=1)
    p.add_argument("--disable-gradient-checkpointing", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reliability-log", action="store_true")
    return p.parse_args(argv)


def write_policy(args) -> Path:
    out = Path(args.output)
    world = args.expected_world_size
    config = GrpoConfig(group_size=args.group_size, clip_epsilon=args.clip_epsilon,
                        learning_rate=args.learning_rate, epochs_per_batch=args.epochs_per_batch,
                        max_grad_norm=args.max_grad_norm, advantage_epsilon=args.advantage_epsilon,
                        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                        checkpoint_every=args.checkpoint_every)
    parent = Path(args.resume_adapter).resolve() if args.resume_adapter else None
    prompt_format = os.environ.get("OM_PROMPT_FORMAT", "olmo_rlzero_math")
    if (out / "policy_train.json").is_file():
        validate_policy_lineage(out, target_steps=args.target_steps, world_size=world,
                                training_objective=args.objective, expected_start_step=args.start_step,
                                expected_parent=parent, expected_model=Path(args.model),
                                expected_seed=args.seed, expected_max_new_tokens=args.max_new_tokens,
                                expected_prompt_format=prompt_format, expected_config=asdict(config),
                                expected_prompts=Path(args.prompts), require_complete_hashes=True)
        print(f"[fake-trainer] validated completed policy: {out}")
        return out
    if args.start_step > 0 and not (parent and args.resume_optimizer):
        raise ValueError("a positive start step requires both parent adapter and optimizer")
    out.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(f"{out.resolve()}|{args.target_steps}|{args.seed}".encode()).hexdigest()
    (out / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": args.lora_rank}) + "\n")
    (out / "adapter_model.safetensors").write_bytes(b"fake-adapter-" + tag.encode())
    (out / "optimizer.pt").write_bytes(b"fake-optimizer-" + tag.encode())
    rng = random.Random(args.seed * 1000 + args.start_step)
    prompts = json.loads(Path(args.prompts).read_text())["train"]
    rho = float(os.environ.get("FAKE_TRAINER_RHO", "0.6"))
    noise_sd = math.sqrt((1 - rho) / rho) if rho > 0 else 1e6
    rel = {r: (out / f"reliability_log.rank{r}.jsonl").open("w") for r in range(world)} if args.reliability_log else {}
    with (out / "grpo_stats.jsonl").open("w") as stats:
        for step in range(args.start_step, args.target_steps):
            stats.write(json.dumps({
                "step": step + 1, "training_objective": args.objective,
                "advantage_normalization": advantage_normalization_for_objective(args.objective),
                "token_normalization": token_normalization_for_objective(args.objective),
                "nonzero_advantage_groups": world, "groups": world, "samples": world * args.group_size,
                "reward_mean": 0.3, "rank_reward_std_mean": 0.4, "loss": 0.01, "grad_norm": 0.5,
                "clip_fraction": 0.0, "mean_ratio": 1.0, "approx_kl": 0.0,
                "step_seconds": 70.0 + (5.0 if args.reliability_log else 0.0),
                "response_tokens": 4000}) + "\n")
            for rank in range(world):
                prompt_index = (args.seed * 37 + step * world + rank) % len(prompts)
                # latent pass rate below one half, so the difficulty score -|p-1/2|
                # is monotone in p and its split-half correlation is about rho
                theta = rng.gauss(0, 1)
                cdf = statistics.NormalDist().cdf
                pa = 0.5 * cdf(theta + noise_sd * rng.gauss(0, 1))
                pb = 0.5 * cdf(theta + noise_sd * rng.gauss(0, 1))
                if rel:
                    rel[rank].write(json.dumps({
                        "step": step + 1, "rank": rank, "prompt_index": prompt_index,
                        "pass_a": pa, "pass_b": pb, "pass": (pa + pb) / 2, "mixed": 0 < (pa + pb) / 2 < 1,
                        "cos_ab": rng.gauss(0, 0.05), "norm_a": 1.0, "norm_b": 1.0, "norm_total": 2.0,
                        "cos_a_others": rng.gauss(0, 0.1), "cos_b_others": rng.gauss(0, 0.1),
                        "cos_total_others": rng.gauss(0, 0.1)}) + "\n")
    for stream in rel.values():
        stream.close()
    manifest = {
        "schema": POLICY_SCHEMA, "training_objective": args.objective,
        "policy_update": policy_update_for_objective(args.objective), "reward_source": "verifier",
        "reference_kl_beta": 0.0, "supervised_loss": False, "positive_only_filter": False,
        "parameterization": "lora", "base_model": str(Path(args.model).resolve()), "world_size": world,
        "start_step": args.start_step, "completed_steps": args.target_steps,
        "samples_per_step": world * config.group_size, "max_new_tokens": args.max_new_tokens,
        "prompt_format": prompt_format, "seed": args.seed, "config": asdict(config),
        "runtime": {"logprob_micro_batch": args.logprob_micro_batch,
                    "gradient_checkpointing": not args.disable_gradient_checkpointing},
        "advantage_normalization": advantage_normalization_for_objective(args.objective),
        "token_normalization": token_normalization_for_objective(args.objective),
        "adapter_sha256": sha256_file(out / "adapter_model.safetensors"),
        "optimizer_sha256": sha256_file(out / "optimizer.pt"),
        "grpo_stats_sha256": sha256_file(out / "grpo_stats.jsonl"),
        "prompts_sha256": sha256_file(Path(args.prompts)),
        "reliability_log": bool(args.reliability_log),
        "parent_policy": str(parent) if parent else None,
        "parent_policy_manifest_sha256": sha256_file(parent / "policy_train.json") if parent else None,
        "parent_adapter_sha256": sha256_file(parent / "adapter_model.safetensors") if parent else None,
        "parent_optimizer_sha256": sha256_file(Path(args.resume_optimizer)) if parent else None,
        "fake_trainer": True,
    }
    (out / "policy_train.json").write_text(json.dumps(manifest, indent=1) + "\n")
    validate_policy_lineage(out, target_steps=args.target_steps, world_size=world,
                            training_objective=args.objective, expected_start_step=args.start_step,
                            expected_parent=parent, expected_prompts=Path(args.prompts),
                            require_complete_hashes=True)
    print(f"[fake-trainer] wrote {args.target_steps - args.start_step} fake updates to {out}")
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # tolerate the launcher's distributed prefix and the real trainer path
    if argv[:1] == ["-m"]:
        argv = argv[argv.index("--model"):] if "--model" in argv else argv
    argv = [a for a in argv if not a.endswith("train_policy_grpo.py")]
    write_policy(parse(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
