"""Read-only recovery check; keep the running E5's hashed scientific code intact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import evidence_downstream as ed
from train_policy_grpo import GrpoConfig, _checkpoint_contract, _latest_checkpoint


def recoverable_checkpoint(out: Path, arm: str) -> tuple[Path, int]:
    out = out.resolve()
    contract = ed.read(out / "experiment.json")
    if contract.get("schema") != ed.SCHEMA or arm not in contract["selectors"]:
        raise ValueError("unknown E5 contract or arm")
    run = Path(contract["source_run"])
    for name, expected in contract["source_hashes"].items():
        if ed.digest(run / name) != expected:
            raise ValueError(f"source changed: {name}")
    for name, expected in contract["code_hashes"].items():
        if ed.digest(ed.ROOT / name) != expected:
            raise ValueError(f"experiment code changed: {name}")
    config = ed.read(run / "run_config.json")
    if ed.digest(Path(config["model"]) / "config.json") != contract["model_config_sha256"]:
        raise ValueError("source model configuration changed")
    prompts = out / "subsets" / f"subset-{arm}.json"
    if ed.digest(prompts) != ed.read(out / "subsets_hashes.json")[arm]:
        raise ValueError("selected prompts changed")
    grpo = GrpoConfig(**{field.removeprefix("grpo_"): config[field]
                       for field in ed.TRAIN_FLAGS.values()
                       if field != "grpo_logprob_micro_batch"}, checkpoint_every=5)
    drift = contract["drift"]
    args = argparse.Namespace(
        objective="grpo", model=config["model"], seed=contract["seed"],
        start_step=drift, target_steps=drift + contract["steps"],
        prompts=str(prompts), max_new_tokens=config["max_new_tokens"],
        resume_adapter=str(run / f"policy_step_{drift}"),
        resume_optimizer=str(run / f"policy_step_{drift}" / "optimizer.pt"),
    )
    expected = _checkpoint_contract(args, grpo, 4)
    checkpoint, step = _latest_checkpoint(out / arm / "policy", args.target_steps, expected)
    if checkpoint is None or step <= drift:
        raise ValueError("no compatible, hash-verified downstream checkpoint for repair")
    return checkpoint, step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=ed.SELECTORS, required=True)
    args = parser.parse_args()
    try:
        checkpoint, step = recoverable_checkpoint(args.out, args.arm)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[repair-blocked] {exc}", file=sys.stderr)
        return 2
    print(f"[repair-ready] {args.arm}: step={step} checkpoint={checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
