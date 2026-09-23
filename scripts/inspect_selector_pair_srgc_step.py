"""Print one saved fixed-On D check, including why it cannot be extracted."""
from __future__ import annotations

import argparse
from pathlib import Path

import selector_pair_srgc_repeat as repeat
import selector_pair_srgc_score as score
from selector_pair_results import srgc_results


def inspect(root: Path, seed: int, start: int, step: int) -> str:
    initial = next((item for item in srgc_results(root).get("decisions", [])
                    if item["seed"] == seed and item["step"] == start), None)
    state = f"s{seed}-t{start}"
    if initial is None:
        return f"{state} step={step}: initial D decision missing"
    checkpoints = repeat.inventory(root, seed, start)
    checkpoint = checkpoints.get(step)
    if checkpoint is None:
        return f"{state} step={step}: On-policy checkpoint not saved"
    directory = repeat.output_dir(root, seed, start, 25) / f"step-{step}"
    expected, _ = repeat.checkpoint_reference(
        root, seed, start, step, checkpoint, initial, 25)
    files = [f"{stage}-{shard}{suffix}"
             for stage in score.STAGES for shard in range(4)
             for suffix in (".json", ".done.json")]
    missing = [name for name in files if not (directory / name).is_file()]
    done = sum((directory / f"{stage}-{shard}.done.json").is_file()
               for stage in score.STAGES for shard in range(4))
    if not (directory / "reference.json").is_file():
        return f"{state} step={step}: reference missing; completed shards={done}/16"
    if missing:
        return (f"{state} step={step}: D incomplete; completed shards={done}/16; "
                f"first missing={missing[0]}")
    try:
        value = repeat.check_projections(directory, root, expected)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return (f"{state} step={step}: 16/16 shards saved but D validation failed: "
                f"{type(exc).__name__}: {exc}")
    return (f"{state} step={step}: D={value['d']:.12g} "
            f"(A={value['d_a']:.12g}, B={value['d_b']:.12g}); "
            "16/16 shards validated")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--start-step", type=int, default=25)
    parser.add_argument("--check-step", type=int, default=100)
    args = parser.parse_args()
    if args.start_step < 0 or args.check_step <= args.start_step:
        parser.error("check step must be after the starting step")
    print(inspect(args.root.resolve(), args.seed, args.start_step, args.check_step))


if __name__ == "__main__":
    main()
