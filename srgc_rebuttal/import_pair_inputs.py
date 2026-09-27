"""Build seed 5–9 ``srgc-inputs-v1`` bundles by reusing the seed 3 and 4 source data.

The selector-pair experiment (seeds 3 and 4) already holds everything a bundle
needs, on the cluster volume:

* ``<run>/prompts.json``: ``train`` (400 candidates) and ``val`` (100 prompts);
  the manuscript's ranking-validation set R is the first half of ``val``;
* ``<run>/rollouts_behavior_train.jsonl``: eight initial-policy rewards per
  candidate (``prompt_idx``, ``rollout_idx``, ``reward``), i.e. the cache;
* ``<pair root>/branches/<any>/switch.json``: ``sources[seed].path`` for each
  run and ``evaluation.test`` (the 300 held-out questions).

No GPU work is needed: prompts are re-rendered with the same RL-Zero math
format the runs used, rewards are copied, and each new seed is assigned one
source split (default alternating 3, 4, 3, 4, 3). The training randomness is
what differs between seeds; the data provenance is written into the bundle.

    python -m srgc_rebuttal.import_pair_inputs --pair-root /group-volume/.../selector-pair-v1 \
        --output-dir srgc_rebuttal/inputs
    # or, without a pair root:
    python -m srgc_rebuttal.import_pair_inputs --source 3=<run dir> --source 4=<run dir> \
        --evaluation <test.json> --output-dir srgc_rebuttal/inputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .build_inputs import MATH_PROMPT, SCHEMA, normalized, record_id
from .plan import validate_inputs
from .runtime import atomic_json

DEFAULT_ASSIGNMENT = {5: 3, 6: 4, 7: 3, 8: 4, 9: 3}
RESPONSES = 8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_rewards(run: Path, count: int) -> list[list[int]]:
    """Eight binary rewards per candidate index, ordered by rollout index."""
    rows = defaultdict(dict)
    with (run / "rollouts_behavior_train.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            reward = float(row["reward"])
            if reward not in (0.0, 1.0):
                raise ValueError(f"non-binary cached reward for prompt {row['prompt_idx']}")
            prompt, rollout = int(row["prompt_idx"]), int(row["rollout_idx"])
            if not 0 <= prompt < count or not 0 <= rollout < RESPONSES:
                raise ValueError("cached prompt or rollout index outside the expected range")
            if rollout in rows[prompt]:
                raise ValueError(f"duplicate rollout {row['rollout_idx']} for prompt {row['prompt_idx']}")
            rows[prompt][rollout] = int(reward)
    out = []
    for index in range(count):
        if sorted(rows.get(index, {})) != list(range(RESPONSES)):
            raise ValueError(f"candidate {index} lacks exactly {RESPONSES} cached rollouts")
        out.append([rows[index][j] for j in range(RESPONSES)])
    return out


def sources_from_pair_root(pair_root: Path) -> tuple[dict[int, Path], dict]:
    switch_files = sorted(pair_root.glob("branches/*/switch.json"))
    if not switch_files:
        raise ValueError(f"no branches/*/switch.json under {pair_root}")
    switch = json.loads(switch_files[0].read_text())
    sources = {int(seed): Path(entry["path"]) for seed, entry in switch["sources"].items()}
    return sources, switch["evaluation"]


def build_bundle(seed: int, source_seed: int, run: Path, evaluation: dict) -> dict:
    prompts = json.loads((run / "prompts.json").read_text())
    train, val, test = prompts["train"], prompts["val"], evaluation["test"]
    if len(val) % 2:
        raise ValueError("validation pool must split into two equal halves (R and A/B)")
    rewards = cached_rewards(run, len(train))
    records, groups = {}, {}
    for name, items in (("candidate_ids", train), ("validation_pool_ids", val), ("evaluation_ids", test)):
        ids = []
        for item in items:
            rid = record_id("math", item["question"])
            records[rid] = {"question": item["question"], "prompt": MATH_PROMPT.format(question=item["question"]),
                            "answer": str(item["answer"])}
            ids.append(rid)
        groups[name] = ids
    ranking = groups["validation_pool_ids"][: len(val) // 2]
    bundle = {"schema": SCHEMA, "dataset": "math500", "records": records, **groups,
              "ranking_validation_ids": ranking,
              "cached_rewards": dict(zip(groups["candidate_ids"], rewards)),
              "provenance": {"experiment_seed": seed, "reused_from_seed": source_seed, "source_run": str(run),
                             "split_seed": source_seed, "prompt_format": "olmo_rlzero_math",
                             "ranking_validation": "first half of the source validation pool (manuscript R split)",
                             "cache": {"file": str(run / "rollouts_behavior_train.jsonl"),
                                       "sha256": sha256(run / "rollouts_behavior_train.jsonl"),
                                       "responses": RESPONSES, "source": "initial-policy behavior rollouts of the source run"},
                             "prompts_sha256": sha256(run / "prompts.json"),
                             "evaluation": evaluation.get("provenance", {}),
                             "verifier": "srgc_rebuttal.verifiers:math_reward"}}
    validate_inputs(bundle)
    return bundle


def parse_assignment(items: list[str] | None) -> dict[int, int]:
    if not items:
        return dict(DEFAULT_ASSIGNMENT)
    out = {}
    for item in items:
        new, old = item.split("=", 1)
        out[int(new)] = int(old)
    return out


def write_bundle(path: Path, bundle: dict):
    if path.exists():
        if json.loads(path.read_text()) != bundle:
            raise ValueError(f"refusing to overwrite different experimental inputs: {path}; use a separate cohort directory")
        return
    atomic_json(path, bundle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair-root", type=Path, help="selector-pair root holding branches/*/switch.json")
    parser.add_argument("--source", action="append", help="SEED=RUN_DIR (repeatable) instead of --pair-root")
    parser.add_argument("--evaluation", type=Path, help="test.json with the 300 evaluation questions (with --source)")
    parser.add_argument("--assign", nargs="*", help="NEW=SOURCE pairs, default 5=3 6=4 7=3 8=4 9=3")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.pair_root:
        sources, evaluation = sources_from_pair_root(args.pair_root)
    elif args.source and args.evaluation:
        sources = {int(k): Path(v) for k, v in (s.split("=", 1) for s in args.source)}
        evaluation = json.loads(args.evaluation.read_text())
    else:
        parser.error("give --pair-root, or --source SEED=RUN_DIR (repeatable) with --evaluation")
    assignment = parse_assignment(args.assign)
    missing = sorted(set(assignment.values()) - set(sources))
    if missing:
        raise SystemExit(f"source seeds {missing} not found; available: {sorted(sources)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed, source_seed in sorted(assignment.items()):
        bundle = build_bundle(seed, source_seed, sources[source_seed], evaluation)
        path = args.output_dir / f"seed-{seed}.json"
        write_bundle(path, bundle)
        solved = sum(1 for r in bundle["cached_rewards"].values() if any(r))
        print(f"PASS: {path} <- seed {source_seed} ({sources[source_seed]}); "
              f"{len(bundle['candidate_ids'])} candidates, {solved} with a cached success")


if __name__ == "__main__":
    main()
