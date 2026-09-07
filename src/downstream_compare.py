#!/usr/bin/env python3
"""Matched downstream update after selection (extension E5, 2026-09-07).

Three subcommands used by ``scripts/run_downstream_compare.sh``:

``subsets``   Write one prompt file per selector from a completed point. Each
              file has the ``{"train": [...], "val": [...]}`` layout the GRPO
              trainer reads, with ``train`` restricted to the selector's top-k
              prompts. Selectors: ``fresh_r`` (matched eight-response R split),
              ``g00``, ``g10``, ``g01``, ``g11``, ``passrate_beta``
              (behavior-reward diversity ``-|rate - 0.5|``), and ``random``
              (seeded uniform draw).
``evaluate``  Sample ``k`` responses for every validation prompt with the base
              model plus an adapter and report the mean verifier reward.
``summarize`` Collect before/after evaluations into a table with paired
              differences against ``fresh_r``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from regime_map import _behavior_rates
from score_artifacts import load_complete_score_artifacts
from select_rules import jittered_topk, topk_count

ESTIMATORS = ("g00", "g10", "g01", "g11")
SELECTORS = ("fresh_r",) + ESTIMATORS + ("passrate_beta", "random")


def selector_scores(run: Path, seed: int) -> dict[str, dict[int, float]]:
    artifacts = load_complete_score_artifacts(run)
    if any("r" not in halves for halves in artifacts.splithalf.values()):
        raise ValueError(f"{run.name}: scores_splithalf.json lacks the matched R split")
    ids = set(artifacts.oracle)
    rates = _behavior_rates(run, ids)
    rng = random.Random(seed + 424_243)
    scores = {"fresh_r": {i: h["r"] for i, h in artifacts.splithalf.items()}}
    scores.update({est: artifacts.offpolicy[est] for est in ESTIMATORS})
    scores["passrate_beta"] = {i: -abs(rates[i] - 0.5) for i in sorted(ids)}
    scores["random"] = {i: rng.random() for i in sorted(ids)}
    return scores


def write_subsets(run: Path, out_dir: Path, frac: float, seed: int) -> dict[str, Path]:
    prompts = json.loads((run / "prompts.json").read_text())
    train = prompts["train"]
    scores = selector_scores(run, seed)
    k = topk_count(len(train), frac)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in SELECTORS:
        selected = sorted(jittered_topk(scores[name], k, seed + 1_000))
        payload = {"train": [train[i] for i in selected], "val": prompts["val"],
                   "selector": name, "selected_idx": selected, "k": k, "source_run": run.name}
        path = out_dir / f"subset-{name}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        written[name] = path
    (out_dir / "subsets_manifest.json").write_text(json.dumps(
        {"source_run": run.name, "frac": frac, "k": k, "seed": seed, "selectors": list(SELECTORS)}, indent=1))
    return written


def evaluate(model: str, adapter: Path | None, prompts_path: Path, k: int, max_new_tokens: int,
             temperature: float, out_path: Path, seed: int) -> dict:
    from rollout import collect_rollouts, load_policy

    prompts = json.loads(prompts_path.read_text())["val"]
    policy, tok = load_policy(model, adapter)
    rollouts = out_path.with_suffix(".jsonl")
    collect_rollouts(policy, tok, prompts, k, max_new_tokens, temperature, rollouts,
                     sampling_seed_base=seed)
    rewards = [float(json.loads(line)["reward"]) for line in rollouts.open()]
    if len(rewards) != len(prompts) * k:
        raise ValueError(f"expected {len(prompts) * k} rollouts, found {len(rewards)}")
    summary = {"model": model, "adapter": str(adapter) if adapter else None, "prompts": len(prompts),
               "k": k, "max_new_tokens": max_new_tokens, "temperature": temperature, "seed": seed,
               "mean_reward": statistics.fmean(rewards), "rollouts": rollouts.name}
    out_path.write_text(json.dumps(summary, indent=1))
    return summary


def summarize(results_dir: Path) -> list[dict]:
    before_path = results_dir / "eval-before.json"
    if not before_path.exists():
        raise FileNotFoundError(f"missing {before_path}")
    before = json.loads(before_path.read_text())["mean_reward"]
    rows = []
    for name in SELECTORS:
        after_path = results_dir / name / "eval-after.json"
        if not after_path.exists():
            continue
        after = json.loads(after_path.read_text())["mean_reward"]
        rows.append({"selector": name, "reward_before": before, "reward_after": after, "reward_change": after - before})
    fresh = next((r["reward_change"] for r in rows if r["selector"] == "fresh_r"), None)
    for row in rows:
        row["change_minus_fresh"] = None if fresh is None else row["reward_change"] - fresh
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("subsets"); p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("evaluate"); p.add_argument("--model", required=True); p.add_argument("--adapter", type=Path)
    p.add_argument("--prompts", type=Path, required=True); p.add_argument("--k", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=2048); p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("summarize"); p.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "subsets":
        for name, path in write_subsets(args.run, args.out, args.frac, args.seed).items():
            print(f"[subset] {name}: {path}")
    elif args.command == "evaluate":
        summary = evaluate(args.model, args.adapter, args.prompts, args.k, args.max_new_tokens,
                           args.temperature, args.out, args.seed)
        print(json.dumps(summary, indent=1))
    else:
        rows = summarize(args.results)
        with (args.results / "downstream_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            delta = row["change_minus_fresh"]
            print(f"{row['selector']:>14} before={row['reward_before']:.4f} after={row['reward_after']:.4f} "
                  f"change={row['reward_change']:+.4f} vs_fresh={'-' if delta is None else f'{delta:+.4f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
