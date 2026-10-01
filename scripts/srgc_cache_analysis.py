#!/usr/bin/env python3
"""SR cache analysis: what the cached success rates look like and which prompts SR trains (no GPU).

The SR arm ranks the 400 candidates once by the cached initial-policy success
rate (eight binary rewards per candidate; score ``-|rate - 0.5|``, seeded tie
breaks) and, at every update, draws 40 candidates and trains the four with the
best cached rank among them. Neither the draw nor the ranking depends on
training, so SR's whole training schedule is fixed by the cache and the seed.
This script reads a seed's input bundle and reports:

  * the cache histogram (how many candidates have 0/8 ... 8/8 successes), the
    number at exactly 4/8 (the only prompts with a measured 50% rate), the
    3/8-5/8 band SR mostly draws from, and the 0/8 and 8/8 prompts whose eight
    equal rewards give GRPO no advantage signal;
  * SR's predicted training schedule from the shared prefix to the endpoint:
    training slots per cache bucket, distinct prompts, how concentrated the
    schedule is, and how many of the exactly-4/8 prompts are ever trained;
  * when ``seed-N/<arm>-progress.json`` histories exist, the cache-rate
    composition of what every recorded or extra arm actually trained
    (On-policy's gradient-ranked four, Random's four, SR's four, the
    direction/refresh controls) and of the SR-GC comparison sets, plus a check
    that the recorded SR history matches the predicted schedule;
  * per update, how much of each update was a real learning signal: the mean
    training reward and gradient norm from the history and, from the training
    receipts in ``cost-receipts/<arm>/``, the number of the four prompts whose
    eight responses all earned the same reward (zero GRPO advantage, no
    gradient). An update count is optimizer steps; this is what they carried.

    python scripts/srgc_cache_analysis.py --plan <group-storage plan>          # every seed of the plan
    python scripts/srgc_cache_analysis.py --input srgc_rebuttal/inputs/seed-5.json --seed 5 [--folder <seed-5 run folder>]

Outputs ``summary.txt``, ``cache.csv`` (one row per candidate),
``composition.csv`` (one row per seed, arm, source and bucket) and
``updates.csv`` (one row per recorded update) under ``<run root>/analysis/sr-cache`` or ``--out``.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srgc_rebuttal.srgc import Config, cached_sr_set, stream_seed  # noqa: E402

SKIP_KEYS = {"seed", "arm", "source", "bucket", "slots"}


def successes_of(cache, responses):
    """Candidate -> number of cached successes; the cache must hold ``responses`` binary rewards each."""
    result = {}
    for prompt, rewards in cache.items():
        values = [int(r) for r in rewards]
        if len(values) != responses or any(v not in (0, 1) for v in values):
            raise ValueError(f"{prompt}: cache must contain {responses} binary rewards")
        result[prompt] = sum(values)
    return result


def histogram(successes, responses):
    counts = Counter(successes.values())
    return [counts.get(k, 0) for k in range(responses + 1)]


def sr_schedule(candidates, cache, seed, config, start, total):
    """(global cached ranking, [(step, four trained prompts)]) exactly as ``Engine`` computes them for the SR arm."""
    candidates = tuple(candidates)
    ranked = cached_sr_set(candidates, cache, len(candidates), seed, config.responses)
    batches = []
    for step in range(start, total):
        rng = np.random.default_rng(stream_seed(seed, step, "candidate-draw"))
        draw = tuple(rng.choice(candidates, config.scoring_prompts, replace=False))
        eligible = set(draw)
        batches.append((step, tuple(i for i in ranked if i in eligible)[:config.training_prompts]))
    return ranked, batches


def composition(ids, successes, responses):
    counts = Counter(successes[i] for i in ids if i in successes)
    return [counts.get(k, 0) for k in range(responses + 1)]


def recorded_arms(folder):
    """{arm label: (history, receipts dir)} from ``<arm>-progress.json`` files; replicates are ``replicate<k>-<arm>``."""
    histories = {}
    if folder is None or not Path(folder).is_dir():
        return histories
    folder = Path(folder)
    files = [*folder.glob("*-progress.json"), *folder.glob("replicate-*/*-progress.json")]
    for path in sorted(files):
        arm = path.name[: -len("-progress.json")]
        receipts = path.parent / "cost-receipts" / arm
        if path.parent != folder:
            arm = f"{path.parent.name.replace('-', '', 1)}-{arm}"
        try:
            history = json.loads(path.read_text()).get("history", [])
        except (OSError, ValueError):
            continue
        if history:
            histories[arm] = (history, receipts)
    return histories


def zero_advantage_prompts(receipts, responses):
    """step -> prompts whose responses all earned the same reward, from finished training receipts; {} without receipts."""
    result = {}
    if receipts is None or not Path(receipts).is_dir():
        return result
    for path in Path(receipts).glob("*.json"):
        try:
            event = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if event.get("phase") != "training" or event.get("state") != "finished":
            continue
        zero = sum(v for k, v in event.get("counts", {}).items() if k.split(".")[-1] == "zero_advantage_responses")
        result[event["checkpoint"]] = zero / responses
    return result


def update_rows(seed, arm, history, saturated):
    rows = []
    for r in history:
        metrics = r.get("metrics") or {}
        step = r["checkpoint"]
        rows.append({"seed": seed, "arm": arm, "step": step, "selector": r.get("selector"),
                     "sample_reward": metrics.get("sample_reward"), "gradient_norm": metrics.get("gradient_norm"),
                     "zero_advantage_prompts": saturated.get(step)})
    return rows


def learning_signal(rows, interval):
    """Per block of ``interval`` updates: mean training reward, zero-gradient updates, saturated-prompt share."""
    blocks = {}
    for row in rows:
        block = blocks.setdefault(row["step"] // interval * interval, {"updates": 0, "reward": [], "zero_gradient": 0,
                                                                        "saturated": [], })
        block["updates"] += 1
        if row["sample_reward"] is not None:
            block["reward"].append(row["sample_reward"])
        if row["gradient_norm"] is not None and row["gradient_norm"] <= 1e-12:
            block["zero_gradient"] += 1
        if row["zero_advantage_prompts"] is not None:
            block["saturated"].append(row["zero_advantage_prompts"])
    return {start: {"updates": b["updates"],
                    "mean_reward": float(np.mean(b["reward"])) if b["reward"] else None,
                    "zero_gradient_updates": b["zero_gradient"],
                    "mean_saturated_prompts": float(np.mean(b["saturated"])) if b["saturated"] else None}
            for start, b in sorted(blocks.items())}


def analyse_seed(seed, data, config, start, total, folder=None):
    responses = config.responses
    candidates = tuple(data["candidate_ids"])
    cache = data["cached_rewards"]
    if set(cache) != set(candidates):
        raise ValueError(f"seed {seed}: the cache does not cover the candidate pool")
    successes = successes_of(cache, responses)
    hist = histogram(successes, responses)
    half = responses // 2
    ranked, batches = sr_schedule(candidates, cache, seed, config, start, total)
    rank_of = {prompt: k for k, prompt in enumerate(ranked)}
    trained = Counter(i for _, batch in batches for i in batch)
    slots = [i for _, batch in batches for i in batch]
    exactly_half = [i for i in candidates if successes[i] == half]
    band = sum(1 for i in candidates if abs(successes[i] - half) <= 1)
    top10 = sum(count for _, count in trained.most_common(10))
    report = {
        "seed": seed, "candidates": len(candidates), "responses": responses,
        "histogram": hist, "exactly_half": len(exactly_half), "band": band,
        "zero": hist[0], "full": hist[responses],
        "mean_rate": float(np.mean([successes[i] / responses for i in candidates])),
        "schedule": {"start": start, "total": total, "updates": len(batches), "slots": len(slots),
                     "composition": composition(slots, successes, responses),
                     "distinct": len(trained), "top10_share": top10 / len(slots) if slots else 0.0,
                     "half_trained": sum(1 for i in exactly_half if i in trained),
                     "mean_trained_rate": float(np.mean([successes[i] / responses for i in slots])) if slots else None},
        "rows": [{"seed": seed, "prompt": i, "successes": successes[i], "rate": successes[i] / responses,
                  "sr_rank": rank_of[i], "predicted_sr_train_count": trained.get(i, 0)} for i in candidates],
        "arms": {}, "sr_mismatch": None,
    }
    predicted = {step: batch for step, batch in batches}
    report["updates"] = []
    for arm, (history, receipts) in recorded_arms(folder).items():
        train_slots = [i for r in history for i in r.get("train_ids", [])]
        rows = update_rows(seed, arm, history, zero_advantage_prompts(receipts, responses))
        report["updates"].extend(rows)
        entry = {"updates": len(history), "composition": composition(train_slots, successes, responses),
                 "mean_trained_rate": float(np.mean([successes[i] / responses for i in train_slots if i in successes]))
                 if train_slots else None,
                 "distinct": len(set(train_slots)),
                 "signal": learning_signal(rows, config.selection_interval),
                 "receipts": any(row["zero_advantage_prompts"] is not None for row in rows)}
        comparison = [i for r in history if r.get("selection_refreshed") for i in r.get("sr_ids", [])]
        if comparison:
            entry["sr_comparison_composition"] = composition(comparison, successes, responses)
        drawn = [i for r in history if r.get("selection_refreshed") for i in r.get("on_ids", [])]
        if drawn:
            entry["candidate_composition"] = composition(drawn, successes, responses)
        report["arms"][arm] = entry
        if arm == "sr":
            mismatch = sum(1 for r in history if r["checkpoint"] in predicted
                           and tuple(r.get("train_ids", ())) != predicted[r["checkpoint"]])
            report["sr_mismatch"] = mismatch
    return report


def buckets_line(values, responses):
    return "  ".join(f"{k}/{responses}:{v:>4}" for k, v in enumerate(values))


def summarize(reports):
    lines = []
    for r in reports:
        n, k = r["candidates"], r["responses"]
        s = r["schedule"]
        lines.append(f"seed {r['seed']}  candidates {n}  cached successes per prompt (out of {k})")
        lines.append("  cache      " + buckets_line(r["histogram"], k))
        lines.append(f"  exactly {k // 2}/{k} (measured 50%): {r['exactly_half']} ({100 * r['exactly_half'] / n:.1f}%)   "
                     f"within one success of 50%: {r['band']} ({100 * r['band'] / n:.1f}%)   "
                     f"0/{k}: {r['zero']}   {k}/{k}: {r['full']}   mean rate {100 * r['mean_rate']:.1f}%")
        lines.append(f"  SR schedule (predicted, updates {s['start']}-{s['total'] - 1}, {s['slots']} training slots)")
        lines.append("  SR trains  " + buckets_line(s["composition"], k))
        lines.append(f"  distinct prompts trained {s['distinct']}/{n}   "
                     f"top-10 prompts hold {100 * s['top10_share']:.1f}% of slots   "
                     f"exactly-50% prompts ever trained {s['half_trained']}/{r['exactly_half']}   "
                     f"mean cached rate of trained prompts {100 * (s['mean_trained_rate'] or 0):.1f}%")
        if r["sr_mismatch"] is not None:
            lines.append(f"  recorded sr history vs predicted schedule: {r['sr_mismatch']} mismatching updates")
        for arm, entry in r["arms"].items():
            lines.append(f"  {arm:<22} trained ({entry['updates']} updates, {entry['distinct']} distinct, "
                         f"mean cached rate {100 * (entry['mean_trained_rate'] or 0):.1f}%)  "
                         + buckets_line(entry["composition"], k))
            if "sr_comparison_composition" in entry:
                lines.append(f"  {'':<22} SR-GC comparison set  " + buckets_line(entry["sr_comparison_composition"], k))
            if "candidate_composition" in entry:
                lines.append(f"  {'':<22} drawn candidates      " + buckets_line(entry["candidate_composition"], k))
            blocks = entry["signal"]
            total_zero = sum(b["zero_gradient_updates"] for b in blocks.values())
            saturated = [b["mean_saturated_prompts"] for b in blocks.values() if b["mean_saturated_prompts"] is not None]
            note = (f"saturated prompts per update {np.mean(saturated):.2f} of 4 (receipts)" if saturated
                    else "saturated prompts unknown (no training receipts)")
            lines.append(f"  {'':<22} learning signal: zero-gradient updates {total_zero}/{entry['updates']}; {note}")
            lines.append(f"  {'':<22} per block (start: updates, mean train reward, saturated prompts): " + "; ".join(
                f"{start}: {b['updates']}, {100 * b['mean_reward']:.1f}%, "
                + (f"{b['mean_saturated_prompts']:.2f}" if b["mean_saturated_prompts"] is not None else "?")
                for start, b in blocks.items() if b["mean_reward"] is not None))
        lines.append("")
    if not reports:
        lines.append("no seed with a complete cache")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(reports, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(reports)
    (out / "summary.txt").write_text(summary)
    with (out / "cache.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "prompt", "successes", "rate", "sr_rank",
                                                    "predicted_sr_train_count"])
        writer.writeheader()
        for r in reports:
            writer.writerows(r["rows"])
    with (out / "composition.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "arm", "source", "bucket", "slots"])
        for r in reports:
            k = r["responses"]
            for bucket, value in enumerate(r["histogram"]):
                writer.writerow([r["seed"], "cache", "cache", f"{bucket}/{k}", value])
            for bucket, value in enumerate(r["schedule"]["composition"]):
                writer.writerow([r["seed"], "sr", "predicted", f"{bucket}/{k}", value])
            for arm, entry in r["arms"].items():
                for bucket, value in enumerate(entry["composition"]):
                    writer.writerow([r["seed"], arm, "recorded", f"{bucket}/{k}", value])
    with (out / "updates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "arm", "step", "selector", "sample_reward", "gradient_norm",
                                                    "zero_advantage_prompts"])
        writer.writeheader()
        for r in reports:
            writer.writerows(r.get("updates", []))
    return summary


def load_bundle(path):
    data = json.loads(Path(path).read_text())
    cache = data.get("cached_rewards") or {}
    return data if cache and set(cache) == set(data.get("candidate_ids", [])) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, help="frozen plan (group-storage path); analyses every seed with a complete cache")
    parser.add_argument("--input", type=Path, help="one input bundle (srgc-inputs-v1) instead of a plan")
    parser.add_argument("--seed", type=int, help="base seed of --input (default: 3)")
    parser.add_argument("--folder", type=Path, help="seed run folder with <arm>-progress.json histories for --input")
    parser.add_argument("--prefix", type=int, default=25, help="shared prefix updates for --input (default 25)")
    parser.add_argument("--total", type=int, default=275, help="endpoint update count for --input (default 275)")
    parser.add_argument("--out", type=Path, help="output directory (default: <run root>/analysis/sr-cache)")
    args = parser.parse_args(argv)
    reports, skipped = [], []
    if args.plan is not None:
        from srgc_rebuttal.plan import input_path, load_plan
        from srgc_rebuttal.runtime import run_root
        plan = load_plan(args.plan)
        root = run_root(args.plan.resolve(), plan)
        config = Config(seed=plan["seeds"][0], scoring_prompts=plan["scoring_prompts_per_set"],
                        training_prompts=plan["training_prompts"], responses=plan["responses"],
                        selection_interval=plan["selection_interval"], check_interval=plan["check_interval"],
                        first_check=plan["first_check"], projection_dim=plan["projection_dim"])
        for seed in plan["seeds"]:
            data = load_bundle(input_path(args.plan, plan, seed))
            if data is None:
                skipped.append(seed)
                continue
            reports.append(analyse_seed(seed, data, config, plan["shared_prefix_updates"], plan["total_updates"],
                                        root / f"seed-{seed}"))
        out = args.out or root / "analysis" / "sr-cache"
    elif args.input is not None:
        data = load_bundle(args.input)
        if data is None:
            parser.error(f"{args.input} has no complete cache; the cache is generated on the cluster before training")
        seed = 3 if args.seed is None else args.seed
        reports.append(analyse_seed(seed, data, Config(seed=seed), args.prefix, args.total, args.folder))
        out = args.out or args.input.resolve().parent / "analysis" / "sr-cache"
    else:
        parser.error("--plan or --input is required")
    summary = write_outputs(reports, out)
    print(summary, end="")
    if skipped:
        print(f"skipped seeds without a complete cache: {skipped} (the cache is generated on the cluster before training)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
