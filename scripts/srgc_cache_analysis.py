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
    3/8-5/8 band, and the 0/8 and 8/8 prompts in the initial cache. Their fresh
    training responses need not have the same rewards;
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
    reward-gradient contribution). This is not held-out evaluation, nor proof
    of a zero parameter change under an optimizer with momentum.

    python scripts/srgc_cache_analysis.py --plan <group-storage plan>          # every seed of the plan
    python scripts/srgc_cache_analysis.py --input srgc_rebuttal/inputs/seed-5.json --seed 5 [--folder <seed-5 run folder>]

Outputs ``summary.txt``, ``cache.csv`` (one row per candidate),
``composition.csv`` (one row per seed, arm, source and bucket) and
``updates.csv`` (one row per recorded update) under ``<run root>/analysis/sr-cache`` or ``--out``.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from srgc_rebuttal.srgc import Config, cached_sr_set, stream_seed  # noqa: E402

def read_object(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def successes_of(cache, responses):
    """Candidate -> number of cached successes; the cache must hold ``responses`` binary rewards each."""
    result = {}
    for prompt, rewards in cache.items():
        if not isinstance(rewards, (list, tuple)) or len(rewards) != responses or any(
                not finite_number(v) or v not in (0, 1) for v in rewards):
            raise ValueError(f"{prompt}: cache must contain {responses} binary rewards")
        result[prompt] = int(sum(rewards))
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
    ids = list(ids)
    unknown = set(ids) - set(successes)
    if unknown:
        raise ValueError(f"recorded prompts are absent from the cache: {sorted(unknown)}")
    counts = Counter(successes[i] for i in ids)
    return [counts.get(k, 0) for k in range(responses + 1)]


def recorded_arms(folder, seed=None, warnings=None):
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
        data = read_object(path)
        if seed is not None and data.get("seed", seed) != seed:
            raise ValueError(f"{path}: seed differs from input bundle")
        if data.get("arm", path.stem.removesuffix("-progress")) != path.stem.removesuffix("-progress"):
            raise ValueError(f"{path}: arm differs from filename")
        if warnings is not None and ("seed" not in data or "arm" not in data):
            warnings.append(f"{path}: missing seed/arm metadata; provenance not verified")
        history = data.get("history")
        if not isinstance(history, list):
            raise ValueError(f"{path}: history must be a list")
        steps = [r.get("checkpoint") if isinstance(r, dict) else None for r in history]
        if any(type(s) is not int or s < 0 for s in steps) or steps != sorted(set(steps)):
            raise ValueError(f"{path}: history checkpoints must be unique and increasing")
        if history:
            histories[arm] = (history, receipts)
    return histories


def zero_advantage_prompts(receipts, responses, training_prompts=None, warnings=None):
    """step -> prompts whose responses all earned the same reward, from finished training receipts; {} without receipts."""
    result = {}
    if receipts is None or not Path(receipts).is_dir():
        return result
    seen = set()
    for path in sorted(Path(receipts).glob("*.json")):
        event = read_object(path)
        if event.get("phase") != "training" or event.get("state") != "finished":
            continue
        step = event.get("checkpoint")
        if type(step) is not int or step < 0:
            raise ValueError(f"{path}: invalid training checkpoint")
        counts = event.get("counts", {})
        if not isinstance(counts, dict):
            raise ValueError(f"{path}: invalid training counts")
        values = [v for k, v in counts.items() if k.split(".")[-1] == "zero_advantage_responses"]
        if any(not finite_number(v) or v < 0 or v % responses for v in values):
            raise ValueError(f"{path}: invalid zero-advantage response count")
        zero = sum(values) / responses if values else None
        if zero is not None and training_prompts is not None and zero > training_prompts:
            raise ValueError(f"{path}: zero-advantage count exceeds training batch")
        # Receipts do not identify which retry produced the persisted optimizer
        # state. Even equal counts cannot establish that association.
        result[step] = None if step in seen else zero
        if warnings is not None and (step in seen or zero is None):
            warnings.append(f"{path}: step {step} has duplicate receipts or missing counts; signal unknown")
        seen.add(step)
    return result


def update_rows(seed, arm, history, saturated):
    rows = []
    for r in history:
        metrics = r.get("metrics") or {}
        step = r["checkpoint"]
        if not isinstance(metrics, dict):
            raise ValueError(f"{arm} step {step}: metrics must be an object")
        for key in ("sample_reward", "gradient_norm"):
            value = metrics.get(key)
            if value is not None and (not finite_number(value) or value < 0 or
                                      (key == "sample_reward" and value > 1)):
                raise ValueError(f"{arm} step {step}: invalid {key}")
        if r.get("completed_updates", step + 1) != step + 1:
            raise ValueError(f"{arm} step {step}: inconsistent completed_updates")
        rows.append({"seed": seed, "arm": arm, "step": step, "completed_updates": step + 1,
                     "selector": r.get("selector"),
                     "sample_reward": metrics.get("sample_reward"), "gradient_norm": metrics.get("gradient_norm"),
                     "zero_advantage_prompts": saturated.get(step)})
    return rows


def learning_signal(rows, interval):
    """Per block of ``interval`` updates: mean training reward, zero-gradient updates, saturated-prompt share."""
    blocks = {}
    for row in rows:
        block = blocks.setdefault(row["step"] // interval * interval, {"updates": 0, "reward": [], "zero_gradient": 0,
                                                                        "saturated": [], "gradient_observed": 0})
        block["updates"] += 1
        if row["sample_reward"] is not None:
            block["reward"].append(row["sample_reward"])
        if row["gradient_norm"] is not None:
            block["gradient_observed"] += 1
            if row["gradient_norm"] <= 1e-12:
                block["zero_gradient"] += 1
        if row["zero_advantage_prompts"] is not None:
            block["saturated"].append(row["zero_advantage_prompts"])
    return {start: {"updates": b["updates"],
                    "mean_reward": float(np.mean(b["reward"])) if b["reward"] else None,
                    "zero_gradient_updates": b["zero_gradient"],
                    "gradient_observed": b["gradient_observed"], "receipt_observed": len(b["saturated"]),
                    "mean_saturated_prompts": float(np.mean(b["saturated"])) if b["saturated"] else None}
            for start, b in sorted(blocks.items())}


def analyse_seed(seed, data, config, start, total, folder=None):
    responses = config.responses
    if not 0 <= start < total or responses <= 0 or responses % 2 or config.selection_interval <= 0:
        raise ValueError("require 0 <= prefix < total, positive even responses and positive interval")
    if data.get("provenance", {}).get("experiment_seed", seed) != seed:
        raise ValueError("seed differs from input provenance")
    candidates = tuple(data["candidate_ids"])
    if any(not isinstance(i, str) for i in candidates) or len(set(candidates)) != len(candidates):
        raise ValueError("candidate ids must be unique strings")
    if not 0 < config.training_prompts <= config.scoring_prompts <= len(candidates):
        raise ValueError("candidate pool is smaller than the requested sampling batch")
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
        "arms": {}, "sr_mismatch": None, "sr_compared": 0, "warnings": [],
        "training_prompts": config.training_prompts,
    }
    predicted = {step: batch for step, batch in batches}
    report["updates"] = []
    for arm, (history, receipts) in recorded_arms(folder, seed, report["warnings"]).items():
        excluded = sum(r["checkpoint"] < start for r in history)
        if any(r["checkpoint"] >= total for r in history):
            raise ValueError(f"{arm}: history extends beyond requested endpoint {total}")
        history = [r for r in history if r["checkpoint"] >= start]
        for r in history:
            ids = r.get("train_ids")
            if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids) or (
                    len(ids) != config.training_prompts or len(set(ids)) != len(ids)):
                raise ValueError(f"{arm} step {r['checkpoint']}: invalid training ids")
        train_slots = [i for r in history for i in r.get("train_ids", [])]
        rows = update_rows(seed, arm, history, zero_advantage_prompts(
            receipts, responses, config.training_prompts, report["warnings"]))
        report["updates"].extend(rows)
        entry = {"updates": len(history), "composition": composition(train_slots, successes, responses),
                 "excluded_prefix_updates": excluded, "missing_updates": total - start - len(history),
                 "mean_trained_rate": float(np.mean([successes[i] / responses for i in train_slots]))
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
            report["sr_compared"] = len(history)
            report["sr_mismatch"] = mismatch if history else None
    return report


def buckets_line(values, responses):
    return "  ".join(f"{k}/{responses}:{v:>4}" for k, v in enumerate(values))


def summarize(reports):
    lines = ["Recorded metrics are training metrics, not held-out evaluation.",
             "Cache buckets describe initial responses, not fresh training rewards."]
    def rate(value):
        return "unknown" if value is None else f"{100 * value:.1f}%"
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
                     f"mean cached rate of trained prompts {rate(s['mean_trained_rate'])}")
        if r["sr_mismatch"] is not None:
            lines.append(f"  recorded sr history vs predicted schedule: {r['sr_mismatch']} mismatching updates "
                         f"out of {r['sr_compared']} compared; {s['updates'] - r['sr_compared']} unobserved")
        else:
            lines.append("  recorded sr history vs predicted schedule: not verified (no overlapping history)")
        for arm, entry in r["arms"].items():
            lines.append(f"  {arm:<22} trained ({entry['updates']} updates, {entry['distinct']} distinct, "
                         f"mean cached rate {rate(entry['mean_trained_rate'])})  "
                         + buckets_line(entry["composition"], k))
            lines.append(f"  {'':<22} missing updates {entry['missing_updates']}; "
                         f"excluded shared-prefix updates {entry['excluded_prefix_updates']}")
            if "sr_comparison_composition" in entry:
                lines.append(f"  {'':<22} SR-GC comparison set  " + buckets_line(entry["sr_comparison_composition"], k))
            if "candidate_composition" in entry:
                lines.append(f"  {'':<22} drawn candidates      " + buckets_line(entry["candidate_composition"], k))
            blocks = entry["signal"]
            total_zero = sum(b["zero_gradient_updates"] for b in blocks.values())
            measured = sum(b["gradient_observed"] for b in blocks.values())
            observed = sum(b["receipt_observed"] for b in blocks.values())
            total_saturated = sum(b["mean_saturated_prompts"] * b["receipt_observed"]
                                  for b in blocks.values() if b["receipt_observed"])
            note = (f"saturated prompts per update {total_saturated / observed:.2f} of {r['training_prompts']} "
                    f"({observed}/{entry['updates']} updates with unambiguous receipts)" if observed
                    else "saturated prompts unknown (no unambiguous training receipts)")
            lines.append(f"  {'':<22} learning signal: zero-gradient updates {total_zero}/{measured} measured "
                         f"({entry['updates'] - measured} unknown); {note}")
            lines.append(f"  {'':<22} per block (start: updates, mean train reward, saturated prompts): " + "; ".join(
                f"{start}: {b['updates']}, {100 * b['mean_reward']:.1f}%, "
                + (f"{b['mean_saturated_prompts']:.2f}" if b["mean_saturated_prompts"] is not None else "?")
                for start, b in blocks.items() if b["mean_reward"] is not None))
        lines.extend(f"  WARNING: {message}" for message in r["warnings"])
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
        writer = csv.DictWriter(handle, fieldnames=["seed", "arm", "step", "completed_updates", "selector", "sample_reward", "gradient_norm",
                                                    "zero_advantage_prompts"])
        writer.writeheader()
        for r in reports:
            writer.writerows(r.get("updates", []))
    return summary


def load_bundle(path):
    data = read_object(path)
    cache = data.get("cached_rewards") or {}
    if not cache:
        return None
    if not isinstance(cache, dict) or set(cache) != set(data.get("candidate_ids", [])):
        raise ValueError(f"{path}: incomplete or invalid cache")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan", type=Path, help="frozen plan (group-storage path); analyses every seed with a complete cache")
    source.add_argument("--input", type=Path, help="one input bundle (srgc-inputs-v1) instead of a plan")
    parser.add_argument("--seed", type=int, help="base seed of --input (default: input provenance)")
    parser.add_argument("--folder", type=Path, help="seed run folder with <arm>-progress.json histories for --input")
    parser.add_argument("--prefix", type=int, default=25, help="shared prefix updates for --input (default 25)")
    parser.add_argument("--total", type=int, default=275, help="endpoint update count for --input (default 275)")
    parser.add_argument("--out", type=Path, help="output directory (default: <run root>/analysis/sr-cache)")
    args = parser.parse_args(argv)
    reports, skipped = [], []
    if args.plan is not None:
        if args.seed is not None or args.folder is not None or args.prefix != 25 or args.total != 275:
            parser.error("--seed, --folder, --prefix and --total apply only to --input")
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
        seed = data.get("provenance", {}).get("experiment_seed") if args.seed is None else args.seed
        if type(seed) is not int:
            parser.error("--seed is required when input provenance has no experiment_seed")
        reports.append(analyse_seed(seed, data, Config(seed=seed), args.prefix, args.total, args.folder))
        out = args.out or args.input.resolve().parent / "analysis" / "sr-cache"
    else:
        parser.error("--plan or --input is required")
    summary = write_outputs(reports, out)
    print(summary, end="")
    if skipped:
        print(f"skipped seeds without a complete cache: {skipped} (the cache is generated on the cluster before training)")
    print(f"-> {out}")
    return 0 if reports and not skipped and not any(r["sr_mismatch"] for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
