"""Prospective four-arm cost experiment, isolated from existing Pair runs.

All arms start from the same saved model AND optimizer. On-policy repeats
gradient-based selection until the end; Switch repeats it only until switching.
Switch uses one fresh reference on its own trajectory, not stored A/B triggers.
No cost/performance ordering is imposed. Use run_selector_pair_cost_measure.sh.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import selector_pair_srgc_score as reference_score

import evidence_downstream as ed
import selection_gate as core
import selection_gate_gpu as base
import selection_switch_gpu as switch
import selection_switch_score as scoring
import selector_pair_gpu as pair_worker

SCHEMA = "offpolicy-selector-pair/prospective-cost-v1"
ARMS = ("random", "cached", "switch", "on_policy")
START = 25
CATEGORIES = ("selection", "online_check", "training", "evaluation")
WORKER = REPO / "scripts/selector_pair_cost_worker.py"
SCOPE = (
    "Fresh four-arm continuations from a common step-25 model and optimizer, with a pre-existing SR cache. "
    "All arms restart at the same block boundaries. Operating allocation = selection + online checks + "
    "training, including phase startup/checkpointing and failed attempts. Evaluation is separate. "
    "Shared prefix, initial cache acquisition, preflight/hash verification, orchestration gaps, and queue "
    "waits are excluded. Function and update timers are nested breakdowns, never added again. "
    "On-policy reranks the full pool at each selection interval until the end. Switch reranks only "
    "until its fresh single-reference signal switches to cached SR. All arms still backpropagate "
    "for training. This repeated-selection protocol differs from the historical fixed-subset experiment; "
    "all rewards and the Switch trigger are measured anew, not copied from the A/B experiment."
)


def separated(output, sources):
    output = output.resolve()
    for source in sources:
        source = source.resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"output must be separate from source: {source}")


def source_plan(root, seed):
    from train_policy_grpo import validate_policy_manifest
    contract_path = root / f"branches/on_policy/states/s{seed}-t25/points/view-25/contract.json"
    c = core.read(contract_path)
    config = c["config"]
    if config["seed"] != seed or config["drift"] != START or config["topk_frac"] != .1:
        raise ValueError("expected Pair seed, step 25 and fixed top-10% subsets")
    run = Path(c["source_run"]).resolve()
    parent, prompts, cache = run / "policy_step_25", run / "prompts.json", run / "rollouts_behavior_train.jsonl"
    manifest = validate_policy_manifest(parent, target_steps=START, world_size=4,
                                        training_objective="grpo", require_complete_hashes=True)
    if (manifest["seed"] != seed or manifest["base_model"] != str(Path(config["model"]).resolve())
            or manifest["config"] != ed._expected_config(config)
            or manifest["max_new_tokens"] != config["max_new_tokens"]
            or manifest["prompt_format"] != config["prompt_format"]):
        raise ValueError("common parent differs from the training configuration")
    pools = core.read(prompts)
    scoring.layout(config, pools)
    if len(pools["train"]) != c["n"] or c["eval_k"] != 8:
        raise ValueError("candidate count or evaluation protocol changed")
    if len(ed.independent_test(pools, {"test": c["evaluation"]["val"],
                                       "provenance": c["evaluation"]["provenance"]})) < 4:
        raise ValueError("independent evaluation needs at least four questions")
    # Validate the whole existing cache without charging this preflight as selection.
    switch.cached_selection(cache, prompts=c["n"], responses=8, seed=seed, selector="difficulty")
    paths = [root / "pair.json", contract_path, prompts, cache, Path(config["model"]) / "config.json",
             *(parent / name for name in ed.POLICY_FILES)]
    return {"seed": seed, "contract": c, "parent": str(parent), "prompts": str(prompts),
            "cache": str(cache), "source_hashes": {str(p.resolve()): base.digest(p) for p in paths}}


def make_plan(args):
    root, output = args.root.resolve(), args.output.resolve()
    separated(output, [root])
    if args.end_step <= START or args.interval <= 0 or args.selection_interval <= 0 or args.replicates < 1:
        raise ValueError("positive interval/replicates and end step > 25 required")
    if not math.isfinite(args.max_gpu_hours) or args.max_gpu_hours <= 0:
        raise ValueError("positive finite per-arm GPU-hour cap required")
    seeds = sorted(set(args.seed or [3, 4]))
    sources = [source_plan(root, seed) for seed in seeds]
    separated(output, [Path(s["parent"]).parent for s in sources] +
              [Path(s["contract"]["config"]["model"]) for s in sources])
    code_paths = [*sorted((REPO / "src").glob("*.py")),
                  REPO / "scripts/selector_pair_cost_measure.py", WORKER,
                  REPO / "scripts/selector_pair_srgc_score.py"]
    return {"schema": SCHEMA, "root": str(root), "output": str(output), "seeds": seeds,
            "start_step": START, "end_step": args.end_step, "interval": args.interval,
            "selection_interval": args.selection_interval,
            "replicates": args.replicates, "max_gpu_hours_per_arm": args.max_gpu_hours,
            "arms": list(ARMS), "sources": sources, "scope": SCOPE,
            "switch_rule": "d<0 and (previous d<0 or (previous-previous d<0 and last-three mean<0))",
            "reference": "A partition only; candidate-a and validation-a; no reference B or A/B averaging",
            "code_sha256": {str(p.relative_to(REPO)): base.digest(p) for p in code_paths}}


def validate_plan(plan):
    if plan["schema"] != SCHEMA:
        raise ValueError("unknown measurement plan")
    for source in plan["sources"]:
        for path, digest in source["source_hashes"].items():
            if base.digest(Path(path)) != digest:
                raise ValueError(f"source changed; original experiment not modified: {path}")
    for name, digest in plan["code_sha256"].items():
        if base.digest(REPO / name) != digest:
            raise ValueError(f"measurement code changed: {name}; use a separate output root")


def hardware_inventory(devices):
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader,nounits",
        "-i", ",".join(devices)], text=True, timeout=20)
    rows = [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(output))]
    if (len(rows) != 4 or any(len(row) != 4 for row in rows)
            or len({row[0] for row in rows}) != 4 or len({tuple(row[1:]) for row in rows}) != 1):
        raise ValueError("four distinct, matching GPUs required")
    return {"devices": list(devices), "gpu_type": rows[0][1], "gpus": rows,
            "same_allocation_required_on_resume": True}


def should_switch(values):
    if any(type(x) not in (float, int) or not math.isfinite(x) for x in values):
        raise ValueError("finite single-reference contrasts required")
    return bool(len(values) >= 2 and values[-1] < 0 and (
        values[-2] < 0 or (len(values) >= 3 and values[-3] < 0 and sum(values[-3:]) / 3 < 0)))


def costs(unit):
    totals = {name: 0. for name in CATEGORIES}
    unknown, failures = [], 0
    for path in sorted((unit / "phases").glob("*/attempt-*/cost.jsonl")):
        category = core.read(path.parent.parent / "category.json")["category"]
        if category not in totals:
            raise ValueError(f"unexpected measurement category: {category}")
        _, events = base.read_cost_events(path.parent)
        summary = core.cost_summary(events)
        if not summary["complete"]:
            unknown.append(str(path))
        for ledger in summary["ledgers"].values():
            totals[category] += ledger["gpu_seconds"]
        failures += sum(e.get("state") == "finished" and e["exit_code"] != 0 for e in events)
    return {"gpu_seconds": totals, "unknown": unknown, "failed_events": failures}


def checked_receipt(path):
    receipt = core.read(path)
    for artifact, digest in receipt["files"].items():
        if base.digest(Path(artifact)) != digest:
            raise ValueError(f"completed phase artifact changed: {artifact}")
    return receipt


def phase(unit, name, category, plan, source, devices, job):
    directory = unit / "phases" / name
    success = directory / "success.json"
    if success.is_file():
        return checked_receipt(success)["result"]
    prior = costs(unit)
    if prior["unknown"]:
        raise ValueError("unclosed measurement cost; refusing to reset or treat it as zero")
    directory.mkdir(parents=True, exist_ok=True)
    base.bind(directory / "category.json", {"category": category})
    attempt = directory / f"attempt-{len(list(directory.glob('attempt-*'))) + 1:04d}"
    attempt.mkdir()
    env = pair_worker.environment(source["contract"])

    def paid(label, *, commands=None, action=None):
        state = costs(unit)
        remaining = plan["max_gpu_hours_per_arm"] * 3600 - sum(state["gpu_seconds"].values())
        if state["unknown"] or remaining <= 0:
            raise ValueError("per-arm cap exhausted or unclosed cost; no automatic reset")
        return base.meter(attempt, label, plan["hardware"]["gpu_type"],
                          commands=commands, action=action, env=env, timeout=remaining / 4,
                          ledger="reporting" if category == "evaluation" else "deployment", devices=4)

    print(f"[cost-measure] {unit.name} {name} -> {attempt}", flush=True)
    result, files = job(attempt, paid)
    base.bind(success, {"result": result, "files": {str(p): base.digest(p) for p in
                                                   [*files, directory / "category.json", attempt / "cost.jsonl"]}})
    return result


def commands(directory, kind, stage, devices):
    return [([sys.executable, str(WORKER), "--root", str(directory), "--kind", kind,
              "--stage", stage, "--shard", str(i)], device) for i, device in enumerate(devices)]


def timing_summary(directory, stages):
    total = {key: 0. for key in ("model_setup", "response_generation", "gradient_computation")}
    files = []
    for stage in stages:
        for shard in range(4):
            path = directory / f"timing-{stage}-{shard}.json"
            row = core.read(path)
            if row["stage"] != stage or row["shard"] != shard or row["gpus"] != 1:
                raise ValueError("timing shard mismatch")
            for key, value in row["seconds"].items():
                if key not in total or type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid function timing")
                total[key] += value
            files.append(path)
    return total, files


def scoring_contract(source, parent, step):
    return {"config": {**source["contract"]["config"], "drift": step}, "parent": str(parent),
            "adapter_sha256": base.digest(parent / "adapter_model.safetensors"),
            "prompts": source["prompts"], "prompts_sha256": base.digest(Path(source["prompts"])),
            "sampling_seed": 701000003 + source["seed"] * 1000003 + step * 7919}


def rank_subset(unit, parent, step, plan, source, devices):
    def job(directory, paid):
        base.bind(directory / "scoring.json", scoring_contract(source, parent, step))
        for stage in ("validation", "candidate"):
            paid(stage, commands=commands(directory, "ranking", stage, devices))
            paid(stage + "-merge", action=lambda stage=stage: scoring.merge(directory, stage))
        timings, files = timing_summary(directory, ("validation", "candidate"))
        return {"indices": core.read(directory / "selected.json")["indices"], "step": step,
                "function_gpu_seconds": timings}, [*files, directory / "selected.json", directory / "scoring.json"]
    return phase(unit, f"on-ranking-{step}", "selection", plan, source, devices, job)


def choose_subset(unit, arm, plan, source, devices):
    pools = core.read(Path(source["prompts"]))

    def cache_job(directory, paid):
        def select():
            value = switch.cached_selection(Path(source["cache"]), prompts=len(pools["train"]),
                                            responses=8, seed=source["seed"], selector="difficulty")
            base.bind(directory / "selected.json", value)
            return value
        value = paid("cached-selection", action=select)
        return {"indices": value["indices"]}, [directory / "selected.json"]

    cached = phase(unit, "cached-selection", "selection", plan, source, devices, cache_job) if arm in (
        "cached", "switch") else None
    if arm == "cached":
        return {"cached": cached["indices"]}
    if arm == "random":
        def random_job(directory, paid):
            def select():
                ids = sorted(random.Random(source["seed"] + 907).sample(
                    range(len(pools["train"])), max(1, int(.1 * len(pools["train"])))))
                base.bind(directory / "selected.json", {"indices": ids})
                return {"indices": ids}
            return paid("random-selection", action=select), [directory / "selected.json"]
        value = phase(unit, "random-selection", "selection", plan, source, devices, random_job)
        return {"random": value["indices"]}

    on = rank_subset(unit, Path(source["parent"]), START, plan, source, devices)
    return {"on_policy": on["indices"], **({"cached": cached["indices"]} if cached else {})}


def check(unit, parent, step, sets, plan, source, devices):
    def job(directory, paid):
        base.bind(directory / "reference.json", {**scoring_contract(source, parent, step),
                  "sets": sets, "state_id": f"s{source['seed']}-t{step}-single-reference"})
        for stage in ("validation-a", "candidate-a"):
            paid(stage, commands=commands(directory, "check", stage, devices))

        def contrast():
            import numpy as np
            candidates = reference_score.projections(directory, "candidate-a")
            direction = np.mean(list(reference_score.projections(directory, "validation-a").values()), axis=0)
            delta = np.mean([candidates[i] for i in sets["on_policy"]], axis=0) - np.mean(
                [candidates[i] for i in sets["cached"]], axis=0)
            d = float(delta @ direction)
            if not math.isfinite(d):
                raise ValueError("non-finite online contrast")
            return d
        d = paid("contrast", action=contrast)
        timings, files = timing_summary(directory, ("validation-a", "candidate-a"))
        return {"step": step, "d": d, "function_gpu_seconds": timings,
                "reference_count": 1}, [*files, directory / "reference.json",
                *(directory / f"{stage}-{i}.json" for stage in ("candidate-a", "validation-a") for i in range(4))]
    return phase(unit, f"check-{step}", "online_check", plan, source, devices, job)


def train_command(policy, parent, subset, source, start, end):
    config = {**source["contract"]["config"], "drift": start}
    args = ed.train_args(config, parent.parent, policy.parent, "unused", end - start)
    for flag, value in (("--prompts", subset), ("--output", policy),
                        ("--resume-adapter", parent), ("--resume-optimizer", parent / "optimizer.pt")):
        args[args.index(flag) + 1] = str(value)
    return [sys.executable, *args]


def training_result(policy, parent, subset, source, start, end):
    from train_policy_grpo import validate_policy_lineage
    cfg = source["contract"]["config"]
    validate_policy_lineage(policy, target_steps=end, world_size=4, training_objective="grpo",
                            expected_start_step=start, expected_parent=parent,
                            expected_model=Path(cfg["model"]), expected_seed=source["seed"],
                            expected_config=ed._expected_config(cfg), expected_prompts=subset,
                            expected_max_new_tokens=cfg["max_new_tokens"],
                            expected_prompt_format=cfg["prompt_format"], require_complete_hashes=True)
    rows = [json.loads(line) for line in (policy / "grpo_stats.jsonl").read_text().splitlines() if line.strip()]
    if [r["step"] for r in rows] != list(range(start + 1, end + 1)):
        raise ValueError("training timers must cover exactly the disjoint update interval")
    seconds = [r["step_seconds"] for r in rows]
    if any(type(s) not in (int, float) or not math.isfinite(s) or s <= 0 for s in seconds):
        raise ValueError("invalid update timer")
    return {"policy": str(policy), "start": start, "end": end, "updates": end - start,
            "update_timer_gpu_seconds": 4 * sum(seconds),
            "response_tokens": sum(r.get("response_tokens", 0) for r in rows)}


def train_block(unit, parent, subset, start, end, plan, source, devices):
    def job(directory, paid):
        policy = directory / "policy"
        paid("train", commands=[(train_command(policy, parent, subset, source, start, end), ",".join(devices))])
        value = training_result(policy, parent, subset, source, start, end)
        return value, [policy / name for name in ed.POLICY_FILES]
    result = phase(unit, f"train-{start}-{end}", "training", plan, source, devices, job)
    training_result(Path(result["policy"]), parent, subset, source, start, end)
    return result


def evaluate(unit, parent, plan, source, devices):
    def job(directory, paid):
        c = source["contract"]
        base.bind(directory / "evaluation.json", {"config": c["config"], "parent": str(parent),
                  "adapter_sha256": base.digest(parent / "adapter_model.safetensors"),
                  "questions": c["evaluation"]["val"], "k": c["eval_k"],
                  "sampling_seed": c["eval_seed"] + 7919 * (plan["end_step"] + 1)})
        paid("evaluation", commands=commands(directory, "evaluation", "evaluation", devices))
        rewards, files = [], [directory / "evaluation.json"]
        n, k = len(c["evaluation"]["val"]), c["eval_k"]
        for shard in range(4):
            path = directory / f"evaluation-{shard}.jsonl"
            done = directory / f"evaluation-{shard}.done.json"
            if core.read(done) != {"evaluation_sha256": base.digest(directory / "evaluation.json"),
                                  "shard": shard, "sha256": base.digest(path)}:
                raise ValueError("evaluation receipt mismatch")
            ed.reward_rows(path, range(n * shard // 4, n * (shard + 1) // 4), k)
            rewards.extend(json.loads(line)["reward"] for line in path.read_text().splitlines() if line.strip())
            files.extend((path, done))
        timings, timing_files = timing_summary(directory, ("evaluation",))
        return {"reward": sum(rewards) / len(rewards), "question_count": n, "k": k,
                "function_gpu_seconds": timings}, [*files, *timing_files]
    return phase(unit, "evaluation", "evaluation", plan, source, devices, job)


def block_boundaries(plan):
    return sorted({START, plan["end_step"],
                   *range(START, plan["end_step"], plan["interval"]),
                   *range(START, plan["end_step"], plan["selection_interval"])})


def run_unit(unit, arm, plan, source, devices):
    sets = choose_subset(unit, arm, plan, source, devices)
    pools = core.read(Path(source["prompts"]))
    for name, indices in sets.items():
        base.bind(unit / f"subset-{name}.json", {"train": [pools["train"][i] for i in indices]})
    parent, trigger, checks, blocks = Path(source["parent"]), None, [], []
    ranking_steps = [START] if arm in ("on_policy", "switch") else []
    boundaries = block_boundaries(plan)
    for start, end in pairwise(boundaries):
        if (start > START and (start - START) % plan["selection_interval"] == 0
                and (arm == "on_policy" or (arm == "switch" and trigger is None))):
            ranked = rank_subset(unit, parent, start, plan, source, devices)
            sets["on_policy"] = ranked["indices"]
            ranking_steps.append(start)
        if arm == "switch" and trigger is None and (start - START) % plan["interval"] == 0:
            checks.append(check(unit, parent, start, sets, plan, source, devices))
            if should_switch([c["d"] for c in checks]):
                trigger = start
        selector = ("cached" if trigger is not None else "on_policy") if arm == "switch" else arm
        subset = unit / (f"subset-on_policy-step-{ranking_steps[-1]}.json" if selector == "on_policy"
                         else f"subset-{selector}.json")
        base.bind(subset, {"train": [pools["train"][i] for i in sets[selector]]})
        block = train_block(unit, parent, subset, start, end, plan, source, devices)
        blocks.append(block)
        parent = Path(block["policy"])
    outcome = evaluate(unit, parent, plan, source, devices)
    result = {"arm": arm, "seed": source["seed"], "switch_step": trigger, "checks": checks,
              "blocks": blocks, "ranking_steps": ranking_steps, "evaluation": outcome,
              "plan_sha256": core.fingerprint(plan),
              "phase_receipts": {str(p): base.digest(p) for p in sorted((unit / "phases").glob("*/success.json"))}}
    base.bind(unit / "result.json", result)
    return result


def units(plan):
    for replica in range(1, plan["replicates"] + 1):
        for source in plan["sources"]:
            # Rotate execution order; report order never asserts a cost ordering.
            offset = (replica - 1 + source["seed"] - 3) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                yield Path(plan["output"]) / f"s{source['seed']}-r{replica}-{arm}", source, replica, arm


def report(plan):
    rows = []
    for unit, source, replica, arm in units(plan):
        charge = costs(unit)
        result = core.read(unit / "result.json") if (unit / "result.json").is_file() else None
        if result and result["plan_sha256"] != core.fingerprint(plan):
            raise ValueError("result belongs to another measurement plan")
        if result:
            for path, digest in result["phase_receipts"].items():
                if base.digest(Path(path)) != digest:
                    raise ValueError("finalized phase receipt changed")
        complete = bool(result and not charge["unknown"])
        totals = charge["gpu_seconds"]
        function = {key: 0. for key in ("model_setup", "response_generation", "gradient_computation")}
        scoring_function = {category: dict(function) for category in ("selection", "online_check")}
        for path in (unit / "phases").glob("*/success.json"):
            receipt = checked_receipt(path)
            category = core.read(path.parent / "category.json")["category"]
            if category not in scoring_function:
                continue
            for key, value in receipt["result"].get("function_gpu_seconds", {}).items():
                function[key] += value
                scoring_function[category][key] += value
        rows.append({"seed": source["seed"], "replica": replica, "arm": arm, "complete": complete,
                     **charge, "operating_gpu_seconds": sum(totals[k] for k in CATEGORIES[:-1]) if complete else None,
                     "function_gpu_seconds_successful_scoring_only": function if complete else None,
                     "scoring_function_gpu_seconds_by_category": scoring_function if complete else None,
                     "update_timer_gpu_seconds": sum(b["update_timer_gpu_seconds"] for b in result["blocks"]) if result else None,
                     "reward": result["evaluation"]["reward"] if result else None,
                     "switch_step": result["switch_step"] if result else None,
                     "ranking_steps": result["ranking_steps"] if result else None,
                     "check_steps": [c["step"] for c in result["checks"]] if result else None,
                     "completed_phases": len(list((unit / "phases").glob("*/success.json")))})
    return {"schema": SCHEMA, "scope": SCOPE, "plan_sha256": core.fingerprint(plan),
            "selection_interval": plan["selection_interval"], "check_interval": plan["interval"],
            "complete": all(r["complete"] for r in rows), "rows": rows,
            "initial_cache_acquisition_gpu_seconds": None}


def render_csv(data):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["seed", "replica", "arm", "complete", *[k + "_gpu_h" for k in CATEGORIES],
                     "operating_gpu_h", "update_timer_gpu_h", "gradient_function_gpu_h", "rollout_function_gpu_h",
                     "selection_gradient_function_gpu_h", "check_gradient_function_gpu_h",
                     "reward_percent", "switch_step", "ranking_count", "ranking_steps", "check_count", "check_steps",
                     "failed_events", "completed_phases"])
    def hours(value):
        return "unknown" if value is None else f"{value / 3600:.9f}"
    for row in sorted(data["rows"], key=lambda r: (r["seed"], r["replica"], ARMS.index(r["arm"]))):
        function = row["function_gpu_seconds_successful_scoring_only"] or {}
        scoring_function = row["scoring_function_gpu_seconds_by_category"] or {}
        steps = []
        for key in ("ranking_steps", "check_steps"):
            steps.extend((len(row[key]), ";".join(map(str, row[key]))) if row[key] is not None
                         else ("unknown", "unknown"))
        writer.writerow([row["seed"], row["replica"], row["arm"], row["complete"],
                         *[hours(row["gpu_seconds"][k]) if row["complete"] else "unknown" for k in CATEGORIES],
                         hours(row["operating_gpu_seconds"]), hours(row["update_timer_gpu_seconds"]),
                         hours(function.get("gradient_computation")), hours(function.get("response_generation")),
                         *[hours(scoring_function.get(k, {}).get("gradient_computation"))
                           for k in ("selection", "online_check")],
                         "unknown" if row["reward"] is None else f"{100 * row['reward']:.6f}",
                         "" if row["switch_step"] is None else row["switch_step"],
                         *steps, row["failed_events"], row["completed_phases"]])
    return out.getvalue()


def default_report_path(plan):
    suffix = f"-s{plan['seeds'][0]}" if len(plan["seeds"]) == 1 else ""
    return Path.home() / f"selector-pair-cost-measure{suffix}-results.txt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "run", "status", "results"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(3, 4), action="append")
    p.add_argument("--end-step", type=int, default=275)
    p.add_argument("--interval", type=int, default=25)
    p.add_argument("--selection-interval", type=int, default=25,
                   help="gradient reranking cadence for On-policy and pre-switch Switch (default: 25)")
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--max-gpu-hours", type=float, default=160., help="per-arm safety cap (default: 160 GPU-hours)")
    p.add_argument("--out", type=Path, help="results TXT (seed-specific filename in home for single-seed runs)")
    args = p.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    separated(args.output, [args.root])
    if args.mode in ("status", "results"):
        plan = core.read(args.output / "plan.json")
        if plan["output"] != str(args.output) or plan["root"] != str(args.root):
            raise ValueError("output plan roots differ")
        if args.seed and sorted(set(args.seed)) != plan["seeds"]:
            raise ValueError("requested seed differs from output plan")
        data = report(plan)
        print(SCOPE + "\n" + render_csv(data))
        if args.mode == "results":
            target = args.out or default_report_path(plan)
            target = target.resolve()
            sources = [args.root, *[Path(s["parent"]).parent for s in plan["sources"]],
                       *[Path(s["contract"]["config"]["model"]) for s in plan["sources"]]]
            if (target.suffix != ".txt" or any(target == s or s in target.parents for s in sources)
                    or (args.output in target.parents and target != args.output / "results.txt")):
                raise ValueError("report must not overwrite source artifacts")
            core.atomic_json(args.output / "results.json", data)
            (args.output / "results.csv").write_text(render_csv(data))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(SCOPE + "\n\n" + render_csv(data) + "\nJSON\n" + json.dumps(data, indent=2) + "\n")
            print(f"[results] {target}")
        return
    plan = make_plan(args)
    if args.mode == "plan":
        print(json.dumps(plan, indent=2))
        return
    devices = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if len(set(devices)) != 4 or len(devices) != 4 or os.environ.get("OM_NODE_LOCK_HELD") != "1":
        raise ValueError("use the shell launcher on an idle four-GPU allocation")
    plan["hardware"] = hardware_inventory(devices)
    from light_selection_gate_gpu import install_signal_handlers
    install_signal_handlers()
    with base.lease(args.output / ".run.lock"):
        base.bind(args.output / "plan.json", plan)
        for unit, source, replica, arm in units(plan):
            validate_plan(plan)
            base.bind(unit / "identity.json", {"seed": source["seed"], "replica": replica, "arm": arm,
                      "plan_sha256": core.fingerprint(plan)})
            run_unit(unit, arm, plan, source, devices)
        core.atomic_json(args.output / "results.json", report(plan))
    print("[complete] run the results command to export the measured costs and rewards")


if __name__ == "__main__":
    main()
