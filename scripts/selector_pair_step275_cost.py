"""GPU time through the common step 275 for the four Figure-2 arms (added 2026-09-25).

CPU only. For seeds 3 and 4 it reports, per arm, the GPU-seconds spent after the
shared step-25 state up to step 275, the step-275 reward, and the difference
against continuing On-policy. Two cost views are given because the arms were
metered differently:

* update-timer: sum of the trainer's per-update seconds x 4 GPUs for updates
  26..275 (comparable across all four arms; excludes startup, checkpointing and
  evaluation, like Table 11 of the manuscript);
* allocation: checkpoint-linked allocated GPU-seconds from each branch's own
  ledger (startup and retries included), where a cost receipt exists.

Selection and diagnosis are listed separately: the On-policy ranking at step 25
(charged in the On branch ledger), and the SR-GC A/B measurements at 25, 50, ...
through the trigger (experimental two-reference cost, not halved). Unknown cost
is reported as unknown, never as zero. Historical cache creation is recovered
separately from the original source log when its provenance is unambiguous.
SR and Switch each pay the same cached-subset preparation once.

    python scripts/selector_pair_step275_cost.py --root PAIR_ROOT --switch-root SWITCH_ROOT \
        --eval-root STEP275_EVAL_ROOT --output OUTPUT_ROOT [--out FILE.txt]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import selector_pair_srgc_repeat as repeat
import selector_pair_switch_rewards as sw
from selector_pair_cache_cost import cache_creation_cost
from selector_pair_online_check_cost import single_reference_checks

import selection_gate as core
import selection_gate_gpu as base
import selector_pair as pair

SCHEMA = "offpolicy-selector-pair/common-step-cost-v1"
STEP = 275
START = 25
GPUS = 4
ARMS = ("random", "on_policy", "cached", "switch")
LABELS = {"random": "Random", "on_policy": "On-policy continued", "cached": "SR continued",
          "switch": "On-policy -> SR (Switch)"}


def step_seconds(stats_path, first, last):
    """Per-update trainer seconds for completed steps first..last, or None if any is missing."""
    if not stats_path.is_file():
        return None
    seconds = {}
    for line in stats_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step, value = row.get("step"), row.get("step_seconds")
        if isinstance(step, int) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            seconds.setdefault(step, float(value))
    wanted = range(first, last + 1)
    if any(step not in seconds for step in wanted):
        return None
    return sum(seconds[step] for step in wanted)


def checkpoint_receipt(policy, step):
    checkpoints = sw.saved_checkpoints(policy)
    path = checkpoints.get(step)
    if path is None or not (path / "cost-receipt.json").is_file():
        return None
    receipt = core.read(path / "cost-receipt.json")
    if receipt.get("step") != step or receipt.get("adapter_sha256") != base.digest(path / "adapter_model.safetensors"):
        raise ValueError(f"cost receipt does not match checkpoint {path}")
    return receipt


def control_cost(directory, step):
    """Branch ledger cost through the saved checkpoint at `step` (selection + setup + training)."""
    receipt = checkpoint_receipt(directory / "policy", step)
    result = {"update_timer_gpu_seconds": None, "allocation_gpu_seconds": None,
              "allocation_scoring_gpu_seconds": None, "allocation_note": None}
    timer = step_seconds(directory / "policy/grpo_stats.jsonl", START + 1, step)
    if timer is not None:
        result["update_timer_gpu_seconds"] = timer * GPUS
    if receipt is None:
        result["allocation_note"] = f"no cost receipt at step {step}; allocation cost unknown"
        return result
    try:
        _, events = base.read_cost_events(directory)
        cost = pair.cost_at_checkpoint(events, receipt)
    except (OSError, ValueError) as exc:
        result["allocation_note"] = f"allocation cost unknown: {exc}"
        return result
    result.update(allocation_gpu_seconds=cost["gpu_seconds"], allocation_scoring_gpu_seconds=cost["scoring_gpu_seconds"],
                  allocation_note="checkpoint-linked ledger cost through the step-275 checkpoint")
    return result


def ledger_total(directory, ledgers=("research", "deployment")):
    if not (directory / "cost.jsonl").exists():
        return None, "no ledger"
    cost = base.cost(directory)
    if not cost["complete"]:
        return None, "unclosed cost event"
    return sum(cost["ledgers"][name]["gpu_seconds"] for name in ledgers if name in cost["ledgers"]), "closed"


def diagnosis_cost(root, seed, trigger):
    """SR-GC A/B measurement cost at 25, 50, ..., trigger (experimental two-reference cost)."""
    initial = core.read(root / f"sr-gc/s{seed}-t25/decision.json")
    checks = [{"step": START, "gpu_seconds": initial["new_measurement_gpu_seconds"], "complete": True,
               "note": "initial measurement; reused R ranking excluded"}]
    for step in range(START + 25, trigger + 1, 25):
        directory = repeat.output_dir(root, seed, START, 25) / f"step-{step}"
        total, note = ledger_total(directory, ("research", "deployment", "reporting"))
        checks.append({"step": step, "gpu_seconds": total, "complete": total is not None, "note": note})
    known = sum(c["gpu_seconds"] for c in checks if c["complete"])
    unknown = [c["step"] for c in checks if not c["complete"]]
    return {"checks": checks, "known_gpu_seconds": known, "unknown_steps": unknown,
            "reused_ranking_gpu_seconds": initial["reused_ranking_gpu_seconds"],
            "scope": "experimental A/B references (two draws); reference-A stage costs are extracted "
                     "separately, never by halving this total"}


def switch_cost(switch_root, plan, on_control):
    """Prefix (On-policy 25->trigger) plus SR suffix (trigger->275) for the executed Switch."""
    seed, trigger = plan["seed"], plan["switch_step"]
    directory = switch_root / f"s{seed}"
    replay = sw.needs_replay(plan)
    prefix = {"mode": "replayed On-policy prefix" if replay else "original On-policy checkpoint"}
    if replay:
        timer = step_seconds(directory / "replay/policy/grpo_stats.jsonl", START + 1, trigger)
        prefix["update_timer_gpu_seconds"] = None if timer is None else timer * GPUS
        totals = [ledger_total(path)[0] for path in (directory / "attempts").glob("replay-*")]
        prefix["allocation_gpu_seconds"] = None if any(t is None for t in totals) or not totals else sum(totals)
        prefix["allocation_note"] = "closed replay allocations 25->trigger (research ledger)"
    else:
        cost = control_cost(Path(plan["controls"]["on_policy"]), trigger)
        prefix.update({k: cost[k] for k in ("update_timer_gpu_seconds", "allocation_gpu_seconds")})
        prefix["allocation_note"] = "On-policy branch ledger through the trigger checkpoint"
    suffix = {}
    timer = step_seconds(directory / "policy/grpo_stats.jsonl", trigger + 1, STEP)
    suffix["update_timer_gpu_seconds"] = None if timer is None else timer * GPUS
    receipt = checkpoint_receipt(directory / "policy", STEP)
    if receipt is not None:
        totals = []
        for path in (directory / "attempts").glob("train-*"):
            try:
                _, events = base.read_cost_events(path)
                totals.append(pair.cost_at_checkpoint(events, receipt)["gpu_seconds"])
            except (OSError, ValueError):
                continue
        suffix["allocation_gpu_seconds"] = totals[0] if len(totals) == 1 else None
        suffix["allocation_note"] = "checkpoint-linked train allocation through step 275" if totals else "receipt without matching allocation"
    else:
        suffix["allocation_gpu_seconds"] = None
        suffix["allocation_note"] = "Switch trainer wrote no step-275 cost receipt; update timers only"
    return {"prefix": prefix, "suffix": suffix,
            "update_timer_gpu_seconds": (None if None in (prefix["update_timer_gpu_seconds"], suffix["update_timer_gpu_seconds"])
                                         else prefix["update_timer_gpu_seconds"] + suffix["update_timer_gpu_seconds"]),
            "allocation_gpu_seconds": (None if None in (prefix["allocation_gpu_seconds"], suffix["allocation_gpu_seconds"])
                                       else prefix["allocation_gpu_seconds"] + suffix["allocation_gpu_seconds"]),
            "on_ranking_gpu_seconds": on_control["allocation_scoring_gpu_seconds"]}


def rewards_at_step(eval_root, switch_root, plan):
    values = {}
    report = eval_root / f"step{STEP}-controls.json"
    if report.is_file():
        for row in core.read(report)["rows"]:
            if row["seed"] == plan["seed"] and row["arm"] in ("random", "on_policy", "cached"):
                values[row["arm"]] = row["reward"]
    point = sw.measured_point(switch_root / f"s{plan['seed']}", plan, "switch", STEP)
    if point:
        values["switch"] = point["reward"]
    return values


def build(root, switch_root, eval_root, seeds):
    rows = []
    for seed in seeds:
        plan = core.read(switch_root / f"s{seed}/plan.json")
        trigger = plan["switch_step"]
        rewards = rewards_at_step(eval_root, switch_root, plan)
        controls = {arm: control_cost(Path(plan["controls"][arm]), STEP) for arm in ("random", "on_policy", "cached")}
        cache = cache_creation_cost(plan["contract"])
        diagnosis = diagnosis_cost(root, seed, trigger)
        online_check = single_reference_checks(root, seed, trigger)
        switch = switch_cost(switch_root, plan, controls["on_policy"])
        for arm in ARMS:
            cost = switch if arm == "switch" else controls[arm]
            extra = diagnosis["known_gpu_seconds"] if arm == "switch" else 0.
            extra_unknown = bool(diagnosis["unknown_steps"]) if arm == "switch" else False
            rows.append({"seed": seed, "arm": arm, "label": LABELS[arm], "trigger": trigger if arm == "switch" else None,
                         "reward": rewards.get(arm),
                         "update_timer_gpu_seconds": cost["update_timer_gpu_seconds"],
                         "allocation_gpu_seconds": cost["allocation_gpu_seconds"],
                         "diagnosis_gpu_seconds": extra, "diagnosis_unknown": extra_unknown,
                         "initial_cache_gpu_seconds": cache["gpu_seconds"] if arm in ("cached", "switch") else 0.,
                         "cache_creation": cache if arm in ("cached", "switch") else None,
                         "sr_preparation_gpu_seconds": (controls["cached"]["allocation_scoring_gpu_seconds"]
                                                        if arm in ("cached", "switch") else 0.),
                         "repeated_selection_gpu_seconds": None if arm in ("on_policy", "switch") else 0.,
                         "online_check_gpu_seconds": online_check["gpu_seconds"] if arm == "switch" else 0.,
                         "online_check": online_check if arm == "switch" else None,
                         "detail": cost, "diagnosis": diagnosis if arm == "switch" else None})
    return {"schema": SCHEMA, "step": STEP, "start": START, "rows": rows,
            "scope": "GPU-seconds after the shared step-25 state through step 275. update-timer = trainer update "
                     "seconds x 4 GPUs (startup, checkpointing and evaluation excluded); allocation = checkpoint-linked "
                     "allocated GPU-seconds from the branch ledgers where a receipt exists. Diagnosis = experimental "
                     "two-reference SR-GC measurements through the trigger, not an operating charge. "
                     "Single-reference check cost uses only separately metered validation-a and candidate-a "
                     "stages through the recorded Switch trigger, including closed retries; B and joint "
                     "A/B CPU aggregation are excluded. This accounting does not change the recorded switch rule. "
                     "The existing runs rank once at the start; On-policy and Switch reuse that initial ranking. "
                     "Selection includes scoring rollouts and gradients, not isolated gradient-kernel time. "
                     "There is no repeated full-pool ranking cost in these logs. Rewards are fractions at step 275; "
                     "cache creation is a separate historical charge before the continuation window. "
                     "missing values are unknown, not zero."}


def fmt_h(seconds):
    return "unknown" if seconds is None else f"{seconds / 3600:.2f}"


def render(data):
    out = io.StringIO()
    cache_notes = []
    check_notes = []
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["seed", "arm", "trigger", "reward_275_percent", "cache_creation_h", "sr_preparation_h", "selection_h", "training_h",
                     "selection_training_subtotal_h", "single_reference_check_h", "ab_validation_h",
                     "allocation_h", "vs_on_reward_pp"])
    for seed in sorted({r["seed"] for r in data["rows"]}):
        rows = {r["arm"]: r for r in data["rows"] if r["seed"] == seed}
        on = rows["on_policy"]
        check = rows["switch"]["online_check"]
        check_notes.append(f"SINGLE_REFERENCE_A seed={seed} through={check['through_step']} "
                           f"complete={check['complete']} gpu_h={fmt_h(check['gpu_seconds'])} "
                           f"known_gpu_h={fmt_h(check['known_gpu_seconds'])} missing_steps={check['unknown_steps']}")
        for point in check["checks"]:
            if point["included_before_switch"]:
                check_notes.append(f"  step={point['step']} gpu_h={fmt_h(point['gpu_seconds'])} "
                                   f"issues={'; '.join(point['issues']) or 'none'}")
        cache = rows["cached"].get("cache_creation") or {}
        cache_notes.append(f"CACHE_CREATION seed={seed} status={cache.get('status', 'unknown')} "
                           f"gpu_h={fmt_h(cache.get('gpu_seconds'))} "
                           f"source={cache.get('source_run', 'unknown')} "
                           f"reason={cache.get('reason', 'none')}")
        for arm in ARMS:
            r = rows[arm]
            timer, alloc, diag = r["update_timer_gpu_seconds"], r["allocation_gpu_seconds"], r["diagnosis_gpu_seconds"]
            diag_text = fmt_h(diag) + ("+unknown" if r["diagnosis_unknown"] else "") if arm == "switch" else "0.00"
            selection = r["detail"].get("on_ranking_gpu_seconds" if arm == "switch"
                                        else "allocation_scoring_gpu_seconds")
            preparation = rows["cached"]["detail"].get("allocation_scoring_gpu_seconds") if arm in ("cached", "switch") else 0.
            if arm == "switch":
                selection = None if selection is None or preparation is None else selection + preparation
            subtotal = None if selection is None or timer is None else selection + timer
            reward = "unknown" if r["reward"] is None else f"{100 * r['reward']:.3f}"
            delta = ("unknown" if r["reward"] is None or on["reward"] is None
                     else f"{100 * (r['reward'] - on['reward']):+.2f}")
            writer.writerow([seed, arm, r["trigger"] or "", reward,
                             fmt_h(r.get("initial_cache_gpu_seconds")),
                             "unknown" if preparation is None else f"{preparation / 3600:.9f}",
                             "unknown" if selection is None else f"{selection / 3600:.6f}",
                             fmt_h(timer), fmt_h(subtotal), fmt_h(r["online_check_gpu_seconds"]),
                             diag_text, fmt_h(alloc), delta])
    return (f"LOGGED GPU TIME AND REWARD AT COMMON STEP {STEP} (after shared step {START})\n"
            + data["scope"] + "\n\n" + out.getvalue()
            + "\nSubtotal = initial selection + update-timer training; not full allocated cost.\n"
            + "The same SR subset preparation is charged once to SR and Switch, as a reused input cost.\n"
            + "Cache creation is separate: recovered from a hash-matched original stage log when available, "
              "otherwise unknown. It is not included in the recorded subtotal.\n"
            + "Single-reference check = validation-a + candidate-a allocated GPU time only, "
              "through the recorded transition. B and joint A/B CPU aggregation are excluded; "
              "missing A receipts are listed, never replaced by half of the A/B total.\n"
            + "A/B validation and allocation are separate views, not added to the subtotal.\n"
            + "These logs cannot establish the cost of repeated gradient-based reselection.\n"
            + "\n" + "\n".join(cache_notes) + "\n\n" + "\n".join(check_notes) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Pair root (selector-pair-v1)")
    parser.add_argument("--switch-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True, help="step-275 control evaluation root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(3, 4), action="append")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    for other in (args.root, args.switch_root, args.eval_root):
        if args.output.resolve() == other.resolve() or other.resolve() in args.output.resolve().parents:
            raise SystemExit("[abort] the output root must be separate from the inputs")
    data = build(args.root.resolve(), args.switch_root.resolve(), args.eval_root.resolve(), args.seed or [3, 4])
    text = render(data) + "\nJSON\n" + json.dumps(data, ensure_ascii=True, allow_nan=False) + "\n"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"step{STEP}-cost.txt").write_text(text)
    (args.output / f"step{STEP}-cost.json").write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    destination = args.out or Path.home() / f"selector-pair-step{STEP}-cost.txt"
    destination.write_text(text)
    print(text, end="")
    print(f"[saved] {destination}")


if __name__ == "__main__":
    main()
