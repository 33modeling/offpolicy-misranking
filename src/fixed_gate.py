"""One fixed-checkpoint reuse-score diagnostic, then a matched E5 continuation.

New output tree and rule; the historical training-pilot arm is not repurposed.
The pilot uses two independent eight-response groups and the same validation
direction as g11. No optimizer runs before the decision. Existing E5 outcomes
are reused only after checking the source, selected subset and training contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import evidence_downstream as ed
import gate_decision as gd
import selection_gate as core
import selection_gate_gpu as runtime

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "offpolicy-fixed-checkpoint-gate/v1"
GPUS = 4


def rule_from(path: Path) -> dict:
    rule = ed.read(path)
    if rule.get("schema") != SCHEMA or rule.get("selector") != "g11":
        raise ValueError("fixed gate requires its own g11 rule, not the old training-pilot rule")
    if type(rule.get("pilot_size")) is not int or rule["pilot_size"] < 4 or rule["pilot_size"] % GPUS:
        raise ValueError("pilot_size must be at least four and divisible by four")
    if rule.get("responses_per_measurement") != 8 or type(rule.get("pilot_seed")) is not int:
        raise ValueError("fixed gate requires eight responses per measurement and an integer seed")
    for key in ("diagnostic_wall_seconds", "remaining_scoring_gpu_seconds", "continuation_wall_seconds", "baseline_wall_seconds", "score_check_atol"):
        if not isinstance(rule.get(key), (int, float)) or not math.isfinite(rule[key]) or rule[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    factor = rule.get("cost_safety_factor")
    if not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor < 1:
        raise ValueError("cost_safety_factor must be at least one")
    gd.validate_rule(gd.default_rule(rule["pilot_size"], rule["r_min"], rule["confidence"]))
    return rule


def bind(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ed.bind(path, value)


def prepare(run: Path, out: Path, e5: Path, rule_path: Path) -> dict:
    run, out, e5 = run.resolve(), out.resolve(), e5.resolve()
    ed.require_separate_output(out, [run, e5])
    if out in run.parents or out in e5.parents:
        raise ValueError("gate output cannot contain source experiments")
    rule = rule_from(rule_path)
    config = ed.read(run / "run_config.json")
    n = len(ed.read(run / "prompts.json")["train"])
    if n < rule["pilot_size"]:
        raise ValueError("candidate pool is smaller than the frozen pilot")
    if config.get("behavior_k") != 8 or config.get("topk_frac") != .1:
        raise ValueError("requires the original eight-response, top-ten-percent source")
    e5_contract = ed.read(e5 / "experiment.json")
    if Path(e5_contract["source_run"]).resolve() != run or e5_contract["steps"] != 100 or e5_contract["eval_k"] != 8:
        raise ValueError("need the matching prepared 100-update, K=8 E5 seed directory")
    if config["seed"] != e5_contract["seed"] or config["drift"] != e5_contract["drift"]:
        raise ValueError("E5 seed/checkpoint differs")
    ed.arm_policy(e5, "before")
    hashes = dict(e5_contract["source_hashes"])
    for name in ("val_groups.pt", "score_protocol.json"):
        hashes[name] = ed.digest(run / name)
    for name, sha in hashes.items():
        if ed.digest(run / name) != sha:
            raise ValueError(f"source changed since E5 preparation: {name}")
    if ed.read(e5 / "subsets_hashes.json") != {
        arm: ed.digest(e5 / "subsets" / f"subset-{arm}.json") for arm in e5_contract["all_subsets"]
    }:
        raise ValueError("E5 subsets differ from their frozen hashes")
    pilot = sorted(random.Random(rule["pilot_seed"] + config["seed"]).sample(range(n), rule["pilot_size"]))
    contract = {"schema": SCHEMA, "rule": rule, "run": str(run), "e5": str(e5), "n": n,
                "config": config, "pilot_ids": pilot, "source_hashes": hashes,
                "e5_sha256": ed.digest(e5 / "experiment.json"),
                "subsets_sha256": ed.digest(e5 / "subsets_hashes.json"),
                "validation_target": "stored R direction, unchanged for both measurements and continuation scoring",
                "training_during_measurement": False,
                "code_hashes": {name: ed.digest(ROOT / "src" / name) for name in
                                ("fixed_gate.py", "fixed_gate_worker.py", "grads.py", "rollout.py")}}
    # Code revisions are provenance, not a reason to discard a completed run.
    target = out / "contract.json"
    if target.exists():
        previous = ed.read(target)
        if {k: v for k, v in previous.items() if k != "code_hashes"} != {k: v for k, v in contract.items() if k != "code_hashes"}:
            raise ValueError("fixed gate contract changed; existing output is preserved")
        return previous
    bind(target, contract)
    return contract


def verify(out: Path) -> dict:
    c = ed.read(out / "contract.json")
    if c.get("schema") != SCHEMA:
        raise ValueError("unsupported fixed gate output")
    for name, sha in c["source_hashes"].items():
        if ed.digest(Path(c["run"]) / name) != sha:
            raise ValueError(f"fixed gate source changed: {name}")
    e5 = Path(c["e5"])
    if ed.digest(e5 / "experiment.json") != c["e5_sha256"] or ed.digest(e5 / "subsets_hashes.json") != c["subsets_sha256"]:
        raise ValueError("E5 contract changed")
    return c


def expected_ids(c: dict, phase: str, shard: int) -> list[int]:
    if phase == "pilot":
        ids = c["pilot_ids"]
    elif phase == "remaining":
        ids = sorted(set(range(c["n"])) - set(c["pilot_ids"]))
    elif phase == "baseline":
        ids = list(range(c["n"]))
    else:
        raise ValueError("unknown fixed gate phase")
    return ids[shard::GPUS]


def read_phase(out: Path, phase: str) -> tuple[dict[int, dict], list[dict]]:
    c = ed.read(out / "contract.json")
    combined, parts = {}, []
    stored = ed.read(Path(c["run"]) / "scores_offpolicy.json")["g11"]
    for shard in range(GPUS):
        path = out / phase / f"shard-{shard}.json"
        part = ed.read(path)
        if part.get("contract_sha256") != ed.digest(out / "contract.json") or part.get("phase") != phase or part.get("shard") != shard:
            raise ValueError("fixed gate shard contract mismatch")
        if part.get("training_updates") != 0:
            raise ValueError("fixed gate measurement must not update the policy")
        ids = expected_ids(c, phase, shard)
        rows = part["rows"]
        if len(rows) != len(ids) or [r["prompt_idx"] for r in rows] != ids:
            raise ValueError("fixed gate shard has missing, duplicate or reordered prompts")
        for r in rows:
            keys = ("primary", "replica") if phase == "pilot" else ("primary",)
            if any(not isinstance(r.get(k), (int, float)) or not math.isfinite(r[k]) or abs(r[k]) > 1.00001 for k in keys):
                raise ValueError("invalid fixed gate score")
            if abs(r["primary"] - float(stored[str(r["prompt_idx"])]["score"])) > c["rule"]["score_check_atol"]:
                raise ValueError("recomputed g11 score differs from original E5 scoring")
            combined[r["prompt_idx"]] = r
        parts.append(part)
    return combined, parts


def projected_remaining_cost(c: dict, parts: list[dict]) -> float | None:
    """Conservative allocation estimate: fixed model loads plus candidate work.

    Does not amortize validation construction or replica generation across
    unseen candidates. Those are shared-input and pilot-only work respectively.
    """
    try:
        rate = max(p["timing"]["primary_seconds"] / len(p["rows"]) for p in parts)
        startup = max(p["timing"]["model_load_seconds"] for p in parts)
        if not all(math.isfinite(v) and v >= 0 for v in (rate, startup)) or rate <= 0:
            return None
        count = c["n"] - len(c["pilot_ids"])
        if count == 0:
            return 0.0
        return GPUS * c["rule"]["cost_safety_factor"] * (startup + math.ceil(count / GPUS) * rate)
    except (KeyError, TypeError, ZeroDivisionError):
        return None


def assess(c: dict, rows: dict[int, dict], cost: float | None) -> dict:
    if set(rows) != set(c["pilot_ids"]):
        raise ValueError("fixed gate requires exactly its frozen pilot IDs")
    rule = c["rule"]
    halves = {i: (rows[i]["primary"], rows[i]["replica"]) for i in c["pilot_ids"]}
    r = gd.pearson([v[0] for v in halves.values()], [v[1] for v in halves.values()])
    valid = math.isfinite(r)
    lo, hi = gd.fisher_bounds(r, len(halves), rule["confidence"]) if valid else (None, None)
    if not valid:
        action, reason = "random", "invalid_scores"
    elif cost is None or not math.isfinite(cost):
        action, reason = "random", "unknown_cost"
    elif cost > rule["remaining_scoring_gpu_seconds"]:
        action, reason = "random", "over_budget"
    elif lo >= rule["r_min"]:
        action, reason = "g11", "reliable"
    else:
        action, reason = "random", "weak" if hi < rule["r_min"] else "unresolved"
    return {"action": action, "reason": reason, "r": r if valid else None, "lower": lo, "upper": hi,
            "pilot_pairs": len(halves), "estimated_remaining_gpu_seconds": cost,
            "inference": "one approximate Fisher interval at a fixed policy/target; not a reward guarantee"}


def allocation(out: Path) -> dict:
    report = runtime.cost(out)
    phases = {}
    path = out / "cost.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        if row["state"] == "finished":
            entry = phases.setdefault(row["phase"], {"gpu_seconds": 0., "wall_seconds": 0.})
            entry["gpu_seconds"] += row["allocated_gpu_seconds"]
            entry["wall_seconds"] += row["seconds"]
    return {"complete": report["complete"], "ledgers": report["ledgers"], "phases": phases,
            "incomplete_events": report["incomplete_events"]}


def phase(out: Path, name: str, cap: float, devices: list[str], env: dict) -> None:
    workers = range(1) if name == "assess" else range(GPUS)
    commands = [([sys.executable, str(ROOT / "src/fixed_gate_worker.py"), "--out", str(out),
                  "--phase", name, "--shard", str(i)], "" if name == "assess" else devices[i]) for i in workers]
    runtime.meter(out, name, env["FIXED_GATE_GPU_TYPE"], commands=commands, env=env,
                  timeout=cap, ledger="research" if name == "baseline" else "deployment")


def sealed(path: Path) -> dict:
    record = ed.read(path)
    expected = core.fingerprint({k: v for k, v in record.items() if k != "record_sha256"})
    if record.get("record_sha256") != expected:
        raise ValueError(f"fixed gate publication changed: {path}")
    return record


def read_decision(out: Path) -> dict:
    record = sealed(out / "decision.json")
    if (record.get("contract_sha256") != ed.digest(out / "contract.json")
            or record.get("diagnostic_sha256") != ed.digest(out / "diagnostic.json")):
        raise ValueError("fixed gate decision belongs to a different contract or diagnostic")
    return record


def select(out: Path, devices: list[str], env: dict) -> dict:
    c = verify(out)
    final_path = out / "decision.json"
    if final_path.exists():
        return read_decision(out)
    diagnosis = out / "diagnostic.json"
    if not diagnosis.exists():
        if (out / "pilot-attempt.json").exists():
            record = {"action": "random", "reason": "interrupted_diagnostic", "pilot_pairs": None,
                      "estimated_remaining_gpu_seconds": None}
        else:
            bind(out / "pilot-attempt.json", {"contract_sha256": ed.digest(out / "contract.json"), "started_at": time.time()})
            try:
                started = time.monotonic()
                phase(out, "pilot", c["rule"]["diagnostic_wall_seconds"], devices, env)
                left = c["rule"]["diagnostic_wall_seconds"] - (time.monotonic() - started)
                if left <= 0:
                    raise TimeoutError("diagnostic deadline expired before validation")
                phase(out, "assess", left, devices, env)
                record = sealed(out / "assessment.json")
            except (TimeoutError, RuntimeError, ValueError, OSError) as exc:
                record = {"action": "random", "reason": "diagnostic_failed", "error": str(exc),
                          "pilot_pairs": None, "estimated_remaining_gpu_seconds": None}
        bind(diagnosis, record)
    record = ed.read(diagnosis)
    if record["action"] == "g11":
        remaining_complete = all((out / "remaining" / f"shard-{s}.json").is_file() for s in range(GPUS))
        if (out / "remaining-attempt.json").exists() and not remaining_complete:
            record = {**record, "action": "random", "reason": "interrupted_remaining_scoring"}
        else:
            if not (out / "remaining-attempt.json").exists():
                bind(out / "remaining-attempt.json", {"started_at": time.time()})
            try:
                if c["n"] > len(c["pilot_ids"]):
                    if not remaining_complete:
                        phase(out, "remaining", c["rule"]["remaining_scoring_gpu_seconds"] / GPUS, devices, env)
                    remaining, _ = read_phase(out, "remaining")
                else:
                    remaining = {}
                pilot, _ = read_phase(out, "pilot")
                scores = {i: r["primary"] for i, r in {**pilot, **remaining}.items()}
                from select_rules import jittered_topk, topk_count
                selected = sorted(jittered_topk(scores, topk_count(c["n"], .1), c["config"]["seed"] + 1000))
                expected = ed.read(Path(c["e5"]) / "subsets/subset-g11.json")["selected_idx"]
                if selected != expected:
                    raise ValueError("recomputed g11 subset differs; cannot reuse this E5 continuation")
            except (TimeoutError, RuntimeError, OSError) as exc:
                record = {**record, "action": "random", "reason": "remaining_scoring_failed", "error": str(exc)}
    if not allocation(out)["complete"]:
        record = {**record, "action": "random", "reason": "unknown_interrupted_allocation_cost"}
    action = record["action"]
    subset = Path(c["e5"]) / "subsets" / f"subset-{action}.json"
    decision = {**record, "schema": SCHEMA, "contract_sha256": ed.digest(out / "contract.json"),
                "diagnostic_sha256": ed.digest(diagnosis), "subset_sha256": ed.digest(subset),
                "selected_idx": ed.read(subset)["selected_idx"], "cost": allocation(out),
                "policy_updated_in_pilot": False,
                "cost_scope": "measured incremental allocations; source cache and ranking validation are shared historical inputs"}
    decision.pop("record_sha256", None)
    decision["record_sha256"] = core.fingerprint(decision)
    bind(final_path, decision)
    print(f"[fixed-gate] {out.parent.name}/{out.name}: {action} ({record['reason']})", flush=True)
    return decision


def rewards(out: Path) -> dict:
    """Read validated, matched fixed-arm outcomes without refitting the rule."""
    c = verify(out)
    decision = read_decision(out)
    e5 = Path(c["e5"])
    chosen = decision["action"]
    subset = e5 / "subsets" / f"subset-{chosen}.json"
    if ed.digest(subset) != decision["subset_sha256"]:
        raise ValueError("chosen E5 subset changed")
    with runtime.lease(e5 / ".summary.lock", blocking=True):
        report = ed.summarize(e5, allow_partial=True)
    values = {r["selector"]: r for r in report["rows"]}
    m, rand, picked = (values.get(k) for k in ("g11", "random", chosen))
    costs = allocation(out)
    phases = costs["phases"]
    baseline_complete = (out / "baseline.done.json").exists()
    if baseline_complete:
        read_phase(out, "baseline")
    baseline = phases.get("baseline", {}).get("gpu_seconds") if baseline_complete else None
    deployment = costs["ledgers"]["deployment"]["gpu_seconds"]
    diagnostic_gpu = sum(phases.get(k, {}).get("gpu_seconds", 0.) for k in ("pilot", "assess"))
    diagnostic_wall = sum(phases.get(k, {}).get("wall_seconds", 0.) for k in ("pilot", "assess"))
    result = {"schema": SCHEMA, "action": chosen, "reason": decision["reason"],
              "seed": c["config"]["seed"], "drift": c["config"]["drift"],
              "complete": all(v is not None for v in (m, rand, picked)),
              "reward_g11": m["reward_after"] if m else None,
              "reward_random": rand["reward_after"] if rand else None,
              "reward_decision": picked["reward_after"] if picked else None,
              "forgone_reward": m["reward_after"] - picked["reward_after"] if m and picked else None,
              "forgone_lower": 0.0 if chosen == "g11" else m.get("random_lower") if m else None,
              "forgone_upper": 0.0 if chosen == "g11" else m.get("random_upper") if m else None,
              "decision_sha256": ed.digest(out / "decision.json"), "cost": costs,
              "cost_comparison_complete": baseline is not None and costs["complete"],
              "r": decision.get("r"), "lower": decision.get("lower"), "upper": decision.get("upper"),
              "diagnostic_gpu_seconds": diagnostic_gpu if costs["complete"] else None,
              "diagnostic_wall_seconds": diagnostic_wall if costs["complete"] else None,
              "remaining_scoring_gpu_seconds": phases.get("remaining", {}).get("gpu_seconds", 0.) if costs["complete"] else None,
              "ungated_scoring_gpu_seconds": baseline,
              "measured_net_scoring_gpu_seconds_saved": baseline - deployment if baseline is not None and costs["complete"] else None,
              "outcome_scope": "contract-checked reuse of fixed-arm outcomes; no claim of newly trained gated policy or equal-total-compute superiority"}
    ed.atomic_json(out / "result.json", result)
    return result


def measure_baseline(out: Path, devices: list[str], env: dict) -> None:
    """Research-only ungated scoring timer; never feeds the frozen decision."""
    c = verify(out)
    read_decision(out)
    done = out / "baseline.done.json"
    if done.exists():
        read_phase(out, "baseline")
        return
    if (out / "baseline-attempt.json").exists():
        print(f"[cost pending] {out}: previous baseline attempt incomplete; no automatic repeated measurement", flush=True)
        return
    bind(out / "baseline-attempt.json", {"started_at": time.time()})
    try:
        phase(out, "baseline", c["rule"]["baseline_wall_seconds"], devices, env)
        read_phase(out, "baseline")
        bind(done, {"contract_sha256": ed.digest(out / "contract.json"), "cost_scope": "ungated incremental g11 scoring only"})
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        bind(out / "baseline-failure.json", {"error": str(exc)})
        print(f"[cost pending] {out}: {exc}; training outcomes remain usable", flush=True)


def complete_missing(out: Path, devices: list[str], env: dict) -> None:
    c = verify(out)
    e5 = Path(c["e5"])
    available = rewards(out)
    if available["complete"]:
        print(f"[reuse] {out.name}: matched E5 outcomes already complete; no training", flush=True)
        return
    experiment = ed.read(e5 / "experiment.json")
    evaluation = ed.read(e5 / "evaluation.json")
    test = out / "test-input.json"
    bind(test, {"test": evaluation["val"], "provenance": evaluation["provenance"]})
    if ed.digest(test) != experiment["eval_input_sha256"]:
        # prepare() binds the original input serialization too. Use its frozen
        # input file when available rather than replacing an existing E5 contract.
        work = e5.parents[3]
        original = work / "inputs/e5-reduced" / f"test-math500-d{c['config']['drift']}.json"
        if not original.is_file() or ed.digest(original) != experiment["eval_input_sha256"]:
            raise ValueError("original E5 test input needed to finish missing arms; existing outcomes preserved")
        test = original
    command = ["bash", str(ROOT / "scripts/run_downstream_independent.sh"), c["run"], str(e5),
               "--eval-prompts", str(test), "--steps", "100", "--eval-k", "8"]
    continuation_env = {**env, "OM_NODE_LOCK_HELD": "1", "OM_E5_CONTROLLER_PID": str(os.getpid()),
                        "DOWNSTREAM_SELECTORS": "random g11", "E5_SKIP_EVAL": "0", "E5_RELIABILITY_LOG": "0"}
    runtime.meter(out, "continuation", env["FIXED_GATE_GPU_TYPE"], commands=[(command, ",".join(devices))],
                  env=continuation_env, timeout=c["rule"]["continuation_wall_seconds"], ledger="reporting")
    rewards(out)


def status(root: Path, drifts=(400, 0), seeds=(0, 1, 2)) -> None:
    print(f"fixed-policy g11 gate: {len(drifts) * len(seeds)} points; {root}")
    for out in (root / f"d{d}" / f"s{s}" for d in drifts for s in seeds):
        if not (out / "contract.json").is_file():
            print(f"{out.parent.name}/{out.name}: not prepared")
            continue
        activity = ""
        if (out / "progress.json").is_file():
            p = ed.read(out / "progress.json")
            age = time.time() - p["updated"]
            state = p["state"] if age <= 120 or p["state"] != "running" else "stale heartbeat, activity unconfirmed"
            activity = f"{p['phase']} {state} {p['seconds']:.0f}s; host={p['host']} last_seen={age:.0f}s"
        suffix = f"; {activity}" if activity else ""
        if (out / "result.json").is_file():
            r = ed.read(out / "result.json")
            print(f"{out.parent.name}/{out.name}: {r['action']} ({r['reason']}) outcomes={'complete' if r['complete'] else 'pending'} cost={'complete' if r['cost_comparison_complete'] else 'pending'} forgone={r['forgone_reward']}{suffix}")
        elif (out / "decision.json").is_file():
            d = ed.read(out / "decision.json")
            print(f"{out.parent.name}/{out.name}: {d['action']} ({d['reason']}), outcomes pending{suffix}")
        elif activity:
            print(f"{out.parent.name}/{out.name}: {activity}")
        else:
            print(f"{out.parent.name}/{out.name}: prepared")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "status", "results", "plan"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--rule", type=Path, default=ROOT / "config/fixed_gate_rule.json")
    parser.add_argument("--drifts", type=int, nargs="+", default=[400, 0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)
    if args.mode == "status":
        status(args.root, args.drifts, args.seeds)
        return 0
    if args.mode == "results":
        rows, failed = [], False
        for out in (args.root / f"d{d}" / f"s{s}" for d in args.drifts for s in args.seeds):
            if (out / "decision.json").is_file():
                try:
                    rows.append(rewards(out))
                except (OSError, ValueError) as exc:
                    print(f"[invalid] {out}: {exc}", file=sys.stderr)
                    failed = True
        args.root.mkdir(parents=True, exist_ok=True)
        incomplete = failed or len(rows) != len(args.drifts) * len(args.seeds) or any(not r["complete"] or not r["cost_comparison_complete"] for r in rows)
        ed.atomic_json(args.root / "results.json", {"rows": rows, "incomplete_or_invalid": incomplete})
        with (args.root / "results.csv").open("w", newline="") as handle:
            fields = ["drift", "seed", "action", "reason", "complete", "cost_comparison_complete", "r", "lower", "upper",
                      "reward_g11", "reward_random", "reward_decision", "forgone_reward", "forgone_lower", "forgone_upper",
                      "diagnostic_wall_seconds", "diagnostic_gpu_seconds", "remaining_scoring_gpu_seconds",
                      "ungated_scoring_gpu_seconds", "measured_net_scoring_gpu_seconds_saved"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        status(args.root, args.drifts, args.seeds)
        return int(failed)
    rule_from(args.rule)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if args.mode == "run" and (len(devices) != GPUS or len(set(devices)) != GPUS or not all(devices)):
        raise ValueError("exactly four distinct allocated GPUs required")
    env = {"FIXED_GATE_GPU_TYPE": os.environ.get("FIXED_GATE_GPU_TYPE", "NVIDIA H100"),
           "OM_NODE_LOCK_HELD": "1", "OM_MATH_VERIFIER": "math_verify"}
    failed = False
    for drift in args.drifts:
        for seed in args.seeds:
            matches = list(args.matrix.glob(f"family-math500-s{seed}/*-s{seed}-math500-d{drift}"))
            e5 = args.work / "runs/e5-reduced" / f"math500-d{drift}" / f"s{seed}"
            out = args.root / f"d{drift}" / f"s{seed}"
            if len(matches) != 1 or not (e5 / "experiment.json").is_file():
                print(f"[skip] d{drift}/s{seed}: matching prepared E5 source not available", flush=True)
                continue
            if args.mode == "plan":
                print(f"d{drift}/s{seed}: g11 fixed-policy pilot -> one decision -> reuse/finish matched E5 arms; {out}")
                continue
            try:
                with runtime.lease(out / ".lock"):
                    if (out / "result.json").is_file():
                        previous = ed.read(out / "result.json")
                        if previous["complete"] and previous["cost_comparison_complete"]:
                            rewards(out)
                            print(f"[done] d{drift}/s{seed}: validated outcome and cost reused", flush=True)
                            continue
                    c = runtime.meter(out, "prepare", env["FIXED_GATE_GPU_TYPE"],
                                      action=lambda: prepare(matches[0], out, e5, args.rule), ledger="deployment")
                    cfg = c["config"]
                    local_env = {**env, "OM_ATTN": cfg.get("attn") or "eager", "OM_GEN_BATCH": str(cfg.get("gen_batch") or 32),
                                 "OM_LORA_TARGETS": cfg.get("lora_targets") or "", "OM_PROMPT_FORMAT": cfg["prompt_format"],
                                 "OM_TOP_P": str(cfg.get("top_p") or 1.), "OM_THINKING": str(cfg.get("thinking") or "off")}
                    select(out, devices, local_env)
                    complete_missing(out, devices, local_env)
                    measure_baseline(out, devices, local_env)
                    rewards(out)
            except BlockingIOError:
                print(f"[busy] d{drift}/s{seed}: another node owns this point; trying next", flush=True)
            except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
                failed = True
                print(f"[failed] d{drift}/s{seed}: {exc}; moving to next point", flush=True)
    return int(failed)


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
