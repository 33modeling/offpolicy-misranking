#!/usr/bin/env python3
"""Compact results of a selection-switch root: one small text file for the phone.

Per branch: state, arm, completed updates, allocation used, deployment phases,
mean held-out reward and the per-question rewards on one line. Per state:
paired question-bootstrap contrasts. Plus the gate model, each held-out gate
decision, and curve summaries when the root carries the convergence gate.
Read-only; needs no GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import selection_gate as core  # noqa: E402
import selection_switch as rule  # noqa: E402
from _status_summary import accounting_label, gate_label, mbpp_suite_label, selector_label, suite_label
from _switch_state_point import resolve_state_point

CONTRASTS = (("selection_full", "random_full"), ("gated", "random_full"), ("gated", "selection_full"),
             ("selection_reduced", "random_reduced"))
SCORING_PHASES = {"fresh-r-validation", "fresh-r-merge-validation", "fresh-r-candidate", "fresh-r-merge-candidate",
                  "difficulty-select", "hard-select"}
COMPACT_LIMIT = 64 * 1024


def read(path):
    try:
        return core.read(path)
    except (OSError, ValueError, RuntimeError):
        return None


def saved_endpoint(directory):
    """Certify only the saved endpoint metadata, never checkpoint lineage."""
    path = directory / "result.json"
    try:
        if not path.exists() and not path.is_symlink():
            return None, "missing", None
        raw = path.read_bytes()
        result = json.loads(raw)
    except (OSError, ValueError, RuntimeError) as exc:
        return None, f"unreadable: {exc}", None
    reason = None
    if not isinstance(result, dict):
        reason = "result is not an object"
    elif result.get("schema") != rule.SCHEMA:
        reason = "wrong result schema"
    elif result.get("complete") is not True:
        reason = "incomplete endpoint"
    else:
        rewards = result.get("rewards")
        if (not isinstance(rewards, dict) or not rewards
                or any(not isinstance(k, str) or len(k) > 20 or not k.isdecimal() for k in rewards)
                or len({int(k) for k in rewards}) != len(rewards)
                or any(not finite_cost(v) or v > 1 for v in rewards.values())):
            reason = "invalid per-question rewards"
        elif object_at(directory / "result.sha256.json").get("sha256") != hashlib.sha256(raw).hexdigest():
            reason = "missing or mismatched result seal"
    return (None, reason, result) if reason else (result, "sealed endpoint metadata", result)


def completed_updates(stop, step):
    completed = stop.get("completed_steps") if isinstance(stop, dict) else None
    return completed-step if type(completed) is int and completed >= step else None


def saved_curve(curve, directory):
    if not isinstance(curve, dict) or curve.get("schema") != rule.SCHEMA:
        return None
    seal = object_at(directory / "result.sha256.json")
    if not seal.get("sha256") or curve.get("result_sha256") != seal["sha256"]:
        return None
    points = curve.get("points")
    if (not isinstance(points, dict) or not points or any(
            not isinstance(v, dict) or type(v.get("updates")) is not int or v["updates"] < 0
            or not finite_cost(v.get("reward")) or v["reward"] > 1 for v in points.values())):
        return None
    return sorted(points.values(), key=lambda v: v["updates"])


def raw_evidence(label, path, value):
    """Bound invalid metadata so one corrupt artifact cannot crowd out results."""
    preview = json.dumps(value, sort_keys=True, ensure_ascii=True)
    suffix = ""
    if len(preview) > 1024:
        preview = preview[:1024]
        suffix = " [truncated; not a measured value]"
    try:
        data = path.read_bytes()
        identity = f"bytes={len(data)} sha256={hashlib.sha256(data).hexdigest()}"
    except (OSError, RuntimeError) as exc:
        identity = f"unreadable={str(exc)[:160]}"
    return f"{label} unverified {preview}{suffix} path={path} {identity}"


def state_point(child, step):
    try:
        return resolve_state_point(child, step)
    except RuntimeError as exc:
        return child / 'points' / f'view-{step}', str(exc)


def phases(directory):
    totals = defaultdict(float)
    # The branch ledger, plus the curve/ sub-ledger where archived-checkpoint curve
    # evaluations are metered (the parent point's curve lives in the state's curve-parent/).
    for path in (directory / "cost.jsonl", directory / "curve" / "cost.jsonl"):
        try:
            lines = path.read_text().splitlines()
        except (OSError, ValueError, RuntimeError):
            # A damaged ledger must not hide another branch's measured rewards.
            # ledger_coverage keeps its total unknown, never a certified zero.
            continue
        seen = set()
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if (row.get("state") != "finished" or not finite_cost(row.get("allocated_gpu_seconds"))
                        or not isinstance(row.get("phase"), str) or not isinstance(row.get("ledger"), str)):
                    continue
                key = row["event_id"]
                if key in seen:
                    continue
                seen.add(key)
                totals[(row["ledger"], row["phase"])] += row["allocated_gpu_seconds"]
            except (ValueError, KeyError, TypeError, AttributeError):
                # Keep readable finished subtotals; coverage marks the total unknown.
                continue
    return totals


def finite_cost(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def ledger_coverage(directory):
    """Read-only completeness check, not recovery or result-seal certification."""
    any_event = False
    try:
        if not (directory / "cost.jsonl").is_file():
            return "unknown: branch ledger missing"
        for path in (directory / "cost.jsonl", directory / "curve/cost.jsonl"):
            if not path.exists() and not path.is_symlink():
                continue
            events = {}
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                event = events.setdefault(row["event_id"], {})
                state = row["state"]
                if state not in {"started", "finished"} or state in event:
                    return "unknown: duplicate/invalid cost event"
                event[state] = row
                any_event = True
            for event in events.values():
                if set(event) != {"started", "finished"}:
                    return "unknown: open/missing cost event"
                start, finish = event["started"], event["finished"]
                if (not finite_cost(finish.get("allocated_gpu_seconds"))
                        or any(start.get(key) != finish.get(key) for key in ("ledger", "phase"))
                        or not isinstance(finish.get("ledger"), str) or not isinstance(finish.get("phase"), str)):
                    return "unknown: invalid cost event"
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
        return "unknown: unreadable cost ledger"
    return "closed events" if any_event else "unknown: empty cost ledger"


def cost_detail(branch):
    totals = branch["phases"]
    subtotal = sum(totals.values())
    coverage = ledger_coverage(branch["directory"])
    known = coverage == "closed events"
    diagnostic = branch["decision"].get("measurement_gpu_seconds")
    budget = branch["decision"].get("budget_gpu_seconds")
    def fmt_cost(value):
        return f"{value:.3f}" if finite_cost(value) else "unknown"
    scoring = sum(value for (_, phase), value in totals.items() if phase in SCORING_PHASES)
    training = sum(value for (_, phase), value in totals.items() if phase == "train")
    evaluation = sum(value for (_, phase), value in totals.items() if phase in {"evaluate", "curve"})
    return ("    COST GPU-s "
            f"scoring={fmt_cost(scoring)} training={fmt_cost(training)} evaluation={fmt_cost(evaluation)} "
            f"other={fmt_cost(subtotal-scoring-training-evaluation)} "
            f"finished_events_subtotal={fmt_cost(subtotal)} "
            f"branch_total_incl_reporting={fmt_cost(subtotal) if known else 'unknown'} "
            f"diagnostic_charge={fmt_cost(diagnostic)} "
            f"action_total_with_diagnostic={fmt_cost(subtotal+diagnostic) if known and finite_cost(diagnostic) else 'unknown'} "
            f"branch_allocation={fmt_cost(budget)} coverage={coverage}")


def paired(a, b, draws=10000, seed=0):
    import numpy as np
    keys = sorted(a, key=int)
    if keys != sorted(b, key=int):
        return None
    d = np.array([100*(a[k]-b[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(draws, len(d)))
    means = np.sort(d[idx].mean(axis=1))
    return float(d.mean()), float(means[int(.025*draws)]), float(means[int(.975*draws)-1])


def fmt(t):
    return "n/a" if t is None else f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]"


def branches(root):
    out = []
    for child in sorted(root.glob("states/s*-t*")):
        match = re.fullmatch(r"s(\d+)-t(\d+)", child.name)
        if not match:
            continue
        seed, step = match.groups()
        point, point_error = state_point(child, int(step))
        for point in [point]:
            for arm in (*rule.TEST_ARMS,):
                directory = point / arm
                if not directory.exists():
                    continue
                result, validation, raw_result = saved_endpoint(directory)
                if point_error:
                    result, validation = None, "state point unresolved: " + point_error
                stop = object_at(directory / "policy/budget_stop.json")
                decision = object_at(directory / "decision.json")
                execution = object_at(directory / "execution.json")
                failure = object_at(directory / "failure.json")
                out.append({"state": f"s{seed}/t{step}", "seed": int(seed), "step": int(step), "arm": arm,
                            "directory": directory, "result": result, "stop": stop, "decision": decision,
                            "validation": validation, "raw_result": raw_result,
                            "execution": execution, "failure": failure, "phases": phases(directory),
                            "discards": sorted(p.name for p in (directory / "discards").glob("*.json")),
                            "waivers": sorted(p.name for p in (directory / "waivers").glob("*.json")),
                            "curve": read(directory / "curve.json")})
    return out


def update_limit(p, rows):
    """Updates one allocation can buy: 110% of budget / median random-arm GPU-seconds per update."""
    units = []
    for b in rows:
        if not b["arm"].startswith("random_") or b["waivers"] or b["discards"] or not b["stop"]:
            continue
        updates = completed_updates(b["stop"], b["step"])
        train = b["phases"].get(("deployment", "train"), 0.)
        if updates is not None and updates > 0 and train > 0:
            units.append(train/updates)
    budget = p.get("budget_gpu_seconds") or 0.
    if not units or not finite_cost(budget) or not budget:
        return float("inf")
    return 1.1*budget/statistics.median(units)


def report(root, *, draws=10000):
    manifest = read(root / "switch.json")
    p = manifest if isinstance(manifest, dict) else {}
    mbpp_named = root.name.startswith("selection-switch-mbpp-")
    is_mbpp = p.get("dataset") == "mbpp" or (mbpp_named and p.get("dataset") is None)
    experiment = mbpp_suite_label(root, p) if is_mbpp else (root.name if mbpp_named else suite_label(root))
    lines = [f"SELECTION SWITCH RESULTS  {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
             f"ROOT {root}", f"EXPERIMENT {experiment}",
             f"SELECTOR {selector_label(p.get('selector', '?'))}  "
             f"ACCOUNTING {accounting_label(p.get('accounting', '?'))}  "
             f"GATE {gate_label(p.get('gate', '?'))}",
             f"BUDGET {p.get('budget_gpu_seconds', '?')}  DATASET {p.get('dataset', '?')}"]
    if not isinstance(manifest, dict):
        lines.append("MANIFEST missing, unreadable or not an object; protocol values are unknown")
    elif mbpp_named and p.get("dataset") not in (None, "mbpp"):
        lines.append("WARNING MBPP-named root has a different frozen dataset; displayed values are from the manifest")
    for child in sorted(root.glob('states/s*-t*')):
        match = re.fullmatch(r's\d+-t(\d+)', child.name)
        if not match:
            lines.append(f"UNVERIFIED STATE skipped malformed directory name: {child.name[:160]}")
        else:
            _point, error = state_point(child, int(match[1]))
            if error:
                lines.append(f"UNVERIFIED STATE {child.name[:160]}: {error[:512]}")
    if p.get("accounting") == "matched":
        lines.append("MATCHED: training allocation matched, total compute not matched; selection scoring is separately charged, not free")
    lines.append("COST SOURCE switch.json budget_gpu_seconds=" + str(p.get("budget_gpu_seconds", "unknown"))
                 + " budget_source=" + json.dumps(p.get("budget_source"), sort_keys=True))
    try:
        lines[1] += "  COMMIT " + subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                                          cwd=Path(__file__).resolve().parents[1]).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    model = object_at(root / "model.json")
    nested = model.get("model")
    ridge = nested.get("ridge") if isinstance(nested, dict) else model.get("ridge")
    if (isinstance(ridge, dict) and isinstance(ridge.get("intercept"), (int, float))
            and math.isfinite(ridge["intercept"]) and isinstance(ridge.get("coef"), list)
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in ridge["coef"])):
        lines.append(f"GATE MODEL intercept={ridge['intercept']:+.5f} coef={[round(c, 5) for c in ridge['coef']]} "
                     f"features={model.get('features', nested.get('features') if isinstance(nested, dict) else None)}")
    else:
        lines.append("GATE MODEL not fitted")
    rows = branches(root)
    limit = update_limit(p, rows)
    lines += ["", f"BRANCHES  (updates = completed_steps - state step; used = deployment GPU-s; flags: INVALID = "
                  f"more than {limit:.0f} updates, which one allocation cannot buy; RESET = rerun from the parent "
                  "policy after a reset receipt, valid on its own ledger; RERUN = waiver without a result)"]
    lines += ["COST totals use finished events, including failed attempts, in each branch's cost.jsonl + curve/cost.jsonl; reporting is included.",
              "Open/missing/invalid ledger totals are unknown, not zero; phase values are finished-event subtotals only.",
              "diagnostic_charge and branch_allocation come from decision.json; diagnostic is shared, so do not sum action totals across arms.",
              "Shared point/curve-parent/cost.jsonl is excluded from branch totals: count it once per point, not once per arm.",
              "These are branch-local recorded costs, not cluster/job billing; shared prefixes and archived/waived costs are not included.",
              "Endpoint schema, complete flag, rewards and byte seal checked; checkpoint lineage and ledger provenance not certified.",
              "Unverified saved endpoint JSON is retained as RAW_ENDPOINT, excluded from rewards and contrasts."]
    by_state = defaultdict(dict)
    for b in rows:
        result, stop = b["result"], b["stop"]
        updates = completed_updates(stop, b["step"])
        flags = []
        if updates is not None and updates > limit:
            flags.append("INVALID")
        if b["discards"]:
            # The reset moved the over-trained attempt's ledger and outputs aside; this
            # result is the rerun's own, from the parent policy within one allocation.
            flags.append("RESET")
        if b["waivers"] and not result:
            flags.append("RERUN")
        if b["failure"]:
            flags.append("FAILED:" + str(b["failure"].get("error", ""))[:60].replace("\n", " "))
        dep = {k[1]: round(v) for k, v in b["phases"].items() if k[0] == "deployment"}
        rep = {k[1]: round(v) for k, v in b["phases"].items() if k[0] == "reporting"}
        scoring = {k[1]: round(v) for k, v in b["phases"].items()
                   if k[0] == "reporting" and k[1] != "evaluate" and k[1] != "curve"}
        mean = 100*statistics.fmean(result["rewards"].values()) if result else None
        used = result.get("used_gpu_seconds") if result else None
        used = used if finite_cost(used) else None
        lines.append(f"{b['state']:8s} {b['arm']:18s} reward={mean:6.2f} " if mean is not None else
                     f"{b['state']:8s} {b['arm']:18s} reward=  none  ")
        lines[-1] += (f"updates={updates if updates is not None else '?':>4} used={round(used) if used else '?':>6} "
                      f"action={b['execution'].get('action', b['decision'].get('action', '?'))} "
                      f"pred={b['decision'].get('prediction')} deployment={dep} reporting={rep}"
                      + (f" scoring={scoring}" if scoring else "") + f" {' '.join(flags)}")
        lines.append(cost_detail(b))
        lines.append(f"    COST PATH {b['directory']} (cost.jsonl; curve/cost.jsonl; decision.json)")
        lines.append(f"    ENDPOINT {b['validation']}")
        if result is None and b['validation'] != 'missing':
            lines.append("    " + raw_evidence("RAW_ENDPOINT", b['directory'] / 'result.json', b['raw_result']))
        if b["curve"]:
            points = saved_curve(b['curve'], b['directory']) if result else None
            if points:
                pts = ", ".join(f"{v['updates']}:{100*v['reward']:.2f}" for v in points)
                lines.append(f"    curve k={b['curve'].get('k')} points(updates:reward) {pts}")
            else:
                lines.append("    " + raw_evidence("RAW_CURVE", b['directory'] / 'curve.json', b['curve']))
        if result and "INVALID" not in flags:
            by_state[b["state"]][b["arm"]] = result["rewards"]
    lines += ["", f"CONTRASTS  (percentage points, paired question bootstrap {draws} draws, 95% interval; "
                  "INVALID branches excluded)"]
    for state in sorted(by_state, key=lambda s: (int(s[1:s.index('/')]), int(s[s.index('/t')+2:]))):
        arms = by_state[state]
        parts = []
        for a, b in CONTRASTS:
            if a in arms and b in arms:
                parts.append(f"{a}-{b}={fmt(paired(arms[a], arms[b], draws))}")
        if parts:
            lines.append(f"{state:8s} " + "  ".join(parts))
    for name in ("development-report.json", "test-report.json"):
        rep = read(root / name)
        if isinstance(rep, dict):
            if (not isinstance(rep.get('rows', []), list) or not isinstance(rep.get('missing_or_failed', []), list)
                    or any(not isinstance(row, dict) or not {'seed', 'step', 'means'} <= row.keys()
                           or not isinstance(row['means'], dict)
                           or any(not finite_cost(v) or v > 1 for v in row['means'].values())
                           or not isinstance(row.get('audit', {}), dict)
                           or not isinstance(row.get('net_update_gain', {}), dict) for row in rep.get('rows', []))
                    or any(not isinstance(row, dict) for row in rep.get('missing_or_failed', []))):
                lines.append(raw_evidence(f"RAW_REPORT {name}", root / name, rep))
                continue
            lines += ["", f"{name}: complete={rep.get('complete')} rows={len(rep.get('rows', []))} "
                          f"missing={[(m.get('seed'), m.get('step')) for m in rep.get('missing_or_failed', [])]}"]
            if "summary" in rep:
                lines.append("summary " + json.dumps(rep["summary"], sort_keys=True))
            for row in rep.get("rows", []):
                gain = row.get("net_update_gain")
                lines.append(f"  s{row['seed']}/t{row['step']} means={{{', '.join(f'{k}:{100*v:.2f}' for k, v in row['means'].items())}}}"
                             + (f" audit={json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row['audit'].items()})}" if row.get('audit') else "")
                             + (f" net_update_gain={json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in gain.items()})}" if gain else ""))
    lines += ["", "REWARDS  (per question, in question order, percent of 8 responses correct; INVALID branches included and flagged)"]
    for b in rows:
        if b["result"]:
            keys = sorted(b["result"]["rewards"], key=int)
            values = " ".join(f"{100*b['result']['rewards'][k]:.1f}" for k in keys)
            updates = completed_updates(b['stop'], b['step'])
            flag = " INVALID" if (b["discards"] or (updates is not None and updates > limit)) else ""
            lines.append(f"{b['state']} {b['arm']}{flag}: {values}")
    return "\n".join(lines) + "\n"


def metadata_id(value):
    if value is None:
        return None
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def object_at(path):
    value = read(path)
    return value if isinstance(value, dict) else {}


def compact_cost(directory):
    totals = phases(directory)
    coverage = ledger_coverage(directory)
    scoring = sum(v for (_, phase), v in totals.items() if phase in SCORING_PHASES)
    training = sum(v for (_, phase), v in totals.items() if phase == "train")
    evaluation = sum(v for (_, phase), v in totals.items() if phase in {"evaluate", "curve"})
    total = sum(totals.values())
    return {"subtotals": [round(v, 3) for v in (scoring, training, evaluation,
             total-scoring-training-evaluation,
             sum(v for (ledger, _), v in totals.items() if ledger == "deployment"),
             sum(v for (ledger, _), v in totals.items() if ledger == "reporting"))],
            "total": round(total, 3) if coverage == "closed events" else None,
            "coverage": coverage}


def compact_contract(c):
    # Hash declared metadata, never load model/optimizer tensors or certify them.
    def identity(keys):
        return metadata_id({key: c[key] for key in keys}) if all(c.get(k) is not None for k in keys) else None
    prefix = c.get("selected_prefix")
    prefix = prefix if isinstance(prefix, dict) else {}
    scope = c.get("scope")
    return {"source_id": identity(("source_hashes",)), "learner_id": identity(("config",)),
            "evaluation_id": identity(("evaluation", "eval_k", "eval_seed")),
            "scope_id": metadata_id({k: v for k, v in scope.items() if k != "selector"})
                        if isinstance(scope, dict) else None,
            "prefix_id": prefix.get("certificate_sha256"),
            **{key: c.get(key) for key in ("n", "eval_k", "budget_gpu_seconds", "max_steps")}}


def compact_branch(directory, step, arm):
    endpoint, validation, _raw = saved_endpoint(directory)
    result = object_at(directory / "result.json")
    stop = object_at(directory / "policy/budget_stop.json")
    decision = object_at(directory / "decision.json")
    curve = object_at(directory / "curve.json")
    result_hash = None
    if result:
        result_hash = hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()
    seal = object_at(directory / "result.sha256.json")
    status = ("missing" if not result else "unsealed" if not seal else
              "sealed-metadata" if seal == {"sha256": result_hash} else "seal-mismatch")
    if validation != "missing" and endpoint is None:
        status = "unverified: " + validation
    rewards = result.get("rewards")
    reward = statistics.fmean(endpoint['rewards'].values()) if endpoint else None
    completed = result.get("completed_steps", stop.get("completed_steps"))
    updates = completed-step if type(completed) is int and completed >= step else None
    points = curve.get("points")
    curve_rows = None
    if endpoint and saved_curve(curve, directory):
        curve_rows = [[v["updates"], v["reward"], v.get("k", curve.get("k"))]
                      for v in sorted(points.values(), key=lambda v: v["updates"])]
    def number(key):
        return decision.get(key) if finite_cost(decision.get(key)) else None
    return {"arm": arm, "result": status,
            "complete_claim": result.get("complete") if type(result.get("complete")) is bool else None,
            "reported_reward": reward, "questions": len(rewards) if isinstance(rewards, dict) else None,
            "question_ids": metadata_id(sorted(rewards)) if isinstance(rewards, dict) and rewards else None,
            "updates": updates, "stop": stop.get("stop_reason"),
            "step_match": result.get("completed_steps") == stop.get("completed_steps") if result and stop else None,
            "cap": number("budget_gpu_seconds"), "diagnosis": number("measurement_gpu_seconds"),
            "cost": compact_cost(directory), "curve": curve_rows,
            "curve_bound": curve.get("result_sha256") == result_hash if curve and result_hash else None,
            "history": {name: len(list((directory / name).glob("*")))
                        for name in ("waivers", "discards", "discarded")},
            "failure": str(object_at(directory / "failure.json").get("error", ""))[:160] or None}


def compact_report(root):
    p = object_at(root / "switch.json")
    if not p:
        raise ValueError("readable switch.json required; no experiment was initialized")
    rows = [{"format": "switch-comparison-compact/v1", "root": str(root),
             "manifest_id": metadata_id(p),
             **{key: p.get(key) for key in ("dataset", "selector", "accounting", "gate", "budget_gpu_seconds")},
             "notes": ["Read-only metadata snapshot, NOT policy/optimizer/lineage validation or an adaptive run.",
                       "All 48 planned arms included. Missing rewards/cost totals are null, never zero.",
                       "Reward unit: fraction. Curve rows: [updates,reward,k]. No interpolation.",
                       "Cost unit: GPU-seconds. subtotals: [scoring,training,evaluation,other,deployment,reporting].",
                       "Subtotals are readable finished events, including failures; only closed ledgers have a total.",
                       "Branch costs include own curve ledger; diagnosis and shared parent evaluation are separate.",
                       "Prefix/history costs excluded. No per-question rewards, bootstrap CI or checkpoint-time cost receipts.",
                       "IDs hash declared metadata; equal IDs are not on-disk artifact certification."]}]
    for seed in (*rule.DEV_SEEDS, *rule.TEST_SEEDS):
        for step in rule.STEPS:
            child = root / "states" / f"s{seed}-t{step}"
            point, point_error = state_point(child, step)
            arms = rule.DEV_ARMS if seed in rule.DEV_SEEDS else rule.TEST_ARMS
            for point in [point]:
                rows.append({"state": f"s{seed}-t{step}", "point": point.name,
                             "ambiguous_point": bool(point_error), "point_error": point_error or None,
                             "contract": compact_contract(object_at(point / "contract.json")),
                             "parent_evaluation": compact_cost(point / "curve-parent")})
                for arm in arms:
                    row = compact_branch(point / arm, step, arm)
                    if point_error:
                        row.update(result='unverified: state point unresolved', reported_reward=None, curve=None)
                    rows.append(row)
    text = "\n".join(json.dumps(row, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
                     for row in rows) + "\n"
    size = len(text.encode())
    if size > COMPACT_LIMIT:
        raise ValueError(f"compact report is {size} bytes, above {COMPACT_LIMIT}; no rows were silently dropped; use full results")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compact", action="store_true", help="comparison metadata only, <=64 KiB; no per-question arrays")
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        if args.compact and args.out and args.out.resolve().is_relative_to(root):
            raise ValueError("compact output must be outside the experiment root")
        text = compact_report(root) if args.compact else report(root, draws=args.draws)
    except Exception as exc:  # noqa: BLE001 - name the failure instead of a bare traceback
        import traceback
        traceback.print_exc()
        print(f"[results failed] {args.root}: {type(exc).__name__}: {exc}", flush=True)
        return 1
    if args.out:
        if args.compact:
            with args.out.open("x") as handle:
                handle.write(text)
        else:
            args.out.write_text(text)
        print(f"[saved] {args.out} ({len(text.encode())//1024} KB)")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
