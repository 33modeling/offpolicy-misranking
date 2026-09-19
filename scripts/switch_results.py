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
import json
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

CONTRASTS = (("selection_full", "random_full"), ("gated", "random_full"), ("gated", "selection_full"),
             ("selection_reduced", "random_reduced"))


def read(path):
    try:
        return core.read(path)
    except (OSError, ValueError):
        return None


def phases(directory):
    totals = defaultdict(float)
    # The branch ledger, plus the curve/ sub-ledger where archived-checkpoint curve
    # evaluations are metered (the parent point's curve lives in the state's curve-parent/).
    for path in (directory / "cost.jsonl", directory / "curve" / "cost.jsonl"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("state") == "finished":
                totals[(row.get("ledger", "?"), row["phase"])] += row.get("allocated_gpu_seconds", 0.)
    return totals


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
        seed, step = child.name[1:].split("-t")
        for point in sorted(child.glob("points/view-*")):
            for arm in (*rule.TEST_ARMS,):
                directory = point / arm
                if not directory.exists():
                    continue
                result = read(directory / "result.json")
                stop = read(directory / "policy/budget_stop.json")
                decision = read(directory / "decision.json") or {}
                execution = read(directory / "execution.json") or {}
                failure = read(directory / "failure.json")
                out.append({"state": f"s{seed}/t{step}", "seed": int(seed), "step": int(step), "arm": arm,
                            "directory": directory, "result": result, "stop": stop, "decision": decision,
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
        updates = b["stop"]["completed_steps"]-b["step"]
        train = b["phases"].get(("deployment", "train"), 0.)
        if updates > 0 and train > 0:
            units.append(train/updates)
    budget = p.get("budget_gpu_seconds") or 0.
    if not units or not budget:
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
    try:
        lines[1] += "  COMMIT " + subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                                          cwd=Path(__file__).resolve().parents[1]).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    model = read(root / "model.json")
    if model:
        ridge = model["model"]["ridge"] if "model" in model else model["ridge"]
        lines.append(f"GATE MODEL intercept={ridge['intercept']:+.5f} coef={[round(c, 5) for c in ridge['coef']]} "
                     f"features={model.get('features', model.get('model', {}).get('features'))}")
    else:
        lines.append("GATE MODEL not fitted")
    rows = branches(root)
    limit = update_limit(p, rows)
    lines += ["", f"BRANCHES  (updates = completed_steps - state step; used = deployment GPU-s; flags: INVALID = "
                  f"more than {limit:.0f} updates, which one allocation cannot buy; RESET = rerun from the parent "
                  "policy after a reset receipt, valid on its own ledger; RERUN = waiver without a result)"]
    by_state = defaultdict(dict)
    for b in rows:
        result, stop = b["result"], b["stop"]
        updates = stop["completed_steps"]-b["step"] if stop else None
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
        lines.append(f"{b['state']:8s} {b['arm']:18s} reward={mean:6.2f} " if mean is not None else
                     f"{b['state']:8s} {b['arm']:18s} reward=  none  ")
        lines[-1] += (f"updates={updates if updates is not None else '?':>4} used={round(used) if used else '?':>6} "
                      f"action={b['execution'].get('action', b['decision'].get('action', '?'))} "
                      f"pred={b['decision'].get('prediction')} deployment={dep} reporting={rep}"
                      + (f" scoring={scoring}" if scoring else "") + f" {' '.join(flags)}")
        if b["curve"]:
            pts = ", ".join(f"{v['updates']}:{100*v['reward']:.2f}" for v in sorted(b['curve']['points'].values(), key=lambda v: v['updates']))
            lines.append(f"    curve k={b['curve']['k']} points(updates:reward) {pts}")
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
        if rep:
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
            flag = " INVALID" if (b["discards"] or (b["stop"] and b["stop"]["completed_steps"]-b["step"] > limit)) else ""
            lines.append(f"{b['state']} {b['arm']}{flag}: {values}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        text = report(args.root.resolve(), draws=args.draws)
    except Exception as exc:  # noqa: BLE001 - name the failure instead of a bare traceback
        import traceback
        traceback.print_exc()
        print(f"[results failed] {args.root}: {type(exc).__name__}: {exc}", flush=True)
        return 1
    if args.out:
        args.out.write_text(text)
        print(f"[saved] {args.out} ({len(text.encode())//1024} KB)")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
