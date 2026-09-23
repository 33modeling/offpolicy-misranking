#!/usr/bin/env python3
"""New experiment: SR-GC decisions frozen before E5-style training on unseen seeds.

Reads completed matrix points and the subsets prepared under the new root.
Writes only decision files under the new root and one results TXT. No file of
an existing experiment is modified.

  srgc_newseeds.py copy-test --source SRC --dest DEST
  srgc_newseeds.py freeze --run POINT --out EXPERIMENT --decision FILE
  srgc_newseeds.py results --root ROOT [--out TXT]
  srgc_newseeds.py status --root ROOT
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

SCHEMA = "srgc-newseeds/v1"
ARMS = ("fresh_r", "passrate_beta")
TRAINED_ARMS = ("random", "passrate_beta", "fresh_r", "g11")
REL_TOL = ABS_TOL = 1e-9
RULE = {
    "name": "SR-GC", "threshold": 0.0, "nonnegative": "on_policy", "negative": "cached",
    "statistic": "mean over A,B of dot(validation_h, mean(on_h) - mean(cached_h))",
    "on_policy_subset": "subsets/subset-fresh_r.json, the subset this experiment trains",
    "cached_subset": "subsets/subset-passrate_beta.json, the subset this experiment trains",
    "references": "candidate groups [2q:3q] and [3q:4q]; validation groups [2vq:3vq] and [3vq:4vq]",
}
HERE = Path(__file__).resolve()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text())


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def git_head() -> str | None:
    env = {**os.environ, "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory",
           "GIT_CONFIG_VALUE_0": "*"}
    try:
        return subprocess.run(["git", "-C", str(HERE.parents[1]), "rev-parse", "HEAD"], env=env,
                              capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def copy_test(source: Path, dest: Path) -> dict:
    """Copy the frozen E5 question set once; the source file is only read."""
    source, dest = source.resolve(), dest.resolve()
    if dest == source:
        raise ValueError("the new experiment must use its own copy of the question set")
    expected = digest(source)
    if dest.exists():
        if digest(dest) != expected:
            raise ValueError(f"copied question set differs from its source: {dest}")
        return {"source": str(source), "dest": str(dest), "sha256": expected}
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
    shutil.copyfile(source, temporary)
    if digest(temporary) != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError("question set changed while copying")
    os.replace(temporary, dest)
    return {"source": str(source), "dest": str(dest), "sha256": expected}


def subsets(out: Path) -> dict:
    hashes = read(out / "subsets_hashes.json")
    sets = {}
    for arm in ARMS:
        path = out / "subsets" / f"subset-{arm}.json"
        if hashes.get(arm) != digest(path):
            raise ValueError(f"prepared subset changed: {path}")
        value = read(path)
        ids = [int(i) for i in value["selected_idx"]]
        if len(ids) != value["k"] or len(set(ids)) != len(ids):
            raise ValueError(f"malformed prepared subset: {path}")
        sets[arm] = sorted(ids)
    return sets


def training_started(out: Path) -> bool:
    return any((out / arm / name).exists() for arm in TRAINED_ARMS for name in ("policy", "evaluation"))


def contrast(run: Path, sets: dict) -> dict:
    import torch

    micro = torch.load(run / "oracle_micro_groups.pt", map_location="cpu", weights_only=True)
    validation = torch.load(run / "val_groups.pt", map_location="cpu", weights_only=True)
    rows = {int(key): value for key, value in micro.items()}
    n = len(rows)
    if set(rows) != set(range(n)):
        raise ValueError("candidate gradient IDs are not 0..n-1")
    full = torch.stack([rows[i].double() for i in range(n)])
    if full.ndim != 3:
        raise ValueError("candidate gradients must be [groups, dim] per prompt")
    groups, dim = full.shape[1], full.shape[2]
    validation = validation.double()
    if groups < 8 or groups % 4 or validation.ndim != 2 or validation.shape[0] < 8 \
            or validation.shape[0] % 4 or validation.shape[1] != dim:
        raise ValueError("gradients do not have the R/A/B partition")
    if any(i < 0 or i >= n for arm in ARMS for i in sets[arm]):
        raise ValueError("subset index outside the candidate pool")
    if not torch.isfinite(full).all() or not torch.isfinite(validation).all():
        raise ValueError("non-finite projected gradients")
    q, vq = groups // 4, validation.shape[0] // 4
    on, cached = torch.tensor(sets["fresh_r"]), torch.tensor(sets["passrate_beta"])
    halves = []
    for start, vstart in ((2 * q, 2 * vq), (3 * q, 3 * vq)):
        dots = full[:, start:start + q].mean(1) @ validation[vstart:vstart + vq].mean(0)
        halves.append(float(dots[on].mean() - dots[cached].mean()))
    d_a, d_b = halves
    d = d_a / 2 + d_b / 2
    return {"d_a": d_a, "d_b": d_b, "d": d, "selector": "cached" if d < 0 else "on_policy"}


def freeze(run: Path, out: Path, path: Path) -> dict:
    run, out = run.resolve(), out.resolve()
    experiment = read(out / "experiment.json")
    if Path(experiment["source_run"]).resolve() != run:
        raise ValueError("prepared experiment belongs to another source point")
    config = read(run / "run_config.json")
    inputs = {name: digest(run / name) for name in ("oracle_micro_groups.pt", "val_groups.pt",
                                                     "prompts.json", "run_config.json")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sets = subsets(out)
        subset_hashes = {arm: digest(out / "subsets" / f"subset-{arm}.json") for arm in ARMS}
        value = contrast(run, sets)
        if path.exists():
            saved = read(path)
            if saved["inputs"] != inputs or saved["subset_sha256"] != subset_hashes or saved["sets"] != sets:
                raise ValueError(f"frozen SR-GC decision inputs changed: {path}")
            if saved["selector"] != value["selector"] or any(
                    not math.isclose(saved[key], value[key], rel_tol=REL_TOL, abs_tol=ABS_TOL)
                    for key in ("d_a", "d_b", "d")):
                raise ValueError(f"recomputed SR-GC contrast differs from the frozen decision: {path}")
            return saved
        record = {
            "schema": SCHEMA, "rule": RULE, "seed": config["seed"], "drift": config["drift"], **value,
            "sets": sets, "on_cached_overlap": len(set(sets["fresh_r"]) & set(sets["passrate_beta"])),
            "prospective": not training_started(out),
            "source_run": str(run), "experiment": str(out), "inputs": inputs,
            "subset_sha256": subset_hashes, "experiment_sha256": digest(out / "experiment.json"),
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "host": os.uname().nodename,
            "code_sha256": digest(HERE), "git_commit": git_head(),
        }
        atomic_json(path, record)
        return record


def outcome(out: Path):
    path = out / "downstream_results.csv"
    if not path.is_file():
        return None
    rows = {row["selector"]: row for row in csv.DictReader(path.open())}
    on, cached = rows.get("fresh_r"), rows.get("passrate_beta")
    if not on or not cached or "" in (on.get("reward_after", ""), cached.get("reward_after", "")):
        return None

    def number(row, key):
        value = row.get(key, "")
        return None if value in ("", None) else float(value)

    difference = number(cached, "reward_after") - number(on, "reward_after")
    return {"reward": {arm: number(row, "reward_after") for arm, row in rows.items()
                       if row.get("reward_after") not in ("", None)},
            "reward_before": number(on, "reward_before"),
            "cached_minus_on": difference,
            "cached_minus_on_interval": [number(cached, "difference_lower"), number(cached, "difference_upper")],
            "vs_random": {arm: number(row, "difference_vs_random") for arm, row in rows.items()},
            "observed": "cached" if difference > 0 else "on_policy" if difference < 0 else "tie"}


def results(root: Path) -> dict:
    states = []
    for path in sorted((root / "decisions").glob("s*-d*.json"),
                       key=lambda p: (read(p)["drift"], read(p)["seed"])):
        decision = read(path)
        observed = outcome(Path(decision["experiment"]))
        states.append({"state": path.stem, "decision_sha256": digest(path),
                       **{key: decision[key] for key in ("seed", "drift", "d_a", "d_b", "d", "selector",
                                                        "prospective", "frozen_at_utc", "on_cached_overlap")},
                       "outcome": observed,
                       "match": None if observed is None or observed["observed"] == "tie"
                       else observed["observed"] == decision["selector"]})
    measured = [s for s in states if s["outcome"] is not None]
    rules = {"SR-GC": lambda s: s["selector"], "always on-policy": lambda s: "on_policy",
             "always cached SR": lambda s: "cached",
             "stage (step 0 on-policy, later cached SR)": lambda s: "on_policy" if s["drift"] == 0 else "cached"}
    arm = {"on_policy": "fresh_r", "cached": "passrate_beta"}
    summary = []
    for name, rule in rules.items():
        decided = [s for s in measured if s["outcome"]["observed"] != "tie"]
        summary.append({"rule": name,
                        "agreement": sum(rule(s) == s["outcome"]["observed"] for s in decided),
                        "decided_states": len(decided),
                        "mean_selected_reward_pct": (100 * sum(s["outcome"]["reward"][arm[rule(s)]] for s in measured)
                                                     / len(measured)) if measured else None})
    return {"schema": SCHEMA + "/results", "root": str(root), "states": states,
            "frozen": len(states), "measured": len(measured),
            "prospective_frozen": sum(bool(s["prospective"]) for s in states), "summary": summary,
            "scope": "New seeds evaluated after SR-GC decisions were frozen from parent-policy gradients. "
                     "Rewards are 100-update endpoints on the copied E5 question set; intervals are "
                     "paired question bootstraps conditional on the trained policies."}


def table(data: dict) -> str:
    def pct(value):
        return "" if value is None else f"{100 * value:.2f}"
    lines = [f"frozen decisions {data['frozen']} (prospective {data['prospective_frozen']}); "
             f"measured outcomes {data['measured']}", "",
             "state,D_A,D_B,D,decision,prospective,on_reward_pct,cached_reward_pct,cached_minus_on_pp,"
             "interval_pp,observed,match"]
    for s in data["states"]:
        o = s["outcome"] or {}
        reward = o.get("reward", {})
        low, high = o.get("cached_minus_on_interval", [None, None])
        interval = "" if low is None or high is None else f"[{100 * low:+.2f} {100 * high:+.2f}]"
        lines.append(",".join([s["state"], f"{s['d_a']:+.6g}", f"{s['d_b']:+.6g}", f"{s['d']:+.6g}",
                               s["selector"], str(s["prospective"]), pct(reward.get("fresh_r")),
                               pct(reward.get("passrate_beta")),
                               "" if not o else f"{100 * o['cached_minus_on']:+.2f}", interval,
                               o.get("observed", "pending"), "" if s["match"] is None else str(s["match"])]))
    lines += ["", "rule,agreement,decided_states,mean_selected_reward_pct"]
    for row in data["summary"]:
        mean = row["mean_selected_reward_pct"]
        lines.append(f"{row['rule']},{row['agreement']},{row['decided_states']},"
                     f"{'' if mean is None else f'{mean:.2f}'}")
    return "\n".join(lines)


STATE_STEPS = (0, 100, 400)
STATE_SEEDS = (3, 4)
STATUS_ARMS = ("random", "passrate_beta", "fresh_r")
STALE_SECONDS = 1800


def _age(path: Path, now: float) -> float | None:
    try:
        return max(0., now - path.stat().st_mtime)
    except OSError:
        return None


def _minutes(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def _last_step(stats: Path) -> int | None:
    """Largest recorded update in grpo_stats.jsonl (tail only; no locks are taken)."""
    try:
        with stats.open("rb") as handle:
            handle.seek(max(0, stats.stat().st_size - 65536))
            lines = handle.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    steps = []
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and isinstance(value.get("step"), int):
            steps.append(value["step"])
    return max(steps) if steps else None


def _shards(out: Path, arm: str) -> int:
    return sum((out / arm / "evaluation" / f"shard-{i}.done.json").is_file() for i in range(4))


def arm_status(out: Path, arm: str, drift: int, now: float, steps: int = 100) -> dict:
    policy = out / arm / "policy"
    shards = _shards(out, arm)
    trained = (policy / "policy_train.json").is_file()
    progress = [policy / "grpo_stats.jsonl", *sorted((out / "logs").glob(f"*{arm}*.log")),
                *sorted((out / arm / "evaluation").glob("shard-*.jsonl.partial"))]
    ages = [age for age in (_age(path, now) for path in progress) if age is not None]
    age = min(ages) if ages else None
    if shards == 4:
        state, detail = "DONE", "trained and evaluated"
    elif trained:
        state, detail = ("EVAL" if shards else "TRAINED"), f"evaluation shards {shards}/4"
    elif policy.exists():
        step = _last_step(policy / "grpo_stats.jsonl")
        done = None if step is None else min(steps, max(0, step - drift))
        state, detail = "TRAIN", f"updates {'?' if done is None else done}/{steps}"
    else:
        state, detail = "WAIT", "not started"
    stalled = state in ("TRAIN", "EVAL") and age is not None and age > STALE_SECONDS
    return {"arm": arm, "state": state, "detail": detail, "last_update": _minutes(age),
            "stalled": stalled}


def status(root: Path, now: float | None = None) -> dict:
    import time
    now = time.time() if now is None else now
    states, totals = [], {"decisions": 0, "parents": 0, "arms_done": 0, "arms": 0, "results": 0}
    for drift in STATE_STEPS:
        for seed in STATE_SEEDS:
            out = root / f"math500-d{drift}" / f"s{seed}"
            decision_path = root / "decisions" / f"s{seed}-d{drift}.json"
            decision = read(decision_path) if decision_path.is_file() else None
            parent = _shards(out, "before")
            arms = [arm_status(out, arm, drift, now) for arm in STATUS_ARMS]
            ready = (out / "downstream_results.csv").is_file()
            totals["decisions"] += decision is not None
            totals["parents"] += parent == 4
            totals["arms_done"] += sum(arm["state"] == "DONE" for arm in arms)
            totals["arms"] += len(arms)
            totals["results"] += ready
            states.append({"state": f"s{seed}-d{drift}", "prepared": (out / "experiment.json").is_file(),
                           "decision": None if decision is None else {
                               key: decision[key] for key in ("d", "selector", "prospective", "frozen_at_utc")},
                           "parent_shards": parent, "arms": arms, "result_ready": ready})
    complete = totals["arms_done"] == totals["arms"] and totals["parents"] == len(states)
    return {"root": str(root), "states": states, "totals": totals, "complete": complete}


def status_text(data: dict) -> str:
    t = data["totals"]
    lines = [f"[srgc-newseeds] {'COMPLETE' if data['complete'] else 'IN PROGRESS'}  "
             f"decisions {t['decisions']}/6  parent evals {t['parents']}/6  "
             f"arms done {t['arms_done']}/{t['arms']}  result files {t['results']}/6",
             f"root {data['root']}", ""]
    for state in data["states"]:
        decision = state["decision"]
        head = (f"SR-GC D={decision['d']:+.6g} -> {decision['selector']}"
                f"{'' if decision['prospective'] else ' (NOT prospective: frozen after training started)'}"
                if decision else "SR-GC not frozen")
        prepared = "" if state["prepared"] else "  [not prepared]"
        lines.append(f"{state['state']:8s} {head}  parent eval {state['parent_shards']}/4{prepared}")
        for arm in state["arms"]:
            flag = "  <- no update for over 30 min" if arm["stalled"] else ""
            lines.append(f"  {arm['arm']:14s} {arm['state']:7s} {arm['detail']:26s} last update {arm['last_update']}{flag}")
    if not data["complete"]:
        waiting = [f"{s['state']}/{a['arm']}" for s in data["states"] for a in s["arms"] if a["state"] == "WAIT"]
        if waiting:
            lines += ["", "not started: " + ", ".join(waiting)]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("copy-test")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--decision", type=Path, required=True)
    p = sub.add_parser("results")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p = sub.add_parser("status")
    p.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "status":
        print(status_text(status(args.root.resolve())))
        return 0
    if args.command == "copy-test":
        print(json.dumps(copy_test(args.source, args.dest)))
    elif args.command == "freeze":
        value = freeze(args.run, args.out, args.decision)
        print(f"[srgc-newseeds] s{value['seed']}-d{value['drift']}: D={value['d']:+.6g} "
              f"selector={value['selector']} prospective={value['prospective']}")
    else:
        sys.path.insert(0, str(HERE.parent))
        from paper_result_text import write_export
        data = results(args.root.resolve())
        write_export("srgc-newseeds", data, table(data), args.out)
        print(f"[srgc-newseeds] {data['frozen']} decisions, {data['measured']} measured outcomes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
