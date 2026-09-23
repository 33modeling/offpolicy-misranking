#!/usr/bin/env python3
"""New experiment: SR-GC decisions frozen before E5-style training on unseen seeds.

Reads completed matrix points and the subsets prepared under the new root.
Writes only decision files under the new root and one results TXT. No file of
an existing experiment is modified.

  srgc_newseeds.py copy-test --source SRC --dest DEST
  srgc_newseeds.py freeze --run POINT --out EXPERIMENT --decision FILE
  srgc_newseeds.py results --root ROOT [--out TXT]
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
    args = parser.parse_args(argv)
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
