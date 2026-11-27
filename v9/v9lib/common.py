"""Shared helpers: V8 read-only access, V9 locators, items, evaluation lookup."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from . import env


class Blocked(Exception):
    """A precondition that is not met yet (reason is printed by `status`)."""


def item(stage, cohort, key, state, reason=None, task=None, **extra):
    return {"stage": stage, "cohort": cohort, "key": key, "state": state, "reason": reason, "task": task, **extra}


def v8c(cohort):
    return env.v8_root() / cohort


def v9c(cohort):
    return env.v9_root() / cohort


def cfg(cohort, role="rebuttal_bank"):
    from switching import spec
    return spec.resolve(cohort, role=role)


def read_json(path):
    return json.loads(Path(path).read_text())


def tau_path():
    return env.v9_root() / "frozen" / "tau.json"


def frozen_tau():
    p = tau_path()
    if not p.is_file():
        raise Blocked("tau is not frozen yet (stage freeze-tau runs first)")
    return int(read_json(p)["tau"])


SEED_FIELDS_CHECKED = ("plan_path", "plan_sha256", "input_bundle", "prefix_path", "prefix_sha256",
                       "prefix_completed_updates", "run_implementation_sha256", "runtime_selection",
                       "implementation_matches_code_root")


def ensure_locator(cohort):
    """v9/<cohort>/locator.json: a copy of the V8 locator (V8 is never written).

    The copy is made once. Seeds present in both must keep the same identity; a changed V8 locator
    entry blocks the cohort instead of silently mixing two trainer identities.
    """
    from switching.common import atomic_write_json
    src = v8c(cohort) / "locator.json"
    dst = v9c(cohort) / "locator.json"
    if not src.is_file():
        raise Blocked(f"V8 locator missing ({src}); the V8 queue has not located this cohort")
    v8 = read_json(src)
    if not dst.is_file():
        atomic_write_json(dst, dict(v8, v9_copied_from=str(src)))
        return read_json(dst)
    v9 = read_json(dst)
    added = []
    for seed, rec in v8["seeds"].items():
        mine = v9["seeds"].get(seed)
        if mine is None:
            # a seed the V8 queue located after V9 made its copy: take it over (V9-only seeds such as the X4 test
            # seeds 10-14 stay as they are); without this its tasks would wait for a prefix forever
            v9["seeds"][seed] = rec
            added.append(seed)
            continue
        diff = [k for k in SEED_FIELDS_CHECKED if rec.get(k) != mine.get(k)]
        if diff:
            raise Blocked(f"V8 locator seed {seed} changed after V9 copied it (fields {diff}); "
                          f"move {dst} aside after checking the V8 queue")
    if added:
        v9.setdefault("v9_seeds_added_from_v8", []).extend(sorted(added, key=int))
        atomic_write_json(dst, v9)
    return v9


def eval_records(root):
    """(run_id, completed_update, split) -> (record, path): first committed record (V8 rule)."""
    idx = {}
    d = Path(root) / "evaluations"
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        r = read_json(p)
        key = (r["run_id"], r["completed_update"], r["split"])
        if key not in idx or r["evaluated_at"] < idx[key][0]["evaluated_at"]:
            idx[key] = (r, p)
    return idx


def evals_merged(cohort):
    """V8 evaluations first (they are the recorded outcomes), then V9's own."""
    out = {k: v for k, v in eval_records(v9c(cohort)).items()}
    out.update(eval_records(v8c(cohort)))
    return out


def plain_evals(merged):
    return {k: v[0] for k, v in merged.items()}


def eval_file_for(cohort, task):
    """V9 evaluation file of an evaluate task, or the identical V8 record (same snapshot+spec)."""
    name = f"{task['snapshot_sha256'][:16]}__{task['evaluation_spec_hash'][:12]}.json"
    for root in (v9c(cohort), v8c(cohort)):
        if (root / "evaluations" / name).is_file():
            return root / "evaluations" / name
    return None


@lru_cache(maxsize=None)
def bundle(path):
    from switching.cluster import load_bundle
    return load_bundle(path)


def seed_bundle(cohort, seed, loc=None):
    loc = loc or read_json(v9c(cohort) / "locator.json")
    return bundle(loc["seeds"][str(seed)]["input_bundle"])


def eval_task(cohort, out_root, run_id, update, *, loc=None):
    """Final-split evaluation task of one retained snapshot (spec identical to V8 eval-final)."""
    from switching import matrix as mx
    c = cfg(cohort)
    man = mx.manifest(out_root, run_id)
    if man is None:
        raise Blocked(f"run {run_id} has no manifest under {out_root}")
    data = seed_bundle(cohort, man["training_seed"], loc)
    tasks = mx.eval_tasks(c, data, out_root, [run_id], split="final", updates={update})
    if not tasks:
        raise Blocked(f"run {run_id} has no retained snapshot at update {update}")
    t = tasks[0]
    t["task_id"] = f"{t['task_id']}.{run_id.split('.')[-1][:6]}"  # unique per run when snapshots coincide
    return t


def eval_item(stage, cohort, out_root, run_id, update, key, *, loc=None):
    """Item for 'reward of run at update': done (with value) or a ready evaluate task."""
    try:
        t = eval_task(cohort, out_root, run_id, update, loc=loc)
    except Blocked as exc:
        return item(stage, cohort, key, "blocked", str(exc))
    # where the evaluated snapshot lives (X12 benchmark evaluation reads the same snapshot)
    snap = {"root": str(out_root), "run_id": run_id, "update": update, "path": t["snapshot_path"],
            "sha256": t["snapshot_sha256"], "training_seed": t["training_seed"]}
    f = eval_file_for(cohort, t)
    if f is not None:
        return item(stage, cohort, key, "done", value=read_json(f)["reward_fraction"], eval_file=str(f), snapshot=snap)
    return item(stage, cohort, key, "ready", task=t, snapshot=snap)


def is_complete(root, run_id):
    from switching import matrix as mx
    return mx.is_complete(root, run_id)


def run_state(root, run_id):
    from switching import matrix as mx
    m = mx.manifest(root, run_id)
    return None if m is None else m["status"]


def bank(cohort, seed, *, evals=None):
    from switching.bank import SeedBank
    return SeedBank(v8c(cohort), cfg(cohort), cohort, seed, "rebuttal_bank",
                    evals=evals if evals is not None else {})


def allow_fake():
    return os.environ.get("V9_ALLOW_FAKE") == "1"
