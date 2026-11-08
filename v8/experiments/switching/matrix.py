"""Experiment matrix: arm x seed tasks with dependencies, dedupe and stable run IDs.

Stages
  bank-trajectory   continued On-policy from the verified prefix, SR-GC D monitored at every
                    check through the last check before the horizon, snapshots every 25 (X1/E1/E4/E5/E7/E9)
  bank-branches     SR from trajectory snapshots at the core grid {50,100,150,200}
  bank-extended     SR from {75,125,175,225,250} (exact lookup for every 25-boundary trigger)
  destination       Uniform (and SR if missing) from the current rule's trigger state tau_s (E3)
  budget-extension  cheap runs continued past the matched horizon inside the frozen budget (E6/E7)
  prospective       full-policy arms from the prefix (E1-E4, E9, E10) on development or test seeds
  eval-final / eval-dev / eval-budget   immutable evaluations of retained snapshots
"""

from __future__ import annotations

from pathlib import Path

from .common import read_json, read_jsonl, sha256_json, short_hash
from .methods import equivalent_key, method, method_hash
from .rules import replay

TRAIN_KEYS = ("run_id", "study_id", "cohort_id", "method_id", "training_seed", "role", "config_hash")


def protocol_core(cfg):
    t = cfg["training"]
    return {"study_id": cfg["study_id"], "cohort_id": cfg["cohort_id"], "model": cfg.get("model"),
            "training": {k: t[k] for k in ("objective", "prefix_updates", "matched_update_endpoint",
                                           "selection_interval", "check_interval", "first_check",
                                           "prompts_per_update", "responses_per_prompt", "scoring_prompts_per_set")}}


def train_task(cfg, *, role, seed, method_id, parent, horizon, stage, deps=(), budget_cap=None, cost_offset=None,
               frozen=None, interval=None, extra=None):
    t = cfg["training"]
    interval = interval or t["selection_interval"]
    m = method(method_id, frozen=frozen, interval=interval, prefix=t["prefix_updates"])
    if extra:
        m = {**m, **extra}
    if cfg["bank"].get("monitor_srgc_on_trajectory") and method_id == "on_monitored":
        m["monitor"] = True
    top = t["hard_safety_update_cap"] if budget_cap else horizon
    retain = sorted(set(range(parent["completed_update"] + cfg["bank"]["retain_every"], top + 1,
                              cfg["bank"]["retain_every"])) | {horizon})
    body = {"protocol": protocol_core(cfg), "method_hash": method_hash(m), "seed": seed, "role": role,
            "parent_sha256": parent.get("sha256"), "parent_update": parent["completed_update"], "horizon": horizon,
            "budget_cap": budget_cap, "hard_cap": t["hard_safety_update_cap"] if budget_cap else None}
    h = sha256_json(body)
    rid = f"{cfg['cohort_id']}.{role}.s{seed}.{method_id}.{h[:10]}"
    return {"kind": "train", "stage": stage, "run_id": rid, "study_id": cfg["study_id"], "cohort_id": cfg["cohort_id"],
            "method_id": method_id, "training_seed": seed, "role": role, "config_hash": h, "method": m,
            "parent": parent, "horizon": horizon, "hard_cap": body["hard_cap"], "budget_cap_gpu_seconds": budget_cap,
            "cost_offset_gpu_seconds": cost_offset, "retain_updates": retain,
            "latest_every": cfg["bank"]["latest_every"], "deps": list(deps),
            "equivalence_key": equivalent_key(m, start_update=parent["completed_update"])}


def snapshot_index(out_root, run_id):
    return {r["completed_update"]: r for r in read_jsonl(Path(out_root) / "runs" / run_id / "snapshots.jsonl")}


def manifest(out_root, run_id):
    p = Path(out_root) / "runs" / run_id / "manifest.json"
    return read_json(p) if p.exists() else None


def is_complete(out_root, run_id):
    m = manifest(out_root, run_id)
    return bool(m) and m["status"] in ("completed_horizon", "completed_budget")


def parent_from_snapshot(out_root, run_id, update):
    snap = snapshot_index(out_root, run_id).get(update)
    if not snap:
        return None
    return {"kind": "snapshot", "run_id": run_id, "completed_update": update,
            "path": str(Path(out_root) / "runs" / run_id / snap["path"]), "sha256": snap["sha256"],
            "state_hash": snap["identity"]["state_hash"]}


def trajectory_task(cfg, locator, seed, role):
    prefix = locator.prefix(seed)
    return train_task(cfg, role=role, seed=seed, method_id="on_monitored", parent=prefix,
                      horizon=cfg["training"]["matched_update_endpoint"], stage="bank-trajectory",
                      interval=locator.interval)


def monitored_observations(out_root, run_id):
    """D at each check from the trajectory; complete through the last check if the run completed.

    In flight (no signals.jsonl yet) only the surviving lineage is read: a check at boundary u belongs to
    the attempt that owns update u (attempt i owns [start_i, start_{i+1})). Different values for one check
    are an error, never resolved by file order.
    """
    rdir = Path(out_root) / "runs" / run_id
    rows = read_jsonl(rdir / "signals.jsonl")
    if not rows:  # in-flight: read attempt logs
        m = manifest(out_root, run_id) or {}
        order = [(a["attempt_id"], int(a["start_update"])) for a in m.get("attempts", [])
                 if a.get("start_update") is not None]

        def owner(u):
            own = None
            for a, start in order:
                if start <= u:
                    own = a
            return own
        for p in sorted((rdir / "attempts").glob("*/updates.jsonl")):
            for u in read_jsonl(p):
                rows.extend(dict(s, _attempt=p.parent.name) for s in u.get("signals") or [])
        if order:
            rows = [r for r in rows if owner(int(r["completed_update"])) == r["_attempt"]]
    obs = {}
    for r in rows:
        if r.get("signal_definition_version") and r["signal_name"] == "srgc_d":
            step = int(r["completed_update"])
            value = r["value"] if r.get("valid", True) else None
            if step in obs and obs[step] != value:
                raise ValueError(f"{run_id}: conflicting D at completed_update {step}: {obs[step]!r} vs {value!r}")
            obs[step] = value
    return sorted(obs.items())


FRESH_ROLES = ("development", "test")


def inspected_seeds():
    """Training seeds whose RLVR switching outcomes were already inspected (seeds.KNOWN, any cohort)."""
    from .seeds import KNOWN
    return sorted({s for k in KNOWN if k["inspected"] for s in k["seeds"]})


def resolve_seeds(cfg, locator, role, seeds):
    """Seeds of a stage. Prospective roles never fall back to the cohort's (inspected) seed list."""
    if role not in FRESH_ROLES:
        return list(seeds or cfg["seeds"]["training"])
    roles = getattr(locator, "role_seeds", lambda r: None)
    if not seeds:
        seeds = roles(role)
        if not seeds:
            raise ValueError(f"role {role}: no seeds; run prepare-seeds (locator seed_roles) or pass --seeds")
    seeds = [int(s) for s in seeds]
    if role == "test":
        reused = sorted(set(seeds) & set(inspected_seeds()))
        if reused:
            raise ValueError(f"role test: seeds {reused} were already inspected and cannot be fresh test seeds")
    other = roles("development" if role == "test" else "test") or []
    overlap = sorted(set(seeds) & {int(s) for s in other})
    if overlap:
        raise ValueError(f"role {role}: seeds {overlap} are assigned to the other prospective role")
    return seeds


def build_stage(stage, cfg, locator, out_root, *, role="rebuttal_bank", seeds=None, frozen=None,
                budget=None, prospective_methods=None):
    t = cfg["training"]
    H = t["matched_update_endpoint"]
    seeds = resolve_seeds(cfg, locator, role, seeds)
    tasks, notes = [], []
    for seed in seeds:
        traj = trajectory_task(cfg, locator, seed, role) if stage != "prefix" else None
        if stage == "prefix":
            P = t["prefix_updates"]
            init = {"kind": "initial", "completed_update": 0, "sha256": None, "path": None, "run_id": None}
            tasks.append(train_task(cfg, role=role, seed=seed, method_id="on_continued", parent=init, horizon=P,
                                    stage=stage, interval=locator.interval))
            continue
        if stage == "bank-trajectory":
            tasks.append(traj)
            continue
        if stage in ("bank-branches", "bank-extended", "destination"):
            grid = cfg["bank"]["core_grid"] if stage == "bank-branches" else cfg["bank"]["extended_grid"]
            if stage == "destination":
                if not is_complete(out_root, traj["run_id"]):
                    notes.append({"seed": seed, "status": "blocked", "reason": "trajectory incomplete"})
                    continue
                r = replay(monitored_observations(out_root, traj["run_id"]), "current",
                           first_check=t["first_check"], interval=t["check_interval"],
                           last_scheduled_check=_last_check(t))
                notes.append({"seed": seed, "current_rule_replay": r})
                if r["status"] != "triggered":
                    continue  # not_triggered: no destination branch (kept as not_triggered in analysis)
                grid = [r["switch_update"]]
            for tau in grid:
                # tau at the prefix boundary branches from the prefix itself (= Early SR / Uniform-25)
                parent = (locator.prefix(seed) if tau == t["prefix_updates"]
                          else parent_from_snapshot(out_root, traj["run_id"], tau))
                kinds = ["sr", "uniform"] if stage == "destination" else ["sr"]
                for kind in kinds:
                    mid = f"bank_{kind}_t{tau:03d}"
                    if parent is None:
                        notes.append({"seed": seed, "method_id": mid, "status": "blocked",
                                      "reason": f"trajectory snapshot {tau} not yet available"})
                        continue
                    tasks.append(train_task(cfg, role=role, seed=seed, method_id=mid, parent=parent, horizon=H,
                                            stage=stage, deps=[traj["run_id"]], interval=locator.interval))
        elif stage == "prospective":
            prefix = locator.prefix(seed)
            for mid in prospective_methods or []:
                extra = None
                if mid == "sampled_time_sr":
                    extra = {"policy": dict(frozen["sampled_schedules"][str(seed)])}
                tasks.append(train_task(cfg, role=role, seed=seed, method_id=mid, parent=prefix, horizon=H,
                                        stage=stage, frozen=frozen, interval=locator.interval, extra=extra))
        elif stage == "budget-extension":
            if budget is None:
                raise ValueError("budget-extension requires a frozen budget (freeze-budget)")
            for run in locator.completed_runs(seed, role):
                if run["method"]["initial_selector"] == "on_policy" and run["method"]["policy"]["policy"] == "never":
                    continue  # the On trajectory is the most expensive arm; never extended
                snap = parent_from_snapshot(out_root, run["run_id"], H)
                offset = locator.cost_offset(run)  # cheapest logical policy using this branch, through H
                own = snapshot_index(out_root, run["run_id"]).get(H, {}).get("own_continuation_gpu_seconds")
                est = None if own is None else own / max(1, H - run["parent"]["completed_update"])
                if snap is None or offset is None or est is None:
                    notes.append({"seed": seed, "run_id": run["run_id"], "status": "blocked",
                                  "reason": "missing 275 snapshot or unknown parent cost"})
                    continue
                mid = run["method_id"]
                extra = {"initial_selector": run["method"]["initial_selector"] if run["method"]["initial_selector"] != "on_policy"
                         else "sr", "policy": {"policy": "never"}, "extension_of": run["run_id"]}
                task = train_task(cfg, role=role, seed=seed, method_id=mid if mid.startswith("bank_") else
                                  "bank_sr_t%03d" % H, parent=snap, horizon=H, stage=stage,
                                  deps=[run["run_id"]], budget_cap=budget, cost_offset=offset,
                                  interval=locator.interval, extra=extra)
                task["est_update_gpu_seconds"] = est
                tasks.append(task)
        else:
            raise ValueError(f"unknown stage {stage}")
    return dedupe(tasks), notes


def _last_check(t):
    H, ci, fc = t["matched_update_endpoint"], t["check_interval"], t["first_check"]
    last = fc
    while last + ci < H:
        last += ci
    return last


def dedupe(tasks):
    seen, out = {}, []
    for task in tasks:
        key = (task["cohort_id"], task["role"], task["training_seed"], task["equivalence_key"],
               task["parent"].get("sha256"), task["horizon"])
        if key in seen:
            seen[key].setdefault("aliases", []).append(task["method_id"])
            continue
        seen[key] = task
        out.append(task)
    return out


def eval_tasks(cfg, data, out_root, run_ids, *, split, updates=None, every=None):
    spec = cfg["evaluation"][split]
    ids = data["evaluation_ids"] if split == "final" else data["dev_evaluation_ids"]
    ids_hash = sha256_json(sorted(ids))
    tasks = []
    for rid in run_ids:
        m = manifest(out_root, rid)
        if not m:
            continue
        for u, snap in sorted(snapshot_index(out_root, rid).items()):
            if updates is not None and u not in updates:
                continue
            if every is not None and u % every:
                continue
            spec_hash = sha256_json({"split": split, "ids": ids_hash, "responses": spec["responses"],
                                     "seed_purpose": spec["seed_purpose"], "verifier": cfg.get("verifier")})
            tasks.append({"kind": "evaluate", "run_id": rid, "training_seed": m["training_seed"],
                          "completed_update": u, "snapshot_path": str(Path(out_root) / "runs" / rid / snap["path"]),
                          "snapshot_sha256": snap["sha256"], "split": split, "question_ids_hash": ids_hash,
                          "responses": spec["responses"], "seed_purpose": spec["seed_purpose"],
                          "evaluation_spec_hash": spec_hash,
                          "task_id": f"eval.{snap['sha256'][:12]}.{spec_hash[:8]}"})
    uniq = {t["task_id"]: t for t in tasks}
    return list(uniq.values())


def write_matrix(path, cfg, stage, role, tasks, notes):
    from .common import atomic_write_json, utc_now
    atomic_write_json(path, {"study_id": cfg["study_id"], "cohort_id": cfg["cohort_id"], "stage": stage,
                             "role": role, "created_at": utc_now(), "n_tasks": len(tasks), "tasks": tasks,
                             "notes": notes, "matrix_hash": short_hash([t.get("run_id") or t.get("task_id") for t in tasks])})
