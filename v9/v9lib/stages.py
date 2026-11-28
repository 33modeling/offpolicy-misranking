"""Stage planners: every stage returns items (done / ready task / blocked with a reason).

Planning reads V8 outputs (read-only) and V9 outputs; it never trains. GPU tasks are V8-format
task dicts executed by v9/child.py inside torchrun; their outputs go under $V9_ROOT/<cohort>/.
"""

from __future__ import annotations

from . import env
from .common import (Blocked, bank, cfg, ensure_locator, eval_item, eval_records, eval_task, frozen_tau, is_complete,
                     item, read_json, run_state, v8c, v9c)


# ================================================================ X1 (Staged on six cohorts)
def x1_items():
    out = []
    try:
        tau = frozen_tau()
    except Blocked as exc:
        return [item("x1", c, "all", "blocked", str(exc)) for c in env.X1_COHORTS]
    for cohort in env.BANK_COHORTS:
        out += _x1_bank(cohort, tau)
    for cohort in env.LLAMA_COHORTS:
        out += _x1_llama(cohort, tau)
    return out


ARMS = {"staged": None, "sr": "fixed_sr_t025", "alignment": "on_continued"}
# Random (uniform 4 of 40 fresh candidates every update) from the 25-update prefix to 275. V8 runs Random only on
# the test seeds 10-14 (X4), so for seeds 5-9 it is new V9 training in every X1 cohort. Its tasks are `late`:
# a worker takes them only when no other V9 stage has a ready task (they feed the appendix column only).
RANDOM_METHOD = "uniform_early"


def x1_arm_policy(arm, tau):
    return f"fixed_sr_t{tau:03d}" if arm == "staged" else ARMS[arm]


def _x1_bank(cohort, tau):
    """Read-only: Staged = V8 bank_sr_t{tau}, SR = bank_sr_t025, Alignment = the On trajectory, at 275."""
    out = []
    if not (v8c(cohort) / "locator.json").is_file():
        return [item("x1", cohort, "all", "blocked", "V8 queue has not located this cohort (no locator.json)")]
    c = cfg(cohort)
    evals = {k: v[0] for k, v in eval_records(v8c(cohort)).items()}
    out += _x1_random_bank(cohort, c)
    for seed in c["seeds"]["training"]:
        try:
            rows = {r["policy_id"]: r for r in bank(cohort, seed, evals=evals).policy_rows()}
        except Exception as exc:  # noqa: BLE001 - the V8 bank of this seed cannot be read: say why
            out.append(item("x1", cohort, f"s{seed}", "blocked", f"V8 bank unreadable: {type(exc).__name__}: {exc}"))
            continue
        for arm in ARMS:
            pid = x1_arm_policy(arm, tau)
            r = rows.get(pid)
            key = f"s{seed}.{arm}"
            if r is None or r.get("status") == "missing_trajectory":
                why = ("waiting for V8 trajectory" if "on_continued" not in rows or
                       rows["on_continued"].get("status") == "missing_trajectory"
                       else f"waiting for V8 branch {pid.replace('fixed_sr', 'bank_sr')} (training)")
                out.append(item("x1", cohort, key, "blocked", why, seed=seed, arm=arm))
            elif r.get("status") == "unavailable":
                out.append(item("x1", cohort, key, "blocked",
                                f"waiting for V8 branch {pid.replace('fixed_sr', 'bank_sr')} (training)", seed=seed, arm=arm))
            elif r.get("reward_fraction") is None:
                out.append(item("x1", cohort, key, "blocked", "waiting for V8 eval-final", seed=seed, arm=arm))
            else:
                out.append(item("x1", cohort, key, "done", value=r["reward_fraction"], seed=seed, arm=arm,
                                run_id=r["run_id"], cost=r.get("continuation_gpu_seconds")))
    return out


def _x1_random_bank(cohort, c):
    """Random arm of a bank cohort: V9 training from the V8 prefix of each seed (V9 locator copy), then eval-final."""
    from switching import matrix as mx
    from switching.cluster import ClusterLocator
    try:
        loc = ensure_locator(cohort)
    except Blocked as exc:
        return [item("x1", cohort, "random", "blocked", str(exc), arm="random", late=True)]
    root = v9c(cohort)
    H = c["training"]["matched_update_endpoint"]
    cl = ClusterLocator(root, c)
    out = []
    for seed in c["seeds"]["training"]:
        key = f"s{seed}.random"
        try:
            prefix = cl.prefix(seed)
        except Exception:  # noqa: BLE001 - no prefix recorded for this seed yet
            out.append(item("x1", cohort, key + ".train", "blocked", "waiting for the V8 prefix of this seed",
                            seed=seed, arm="random", late=True))
            continue
        task = mx.train_task(c, role="v9_x1", seed=seed, method_id=RANDOM_METHOD, parent=prefix, horizon=H,
                             stage="x1-random", interval=cl.interval)
        out += _train_then_eval("x1", cohort, root, task, H, key, seed=seed, arm="random", loc=loc, late=True)
    return out


def _x1_llama(cohort, tau):
    """New training: On trajectory from the 25-update prefix to 275 (snapshots every 25), SR branch at tau,
    SR from the prefix; final evaluations at 275."""
    from switching import matrix as mx
    from switching.cluster import ClusterLocator
    from . import llama
    try:
        loc = llama.locate(cohort)
    except Blocked as exc:
        return [item("x1", cohort, "all", "blocked", str(exc))]
    c = cfg(cohort)
    root = v9c(cohort)
    H = c["training"]["matched_update_endpoint"]
    out = []
    cl = ClusterLocator(root, c)
    for seed in c["seeds"]["training"]:
        try:
            prefix = cl.prefix(seed)
        except Exception as exc:  # noqa: BLE001 - one seed without a prefix must not stop the whole X1 stage
            out += [item("x1", cohort, f"s{seed}.{arm}", "blocked", f"no Llama prefix for this seed: {exc}", seed=seed,
                         arm=arm, **({"late": True} if arm == "random" else {}))
                    for arm in ("alignment", "sr", "random", "staged")]
            continue
        traj = mx.train_task(c, role="v9_x1", seed=seed, method_id="on_continued", parent=prefix, horizon=H,
                             stage="x1-llama")
        runs = {"alignment": traj}
        snap = mx.parent_from_snapshot(root, traj["run_id"], tau)
        runs["sr"] = mx.train_task(c, role="v9_x1", seed=seed, method_id="bank_sr_t025", parent=prefix, horizon=H,
                                   stage="x1-llama")
        runs["random"] = mx.train_task(c, role="v9_x1", seed=seed, method_id=RANDOM_METHOD, parent=prefix, horizon=H,
                                       stage="x1-llama")
        if snap is not None:
            runs["staged"] = mx.train_task(c, role="v9_x1", seed=seed, method_id=f"bank_sr_t{tau:03d}", parent=snap,
                                           horizon=H, stage="x1-llama", deps=[traj["run_id"]])
        else:
            out.append(item("x1", cohort, f"s{seed}.staged", "blocked",
                            f"waiting for the V9 Llama trajectory snapshot at {tau}", seed=seed, arm="staged"))
        for arm, task in runs.items():
            out += _train_then_eval("x1", cohort, root, task, H, f"s{seed}.{arm}", seed=seed, arm=arm, loc=loc,
                                    **({"late": True} if arm == "random" else {}))
    return out


def _train_then_eval(stage, cohort, root, task, update, key, *, loc=None, **extra):
    if not is_complete(root, task["run_id"]):
        st = run_state(root, task["run_id"])
        if st and st.startswith("failed"):
            return [item(stage, cohort, key + ".train", "ready", f"previous attempt {st}; resumes", task=task, **extra)]
        return [item(stage, cohort, key + ".train", "ready", task=task, **extra)]
    ev = eval_item(stage, cohort, root, task["run_id"], update, key + ".eval", loc=loc)
    ev.update(extra, run_id=task["run_id"])
    return [item(stage, cohort, key + ".train", "done", run_id=task["run_id"], **extra), ev]


# ================================================================ X2 (equal compute at B = Staged cost)
BUDGET_RULE = ("V9: mean continuation algorithm cost (GPU-seconds) of Staged = fixed_sr_t{tau} (trajectory to tau + "
               "SR branch to 275, SR-GC monitor cost excluded) over seeds 5-9; cost-only, evaluation files not read")


def x2_budget(cohort, tau):
    from switching.freeze import freeze_budget, write_frozen_budget
    from switching.common import utc_now
    c = cfg(cohort)
    path = v9c(cohort) / "frozen" / "budget.json"
    if path.is_file():
        return read_json(path)
    rows = []
    for seed in c["seeds"]["training"]:
        rows += bank(cohort, seed, evals={}).policy_rows()
    try:
        b = freeze_budget(rows, policy_id=f"fixed_sr_t{tau:03d}", scope="continuation_algorithm",
                          files_read=[str(v8c(cohort) / "runs")], expected_seeds=c["seeds"]["training"])
    except ValueError as exc:
        raise Blocked(f"budget not freezable yet: {exc}") from None
    b.update(rule=BUDGET_RULE, gpu_type=c["runtime"]["gpu_type"], gpu_count=c["runtime"]["world_size"],
             v9_tau=tau, source_root=str(v8c(cohort)),
             note="cost-only; V8 evaluation files of this cohort exist on the volume but were not read",
             frozen_at=utc_now())
    b, _ = write_frozen_budget(path, b)
    return b


def x2_items():
    try:
        tau = frozen_tau()
    except Blocked as exc:
        return [item("x2", c, "all", "blocked", str(exc)) for c in env.X2_COHORTS]
    out = []
    for cohort in env.X2_COHORTS:
        try:
            out += _x2_cohort(cohort, tau)
        except Blocked as exc:
            out.append(item("x2", cohort, "all", "blocked", str(exc)))
    return out


def x2_cheap_taus(cohort, tau):
    c = cfg(cohort)
    grid = sorted(set(c["bank"]["core_grid"]) | set(c["bank"]["extended_grid"]))
    return [t for t in grid if t < tau]  # SR (t025) and any earlier switch are cheaper than Staged(tau)


def _x2_cohort(cohort, tau):
    from switching import matrix as mx
    from switching.budget import select_endpoint
    loc = ensure_locator(cohort)
    B = x2_budget(cohort, tau)["value"]
    c = cfg(cohort)
    H = c["training"]["matched_update_endpoint"]
    root, src = v9c(cohort), v8c(cohort)
    out = [item("x2", cohort, "budget", "done", f"B={B / 3600:.3f} GPU-h ({B:.1f} GPU-s), Staged tau={tau}", value=B)]
    for seed in c["seeds"]["training"]:
        b = bank(cohort, seed)
        # cheap arms extended to B (SR = bank_sr_t025 at least)
        for t in x2_cheap_taus(cohort, tau):
            arm = "sr" if t == c["training"]["prefix_updates"] else f"switch{t:03d}"
            key = f"s{seed}.{arm}@B"
            man = b.branches.get(("sr", t))
            if man is None or man["status"] not in ("completed_horizon", "completed_budget"):
                out.append(item("x2", cohort, key, "blocked", f"waiting for V8 branch bank_sr_t{t:03d}", seed=seed, arm=arm))
                continue
            snap = mx.parent_from_snapshot(src, man["run_id"], H)
            offset = b.composed("fixed_sr", t, man, include_check=False, upto=H)["continuation_algorithm"]
            own = mx.snapshot_index(src, man["run_id"]).get(H, {}).get("own_continuation_gpu_seconds")
            if snap is None or offset is None or own is None:
                out.append(item("x2", cohort, key, "blocked", "V8 branch has no 275 snapshot or an unmeasured cost",
                                seed=seed, arm=arm))
                continue
            task = mx.train_task(c, role="v9_budget", seed=seed, method_id=f"bank_sr_t{t:03d}", parent=snap, horizon=H,
                                 stage="x2-extension", budget_cap=B, cost_offset=offset,
                                 extra={"initial_selector": "sr", "policy": {"policy": "never"},
                                        "extension_of": man["run_id"], "v9_budget_sha": B})
            task["est_update_gpu_seconds"] = own / max(1, H - man["task"]["parent"]["completed_update"])
            if offset + task["est_update_gpu_seconds"] > B:
                # not one more update fits in B: no extension run (queues bound to the pre-V8-161 runner fail a
                # zero-update extension); the endpoint is chosen on the branch itself from cost metadata
                be = b.budget_endpoint("fixed_sr", t, man, B, include_check=False)
                if be["status"] in ("selected", "capped_at_horizon"):
                    ep = be["endpoint"]
                    ev = eval_item("x2", cohort, src, ep["run_id"], ep["completed_update"], key + ".eval", loc=loc)
                    ev.update(seed=seed, arm=arm, endpoint_update=ep["completed_update"], cost=ep["cumulative_cost"],
                              note="no update past 275 fits in B")
                    out.append(ev)
                else:
                    out.append(item("x2", cohort, key, "blocked", f"budget endpoint {be['status']}", seed=seed, arm=arm))
                continue
            if not is_complete(root, task["run_id"]):
                out.append(item("x2", cohort, key + ".train", "ready", task=task, seed=seed, arm=arm))
                continue
            rows = [{"completed_update": H, "cumulative_cost": offset, "run_id": man["run_id"], "root": "v8"}]
            for u, s in mx.snapshot_index(root, task["run_id"]).items():
                if u > H:
                    own_u = s.get("own_continuation_gpu_seconds")
                    rows.append({"completed_update": u, "run_id": task["run_id"], "root": "v9",
                                 "cumulative_cost": None if own_u is None else offset + own_u})
            sel = select_endpoint(rows, B)
            if sel["status"] != "selected":
                out.append(item("x2", cohort, key, "blocked", f"budget endpoint {sel['status']}", seed=seed, arm=arm))
                continue
            ep = sel["endpoint"]
            ev = eval_item("x2", cohort, src if ep["root"] == "v8" else root, ep["run_id"], ep["completed_update"],
                           key + ".eval", loc=loc)
            ev.update(seed=seed, arm=arm, endpoint_update=ep["completed_update"], cost=ep["cumulative_cost"])
            out += [item("x2", cohort, key + ".train", "done", seed=seed, arm=arm), ev]
        # Alignment (On) stopped at B: latest trajectory snapshot within B (metadata only)
        for arm, family, tau_arm, branch in (("alignment", "on_continued", None, None),
                                             ("staged", "fixed_sr", tau, b.branches.get(("sr", tau)))):
            key = f"s{seed}.{arm}@B"
            if b.traj is None or (arm == "staged" and branch is None):
                out.append(item("x2", cohort, key, "blocked", "waiting for V8 trajectory/branch", seed=seed, arm=arm))
                continue
            be = b.budget_endpoint(family, tau_arm, branch, B, include_check=False)
            if be["status"] == "capped_at_horizon":
                ep = be["endpoint"]
                rid = branch["run_id"] if branch is not None else b.traj["run_id"]
                ev = eval_item("x2", cohort, src, rid, H, key + ".eval", loc=loc)
                ev.update(seed=seed, arm=arm, endpoint_update=H, cost=ep["cumulative_cost"],
                          note="ran to 275 below B (endpoint at the horizon)")
                out.append(ev)
            elif be["status"] == "selected":
                ep = be["endpoint"]
                ev = eval_item("x2", cohort, src, ep["run_id"], ep["completed_update"], key + ".eval", loc=loc)
                ev.update(seed=seed, arm=arm, endpoint_update=ep["completed_update"], cost=ep["cumulative_cost"])
                out.append(ev)
            else:
                out.append(item("x2", cohort, key, "blocked", f"budget endpoint {be['status']}", seed=seed, arm=arm))
    return out


# ================================================================ X3 (horizon 400)
def x3_items():
    try:
        tau = frozen_tau()
    except Blocked as exc:
        return [item("x3", c, "all", "blocked", str(exc)) for c in env.X3_COHORTS]
    out = []
    for cohort in env.X3_COHORTS:
        try:
            out += _x3_cohort(cohort, tau)
        except Blocked as exc:
            out.append(item("x3", cohort, "all", "blocked", str(exc)))
    return out


def _x3_cohort(cohort, tau):
    from switching import matrix as mx
    loc = ensure_locator(cohort)
    c = cfg(cohort)
    H, T = c["training"]["matched_update_endpoint"], env.HORIZON_X3
    root, src = v9c(cohort), v8c(cohort)
    out = []
    for seed in c["seeds"]["training"]:
        b = bank(cohort, seed)
        sources = {"alignment": (b.traj, "on_continued"),
                   "sr": (b.branches.get(("sr", c["training"]["prefix_updates"])), "bank_sr_t025"),
                   "staged": (b.branches.get(("sr", tau)), f"bank_sr_t{tau:03d}")}
        for arm, (man, mid) in sources.items():
            key = f"s{seed}.{arm}"
            if man is None or man["status"] not in ("completed_horizon", "completed_budget"):
                out.append(item("x3", cohort, key, "blocked", f"waiting for V8 {'trajectory' if arm == 'alignment' else mid} "
                                "to reach 275", seed=seed, arm=arm))
                continue
            snap = mx.parent_from_snapshot(src, man["run_id"], H)
            if snap is None:
                out.append(item("x3", cohort, key, "blocked", "V8 run has no retained 275 snapshot", seed=seed, arm=arm))
                continue
            task = mx.train_task(c, role="v9_horizon", seed=seed, method_id=mid, parent=snap, horizon=T,
                                 stage="x3-horizon", extra={"extension_of": man["run_id"]})
            out += _train_then_eval("x3", cohort, root, task, T, key, seed=seed, arm=arm, loc=loc)
    return out


# ================================================================ X4 (fresh seeds 10-14)
X4_ARMS = {"random": "uniform_early", "sr": "sr_early", "alignment": "on_continued"}


def x4_prepare(cohort):
    """Fresh-seed bundles and roles in the V9 copy of the locator (V8 prepare-seeds logic, V9 root)."""
    import argparse
    import contextlib
    import io
    import sys
    from switching import cluster
    loc = ensure_locator(cohort)
    roles = loc.get("seed_roles") or {}
    if [int(s) for s in roles.get("test", [])] == list(env.X4_SEEDS) and all(
            str(s) in loc["seeds"] for s in env.X4_SEEDS):
        return loc
    args = argparse.Namespace(v8_root=str(env.v9_root()), cohort=cohort, code_root=str(env.code_root()),
                              source_seed=None)
    with contextlib.redirect_stdout(io.StringIO()):
        cluster.cmd_prepare_seeds(args)
    loc = read_json(v9c(cohort) / "locator.json")
    from .common import allow_fake
    if allow_fake():  # local CPU check: fresh seeds inherit the fake backend of their source seed
        from switching.common import atomic_write_json
        for rec in loc["seeds"].values():
            src = loc["seeds"].get(str(rec.get("runtime_source_seed")))
            if src and src.get("fake") and not rec.get("fake"):
                rec["fake"] = src["fake"]
        atomic_write_json(v9c(cohort) / "locator.json", loc)
    test = [int(s) for s in (loc.get("seed_roles") or {}).get("test", [])]
    if test != list(env.X4_SEEDS):
        raise Blocked(f"seed registry chose test seeds {test}, not {list(env.X4_SEEDS)}")
    _ = sys
    return loc


def x4_items():
    try:
        tau = frozen_tau()
    except Blocked as exc:
        return [item("x4", env.X4_COHORT, "all", "blocked", str(exc))]
    cohort = env.X4_COHORT
    try:
        return _x4(cohort, tau)
    except Blocked as exc:
        return [item("x4", cohort, "all", "blocked", str(exc))]


def _x4(cohort, tau):
    from switching import matrix as mx
    from switching.cluster import ClusterLocator
    loc = x4_prepare(cohort)
    c = cfg(cohort, role="test")
    root = v9c(cohort)
    H = c["training"]["matched_update_endpoint"]
    seeds = list(env.X4_SEEDS)
    cl = ClusterLocator(root, c)
    out = []
    prefixes, _ = mx.build_stage("prefix", c, cl, root, role="test", seeds=seeds)
    arms = dict(X4_ARMS, staged=f"fixed_sr_t{tau:03d}")
    for task in prefixes:
        seed = task["training_seed"]
        if not is_complete(root, task["run_id"]):
            out.append(item("x4", cohort, f"s{seed}.prefix", "ready", task=task, seed=seed))
            out += [item("x4", cohort, f"s{seed}.{a}", "blocked", "waiting for the V9 prefix of this seed", seed=seed,
                         arm=a) for a in arms]
            continue
        out.append(item("x4", cohort, f"s{seed}.prefix", "done", seed=seed))
        tasks, _ = mx.build_stage("prospective", c, cl, root, role="test", seeds=[seed],
                                  prospective_methods=list(arms.values()))
        by = {t["method_id"]: t for t in tasks}
        for arm, mid in arms.items():
            t = by.get(mid) or next(t for t in tasks if mid in t.get("aliases", []))
            out += _train_then_eval("x4", cohort, root, t, H, f"s{seed}.{arm}", seed=seed, arm=arm, loc=loc)
    return out


# ================================================================ X5 (measurement)
def x5_targets():
    """(cohort, seeds): OLMo MATH every bank seed; the other V8 cohorts their first seed (diagnostic)."""
    out = []
    for cohort in env.BANK_COHORTS:
        if not (v8c(cohort) / "locator.json").is_file():
            continue
        seeds = cfg(cohort)["seeds"]["training"]
        out.append((cohort, seeds if cohort == "olmo_math" else seeds[:1]))
    return out


def x5_items():
    from switching import matrix as mx
    from switching.common import sha256_json
    from .measure import MEASURE_SPEC, HALF
    out = []
    targets = x5_targets()
    if not targets:
        return [item("x5", "olmo_math", "all", "blocked", "no V8 cohort located yet")]
    for cohort, seeds in targets:
        try:
            loc = ensure_locator(cohort)
        except Blocked as exc:
            out.append(item("x5", cohort, "all", "blocked", str(exc)))
            continue
        P = cfg(cohort)["training"]["prefix_updates"]
        for seed in seeds:
            try:
                b = bank(cohort, seed)
            except Exception as exc:  # noqa: BLE001
                out.append(item("x5", cohort, f"s{seed}", "blocked", f"V8 bank unreadable: {exc}"))
                continue
            for u in env.X5_UPDATES:
                key = f"s{seed}.u{u:03d}"
                if u == P:
                    rec = loc["seeds"][str(seed)]
                    if not rec.get("prefix_sha256"):
                        out.append(item("x5", cohort, key, "blocked", "no prefix checkpoint", seed=seed, update=u))
                        continue
                    snap = {"path": rec["prefix_path"], "sha256": rec["prefix_sha256"]}
                    source = {"kind": "prefix", "run_id": None}
                else:
                    snap = None
                    if b.traj is not None:
                        s = mx.snapshot_index(v8c(cohort), b.traj["run_id"]).get(u)
                        if s:
                            snap = {"path": str(v8c(cohort) / "runs" / b.traj["run_id"] / s["path"]), "sha256": s["sha256"]}
                    if snap is None:
                        out.append(item("x5", cohort, key, "blocked", f"waiting for V8 trajectory snapshot {u}",
                                        seed=seed, update=u))
                        continue
                    source = {"kind": "on_trajectory", "run_id": b.traj["run_id"]}
                spec = {"spec": MEASURE_SPEC, "half": HALF, "snapshot": snap["sha256"]}
                task = {"kind": "measure", "task_id": f"x5.{cohort}.s{seed}.u{u:03d}.{sha256_json(spec)[:8]}",
                        "cohort_id": cohort, "training_seed": seed, "completed_update": u,
                        "snapshot_path": snap["path"], "snapshot_sha256": snap["sha256"], "source": source,
                        "responses_half": HALF, "validation_responses": 8, "out_dir": f"x5/s{seed}/u{u:03d}",
                        "beta": {"policy": "pretrained base model without the LoRA adapter (theta_0, the SR-cache policy)",
                                 "evidence": "srgc_rebuttal/build_cache.py docstring; srgc_rebuttal/import_pair_inputs.py "
                                             "'initial-policy behavior rollouts'", "log_prob": "peft disable_adapter()"}}
                if (v9c(cohort) / task["out_dir"] / "summary.json").is_file():
                    out.append(item("x5", cohort, key, "done", seed=seed, update=u))
                else:
                    out.append(item("x5", cohort, key, "ready", task=task, seed=seed, update=u))
    return out


# ================================================================ X8 (refresh-frequency control)
DONE_STATUS = ("completed_horizon", "completed_budget")


def x8_items():
    try:
        tau = frozen_tau()
    except Blocked as exc:
        return [item("x8", c, "all", "blocked", str(exc)) for c in env.X8_COHORTS]
    out = []
    for cohort in env.X8_COHORTS:
        try:
            out += _x8_cohort(cohort, tau)
        except Blocked as exc:
            out.append(item("x8", cohort, "all", "blocked", str(exc)))
    return out


def _x8_cohort(cohort, tau):
    """SR-hold and Random-hold from each seed's 25-update prefix; Staged-hold from the V8 On trajectory at tau.
    The candidates of a hold refresh are the draw of the Alignment refresh at the same update (same stream seed)."""
    from switching import matrix as mx
    from switching.cluster import ClusterLocator
    loc = ensure_locator(cohort)
    c = cfg(cohort)
    root, src = v9c(cohort), v8c(cohort)
    H = c["training"]["matched_update_endpoint"]
    cl = ClusterLocator(root, c)
    k = cl.interval
    out = []
    for seed in c["seeds"]["training"]:
        try:
            prefix = cl.prefix(seed)
        except Exception:  # noqa: BLE001 - no prefix recorded for this seed yet
            out += [item("x8", cohort, f"s{seed}.{a}", "blocked", "waiting for the V8 prefix of this seed", seed=seed, arm=a)
                    for a in ("sr_hold", "random_hold")]
            prefix = None
        if prefix is not None:
            for arm, mid in (("sr_hold", f"sr_hold{k:03d}"), ("random_hold", f"uniform_hold{k:03d}")):
                task = mx.train_task(c, role="v9_x8", seed=seed, method_id=mid, parent=prefix, horizon=H,
                                     stage="x8-hold", interval=k)
                out += _train_then_eval("x8", cohort, root, task, H, f"s{seed}.{arm}", seed=seed, arm=arm, loc=loc)
        key = f"s{seed}.staged_hold"
        try:
            b = bank(cohort, seed)
        except Exception as exc:  # noqa: BLE001
            out.append(item("x8", cohort, key, "blocked", f"V8 bank unreadable: {exc}", seed=seed, arm="staged_hold"))
            continue
        snap = None if b.traj is None else mx.parent_from_snapshot(src, b.traj["run_id"], tau)
        if snap is None:
            out.append(item("x8", cohort, key, "blocked", f"waiting for the V8 trajectory snapshot at {tau}",
                            seed=seed, arm="staged_hold"))
            continue
        task = mx.train_task(c, role="v9_x8", seed=seed, method_id=f"bank_srhold_t{tau:03d}", parent=snap, horizon=H,
                             stage="x8-hold", interval=k, deps=[])
        out += _train_then_eval("x8", cohort, root, task, H, key, seed=seed, arm="staged_hold", loc=loc)
    return out


# ================================================================ X9 (online SR baseline)
def x9_items():
    cohort = env.X9_COHORT
    try:
        frozen_tau()
        return _x9_cohort(cohort)
    except Blocked as exc:
        return [item("x9", cohort, "all", "blocked", str(exc))]


def _x9_cohort(cohort):
    """SR with its cache rebuilt from the current policy at every refresh (pool x 8 responses), from the prefix."""
    from switching import matrix as mx
    from switching.cluster import ClusterLocator
    loc = ensure_locator(cohort)
    c = cfg(cohort)
    root = v9c(cohort)
    H = c["training"]["matched_update_endpoint"]
    cl = ClusterLocator(root, c)
    out = []
    for seed in c["seeds"]["training"]:
        try:
            prefix = cl.prefix(seed)
        except Exception:  # noqa: BLE001
            out.append(item("x9", cohort, f"s{seed}.online_sr", "blocked", "waiting for the V8 prefix of this seed",
                            seed=seed, arm="online_sr"))
            continue
        task = mx.train_task(c, role="v9_x9", seed=seed, method_id=f"sr_early_refreshed_k{cl.interval:03d}",
                             parent=prefix, horizon=H, stage="x9-online", interval=cl.interval)
        out += _train_then_eval("x9", cohort, root, task, H, f"s{seed}.online_sr", seed=seed, arm="online_sr", loc=loc)
    return out


# ================================================================ X11 (dynamic-sampling baseline)
def x11_items():
    try:
        frozen_tau()
    except Blocked as exc:
        return [item("x11", c, "all", "blocked", str(exc)) for c in env.X11_COHORTS]
    out = []
    for cohort in env.X11_COHORTS:
        try:
            out += _x11_cohort(cohort)
        except Blocked as exc:
            out.append(item("x11", cohort, "all", "blocked", str(exc)))
    return out


def _x11_cohort(cohort):
    """DAPO-style dynamic sampling from each seed's 25-update prefix to 275: each update keeps the first four of the
    40 drawn candidates whose current-policy rollouts have mixed rewards (V8 method dynsample_early)."""
    from switching import matrix as mx
    from switching.cluster import ClusterLocator
    loc = ensure_locator(cohort)
    c = cfg(cohort)
    root = v9c(cohort)
    H = c["training"]["matched_update_endpoint"]
    cl = ClusterLocator(root, c)
    out = []
    for seed in c["seeds"]["training"]:
        key = f"s{seed}.dynamic_sampling"
        try:
            prefix = cl.prefix(seed)
        except Exception:  # noqa: BLE001
            out.append(item("x11", cohort, key, "blocked", "waiting for the V8 prefix of this seed", seed=seed,
                            arm="dynamic_sampling"))
            continue
        task = mx.train_task(c, role="v9_x11", seed=seed, method_id="dynsample_early", parent=prefix, horizon=H,
                             stage="x11-dynamic", interval=cl.interval)
        out += _train_then_eval("x11", cohort, root, task, H, key, seed=seed, arm="dynamic_sampling", loc=loc)
    return out


# ================================================================ X13 (Random -> SR warm-up control)
def x13_items():
    try:
        tau = frozen_tau()
    except Blocked as exc:
        return [item("x13", c, "all", "blocked", str(exc)) for c in env.X13_COHORTS]
    out = []
    for cohort in env.X13_COHORTS:
        try:
            out += _x13_cohort(cohort, tau)
        except Blocked as exc:
            out.append(item("x13", cohort, "all", "blocked", str(exc)))
    return out


def _x13_cohort(cohort, tau):
    """Random until tau, then SR to 275: an SR branch (bank_sr_t{tau}) from the update-tau snapshot of the X1 Random
    run of the same seed (V9 root). Compared with Staged (Alignment until tau, then SR) of the same seed."""
    from switching import matrix as mx
    from switching.cluster import ClusterLocator
    loc = ensure_locator(cohort)
    c = cfg(cohort)
    root = v9c(cohort)
    H = c["training"]["matched_update_endpoint"]
    cl = ClusterLocator(root, c)
    out = []
    for seed in c["seeds"]["training"]:
        key = f"s{seed}.random_then_sr"
        try:
            prefix = cl.prefix(seed)
        except Exception:  # noqa: BLE001
            out.append(item("x13", cohort, key, "blocked", "waiting for the V8 prefix of this seed", seed=seed,
                            arm="random_then_sr"))
            continue
        rnd = mx.train_task(c, role="v9_x1", seed=seed, method_id=RANDOM_METHOD, parent=prefix, horizon=H,
                            stage="x1-random", interval=cl.interval)
        snap = mx.parent_from_snapshot(root, rnd["run_id"], tau)
        if snap is None:
            out.append(item("x13", cohort, key, "blocked", f"waiting for the X1 Random run to reach update {tau}",
                            seed=seed, arm="random_then_sr"))
            continue
        task = mx.train_task(c, role="v9_x13", seed=seed, method_id=f"bank_sr_t{tau:03d}", parent=snap, horizon=H,
                             stage="x13-warmup", interval=cl.interval)
        out += _train_then_eval("x13", cohort, root, task, H, key, seed=seed, arm="random_then_sr", loc=loc)
    return out


# ================================================================ X14 (SR -> Alignment, order reversed)
def x14_items():
    try:
        tau = frozen_tau()
    except Blocked as exc:
        return [item("x14", c, "all", "blocked", str(exc)) for c in env.X14_COHORTS]
    out = []
    for cohort in env.X14_COHORTS:
        try:
            out += _x14_cohort(cohort, tau)
        except Blocked as exc:
            out.append(item("x14", cohort, "all", "blocked", str(exc)))
    return out


SR_BRANCH = "fixed_sr_t025"  # V8 policy id of the SR arm (branch after update 25, bank_sr_t025)


def _x14_cohort(cohort, tau):
    """SR until tau, then Alignment to 275: on_continued from the update-tau snapshot of the V8 SR branch of the
    same seed (V8 root, read only); the new run is written to the V9 root."""
    from switching import matrix as mx
    from switching.cluster import ClusterLocator
    loc = ensure_locator(cohort)
    c = cfg(cohort)
    root, src = v9c(cohort), v8c(cohort)
    H = c["training"]["matched_update_endpoint"]
    k = ClusterLocator(root, c).interval
    out = []
    for seed in c["seeds"]["training"]:
        key = f"s{seed}.sr_then_alignment"
        try:
            rows = {r["policy_id"]: r for r in bank(cohort, seed, evals={}).policy_rows()}
        except Exception as exc:  # noqa: BLE001
            out.append(item("x14", cohort, key, "blocked", f"V8 bank unreadable: {exc}", seed=seed,
                            arm="sr_then_alignment"))
            continue
        rid = (rows.get(SR_BRANCH) or {}).get("run_id")
        snap = None if not rid else mx.parent_from_snapshot(src, rid, tau)
        if snap is None:
            out.append(item("x14", cohort, key, "blocked", f"waiting for the V8 SR branch snapshot at {tau}",
                            seed=seed, arm="sr_then_alignment"))
            continue
        task = mx.train_task(c, role="v9_x14", seed=seed, method_id="on_continued", parent=snap, horizon=H,
                             stage="x14-order", interval=k)
        out += _train_then_eval("x14", cohort, root, task, H, key, seed=seed, arm="sr_then_alignment", loc=loc)
    return out


# ================================================================ X12 (public benchmarks)
def _x12_snapshot(it):
    """The update-275 snapshot behind an X1/X11 result: from its evaluation item, or (V8 bank arms, read-only)
    the snapshot of the V8 run at the matched endpoint; None while the run has not reached it."""
    if it.get("snapshot"):
        return it["snapshot"]
    if it["state"] != "done" or not it.get("run_id") or str(it["key"]).endswith((".train", ".eval")):
        return None
    cohort = it["cohort"]
    H = cfg(cohort)["training"]["matched_update_endpoint"]
    try:
        t = eval_task(cohort, v8c(cohort), it["run_id"], H)
    except Blocked:
        return None
    return {"root": str(v8c(cohort)), "run_id": it["run_id"], "update": H, "path": t["snapshot_path"],
            "sha256": t["snapshot_sha256"], "training_seed": t["training_seed"]}


def x12_items():
    from . import bench
    from switching.common import sha256_json
    out = []
    for src_stage, arms in env.X12_ARMS.items():
        try:
            items = PLANNERS[src_stage]()
        except Exception as exc:  # noqa: BLE001
            out.append(item("x12", "all", f"{src_stage}.all", "blocked", f"{src_stage} planning error: {exc}"))
            continue
        for it in items:
            k = str(it["key"])
            if it.get("arm") not in arms or it.get("seed") is None:
                continue
            if k.endswith(".train") and it["state"] == "done":
                continue  # its .eval item carries the snapshot
            cohort, seed, arm = it["cohort"], it.get("seed"), it["arm"]
            snap, why = _x12_snapshot(it), f"waiting for the {src_stage} {arm} run to reach 275"
            for suite in bench.suites_for(cohort):
                key = f"s{seed}.{arm}.{suite}"
                if snap is None:
                    out.append(item("x12", cohort, key, "blocked", why, seed=seed, arm=arm, suite=suite))
                    continue
                try:
                    loc = read_json(v9c(cohort) / "locator.json")
                except OSError:
                    loc = ensure_locator(cohort)
                seed_rec = loc["seeds"][str(seed)]
                n = bench.SUITES[suite]["responses"]
                spec = {"spec": bench.BENCH_SPEC, "suite": suite, "responses": n, "seed_purpose": f"x12-{suite}",
                        "source": {k: bench.SUITES[suite][k] for k in ("repo", "revision", "file")}}
                task = {"kind": "bench", "task_id": f"x12.{suite}.{snap['sha256'][:12]}.{snap['run_id'].split('.')[-1][:6]}",
                        "cohort_id": cohort, "run_id": snap["run_id"], "training_seed": snap["training_seed"],
                        "completed_update": snap["update"], "snapshot_path": snap["path"],
                        "snapshot_sha256": snap["sha256"], "suite": suite, "responses": n,
                        "bench_spec_hash": sha256_json(spec), "bench_spec": spec, "source_stage": src_stage, "arm": arm,
                        "bundles": sorted({r["input_bundle"] for r in loc["seeds"].values() if r.get("input_bundle")}),
                        "fake": bool(seed_rec.get("fake"))}
                f = bench.result_path(v9c(cohort), task)
                if f.is_file():
                    out.append(item("x12", cohort, key, "done", value=read_json(f)["reward_fraction"], seed=seed,
                                    arm=arm, suite=suite, result_file=str(f)))
                else:
                    out.append(item("x12", cohort, key, "ready", task=task, seed=seed, arm=arm, suite=suite))
    return out


# ================================================================ X10 (budget sensitivity of the tau rule)
def x10_budgets(cohort):
    """{tag: GPU-seconds} at env.X10_FRACTIONS of the frozen V8 budget (cost-only, frozen before evaluation)."""
    path = v8c(cohort) / "frozen" / "budget.json"
    if not path.is_file():
        raise Blocked("V8 budget of this cohort is not frozen")
    base = read_json(path)["value"]
    return {f"b{round(100 * f):03d}": base * f for f in env.X10_FRACTIONS}, base


def x10_items():
    cohort = env.X10_COHORT
    try:
        frozen_tau()
        return _x10_cohort(cohort)
    except Blocked as exc:
        return [item("x10", cohort, "all", "blocked", str(exc))]


def _x10_cohort(cohort):
    loc = ensure_locator(cohort)
    c = cfg(cohort)
    P = c["training"]["prefix_updates"]
    budgets, _ = x10_budgets(cohort)
    taus = sorted(set(c["bank"]["core_grid"]) | set(c["bank"]["extended_grid"]) | {P})
    out = []
    for tag, B in budgets.items():
        for seed in c["seeds"]["training"]:
            try:
                b = bank(cohort, seed)
            except Exception as exc:  # noqa: BLE001
                out.append(item("x10", cohort, f"{tag}.s{seed}", "blocked", f"V8 bank unreadable: {exc}", seed=seed))
                continue
            arms = [("alignment", "on_continued", None, None)]
            arms += [("sr" if t == P else f"switch{t:03d}", "fixed_sr", t, b.branches.get(("sr", t))) for t in taus]
            for arm, family, t, man in arms:
                extra = dict(seed=seed, arm=arm, budget=tag, tau=t)
                out += _x10_endpoint(cohort, c, b, loc, B, family, t, man, f"{tag}.s{seed}.{arm}", extra)
    return out


def _x10_endpoint(cohort, c, b, loc, B, family, t, man, key, extra):
    """Reward at the last saved state within B. V8 runs (trajectory, branch, V8 budget extension) are read from
    metadata; when they end with budget left (no snapshot past B) and below the update cap, a V9 extension with
    budget cap B continues from their last saved state."""
    import math
    from switching import matrix as mx
    from switching.budget import select_endpoint
    root, src = v9c(cohort), v8c(cohort)
    cap = c["training"]["hard_safety_update_cap"]
    if b.traj is None or (family == "fixed_sr" and (man is None or man["status"] not in DONE_STATUS)):
        what = "trajectory" if b.traj is None else f"branch bank_sr_t{t:03d}"
        return [item("x10", cohort, key, "blocked", f"waiting for V8 {what}", **extra)]
    be = b.budget_endpoint(family, t, man, B, include_check=False)
    last = b.budget_endpoint(family, t, man, math.inf, include_check=False).get("endpoint")
    if be.get("status") == "selected" and (be.get("overshoot") is not None or
                                          be["endpoint"]["completed_update"] >= cap):
        ep = be["endpoint"]
        ev = eval_item("x10", cohort, src, ep["run_id"], ep["completed_update"], key + ".eval", loc=loc)
        ev.update(extra, endpoint_update=ep["completed_update"], cost=ep["cumulative_cost"])
        return [ev]
    if last is None or last.get("cumulative_cost") is None:
        return [item("x10", cohort, key, "blocked", f"budget endpoint {be.get('status')} (unmeasured cost)", **extra)]
    if last["cumulative_cost"] > B:
        return [item("x10", cohort, key, "blocked", f"budget endpoint {be.get('status')}", **extra)]
    # the recorded runs end below B: continue from their last saved state with budget cap B
    u0, rid0 = last["completed_update"], last["run_id"]
    if u0 >= cap:
        ev = eval_item("x10", cohort, src, rid0, u0, key + ".eval", loc=loc)
        ev.update(extra, endpoint_update=u0, cost=last["cumulative_cost"], note="update cap")
        return [ev]
    snap = mx.parent_from_snapshot(src, rid0, u0)
    m0 = mx.manifest(src, rid0)
    own = mx.snapshot_index(src, rid0).get(u0, {}).get("own_continuation_gpu_seconds")
    if snap is None or m0 is None or own is None:
        return [item("x10", cohort, key, "blocked", f"V8 run {rid0} has no retained snapshot or cost at {u0}", **extra)]
    per_update = own / max(1, u0 - m0["task"]["parent"]["completed_update"])
    if last["cumulative_cost"] + per_update > B:  # not one more update fits: the last state is the endpoint
        ev = eval_item("x10", cohort, src, rid0, u0, key + ".eval", loc=loc)
        ev.update(extra, endpoint_update=u0, cost=last["cumulative_cost"], note="no further update fits in B")
        return [ev]
    if family == "on_continued":
        mid, mextra = "on_continued", {"extension_of": rid0, "v9_budget_sha": B}
    else:
        mid, mextra = f"bank_sr_t{t:03d}", {"initial_selector": "sr", "policy": {"policy": "never"},
                                             "extension_of": rid0, "v9_budget_sha": B}
    task = mx.train_task(c, role="v9_x10", seed=extra["seed"], method_id=mid, parent=snap, horizon=cap,
                         stage="x10-extension", budget_cap=B, cost_offset=last["cumulative_cost"], extra=mextra)
    task["est_update_gpu_seconds"] = per_update
    if not is_complete(root, task["run_id"]):
        st = run_state(root, task["run_id"])
        note = f"previous attempt {st}; resumes" if st and st.startswith("failed") else None
        return [item("x10", cohort, key + ".train", "ready", note, task=task, **extra)]
    rows = [dict(last, root="v8")]
    for u, s in mx.snapshot_index(root, task["run_id"]).items():
        if u > u0:
            o = s.get("own_continuation_gpu_seconds")
            rows.append({"completed_update": u, "run_id": task["run_id"], "root": "v9",
                         "cumulative_cost": None if o is None else last["cumulative_cost"] + o})
    sel = select_endpoint(rows, B)
    if sel["status"] != "selected":
        return [item("x10", cohort, key, "blocked", f"budget endpoint {sel['status']}", **extra)]
    ep = sel["endpoint"]
    ev = eval_item("x10", cohort, src if ep["root"] == "v8" else root, ep["run_id"], ep["completed_update"],
                   key + ".eval", loc=loc)
    ev.update(extra, endpoint_update=ep["completed_update"], cost=ep["cumulative_cost"])
    return [item("x10", cohort, key + ".train", "done", **extra), ev]


PLANNERS = {"x11": x11_items, "x12": x12_items, "x1": x1_items, "x2": x2_items, "x3": x3_items, "x4": x4_items, "x5": x5_items, "x8": x8_items,
            "x9": x9_items, "x10": x10_items, "x13": x13_items, "x14": x14_items}
