"""Local CPU smoke: every experiment path end to end on the code repo's own CPU backends.

toy  : ToyBackend (400 candidates, 40/4/8, prefix 25, horizon 275) -- prefixes, original
       code-repo arms (for E0 audit + replay censoring), Branch Bank (trajectory with D
       monitor, core+extended branches, E3 destination, budget extension), budget freeze,
       development matrix + tuning + tau distribution + frozen study, test matrix (11 arms
       incl. E9 sampled time and E10 refreshed SR), E8 external switch_repeat, interruption
       and resume, evaluations, analysis, report.
tiny : TorchBackend on a random 4-layer OLMo-3 + LoRA (real GRPO/AdamW/autograd scoring),
       prefix 5, horizon 30, cadence 5: branching state identity, sibling identity, ledger.
Outputs are software checks only, never paper results.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

from . import analyze, backends, coderepo, matrix as mx, report, snapshots, spec
from .bank import SeedBank, eval_index, prospective_rows, sample_schedules, tau_distribution
from .cache import CacheStore
from .common import atomic_write_json, read_json, sha256_file, sha256_json, utc_now, write_once_json
from .engine import make_engine_class
from .freeze import freeze_budget, freeze_study, write_frozen_budget
from .ledger import Ledger, component_totals, physical_gpu_seconds
from .runner import Context, Interrupted, execute_eval, execute_train
from .seeds import choose, registry
from .tune import frozen_tuning, select_dev

HERE = Path(__file__).resolve().parents[1]
MANUSCRIPT = HERE.parents[1]


class Locator:
    def __init__(self, out_root, cfg, interval):
        self.out_root, self.cfg, self.interval = Path(out_root), cfg, interval
        self.prefix_costs = {}

    def prefix(self, seed):
        d = self.out_root / "prefixes" / f"seed-{seed}"
        ready = read_json(d / "prefix-ready.json")
        return {"kind": "prefix", "path": str(d / "prefix.pt"), "sha256": ready["checkpoint_sha256"],
                "completed_update": ready["completed_updates"], "run_id": None}

    def completed_runs(self, seed, role):
        out = []
        for m in sorted((self.out_root / "runs").glob("*/manifest.json")):
            man = read_json(m)
            if (man["training_seed"], man["role"], man["cohort_id"]) == (seed, role, self.cfg["cohort_id"]) and \
                    man["status"] in ("completed_horizon",) and man["method_id"].startswith(("bank_sr_", "bank_uniform_")) and \
                    not man["task"]["method"].get("extension_of"):
                out.append({**man["task"], "status": man["status"]})
        return out

    def cost_offset(self, run):
        bank = SeedBank(self.out_root, self.cfg, self.cfg["cohort_id"], run["training_seed"], run["role"])
        tau = run["parent"]["completed_update"]
        dest = "uniform" if run["method_id"].startswith("bank_uniform_") else "sr"
        branch = bank.branches.get((dest, tau))
        if branch is None or branch["run_id"] != run["run_id"]:
            return None  # not a branch of this seed's trajectory: unknown parent cost
        return bank.composed("fixed_uniform" if dest == "uniform" else "fixed_sr", tau, branch, include_check=False,
                             upto=self.cfg["training"]["matched_update_endpoint"])["continuation_algorithm"]


def toy_ctx(out_root, seed, srgc, PE, feat, ans, data, H):
    ctx = Context(srgc=srgc, engine_class=PE, backend=None, data=data, config=srgc.Config(seed=seed, projection_dim=64),
                  out_root=out_root, measurement_status="simulated_cpu", gpu_type="virtual-4gpu-cpu",
                  cache_store=CacheStore(Path(out_root) / "caches"),
                  cache_context={"prompt_hash": "toy", "generation_spec": "toy-categorical", "verifier_hash": "toy"})
    ctx.backend = backends.toy_backend(feat, ans, seed=seed, record=ctx.on_meter_event)
    ctx.meter = ctx.backend.cost_meter
    return ctx


def make_prefix(ctx, seed, prefix_updates, srgc):
    d = ctx.out_root / "prefixes" / f"seed-{seed}"
    if (d / "prefix-ready.json").exists():
        return
    led = Ledger(d / "events.jsonl", run_id=f"prefix-s{seed}", attempt_id="a0", measurement_status=ctx.measurement_status)
    ctx.event_sink = led.on_meter_event
    e = srgc.Engine(ctx.backend, ctx.data["candidate_ids"], ctx.data["ranking_validation_ids"], ctx.data["cached_rewards"],
                    arm="on_policy", config=ctx.config)
    e.run_until(prefix_updates)
    ctx.event_sink = None
    d.mkdir(parents=True, exist_ok=True)
    snapshots.save(d / "prefix.pt", e.state_dict(), immutable=False)  # raw code-repo state, like prefix.pt
    atomic_write_json(d / "prefix-ready.json", {"completed_updates": e.step, "checkpoint_sha256": sha256_file(d / "prefix.pt")})


def prefix_cost(out_root, seed):
    from .common import read_jsonl
    ev = read_jsonl(Path(out_root) / "prefixes" / f"seed-{seed}" / "events.jsonl")
    if not ev:
        return None  # no ledger: unknown, never zero
    tot, missing = component_totals(ev)
    return None if missing else sum(v for k, v in tot.items() if k in ("on_selection", "training"))


def original_arms(ctx, seed, srgc, H):
    """Code-repo arms from the prefix, written like run_experiment endpoints (E0 audit input)."""
    root = ctx.out_root / "original"
    d = root / f"seed-{seed}"
    prefix = snapshots.load_engine_state(ctx.out_root / "prefixes" / f"seed-{seed}" / "prefix.pt")
    ready = read_json(ctx.out_root / "prefixes" / f"seed-{seed}" / "prefix-ready.json")
    atomic_write_json(d / "prefix-ready.json", ready)
    for arm in ("random", "sr", "on_policy", "switch"):
        if (d / f"{arm}-endpoint.json").exists():
            continue
        e = srgc.Engine(ctx.backend, ctx.data["candidate_ids"], ctx.data["ranking_validation_ids"],
                        ctx.data["cached_rewards"], arm=arm, config=ctx.config)
        e.load_state_dict(prefix, fork_arm=arm)
        e.run_until(H)
        per_q = ctx.backend.evaluate(ctx.data["evaluation_ids"], seed=srgc.stream_seed(seed, H, "reporting-evaluation"),
                                     responses=8)
        atomic_write_json(d / f"{arm}-endpoint.json", {
            "arm": arm, "seed": seed, "total_updates": e.step, "switched_at": e.switched_at,
            "prefix_checkpoint_sha256": ready["checkpoint_sha256"], "reward": sum(per_q.values()) / len(per_q),
            "per_question_reward": per_q, "cost_measurement_complete": True,
            "checks": [{"step": r["checkpoint"], "d": r["d"]} for r in e.history if r["d"] is not None],
            "implementation_sha256": coderepo.implementation_digest()})
    return root


def run_tasks(tasks, ctx_for_seed, log):
    done = []
    for t in tasks:
        ctx = ctx_for_seed(t["training_seed"])
        try:
            r = execute_train(t, ctx) if t["kind"] == "train" else execute_eval(t, ctx)
        except Interrupted as exc:
            log.append({"event": "interrupted", "run_id": t["run_id"], "detail": str(exc)})
            r = execute_train({k: v for k, v in t.items() if k != "debug_stop_after_update"}, ctx)
            log.append({"event": "resumed", "run_id": t["run_id"], "result": r})
        done.append(r)
    return done


def toy(out_root, *, bank_seeds=(5, 6, 7), dev_seeds=None, test_seeds=None, fast=False):
    t0 = time.time()
    out_root = Path(out_root)
    srgc = coderepo.srgc()
    PE = make_engine_class(srgc)
    feat, ans, data = backends.toy_problem(0)
    cfg = spec.resolve("olmo_math", role="smoke", extra={"cohort_id": "smoke_toy", "model": {
        "repository": "toy-categorical (srgc_rebuttal.toy_backend)", "revision": "code-repo"},
        "seeds": {"training": list(bank_seeds)}, "analysis": {"tie_tolerance_pp": 0.25}})
    validation = spec.validate(cfg, mode="smoke")
    H, P = cfg["training"]["matched_update_endpoint"], cfg["training"]["prefix_updates"]
    if fast:
        cfg["bank"]["extended_grid"] = [25, 75]
    rows_reg, scanned = registry(coderepo.code_root(), MANUSCRIPT)
    chosen = choose(rows_reg, "olmo_math", n_test=3, n_dev=3)
    dev_seeds = list(dev_seeds or chosen["development"])
    test_seeds = list(test_seeds or chosen["test"])
    loc = Locator(out_root, cfg, 25)
    ctxs = {}

    def ctx_for(seed):
        if seed not in ctxs:
            ctxs[seed] = toy_ctx(out_root, seed, srgc, PE, feat, ans, data, H)
        return ctxs[seed]

    log = [{"event": "validation", **validation}]
    for s in [*bank_seeds, *dev_seeds, *test_seeds]:
        make_prefix(ctx_for(s), s, P, srgc)
    orig_root = None
    for s in bank_seeds:
        orig_root = original_arms(ctx_for(s), s, srgc, H)

    # ---- Branch Bank (rebuttal role, seeds already inspected)
    role = "rebuttal_bank"
    traj, _ = mx.build_stage("bank-trajectory", cfg, loc, out_root, role=role, seeds=list(bank_seeds))
    traj[0]["debug_stop_after_update"] = 120  # exercise SIGTERM-like interruption + resume
    run_tasks(traj, ctx_for, log)
    for stage in ("bank-branches", "bank-extended", "destination"):
        tasks, notes = mx.build_stage(stage, cfg, loc, out_root, role=role, seeds=list(bank_seeds))
        log.append({"event": f"matrix:{stage}", "n": len(tasks), "notes": notes})
        run_tasks(tasks, ctx_for, log)
    # endpoint evaluations (final split at H) and dev curves on the trajectory + core branches (E5/E7)
    run_ids = sorted(p.parent.name for p in (out_root / "runs").glob("*/manifest.json"))
    run_tasks(mx.eval_tasks(cfg, data, out_root, run_ids, split="final", updates={H}), ctx_for, log)
    run_tasks(mx.eval_tasks(cfg, data, out_root, run_ids, split="dev", every=25), ctx_for, log)
    # budget: cost-only freeze, then extension and budget-endpoint evaluations
    pre = {s: prefix_cost(out_root, s) for s in bank_seeds}
    rows = []
    for s in bank_seeds:
        rows += SeedBank(out_root, cfg, cfg["cohort_id"], s, role, prefix_cost=pre[s], cache_cost=None,
                         evals={}).policy_rows()
    budget = freeze_budget(rows, scope="cold_start_algorithm", files_read=[str(out_root / "runs")])
    budget, _ = write_frozen_budget(out_root / "frozen" / "budget_rebuttal_bank.json", budget)
    ext, notes = mx.build_stage("budget-extension", cfg, loc, out_root, role=role, seeds=list(bank_seeds),
                                budget=budget["value"])
    log.append({"event": "matrix:budget-extension", "n": len(ext), "notes": notes})
    run_tasks(ext, ctx_for, log)
    run_ids = sorted(p.parent.name for p in (out_root / "runs").glob("*/manifest.json"))
    run_tasks(mx.eval_tasks(cfg, data, out_root, run_ids, split="final", every=25), ctx_for, log)
    evals = eval_index(out_root)
    bank_rows = []
    for s in bank_seeds:
        bank_rows += SeedBank(out_root, cfg, cfg["cohort_id"], s, role, prefix_cost=pre[s], evals=evals).policy_rows(
            budget=budget["value"])

    # ---- prospective development (fresh seeds): fixed grid, plateau eps, refresh K, srgc tau distribution
    dev_methods = ["fixed_sr_t025", "fixed_sr_t050", "fixed_sr_t100", "fixed_sr_t150", "fixed_sr_t200",
                   "plateau_sr_eps0000", "plateau_sr_eps0005", "plateau_sr_eps0010",
                   "sr_early_refreshed_k050", "sr_early_refreshed_k100", "srgc_current", "on_continued"]
    dtasks, _ = mx.build_stage("prospective", cfg, loc, out_root, role="development", seeds=dev_seeds,
                               prospective_methods=dev_methods)
    run_tasks(dtasks, ctx_for, log)
    dev_ids = [t["run_id"] for t in dtasks]
    run_tasks(mx.eval_tasks(cfg, data, out_root, dev_ids, split="dev", updates={H}), ctx_for, log)
    dev_rows = prospective_rows(out_root, cfg, cfg["cohort_id"], "development", dev_seeds, split="dev",
                                prefix_cost=None)
    tol = cfg["analysis"]["tie_tolerance_pp"]
    fixed_rows = [dict(r, tau=int(r["policy_id"][-3:])) for r in dev_rows if r["policy_id"].startswith("fixed_sr_t")]
    fixed = select_dev(fixed_rows, candidates=[25, 50, 100, 150, 200], key="tau", tie_tolerance_pp=tol)
    pl_rows = [dict(r, eps=int(r["policy_id"][-4:]) / 1000) for r in dev_rows if r["policy_id"].startswith("plateau_sr_eps")]
    plateau = select_dev(pl_rows, candidates=[0.0, 0.005, 0.01], key="eps", tie_tolerance_pp=tol)
    rf_rows = [dict(r, k=int(r["policy_id"][-3:])) for r in dev_rows if r["policy_id"].startswith("sr_early_refreshed_k")]
    refresh = select_dev(rf_rows, candidates=[50, 100], key="k", tie_tolerance_pp=tol)
    dist = tau_distribution([r for r in dev_rows if r["policy_id"] == "srgc_current"])
    tuning = frozen_tuning(cohort=cfg["cohort_id"], dev_seeds=dev_seeds,
                           dev_eval_hash=sha256_json(sorted(data["dev_evaluation_ids"])), fixed=fixed, plateau=plateau,
                           refresh=refresh, tau_distribution=dist, dev_run_ids=dev_ids,
                           code_identity=coderepo.identity(), config_hash=spec.config_hash(cfg),
                           test_runs_seen=sorted(p.parent.name for p in (out_root / "runs").glob("*/manifest.json")
                                                 if read_json(p)["role"] == "test"))
    write_once_json(out_root / "frozen" / "frozen_tuning.json", tuning)
    schedules = sample_schedules(dist, test_seeds, draws=cfg["e9"]["draws_per_seed"])
    write_once_json(out_root / "frozen" / "sampled_schedules.json", schedules)
    freeze_study(out_root / "frozen" / "frozen_study.json", cfgs={cfg["cohort_id"]: cfg},
                 seeds={"development": dev_seeds, "test": test_seeds, "rebuttal_bank": list(bank_seeds)},
                 tuning_path=out_root / "frozen" / "frozen_tuning.json", schedules=schedules,
                 code_identity=coderepo.identity())

    # ---- prospective test (fresh seeds; tuned values read-only)
    frozen = dict(tuning, sampled_schedules={k: {"policy": "sampled_time", "sampled_update": v[0]["sampled_update"],
                  "distribution_hash": v[0]["distribution_hash"], "schedule_seed": v[0]["schedule_seed"]}
                  for k, v in schedules.items()})
    test_methods = ["on_continued", "sr_early", "uniform_early", "fixed_sr_tuned", "srgc_current", "plateau_sr_tuned",
                    "srgc_uniform_destination", "srgc_one_negative", "srgc_two_negative", "sampled_time_sr",
                    f"sr_early_refreshed_k{tuning['refresh_interval']:03d}"]
    ttasks, _ = mx.build_stage("prospective", cfg, loc, out_root, role="test", seeds=test_seeds, frozen=frozen,
                               prospective_methods=test_methods)
    run_tasks(ttasks, ctx_for, log)
    run_tasks(mx.eval_tasks(cfg, data, out_root, [t["run_id"] for t in ttasks], split="final", every=25), ctx_for, log)
    test_rows = [dict(r, policy_id="srgc_current" if r["policy_id"] == "srgc_current" else r["policy_id"])
                 for r in prospective_rows(out_root, cfg, cfg["cohort_id"], "test", test_seeds,
                                           prefix_cost=None, budget=budget["value"])]

    # ---- E8 optional: the code repo's own bidirectional arm (switch_repeat), external
    e8 = []
    try:
        from scripts.srgc_switch_repeat import SwitchRepeatEngine
        s = bank_seeds[0]
        c = ctx_for(s)
        e = SwitchRepeatEngine(c.backend, data["candidate_ids"], data["ranking_validation_ids"], data["cached_rewards"],
                               arm="switch_repeat", config=c.config)
        e.load_state_dict(snapshots.load_engine_state(out_root / "prefixes" / f"seed-{s}" / "prefix.pt"),
                          fork_arm="switch_repeat")
        e.run_until(H)
        e8.append({"seed": s, "transitions": getattr(e, "transitions", None), "status": "ran_external_switch_repeat"})
    except Exception as exc:  # noqa: BLE001
        e8.append({"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"})

    # ---- E0 audit on the toy "original" arms + local split audit of real input bundles
    from .audit import split_overlap
    from .runner import integer_counts
    orig = []
    for s in bank_seeds:
        for arm in ("random", "sr", "on_policy", "switch"):
            ep = read_json(orig_root / f"seed-{s}" / f"{arm}-endpoint.json")
            k = integer_counts(ep["per_question_reward"], 8)
            orig.append({"seed": s, "arm": arm, "recomputed": sum(k.values()) / (8 * len(k)), "reported": ep["reward"],
                         "switched_at": ep["switched_at"]})
    # replay on the ORIGINAL switch log: rules that would fire after its own trigger are censored
    from .rules import replay
    censor_demo = []
    for s in bank_seeds:
        ep = read_json(orig_root / f"seed-{s}" / "switch-endpoint.json")
        obs = [(c["step"], c["d"]) for c in ep["checks"]]
        through = ep["switched_at"] if ep["switched_at"] is not None else 250
        for rule in ("one_negative", "two_negative", "current", "three_negative"):
            censor_demo.append({"seed": s, "source": "original_switch_log", "observed_through": through,
                                **replay(obs, rule, observed_through=through)})
    bundles = sorted(coderepo.code_root().glob("srgc_rebuttal/inputs/*seed-*.json"))
    overlap = split_overlap(bundles)

    # ---- analysis
    an = analyze_all(out_root, cfg, bank_rows, test_rows, list(bank_seeds), test_seeds, data, evals=eval_index(out_root))
    an["e8_external"] = e8
    an["e0_original_arms_recomputed"] = orig
    an["replay_on_original_switch_logs"] = censor_demo
    an["split_overlap_real_bundles"] = overlap
    an["seed_registry_choice"] = chosen
    an["physical_expenditure"] = physical_gpu_seconds(all_events(out_root))
    atomic_write_json(out_root / "analysis" / "analysis.json", an)
    (out_root / "analysis" / "report.md").write_text(report.markdown({k: an[k] for k in (
        "primary_rebuttal", "primary_test", "status_counts", "e3", "e9", "e10")}))
    atomic_write_json(out_root / "smoke_log.json", log)
    return {"seconds": time.time() - t0, "bank_rows": len(bank_rows), "test_rows": len(test_rows), "analysis": an}


def all_events(out_root):
    from .common import read_jsonl
    ev = []
    for p in Path(out_root).rglob("events.jsonl"):
        ev.extend(read_jsonl(p))
    for p in (Path(out_root) / "evaluations" / "events").glob("*.jsonl"):
        ev.extend(read_jsonl(p))
    return ev


def analyze_all(out_root, cfg, bank_rows, test_rows, bank_seeds, test_seeds, data, *, evals):
    an = {"status": "software check only (CPU smoke); not paper results"}
    an["primary_rebuttal"] = analyze.loso_contrast(bank_rows, bank_seeds, taus=cfg["bank"]["core_grid"])
    t = analyze.by_policy(test_rows)
    an["primary_test"] = analyze.paired(t.get("srgc_current", {}), t.get("fixed_sr_tuned", {}), test_seeds,
                                        label="srgc_current - fixed_sr_tuned (fresh test seeds)")
    tb = analyze.by_policy(test_rows, "budget_reward_fraction")
    an["primary_test_budget"] = analyze.paired(tb.get("srgc_current", {}), tb.get("fixed_sr_tuned", {}), test_seeds,
                                               label="at frozen budget")
    an["primary_test_bootstrap"] = analyze.bootstrap_ci(an["primary_test"]["per_seed_delta_pp"].values())
    an["secondary_rebuttal"] = analyze.secondary(bank_rows, bank_seeds)
    an["secondary_test"] = analyze.secondary(test_rows, test_seeds)
    an["status_counts"] = analyze.status_counts(bank_rows)
    an["e3"] = analyze.e3_destination(bank_rows, bank_seeds)
    an["e4"] = analyze.e4_rules(bank_rows)
    sig, curve = {}, {}
    for s in bank_seeds:
        bank = SeedBank(out_root, cfg, cfg["cohort_id"], s, "rebuttal_bank", evals=evals)
        sig[s] = dict(bank.d_observations()[0])
        for (dest, tau), man in bank.branches.items():
            for u in range(tau, cfg["training"]["matched_update_endpoint"] + 1, 25):
                r = evals.get((man["run_id"], u, "dev"))
                if r and dest == "sr":
                    curve[(s, f"fixed_sr_t{tau:03d}", u)] = r["reward_fraction"]
        for u in range(25, cfg["training"]["matched_update_endpoint"] + 1, 25):
            r = evals.get((bank.traj["run_id"], u, "dev"))
            curve[(s, "on_continued", u)] = r["reward_fraction"] if r else None
    an["e5"] = analyze.e5_decision_utility(sig, curve, cfg["evaluation"]["e5_parents"],
                                           cfg["evaluation"]["e5_short_horizons"], cfg["training"]["matched_update_endpoint"])
    an["e7_curves"] = [{"seed": k[0], "policy": k[1], "completed_update": k[2], "dev_reward": v,
                        "status": "measured" if v is not None else "unmeasured"} for k, v in sorted(curve.items(), key=str)]
    an["e9"] = analyze.paired(t.get("srgc_current", {}), t.get("sampled_time_sr", {}), test_seeds,
                              label="srgc_current - sampled_time_sr")
    k = next((p for p in t if p.startswith("sr_early_refreshed_k")), None)
    an["e10"] = analyze.paired(t.get(k, {}), t.get("sr_early", {}), test_seeds, label=f"{k} - sr_early") if k else None
    an["stratified_sign_test_example"] = analyze.stratified_sign_test(
        {"smoke_bank": an["primary_rebuttal"]["primary_contrast"]["per_seed_delta_pp"] if "primary_contrast" in an["primary_rebuttal"] else {},
         "smoke_test": an["primary_test"]["per_seed_delta_pp"]})
    analyze.write_csv(Path(out_root) / "analysis" / "policy_outcomes_bank.csv", bank_rows)
    analyze.write_csv(Path(out_root) / "analysis" / "policy_outcomes_test.csv", test_rows)
    analyze.write_csv(Path(out_root) / "analysis" / "e5_decision_utility.csv", an["e5"]["rows"])
    analyze.write_csv(Path(out_root) / "analysis" / "e7_reward_vs_updates.csv", an["e7_curves"])
    an["pareto_bank"] = analyze.pareto_plot(bank_rows, Path(out_root) / "analysis" / "pareto_bank.png", title="smoke bank")
    an["pareto_test"] = analyze.pareto_plot(test_rows, Path(out_root) / "analysis" / "pareto_test.png", title="smoke test")
    contrast = an["primary_test"]
    an["sentence_primary_test"] = report.primary_sentence(contrast, budget="(matched update 275)", unit="",
                                                          scope_components="selection, SR-GC checks and training "
                                                          "(prefix and cache excluded in smoke)",
                                                          missing_summary=str(contrast["missing_seeds"]), costs_partial=True)
    return an


def tiny(out_root):
    """TorchBackend branch-identity smoke on a random tiny OLMo-3 (CPU)."""
    t0 = time.time()
    out_root = Path(out_root)
    srgc = coderepo.srgc()
    PE = make_engine_class(srgc)
    records, data = backends.tiny_problem()
    cfg = spec.resolve("olmo_math", role="smoke", extra={
        "cohort_id": "smoke_tiny_olmo3", "model": {"repository": "random tiny Olmo3ForCausalLM", "revision": "none"},
        "training": {"prefix_updates": 5, "matched_update_endpoint": 30, "hard_safety_update_cap": 40,
                     "selection_interval": 5, "check_interval": 5, "first_check": 5, "scoring_prompts_per_set": 8,
                     "prompts_per_update": 2, "responses_per_prompt": 4},
        "bank": {"core_grid": [10, 15, 20], "extended_grid": [25], "retain_every": 5},
        "plateau": {"window_updates": 5, "check_interval": 5}, "seeds": {"training": [5]}})
    seed = 5
    ctx = Context(srgc=srgc, engine_class=PE, backend=None, data=None, config=srgc.Config(
        seed=seed, scoring_prompts=8, training_prompts=2, responses=4, candidate_group_size=2, projection_dim=16,
        selection_interval=5, check_interval=5, first_check=5), out_root=out_root, measurement_status="measured_cpu")
    ctx.backend = backends.tiny_backend(records, record=ctx.on_meter_event)
    ctx.meter = ctx.backend.cost_meter
    cache = backends.initial_cache(ctx.backend, data["candidate_ids"], responses=4, seed=seed, srgc=srgc)
    ctx.data = dict(data, cached_rewards=cache)
    make_prefix(ctx, seed, 5, srgc)
    loc = Locator(out_root, cfg, 5)
    log = []
    for stage in ("bank-trajectory", "bank-branches", "destination"):
        tasks, notes = mx.build_stage(stage, cfg, loc, out_root, role="rebuttal_bank", seeds=[seed])
        log.append({"stage": stage, "n": len(tasks), "notes": notes})
        for t in tasks:
            t["retain_updates"] = sorted(set(t["retain_updates"]) | set(range(t["parent"]["completed_update"] + 5, 31, 5)))
            execute_train(t, ctx)
    run_ids = sorted(p.parent.name for p in (out_root / "runs").glob("*/manifest.json"))
    for t in mx.eval_tasks(cfg, ctx.data, out_root, run_ids, split="final", updates={30}):
        execute_eval(t, ctx)
    # identity: each branch's recorded parent identity equals the trajectory snapshot identity at its fork
    checks = []
    for rid in run_ids:
        bs = Path(out_root) / "runs" / rid / "branch_start.json"
        if not bs.exists():
            continue
        b = read_json(bs)
        if b["parent"]["kind"] == "snapshot":
            snap = {r["completed_update"]: r for r in mx.read_jsonl(Path(out_root) / "runs" / b["parent"]["run_id"] / "snapshots.jsonl")}[b["parent"]["completed_update"]]
            checks.append({"run_id": rid, "fork_update": b["parent"]["completed_update"],
                           "state_identity_equal": snap["identity"]["state_hash"] == b["parent_identity"]["state_hash"],
                           "branch_identity_equal": snap["branch_identity"]["branch_hash"] == b["parent_branch_identity"]["branch_hash"]})
    out = {"seconds": time.time() - t0, "log": log, "identity_checks": checks,
           "evaluations": len(list((out_root / "evaluations").glob("*.json")))}
    atomic_write_json(out_root / "tiny_summary.json", out)
    return out


def main(out_root, *, keep=False, fast=False):
    out_root = Path(out_root)
    if out_root.exists() and not keep:
        shutil.rmtree(out_root)
    result = {"started_at": utc_now()}
    toy_res = toy(out_root / "toy", fast=fast)
    an = toy_res["analysis"]
    result["toy"] = {"seconds": round(toy_res["seconds"], 1), "bank_rows": toy_res["bank_rows"],
                     "test_rows": toy_res["test_rows"], "status_counts": an["status_counts"],
                     "loso_tau_hat": {s: v.get("tau_hat") for s, v in an["primary_rebuttal"]["loso"]["per_seed"].items()}
                     if "loso" in an["primary_rebuttal"] else None,
                     "primary_test_n": an["primary_test"]["n_training_seeds"],
                     "e3": an["e3"], "e8": an["e8_external"], "pareto": [an["pareto_bank"]["status"], an["pareto_test"]["status"]],
                     "physical_complete": an["physical_expenditure"]["complete"],
                     "censored_on_original_logs": sum(r["status"] == "censored" for r in an["replay_on_original_switch_logs"]),
                     "split_overlap_bundles": len(an["split_overlap_real_bundles"]),
                     "split_overlap_all_clean": all(r["validate_inputs"] and not r["ranking_vs_final_overlap"]
                                                    and not r["dev_vs_final_overlap"] for r in an["split_overlap_real_bundles"])}
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
        result["tiny"] = tiny(out_root / "tiny")
    except ImportError as exc:
        result["tiny"] = {"status": "skipped", "reason": str(exc)}
    result["finished_at"] = utc_now()
    atomic_write_json(out_root / "SMOKE_RESULT.json", result)
    return result
