"""Cluster-side CPU commands (run by run_h100.sh on a node with /group-volume mounted).

Layout under V8_ROOT (default /group-volume/minsoo3.kim/offpolicy-misranking/v8):
  <cohort>/locator.json            plan, run roots, prefix hashes, identities (from the code repo)
  <cohort>/matrix/<stage>.json     tasks of a stage (regenerated idempotently)
  <cohort>/runs/<run_id>/...       see runner.py
  <cohort>/evaluations/*.json      immutable evaluation records (+ events/ ledgers)
  <cohort>/frozen/*.json           budget, tuning, schedules, frozen study (write-once)
  <cohort>/analysis/...            tables, figures, report (recomputable from raw)
  audit/                           E0 outputs
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import coderepo, matrix as mx, spec
from .common import atomic_write_json, read_json, sha256_file, utc_now, write_once_json

DEFAULT_ROOT = "/group-volume/minsoo3.kim/offpolicy-misranking/v8"


def v8_root(args=None):
    root = Path(getattr(args, "v8_root", None) or os.environ.get("V8_ROOT", DEFAULT_ROOT))
    group = Path(os.environ.get("GROUP_VOLUME", "/group-volume"))
    if not os.environ.get("V8_ALLOW_ANY_ROOT") and not root.resolve().is_relative_to(group.resolve()):
        raise ValueError(f"V8_ROOT must be inside {group} (user volume holds code only): {root}")
    return root


def resolve_plan(cohort_cfg, seed, code_root):
    """Same plan resolution as the code repo's extra-arm launcher (run_srgc_sr_refresh.sh)."""
    coderepo.activate(code_root)
    src = cohort_cfg["runtime"]["source_plan"]
    if os.environ.get(f"V8_PLAN_{cohort_cfg['cohort_id'].upper()}"):
        return Path(os.environ[f"V8_PLAN_{cohort_cfg['cohort_id'].upper()}"]).resolve()
    kind, dataset = src.split(":", 1)
    if kind == "auto":
        from srgc_pair_inputs import default_plan
        from srgc_shared_storage import route_plan
        from scripts.srgc_extra_plan import select_plan
        source = default_plan(Path(code_root), dataset, os.environ, writing=False)
        return select_plan(route_plan(source, writing=False), seed)
    if kind == "qwen":
        from srgc_qwen35_storage import default_root
        return (default_root(os.environ) / "experiments" / f"qwen35-9b-{dataset}.json").resolve()
    raise ValueError(f"no plan resolver for {src}; set V8_PLAN_{cohort_cfg['cohort_id'].upper()}")


def cmd_locate(args):
    root = coderepo.code_root(args.code_root)
    cfg = spec.resolve(args.cohort, role="rebuttal_bank")
    coderepo.activate(root)
    from srgc_rebuttal.plan import input_path, load_plan
    from srgc_rebuttal.runtime import run_root
    digest = coderepo.implementation_digest(root)
    out = {"cohort": args.cohort, "code": coderepo.identity(root), "located_at": utc_now(), "seeds": {}}
    for seed in cfg["seeds"]["training"]:
        plan_path = resolve_plan(cfg, seed, root)
        plan = load_plan(plan_path)
        folder = run_root(plan_path, plan) / f"seed-{seed}"
        ready = folder / "prefix-ready.json"
        run = folder / "run.json"
        rec = {"plan_path": str(plan_path), "plan_sha256": sha256_file(plan_path), "folder": str(folder),
               "model": plan.get("model"), "model_revision": plan.get("model_revision"),
               "input_bundle": str(input_path(plan_path, plan, seed))}
        if ready.exists():
            r = read_json(ready)
            rec.update(prefix_path=str(folder / "prefix.pt"), prefix_sha256=r["checkpoint_sha256"],
                       prefix_completed_updates=r["completed_updates"])
        if run.exists():
            impl = read_json(run).get("implementation_sha256")
            rec.update(run_implementation_sha256=impl, implementation_matches_code_root=impl == digest)
        if cfg["model"]["revision"] and plan.get("model_revision") not in (None, cfg["model"]["revision"]):
            rec["model_revision_mismatch"] = True
        out["seeds"][str(seed)] = rec
    path = v8_root(args) / args.cohort / "locator.json"
    atomic_write_json(path, out)
    bad = [s for s, r in out["seeds"].items() if not r.get("implementation_matches_code_root") or
           "prefix_sha256" not in r or r.get("model_revision_mismatch")]
    print(json.dumps({"written": str(path), "problems": bad}, indent=2))
    if bad and not args.allow_mismatch:
        raise SystemExit("locator problems: point V8_CODE_ROOT at the frozen checkout that produced these runs "
                         "(implementation_sha256 must match run.json) or pass --allow-mismatch for read-only use")


class ClusterLocator:
    def __init__(self, cohort_root, cfg, interval=25):
        self.root, self.cfg, self.interval = Path(cohort_root), cfg, interval
        self.loc = read_json(self.root / "locator.json")

    def prefix(self, seed):
        r = self.loc["seeds"][str(seed)]
        if r.get("prefix_sha256"):
            return {"kind": "prefix", "path": r["prefix_path"], "sha256": r["prefix_sha256"],
                    "completed_update": r["prefix_completed_updates"], "run_id": None}
        # fresh seed: the v8 prefix run (stage `prefix`) provides the snapshot at the prefix boundary
        P = self.cfg["training"]["prefix_updates"]
        for m in sorted((self.root / "runs").glob("*/manifest.json")):
            man = read_json(m)
            if (man["training_seed"], man["method_id"]) == (seed, "on_continued") and \
                    man["task"]["parent"]["kind"] == "initial" and man["status"] == "completed_horizon":
                p = mx.parent_from_snapshot(self.root, man["run_id"], P)
                if p:
                    return p
        raise LookupError(f"seed {seed}: no prefix yet (run stage `prefix` first)")

    def completed_runs(self, seed, role):
        from .smoke import Locator
        return Locator.completed_runs(self, seed, role)  # same scan of <cohort>/runs

    @property
    def out_root(self):
        return self.root

    def cost_offset(self, run):
        from .smoke import Locator
        return Locator.cost_offset(self, run)


def cmd_make_matrix(args):
    root = v8_root(args) / args.cohort
    cfg = spec.resolve(args.cohort, role=args.role)
    loc = ClusterLocator(root, cfg)
    frozen = read_json(root / "frozen" / "frozen_tuning.json") if (root / "frozen" / "frozen_tuning.json").exists() else None
    budget = read_json(root / "frozen" / "budget.json")["value"] if (root / "frozen" / "budget.json").exists() else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    if args.stage.startswith("eval-"):
        data = load_bundle(loc.loc["seeds"][str((seeds or cfg["seeds"]["training"])[0])]["input_bundle"])
        run_ids = sorted(p.parent.name for p in (root / "runs").glob("*/manifest.json"))
        split = "dev" if args.stage == "eval-dev" else "final"
        H = cfg["training"]["matched_update_endpoint"]
        updates = {H} if args.stage == "eval-final" else None
        every = cfg["evaluation"]["curve_every"] if args.stage in ("eval-dev", "eval-curves-final") else None
        tasks, notes = mx.eval_tasks(cfg, data, root, run_ids, split=split, updates=updates, every=every), []
        if args.stage == "eval-budget":
            tasks = budget_eval_tasks(root, cfg, data, budget)
    else:
        methods = args.methods.split(",") if args.methods else None
        tasks, notes = mx.build_stage(args.stage, cfg, loc, root, role=args.role, seeds=seeds, frozen=frozen,
                                      budget=budget, prospective_methods=methods)
    path = root / "matrix" / f"{args.stage}.{args.role}.json"
    mx.write_matrix(path, cfg, args.stage, args.role, tasks, notes)
    print(json.dumps({"matrix": str(path), "tasks": len(tasks), "notes": notes[:20]}, indent=2, default=str))


def budget_eval_tasks(root, cfg, data, budget):
    """Evaluate (final split) exactly the snapshots selected by metadata at the frozen budget."""
    from .bank import SeedBank
    if budget is None:
        raise SystemExit("freeze the budget first (freeze-budget)")
    keep = set()
    for seed in cfg["seeds"]["training"]:
        bank = SeedBank(root, cfg, cfg["cohort_id"], seed, "rebuttal_bank", evals={})
        for row in bank.policy_rows(budget=budget):
            if row.get("budget_status") == "selected":
                keep.add((row["run_id"], row["budget_endpoint_update"]))
    tasks = []
    for rid, u in sorted(keep):
        tasks += mx.eval_tasks(cfg, data, root, [rid], split="final", updates={u})
    return tasks


def load_bundle(path):
    d = json.loads(Path(path).read_text())
    ranking = set(d["ranking_validation_ids"])
    d["dev_evaluation_ids"] = [i for i in d["validation_pool_ids"] if i not in ranking]
    return d


def task_done(root, task):
    if task["kind"] == "evaluate":
        return (root / "evaluations" / f"{task['snapshot_sha256'][:16]}__{task['evaluation_spec_hash'][:12]}.json").exists()
    return mx.is_complete(root, task["run_id"])


def cmd_pending(args):
    root = v8_root(args) / args.cohort
    m = read_json(root / "matrix" / f"{args.stage}.{args.role}.json")
    done = {t.get("run_id") for t in m["tasks"] if t["kind"] == "train" and task_done(root, t)}
    for t in m["tasks"]:
        if task_done(root, t):
            continue
        if any(d not in done and not mx.is_complete(root, d) for d in t.get("deps", [])):
            continue
        print(t.get("task_id") or t["run_id"])


def cmd_freeze_budget(args):
    from .bank import SeedBank
    from .freeze import freeze_budget, write_frozen_budget
    root = v8_root(args) / args.cohort
    cfg = spec.resolve(args.cohort, role="rebuttal_bank")
    rows = []
    for seed in cfg["seeds"]["training"]:
        rows += SeedBank(root, cfg, args.cohort, seed, "rebuttal_bank", evals={},
                         prefix_cost=prefix_cost_from_receipts(root, seed)).policy_rows()
    b = freeze_budget(rows, files_read=[str(root / "runs")])
    b["gpu_type"], b["gpu_count"] = cfg["runtime"]["gpu_type"], cfg["runtime"]["world_size"]
    b, how = write_frozen_budget(root / "frozen" / "budget.json", b)
    print(json.dumps({"status": how, **b}, indent=2))


def prefix_cost_from_receipts(root, seed):
    """Shared-prefix GPU-seconds from the code repo's own receipts (cost-receipts/shared-prefix)."""
    try:
        loc = read_json(Path(root) / "locator.json")["seeds"][str(seed)]
        coderepo.activate()
        from srgc_rebuttal.cost_ledger import PhaseLedger
        receipts = Path(loc["folder"]) / "cost-receipts" / "shared-prefix"
        if not receipts.is_dir():
            return None  # no receipts: unknown, never zero
        t = PhaseLedger(receipts).totals()
        if not t["complete"] or t["recorded_phases"] == 0:
            return None
        return t["known_gpu_seconds"]["selection_gpu_seconds"] + t["known_gpu_seconds"]["training_gpu_seconds"]
    except Exception:  # noqa: BLE001 - unknown stays unknown
        return None


def cmd_profile(args):
    """Cost profile from EXISTING endpoint receipts of the original arms (no rewards read)."""
    root = v8_root(args) / args.cohort
    loc = read_json(root / "locator.json")
    out = {"cohort": args.cohort, "seeds": {}, "note": "costs only; reward fields are not read"}
    for seed, r in loc["seeds"].items():
        folder = Path(r["folder"])
        row = {}
        for arm in ("random", "sr", "on_policy", "switch"):
            p = folder / f"{arm}-endpoint.json"
            if p.exists():
                e = json.loads(p.read_text())
                row[arm] = {"costs": e.get("costs"), "evaluation_gpu_seconds": e.get("evaluation_gpu_seconds"),
                            "complete": e.get("cost_measurement_complete"), "switched_at": e.get("switched_at"),
                            "total_updates": e.get("total_updates")}
        row["prefix_gpu_seconds"] = prefix_cost_from_receipts(root, seed)
        out["seeds"][seed] = row
    atomic_write_json(root / "cost_profile.json", out)
    print(json.dumps({"written": str(root / "cost_profile.json")}, indent=2))


def cmd_collect(args):
    """Rows for analysis: bank policies (rebuttal role) and prospective runs (development/test)."""
    from .analyze import write_csv
    from .bank import SeedBank, eval_index, prospective_rows
    root = v8_root(args) / args.cohort
    cfg = spec.resolve(args.cohort, role="rebuttal_bank")
    budget = read_json(root / "frozen" / "budget.json")["value"] if (root / "frozen" / "budget.json").exists() else None
    evals = eval_index(root)
    rows = []
    for seed in cfg["seeds"]["training"]:
        rows += SeedBank(root, cfg, args.cohort, seed, "rebuttal_bank", evals=evals,
                         prefix_cost=prefix_cost_from_receipts(root, seed)).policy_rows(budget=budget)
    write_csv(root / "analysis" / "policy_outcomes_bank.csv", rows)
    atomic_write_json(root / "analysis" / "policy_outcomes_bank.json", rows)
    for role in ("development", "test"):
        seeds = sorted({read_json(m)["training_seed"] for m in (root / "runs").glob("*/manifest.json")
                        if read_json(m)["role"] == role})
        pr = prospective_rows(root, cfg, args.cohort, role, seeds, evals=evals, budget=budget,
                              split="dev" if role == "development" else "final")
        write_csv(root / "analysis" / f"policy_outcomes_{role}.csv", pr)
        atomic_write_json(root / "analysis" / f"policy_outcomes_{role}.json", pr)
    print(json.dumps({"bank_rows": len(rows)}, indent=2))


def cmd_analyze(args):
    from . import analyze, report
    root = v8_root(args) / args.cohort
    cfg = spec.resolve(args.cohort, role="rebuttal_bank")
    rows = read_json(root / "analysis" / "policy_outcomes_bank.json")
    seeds = cfg["seeds"]["training"]
    an = {"status": "from recorded outputs", "primary_rebuttal": analyze.loso_contrast(rows, seeds, taus=cfg["bank"]["core_grid"]),
          "primary_rebuttal_extended": analyze.loso_contrast(rows, seeds),
          "secondary": analyze.secondary(rows, seeds), "status_counts": analyze.status_counts(rows),
          "e3": analyze.e3_destination(rows, seeds), "e4": analyze.e4_rules(rows)}
    test = root / "analysis" / "policy_outcomes_test.json"
    if test.exists():
        tr = read_json(test)
        ts = sorted({r["seed"] for r in tr})
        t = analyze.by_policy(tr)
        an["primary_test"] = analyze.paired(t.get("srgc_current", {}), t.get("fixed_sr_tuned", {}), ts,
                                            label="srgc_current - fixed_sr_tuned")
        an["secondary_test"] = analyze.secondary(tr, ts)
    an["pareto"] = analyze.pareto_plot(rows, root / "analysis" / "pareto_bank.png", title=args.cohort)
    atomic_write_json(root / "analysis" / "analysis.json", an)
    (root / "analysis" / "report.md").write_text(report.markdown(an))
    print(json.dumps({"written": str(root / "analysis")}, indent=2))


def cmd_prepare_seeds(args):
    """Fresh development/test seeds: same split (split seed 0) and initial-policy cache as the source bundle.

    Integration assumption to confirm on the cluster: the seeds 5-9 bundles share one split and
    one initial-policy cache protocol, so a new seed reuses them and differs only in training RNG.
    """
    from .seeds import choose, registry
    root = v8_root(args) / args.cohort
    loc_path = root / "locator.json"
    loc = read_json(loc_path)
    code = coderepo.code_root(args.code_root)
    rows, _ = registry(code, Path(__file__).resolve().parents[3])
    chosen = choose(rows, args.cohort if args.cohort in {r["cohort"] for r in rows} else "olmo_math")
    source_seed = args.source_seed or min(int(s) for s, r in loc["seeds"].items() if r.get("prefix_sha256"))
    src = loc["seeds"][str(source_seed)]
    bundle = json.loads(Path(src["input_bundle"]).read_text())
    for role in ("development", "test"):
        for seed in chosen[role]:
            if str(seed) in loc["seeds"]:
                continue
            b = dict(bundle, provenance=dict(bundle["provenance"], experiment_seed=seed,
                     v8_source_bundle=src["input_bundle"], v8_source_bundle_sha256=sha256_file(src["input_bundle"]),
                     v8_note="split and initial-policy cache reused from the source bundle; training RNG = seed"))
            path = root / "inputs" / f"seed-{seed}.json"
            atomic_write_json(path, b)
            loc["seeds"][str(seed)] = {k: src[k] for k in ("plan_path", "plan_sha256", "model", "model_revision",
                                                          "run_implementation_sha256", "implementation_matches_code_root")
                                       if k in src}
            loc["seeds"][str(seed)].update(input_bundle=str(path), role=role, folder=None)
    loc["seed_roles"] = chosen
    atomic_write_json(loc_path, loc)
    print(json.dumps({"seed_roles": chosen, "source_seed": source_seed}, indent=2))


def register(sub):
    def add(name, fn, *opts):
        s = sub.add_parser(name)
        s.add_argument("--cohort", required=True)
        s.add_argument("--v8-root", default=None)
        for o in opts:
            o(s)
        s.set_defaults(fn=fn)
        return s
    add("locate", cmd_locate, lambda s: s.add_argument("--allow-mismatch", action="store_true"))
    add("make-matrix", cmd_make_matrix, lambda s: s.add_argument("--stage", required=True),
        lambda s: s.add_argument("--role", default="rebuttal_bank"), lambda s: s.add_argument("--seeds"),
        lambda s: s.add_argument("--methods"))
    add("pending", cmd_pending, lambda s: s.add_argument("--stage", required=True),
        lambda s: s.add_argument("--role", default="rebuttal_bank"))
    add("freeze-budget", cmd_freeze_budget)
    add("prepare-seeds", cmd_prepare_seeds, lambda s: s.add_argument("--source-seed", type=int))
    add("profile", cmd_profile)
    add("collect", cmd_collect)
    add("analyze", cmd_analyze)
