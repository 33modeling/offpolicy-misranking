"""Build a fake $WORK with V8 Branch Bank outputs for local V9 checks (CPU only, never paper data).

    V9_ALLOW_FAKE=1 python v9/fake/make_fake_work.py <group_volume_dir>

Layout (same as the cluster): <group>/u/offpolicy-misranking/v8/<cohort>/{locator.json,runs,evaluations,frozen}
written by the V8 code itself (matrix.build_stage + runner.execute_train/execute_eval + cluster freeze-budget):
  olmo_math  ToyBackend, seeds 5-9: On trajectory (D monitor), SR branches at every grid tau, V8 budget freeze,
             V8 budget extension, eval-final and eval-budget  -> freeze-tau / x1 / x2 / x3 / x4 / x5 input
  olmo_mbpp  ToyBackend, seeds 5-9: trajectory, branches 25 and tau, eval-final                -> x1 / x2
  qwen_mbpp  ToyBackend, seeds 5-9: trajectory, branches 25 and tau; eval-final for seeds 5-7 only
             (8, 9 stay 'waiting for V8 eval-final')
  qwen_math  tiny random OLMo-3 + LoRA (TorchBackend, autograd), seed 5 only: trajectory, branches 25 and tau,
             eval-final (6-9 stay 'waiting for V8 trajectory'); x5 runs the real TorchBackend probe here
  llama_math fake locator only (toy prefixes) under v9-fake-inputs/: V9 trains it (X1 Llama path)
  llama_mbpp nothing: 'Llama plan not found'
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def toy_bundle(path, problem_seed=0):
    from switching import backends
    _, answers, data = backends.toy_problem(problem_seed)
    ids = data["candidate_ids"] + data["validation_pool_ids"] + data["evaluation_ids"]
    b = {"schema": "srgc-inputs-v1", "dataset": "toy", "records": {i: {"prompt": i, "question": i, "answer": str(answers[i])}
                                                                     for i in ids},
         "candidate_ids": data["candidate_ids"], "validation_pool_ids": data["validation_pool_ids"],
         "ranking_validation_ids": data["ranking_validation_ids"], "evaluation_ids": data["evaluation_ids"],
         "cached_rewards": data["cached_rewards"],
         "provenance": {"split_seed": 0, "cache": {"source": "toy make_problem initial policy (seed 0)"},
                        "note": "FAKE local software check"}}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(b))
    return b


def tiny_bundle(path, n_candidates=80, n_val=8, n_eval=6, seed=5):
    from switching import backends, coderepo
    srgc = coderepo.srgc()
    ids = [f"t{i:03d}" for i in range(n_candidates + n_val + n_eval)]
    records = {i: {"prompt": f"problem {i}", "answer": "3", "question": f"problem {i}"} for i in ids}
    cand, val, ev = ids[:n_candidates], ids[n_candidates:n_candidates + n_val], ids[n_candidates + n_val:]
    backend = backends.tiny_backend(records)
    cache = backends.initial_cache(backend, cand, responses=8, seed=seed, srgc=srgc)
    b = {"schema": "srgc-inputs-v1", "dataset": "tiny", "records": records, "candidate_ids": cand,
         "validation_pool_ids": val, "ranking_validation_ids": val[:4], "evaluation_ids": ev,
         "cached_rewards": cache, "provenance": {"split_seed": 0, "cache": {"source": "tiny base model (no adapter)"},
                                                  "note": "FAKE local software check"}}
    Path(path).write_text(json.dumps(b))
    return b


def make_prefix(ctx, folder, srgc):
    from switching import snapshots
    from switching.common import atomic_write_json, sha256_file
    folder = Path(folder)
    if (folder / "prefix-ready.json").exists():
        return json.loads((folder / "prefix-ready.json").read_text())
    e = srgc.Engine(ctx.backend, ctx.data["candidate_ids"], ctx.data["ranking_validation_ids"], ctx.data["cached_rewards"],
                    arm="on_policy", config=ctx.config)
    e.run_until(25)
    folder.mkdir(parents=True, exist_ok=True)
    snapshots.save(folder / "prefix.pt", e.state_dict(), immutable=False)
    ready = {"completed_updates": e.step, "checkpoint_sha256": sha256_file(folder / "prefix.pt")}
    atomic_write_json(folder / "prefix-ready.json", ready)
    return ready


def locator(cohort, work, seeds, backend, bundle_path, prefixes):
    loc = {"cohort": cohort, "located_at": "fake", "code": {"code_root": os.environ["V8_CODE_ROOT"]}, "seeds": {}}
    for s in seeds:
        rec = {"plan_path": "FAKE", "plan_sha256": "FAKE", "folder": str(work / "fake-runs" / cohort / f"seed-{s}"),
               "model": f"FAKE-{backend}", "model_revision": None, "input_bundle": str(bundle_path),
               "run_implementation_sha256": "FAKE", "implementation_matches_code_root": True,
               "runtime_selection": {"mode": "fake"}, "fake": {"backend": backend, "problem_seed": 0}}
        if s in prefixes:
            rec.update(prefix_path=str(work / "fake-runs" / cohort / f"seed-{s}" / "prefix.pt"),
                       prefix_sha256=prefixes[s]["checkpoint_sha256"], prefix_completed_updates=25)
        loc["seeds"][str(s)] = rec
    return loc


def build_cohort(work, cohort, *, backend, seeds, taus, eval_seeds, budget=False, log=print):
    from switching import cluster, matrix as mx, spec
    from switching.cluster import ClusterLocator, load_bundle
    from switching.common import atomic_write_json
    from switching.runner import execute_eval, execute_train
    from fake.fake_child import build_ctx
    from switching import coderepo
    srgc = coderepo.srgc()
    v8 = work / "v8"
    root = v8 / cohort
    bpath = work / "fake-inputs" / f"{cohort}-bundle.json"
    if not bpath.exists():
        (toy_bundle if backend == "toy" else tiny_bundle)(bpath)
    data = load_bundle(bpath)
    prefixes = {}
    for s in seeds:
        seed_loc = {"fake": {"backend": backend, "problem_seed": 0}}
        ctx, _ = build_ctx(root, s, seed_loc, data)
        prefixes[s] = make_prefix(ctx, work / "fake-runs" / cohort / f"seed-{s}", srgc)
    cfg_seeds = spec.resolve(cohort, role="rebuttal_bank")["seeds"]["training"]
    atomic_write_json(root / "locator.json", locator(cohort, work, cfg_seeds, backend, bpath, prefixes))
    cfg = spec.resolve(cohort, role="rebuttal_bank")
    cfg["bank"]["core_grid"] = [t for t in taus if t != 25]
    cfg["bank"]["extended_grid"] = [25] if 25 in taus else []
    loc = ClusterLocator(root, cfg)
    ctxs = {}

    def run(task):
        s = task["training_seed"]
        if s not in ctxs:
            ctxs[s] = build_ctx(root, s, {"fake": {"backend": backend, "problem_seed": 0}}, data)[0]
        return execute_eval(task, ctxs[s]) if task["kind"] == "evaluate" else execute_train(task, ctxs[s])
    t0 = time.time()
    for stage in ("bank-trajectory", "bank-branches", "bank-extended"):
        tasks, _ = mx.build_stage(stage, cfg, loc, root, seeds=seeds)
        for t in tasks:
            run(t)
        log(f"  {cohort} {stage}: {len(tasks)} runs ({time.time() - t0:.0f}s)")
    H = cfg["training"]["matched_update_endpoint"]
    if budget:
        args = argparse.Namespace(v8_root=str(v8), cohort=cohort)
        cluster.cmd_freeze_budget(args)
        B = json.loads((root / "frozen" / "budget.json").read_text())["value"]
        tasks, notes = mx.build_stage("budget-extension", cfg, loc, root, seeds=seeds, budget=B)
        # V8 runner fails a budget extension that cannot take a single update (offset + one update > B:
        # _finalize finds no retained snapshot at the parent update); skip those, as V9 x2 does.
        tasks = [t for t in tasks if t["cost_offset_gpu_seconds"] + t["est_update_gpu_seconds"] <= B]
        for t in tasks:
            run(t)
        log(f"  {cohort} budget-extension: {len(tasks)} runs ({time.time() - t0:.0f}s)")
    run_ids = sorted(p.parent.name for p in (root / "runs").glob("*/manifest.json"))
    evs = [t for t in mx.eval_tasks(cfg, data, root, run_ids, split="final", updates={H})
           if t["training_seed"] in eval_seeds]
    if budget:
        evs += cluster.budget_eval_tasks(root, cfg, data, B)
        # cluster.budget_eval_tasks evaluates (policy row run_id, endpoint update); an endpoint inside a budget
        # extension belongs to the extension run, so it is missed there. Evaluate the endpoints themselves.
        from switching.bank import SeedBank
        for s in seeds:
            b = SeedBank(root, cfg, cohort, s, "rebuttal_bank", evals={})
            for (dest, t), man in b.branches.items():
                if dest != "sr":
                    continue
                be = b.budget_endpoint("fixed_sr", t, man, B, include_check=False)
                if be["status"] == "selected":
                    ep = be["endpoint"]
                    evs += mx.eval_tasks(cfg, data, root, [ep["run_id"]], split="final", updates={ep["completed_update"]})
    for t in {t["task_id"]: t for t in evs}.values():
        run(t)
    log(f"  {cohort} evaluations: {len(evs)} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    group = Path(args.group).resolve()
    os.environ["GROUP_VOLUME"] = str(group)
    os.environ.setdefault("OM_USER", "u")
    from v9lib import env
    env.activate()
    work = env.work()
    work.mkdir(parents=True, exist_ok=True)
    os.environ["V8_ROOT"] = str(work / "v8")
    only = set(args.only.split(",")) if args.only else None
    full = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    if only is None or "olmo_math" in only:
        build_cohort(work, "olmo_math", backend="toy", seeds=[5, 6, 7, 8, 9], taus=full, eval_seeds={5, 6, 7, 8, 9},
                     budget=True)
    from v9lib import tau
    tau_value = tau.compute()["tau"]
    print(f"  fake tau (computed, not frozen) = {tau_value}")
    branches = sorted({25, tau_value})
    if only is None or "olmo_mbpp" in only:
        build_cohort(work, "olmo_mbpp", backend="toy", seeds=[5, 6, 7, 8, 9], taus=branches, eval_seeds={5, 6, 7, 8, 9})
    if only is None or "qwen_mbpp" in only:
        build_cohort(work, "qwen_mbpp", backend="toy", seeds=[5, 6, 7, 8, 9], taus=branches, eval_seeds={5, 6, 7})
    if only is None or "qwen_math" in only:
        build_cohort(work, "qwen_math", backend="tiny", seeds=[5], taus=branches, eval_seeds={5})
    # Llama MATH: original-run stand-in = prefixes only (V9 trains it); Llama MBPP: absent
    from switching.common import atomic_write_json
    from switching.cluster import load_bundle
    from fake.fake_child import build_ctx
    from switching import coderepo
    bpath = work / "fake-inputs" / "llama_math-bundle.json"
    if not bpath.exists():
        toy_bundle(bpath)
    data = load_bundle(bpath)
    prefixes = {}
    for s in (5, 6, 7, 8, 9):
        ctx, _ = build_ctx(work / "fake-runs" / "llama_math", s, {"fake": {"backend": "toy"}}, data)
        prefixes[s] = make_prefix(ctx, work / "fake-runs" / "llama_math" / f"seed-{s}", coderepo.srgc())
    atomic_write_json(work / "v9-fake-inputs" / "llama_math-locator.json",
                      locator("llama_math", work, [5, 6, 7, 8, 9], "toy", bpath, prefixes))
    print(json.dumps({"work": str(work), "v8": str(work / "v8")}))


if __name__ == "__main__":
    main()
