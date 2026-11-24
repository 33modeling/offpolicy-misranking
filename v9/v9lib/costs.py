"""Cost ledger for collect (X1.cost, X1.bankcost, X6): GPU-seconds from recorded ledger events and receipts only.

Scopes (the manuscript's Table 3 and appendix E.3):
  per refresh   Alignment candidate scoring (on_selection) per selection round of the On trajectory, updates [P, H)
  incremental   continuation algorithm cost from the shared 25-update state to update 275: selection, subset
                preparation and training; prefix, reward cache and evaluation excluded. Staged = trajectory to tau
                + SR branch to 275 (SR-GC monitor checks excluded, as in the X2 budget rule).
  cold start    incremental + shared prefix + reward cache (cache only for the arms that read it: SR, Staged)
  cache         one-time reward-cache generation of the seed's input bundle (srgc_rebuttal build_cache receipts)
Algorithm costs use the surviving lineage of each run (work redone after a restart is dropped), so the
V9-trained Llama times exclude restarts; `physical_*` keeps the redone work. Unknown is None, never zero.
"""

from __future__ import annotations

from pathlib import Path

from . import env
from .common import cfg, v8c, v9c

CACHE_ARMS = ("sr", "staged")


def _events(rdir, lineage=True):
    from switching.runner import run_events
    return run_events(rdir, lineage=lineage)


def _family(root, run_id):
    from switching.matrix import manifest
    m = manifest(root, run_id)
    return None if m is None else m["task"]["method"]["family"]


def run_cost(root, run_id, *, lo=None, hi=None, family=None):
    """Continuation algorithm GPU-seconds of one run over updates [lo, hi); None if unknown."""
    from switching.ledger import REQUIRED, component_totals, views
    if not run_id:
        return None
    rdir = Path(root) / "runs" / run_id
    family = family or _family(root, run_id)
    if family is None or family not in REQUIRED or not rdir.is_dir():
        return None
    tot, missing = component_totals(_events(rdir), min_update=lo, max_update=hi)
    need = REQUIRED[family] - {"prefix", "cache_build"}
    if any(m.get("component") in need for m in missing) or not tot:
        return None
    return views(tot, family)["continuation_algorithm"]


def physical_cost(root, run_id):
    """All GPU time the run spent, restarted attempts included; None if any interval is unknown."""
    from switching.ledger import physical_gpu_seconds
    from switching.runner import run_starts
    rdir = Path(root) / "runs" / (run_id or "")
    if not run_id or not rdir.is_dir():
        return None
    return physical_gpu_seconds(_events(rdir, lineage=False), run_starts(rdir))["gpu_seconds"]


def refresh_cost(root, run_id, lo, hi):
    """Mean on_selection GPU-seconds per selection round in [lo, hi)."""
    from switching.ledger import component_totals
    rdir = Path(root) / "runs" / (run_id or "")
    if not run_id or not rdir.is_dir():
        return None
    events = _events(rdir)
    tot, missing = component_totals(events, min_update=lo, max_update=hi)
    rounds = {e["completed_update"] for e in events
              if e.get("component") == "on_selection" and e.get("attribution") != "exclusive_stage"
              and e.get("completed_update") is not None and lo <= e["completed_update"] < hi}
    if not rounds or any(m.get("component") == "on_selection" for m in missing):
        return None
    return tot.get("on_selection", 0.0) / len(rounds)


def receipts(cohort, seed):
    """(prefix GPU-s, cache GPU-s, cache detail) of the seed: V8 receipts, else the V9 copy (Llama)."""
    from switching.cluster import cache_cost_from_receipts, prefix_cost_from_receipts
    pre = cache = None
    detail = {}
    for root in (v8c(cohort), v9c(cohort)):
        if not (root / "locator.json").is_file():
            continue
        d = {}
        try:
            pre = pre if pre is not None else prefix_cost_from_receipts(root, seed)
            c = cache_cost_from_receipts(root, seed, d)
        except Exception as exc:  # noqa: BLE001 - unknown stays unknown, with its reason
            c, d = None, {"status": f"{type(exc).__name__}: {exc}"}
        if cache is None:
            cache, detail = c, d
    if not detail:
        detail = {"status": "no locator.json for this cohort"}
    return pre, cache, detail


def trajectory(cohort, seed, items):
    """(root, run_id) of the On trajectory of a seed: V8 bank trajectory, else the V9 Alignment run (Llama)."""
    if cohort in env.BANK_COHORTS:
        try:
            from .common import bank
            b = bank(cohort, seed)
            if b.traj is not None:
                return v8c(cohort), b.traj["run_id"]
        except Exception:  # noqa: BLE001
            return None, None
        return None, None
    it = next((i for i in items if i.get("arm") == "alignment" and i.get("seed") == seed and i.get("run_id")), None)
    return (v9c(cohort), it["run_id"]) if it else (None, None)


def _trained(items, cohort, seed, arm):
    """V9-trained run of an arm whose training finished: its run_id, else None."""
    for i in items:
        if (i["cohort"], i.get("seed"), i.get("arm")) == (cohort, seed, arm) and str(i["key"]).endswith(".train") \
                and i["state"] == "done":
            return i.get("run_id")
    return None


def x1_costs(tau, items):
    """{cohort: {seed: {arm: incremental GPU-s | None, ..., per_refresh, prefix, cache, cold_start{arm}}}}."""
    out = {}
    for cohort in env.X1_COHORTS:
        its = [i for i in items if i["cohort"] == cohort]
        try:
            c = cfg(cohort)
            P, H = c["training"]["prefix_updates"], c["training"]["matched_update_endpoint"]
        except Exception:  # noqa: BLE001 - cohort not configured here: everything pending
            P, H = 25, 275
        rows = {}
        for seed in (5, 6, 7, 8, 9):
            r = {"incremental": {}, "physical": {}}
            for arm in ("random", "alignment", "sr", "staged"):
                bank_it = next((i for i in its if i.get("seed") == seed and i.get("arm") == arm
                                and i["key"] == f"s{seed}.{arm}"), None)
                if bank_it is not None:  # V8 branched run, read-only (bank policy row cost)
                    r["incremental"][arm] = bank_it.get("cost") if bank_it["state"] == "done" else None
                    continue
                rid = _trained(its, cohort, seed, arm)
                root = v9c(cohort)
                if rid is None:
                    r["incremental"][arm] = None
                elif arm == "staged":
                    troot, trid = trajectory(cohort, seed, its)
                    a = run_cost(troot, trid, hi=tau) if trid else None
                    b = run_cost(root, rid, hi=H)
                    r["incremental"][arm] = None if a is None or b is None else a + b
                    r["physical"][arm] = physical_cost(root, rid)
                else:
                    r["incremental"][arm] = run_cost(root, rid, hi=H)
                    r["physical"][arm] = physical_cost(root, rid)
            troot, trid = trajectory(cohort, seed, its)
            r["per_refresh_alignment"] = refresh_cost(troot, trid, P, H) if trid else None
            pre, cache, detail = receipts(cohort, seed)
            r.update(prefix=pre, cache=cache, cache_detail=detail)
            r["cold_start"] = {}
            for arm, v in r["incremental"].items():
                parts = [v, pre] + ([cache] if arm in CACHE_ARMS else [])
                r["cold_start"][arm] = None if any(p is None for p in parts) else sum(parts)
            rows[seed] = r
        out[cohort] = rows
    return out
