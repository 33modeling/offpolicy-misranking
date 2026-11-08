"""Regression tests for the 2026-11-08 science audit (SCI-01 .. SCI-18).

Each test reproduces the audited failure on small synthetic inputs whose correct answer is known by hand.
Software checks only; no GPU, no paper results.
"""

import contextlib
import io
import json
import os
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("V8_FSYNC", "0")
from helpers import HAS_CODE, tmpdir, toy_setup  # noqa: E402
from switching.common import atomic_write_json  # noqa: E402

H = 275
CFG = {"training": {"matched_update_endpoint": H, "check_interval": 25, "first_check": 25},
       "plateau": {"window_updates": 25, "consecutive_checks": 2, "check_interval": 25}}
SIG = "srgc-40v40-validation-dot-v1"


def manifest(root, rid, mid, parent, *, seed=5, role="rebuttal_bank", status="completed_horizon", family="x",
             attempts=(("a", 25),), **task):
    atomic_write_json(Path(root) / "runs" / rid / "manifest.json", {
        "run_id": rid, "cohort_id": "c", "training_seed": seed, "role": role, "method_id": mid, "status": status,
        "config_hash": "h", "attempts": [{"attempt_id": a, "start_update": s} for a, s in attempts],
        "task": {"run_id": rid, "method_id": mid, "training_seed": seed, "role": role,
                 "method": {"family": family, **task.pop("method", {})}, "parent": parent, **task}})


def events(root, rid, rows, attempt="a"):
    d = Path(root) / "runs" / rid / "attempts" / attempt
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "events.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({"run_id": rid, "attempt_id": attempt, "attribution": "interval", **r}) + "\n")


def monitored_trajectory_events(rid):
    """25..250: On refresh 10 GPU-s + SR-GC comparison stage 4 GPU-s per check; 1 GPU-s per training update."""
    rows = [{"event_id": f"t{u}", "completed_update": u, "source_phase": "training", "component": "training",
             "gpu_seconds": 1.0} for u in range(25, H)]
    for u in range(25, 251, 25):
        rows.append({"event_id": f"s{u}", "completed_update": u, "source_phase": "selection",
                     "component": "on_selection", "gpu_seconds": 14.0})
        rows.append({"event_id": f"s{u}:sr", "completed_update": u, "parent_event_id": f"s{u}",
                     "component": "sr_comparison", "gpu_seconds": 4.0, "attribution": "exclusive_stage"})
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class SCI01RefreshCost(unittest.TestCase):
    def test_views_count_refresh_for_refreshed_families(self):
        from switching.ledger import views
        v = views({"training": 250.0, "sr_preparation": 5.0, "cache_refresh": 300.0}, "sr_refreshed")
        self.assertEqual(v["continuation_algorithm"], 555.0)
        self.assertEqual(views({"training": 10.0, "cache_refresh": 3.0}, "srgc")["continuation_algorithm"], 13.0)

    @unittest.skipUnless(HAS_CODE, "code repository not available")
    def test_engine_meters_refresh_as_cache_refresh(self):
        from switching.engine import fork_arm_for
        from switching.ledger import Ledger, component_totals, views
        from switching.methods import method
        from switching.policies import build_policy
        srgc, PE, data, cfg, backend, prefix = toy_setup()
        d = tmpdir()
        led = Ledger(d / "events.jsonl", run_id="r", attempt_id="a")
        m = method("sr_early_refreshed_k050")
        e = PE(backend(record=led.on_meter_event), data["candidate_ids"], data["ranking_validation_ids"],
               data["cached_rewards"], method=m, policy=build_policy(m["policy"]), config=cfg)
        e.load_state_dict(prefix, fork_arm=fork_arm_for(m))
        e.run_until(76)  # entry 25, K 50 -> refresh at 75
        self.assertEqual([c["event"] for c in e.cache_log], ["committed"])
        rows = led.log.rows()
        refresh = [r for r in rows if r["component"] == "cache_refresh" and r.get("attribution") != "exclusive_stage"]
        self.assertEqual([r["completed_update"] for r in refresh], [75])
        self.assertFalse([r for r in rows if r["component"] == "cache_build"])
        tot, missing = component_totals(rows)
        self.assertFalse(missing)
        v = views(tot, "sr_refreshed")
        self.assertAlmostEqual(v["continuation_algorithm"],
                               tot["training"] + tot.get("sr_preparation", 0.0) + tot["cache_refresh"])
        self.assertGreater(tot["cache_refresh"], 0.0)


class SCI02SeedRoles(unittest.TestCase):
    class Loc:
        def __init__(self, roles=None):
            self.roles, self.interval = roles or {}, 25

        def prefix(self, seed):
            return {"kind": "prefix", "path": f"/x/{seed}.pt", "sha256": "ab" * 32, "completed_update": 25,
                    "run_id": None}

        def role_seeds(self, role):
            return self.roles.get(role)

    def cfg(self):
        from switching import spec
        return spec.resolve("olmo_math", role="test")

    def test_test_role_never_defaults_to_inspected_seeds(self):
        from switching import matrix as mx
        with self.assertRaises(ValueError):  # no prepare-seeds roles, no --seeds
            mx.build_stage("prospective", self.cfg(), self.Loc(), tmpdir(), role="test", prospective_methods=["sr_early"])
        tasks, _ = mx.build_stage("prospective", self.cfg(), self.Loc({"test": [10, 11], "development": [15]}),
                                  tmpdir(), role="test", prospective_methods=["sr_early"])
        self.assertEqual(sorted({t["training_seed"] for t in tasks}), [10, 11])
        with self.assertRaises(ValueError):  # inspected seed as a fresh test seed
            mx.build_stage("prospective", self.cfg(), self.Loc(), tmpdir(), role="test", seeds=[5],
                           prospective_methods=["sr_early"])
        with self.assertRaises(ValueError):  # development seed overlapping the test assignment
            mx.build_stage("prospective", self.cfg(), self.Loc({"test": [10]}), tmpdir(), role="development",
                           seeds=[10], prospective_methods=["sr_early"])
        with self.assertRaises(ValueError):
            mx.build_stage("prefix", self.cfg(), self.Loc(), tmpdir(), role="development")

    def test_cluster_locator_reads_prepare_seeds_roles(self):
        from switching.cluster import ClusterLocator
        d = tmpdir()
        atomic_write_json(d / "locator.json", {"seeds": {}, "seed_roles": {"test": [10, 11, 12], "development": [15]}})
        loc = ClusterLocator(d, self.cfg())
        self.assertEqual(loc.role_seeds("test"), [10, 11, 12])
        atomic_write_json(d / "locator.json", {"seeds": {}})
        self.assertIsNone(ClusterLocator(d, self.cfg()).role_seeds("test"))


def prospective_run(root, rid, mid, *, family, role="test", seed=10, gpu=None, aliases=None, status="completed_horizon",
                    transition=None, no_switch=None):
    atomic_write_json(Path(root) / "runs" / rid / "manifest.json", {
        "run_id": rid, "cohort_id": "c", "training_seed": seed, "role": role, "method_id": mid, "status": status,
        "attempts": [{"attempt_id": "a1", "start_update": 25}], "transition_update": transition, "no_switch": no_switch,
        "task": {"method": {"family": family}, "parent": {"kind": "prefix"}, **({"aliases": aliases} if aliases else {})}})
    events(root, rid, [{"event_id": f"t{u}", "completed_update": u, "source_phase": "training", "component": "training",
                        "gpu_seconds": (gpu or {}).get(u, 1.0),
                        "measurement_status": "interrupted" if (gpu or {}).get(u, 1.0) is None else "measured"}
                       for u in range(25, H)], attempt="a1")


class SCI03ProspectiveUnknownCost(unittest.TestCase):
    def test_unmeasured_event_makes_cold_start_unknown(self):
        from switching.bank import prospective_rows
        d = tmpdir()
        prospective_run(d, "c.test.s10.on_continued.x", "on_continued", family="on_continued", gpu={100: None})
        r = prospective_rows(d, CFG, "c", "test", [10], evals={}, prefix_cost=100.0, cache_cost=50.0)[0]
        self.assertIsNone(r["continuation_gpu_seconds"])
        self.assertIsNone(r["cold_start_gpu_seconds"])                # was 349.0 presented as complete
        self.assertEqual(r["cold_start_lower_bound_gpu_seconds"], 349.0)
        self.assertIn("unmeasured_events", r["cost_missing"])

    def test_measured_run_unchanged(self):
        from switching.bank import prospective_rows
        d = tmpdir()
        prospective_run(d, "c.test.s10.on_continued.x", "on_continued", family="on_continued")
        r = prospective_rows(d, CFG, "c", "test", [10], evals={}, prefix_cost=100.0, cache_cost=50.0)[0]
        self.assertEqual((r["continuation_gpu_seconds"], r["cold_start_gpu_seconds"]), (250.0, 350.0))


class SCI04Aliases(unittest.TestCase):
    def evals(self, rid, value):
        return {(rid, H, "final"): {"reward_fraction": value, "n_questions": 1, "responses_per_question": 8}}

    def test_alias_rows_from_manifest_and_matrix(self):
        from switching.analyze import by_policy, paired
        from switching.bank import prospective_rows
        d = tmpdir()
        prospective_run(d, "sr", "sr_early", family="sr_early", aliases=["fixed_sr_tuned"])
        prospective_run(d, "on", "on_continued", family="on_continued", no_switch=True)
        prospective_run(d, "sw", "srgc_current", family="srgc", transition=50)
        # the multi-node coordinator strips aliases from tasks: the stage matrix still has them
        atomic_write_json(d / "matrix" / "prospective.test.json",
                          {"tasks": [{"run_id": "on", "method_id": "on_continued", "aliases": ["sampled_time_sr"]}]})
        ev = {**self.evals("sr", .40), **self.evals("on", .30), **self.evals("sw", .45)}
        rows = prospective_rows(d, CFG, "c", "test", [10], evals=ev)
        t = by_policy(rows)
        self.assertEqual(t["fixed_sr_tuned"], {10: .40})
        self.assertEqual(t["sampled_time_sr"], {10: .30})   # no-switch draw kept (its outcome = On continued)
        self.assertEqual({r["policy_id"]: r.get("alias_of") for r in rows}["fixed_sr_tuned"], "sr_early")
        self.assertEqual(paired(t["srgc_current"], t["fixed_sr_tuned"], [10])["n_training_seeds"], 1)


class SCI05CadenceCost(unittest.TestCase):
    def bank_root(self):
        d = tmpdir()
        traj = "c.rebuttal_bank.s5.on_monitored.t"
        manifest(d, traj, "on_monitored", {"kind": "prefix", "sha256": "p", "completed_update": 25})
        events(d, traj, monitored_trajectory_events(traj))
        write_jsonl(d / "runs" / traj / "signals.jsonl",
                    [{"completed_update": s, "signal_name": "srgc_d", "signal_definition_version": SIG,
                      "value": -1.0 if s in (50, 75, 100) else 1.0, "valid": True} for s in range(25, 251, 25)])
        write_jsonl(d / "runs" / traj / "training_stats.jsonl",
                    [{"completed_update": u, "reward_sum": 16.0, "valid_response_count": 32} for u in range(1, H + 1)])
        for tau in (75, 100):
            rid = f"c.rebuttal_bank.s5.bank_sr_t{tau:03d}.b"
            manifest(d, rid, f"bank_sr_t{tau:03d}", {"kind": "snapshot", "run_id": traj, "completed_update": tau},
                     attempts=(("a", tau),))
            events(d, rid, [{"event_id": f"b{u}", "completed_update": u, "source_phase": "training",
                             "component": "training", "gpu_seconds": 1.0} for u in range(tau, H)])
        return d, traj

    def test_component_filter(self):
        from switching.ledger import check_schedule_filter, component_totals, views
        ev = [dict(e, run_id="t", attempt_id="a", attribution=e.get("attribution", "interval"))
              for e in monitored_trajectory_events("t")]
        v = lambda keep: views(component_totals(ev, max_update=100, include_selection_at=100, keep=keep)[0],  # noqa: E731
                               "srgc")["continuation_algorithm"]
        self.assertEqual(v(None), 131.0)                          # 75 training + 4x10 On + 4x4 SR-GC (cadence 25)
        self.assertEqual(v(check_schedule_filter(50, 50)), 123.0)  # cadence 50 runs SR-GC at 50 and 100 only

    def test_policy_rows_charge_c50_only_its_checks(self):
        from switching.bank import SeedBank
        d, _ = self.bank_root()
        rows = {r["policy_id"]: r for r in SeedBank(d, CFG, "c", 5, "rebuttal_bank", evals={}).policy_rows()}
        self.assertEqual((rows["srgc_current"]["switch_update"], rows["srgc_current_c50"]["switch_update"]), (75, 100))
        self.assertEqual(rows["srgc_current_c50"]["continuation_gpu_seconds"], 123.0 + 175.0)  # was 131 + 175
        self.assertEqual(rows["srgc_current"]["continuation_gpu_seconds"], 50 + 30 + 12 + 200.0)


class SCI06TrajectoryIdentity(unittest.TestCase):
    def test_two_trajectories_refused_and_foreign_branch_ignored(self):
        from switching.bank import SeedBank
        d = tmpdir()
        manifest(d, "c.rebuttal_bank.s5.on_monitored.aaaa", "on_monitored", {"kind": "prefix", "sha256": "p"})
        manifest(d, "c.rebuttal_bank.s5.on_monitored.bbbb", "on_monitored", {"kind": "prefix", "sha256": "p"})
        with self.assertRaises(ValueError):
            SeedBank(d, CFG, "c", 5, "rebuttal_bank", evals={})
        shutil.rmtree(d / "runs" / "c.rebuttal_bank.s5.on_monitored.aaaa")
        manifest(d, "c.rebuttal_bank.s5.bank_sr_t050.1", "bank_sr_t050",
                 {"kind": "snapshot", "run_id": "c.rebuttal_bank.s5.on_monitored.bbbb", "completed_update": 50})
        manifest(d, "c.rebuttal_bank.s5.bank_sr_t100.2", "bank_sr_t100",
                 {"kind": "snapshot", "run_id": "c.rebuttal_bank.s5.on_monitored.aaaa", "completed_update": 100})
        manifest(d, "c.rebuttal_bank.s5.bank_sr_t025.3", "bank_sr_t025", {"kind": "prefix", "sha256": "p"})
        b = SeedBank(d, CFG, "c", 5, "rebuttal_bank", evals={})
        self.assertEqual(sorted(b.branches), [("sr", 25), ("sr", 50)])


class SCI07AnalysisCoverage(unittest.TestCase):
    def test_budget_key_uses_selected_endpoints_only(self):
        from switching.analyze import by_policy
        rows = [{"policy_id": "a", "seed": 5, "status": "triggered", "budget_status": "selected", "budget_reward_fraction": .5},
                {"policy_id": "a", "seed": 6, "status": "triggered", "budget_status": "capped_at_horizon",
                 "budget_reward_fraction": .6}]
        self.assertEqual(by_policy(rows, "budget_reward_fraction"), {"a": {5: .5}})

    def test_test_rows_pair_tuned_sampled_refreshed_and_show_missing(self):
        from switching.analyze import secondary
        rows = [{"policy_id": p, "seed": s, "status": "ran", "reward_fraction": v, "switch_update": None}
                for s in (10, 11) for p, v in (("srgc_current", .5), ("plateau_sr_tuned", .4), ("sampled_time_sr", .45),
                                               ("sr_early", .41), ("sr_early_refreshed_k050", .42))]
        out = {r["label"]: r for r in secondary(rows, [10, 11])}
        for label in ("srgc_current - plateau_sr_tuned", "srgc_current - sampled_time_sr",
                      "sr_early_refreshed_k050 - sr_early"):
            self.assertEqual(out[label]["n_training_seeds"], 2, label)
        self.assertEqual(out["srgc_current - uniform_early"]["status"], "missing_policy")

    def test_cmd_analyze_reports_budget_and_physical(self):
        from switching import cluster
        d = tmpdir()
        root = d / "olmo_math"
        rows = []
        for s in (5, 6, 7, 8, 9):
            for t, v in ((50, .40), (100, .41), (150, .39), (200, .38)):
                rows.append({"policy_id": f"fixed_sr_t{t:03d}", "seed": s, "status": "triggered", "reward_fraction": v,
                             "budget_status": "selected", "budget_reward_fraction": v - .01})
            rows.append({"policy_id": "srgc_current", "seed": s, "status": "triggered", "reward_fraction": .42,
                         "switch_update": 50, "budget_status": "selected", "budget_reward_fraction": .415})
        atomic_write_json(root / "analysis" / "policy_outcomes_bank.json", rows)
        atomic_write_json(root / "frozen" / "budget.json", {"value": 1.0, "unit": "gpu_seconds", "scope": "c", "n_seeds": 5})
        os.environ["V8_ALLOW_ANY_ROOT"] = "1"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cluster.cmd_analyze(SimpleNamespace(cohort="olmo_math", v8_root=str(d)))
        finally:
            os.environ.pop("V8_ALLOW_ANY_ROOT", None)
        an = json.loads((root / "analysis" / "analysis.json").read_text())
        self.assertEqual(an["primary_rebuttal_budget"]["primary_contrast"]["n_training_seeds"], 5)
        self.assertIn("physical_expenditure", an)
        self.assertEqual(an["endpoints"]["matched_update"], 275)


class SCI08BudgetCap(unittest.TestCase):
    def test_horizon_stop_is_capped_not_selected(self):
        from switching.bank import capped
        from switching.budget import select_endpoint
        snaps = [{"completed_update": 250, "cumulative_cost": 90.0}, {"completed_update": 275, "cumulative_cost": 100.0}]
        self.assertEqual(capped(select_endpoint(snaps, 120.0), snaps, stopped_by_budget=False)["status"],
                         "capped_at_horizon")
        self.assertEqual(capped(select_endpoint(snaps, 120.0), snaps, stopped_by_budget=True)["status"], "selected")
        self.assertEqual(capped(select_endpoint(snaps, 95.0), snaps, stopped_by_budget=False)["status"], "selected")

    def test_prospective_horizon_run_is_capped(self):
        from switching.bank import prospective_rows
        d = tmpdir()
        prospective_run(d, "u", "uniform_early", family="uniform_early")
        write_jsonl(d / "runs" / "u" / "snapshots.jsonl",
                    [{"completed_update": u, "own_continuation_gpu_seconds": float(u - 25), "sha256": str(u)}
                     for u in (250, 275)])
        r = prospective_rows(d, CFG, "c", "test", [10], evals={}, budget=1000.0)[0]
        self.assertEqual(r["budget_status"], "capped_at_horizon")

    def test_uniform_branches_are_extended(self):
        from switching import smoke
        d = tmpdir()
        traj = "c.rebuttal_bank.s5.on_monitored.t"
        manifest(d, traj, "on_monitored", {"kind": "prefix", "sha256": "p", "completed_update": 25})
        events(d, traj, monitored_trajectory_events(traj))
        rid = "c.rebuttal_bank.s5.bank_uniform_t050.u"
        manifest(d, rid, "bank_uniform_t050", {"kind": "snapshot", "run_id": traj, "completed_update": 50},
                 attempts=(("a", 50),), method={"initial_selector": "uniform"})
        events(d, rid, [{"event_id": f"b{u}", "completed_update": u, "source_phase": "training",
                         "component": "training", "gpu_seconds": 1.0} for u in range(50, H)])
        cfg = dict(CFG, cohort_id="c")
        loc = smoke.Locator(d, cfg, 25)
        runs = loc.completed_runs(5, "rebuttal_bank")
        self.assertEqual([r["run_id"] for r in runs], [rid])
        # fixed_uniform at 50: training 25..49 (25) + On refreshes at 25 (10) + uniform branch 225 updates
        self.assertEqual(loc.cost_offset(runs[0]), 25 + 10 + 225.0)


class SCI09AuditSwitch(unittest.TestCase):
    def endpoint_root(self, switched_at):
        d = tmpdir() / "seed-7"
        d.mkdir(parents=True)
        atomic_write_json(d / "prefix-ready.json", {"checkpoint_sha256": "p"})
        per_q = {f"q{i}": (1.0 if i % 4 == 0 else 0.0) for i in range(300)}
        atomic_write_json(d / "switch-endpoint.json", {"per_question_reward": per_q, "reward": .25, "total_updates": 275,
                                                       "switched_at": switched_at, "prefix_checkpoint_sha256": "p",
                                                       "cost_measurement_complete": True})
        return d.parent

    def test_never_switched_and_unreported_switch_times(self):
        from switching.audit import audit_cell
        cell = {"cohort": "gemma_mbpp", "seed": 7, "arm": "switch", "reward_fraction": .25,
                "precision": "one_decimal_percent", "switch_update": None, "never_switched": True}
        self.assertEqual(audit_cell(cell, self.endpoint_root(100))["status"], "inconsistent")
        self.assertEqual(audit_cell(cell, self.endpoint_root(None))["status"], "verified")
        counts = dict(cell, cohort="olmo_math", precision="exact_counts", never_switched=False)
        self.assertEqual(audit_cell(counts, self.endpoint_root(50))["status"], "incomplete_provenance")


class SCI10TauDistribution(unittest.TestCase):
    def test_failed_dev_run_is_not_no_switch(self):
        from switching.bank import tau_distribution
        ran = [{"policy_id": "srgc_current", "role": "development", "seed": 15, "status": "ran", "switch_update": 50},
               {"policy_id": "srgc_current", "role": "development", "seed": 16, "status": "ran", "switch_update": None,
                "no_switch": True}]
        d = tau_distribution(ran)
        self.assertEqual({x["switch_update"]: x["probability"] for x in d["support"]}, {50: .5, None: .5})
        with self.assertRaises(ValueError):
            tau_distribution(ran + [{"policy_id": "srgc_current", "role": "development", "seed": 17, "status": "failed",
                                     "switch_update": None}])


class SCI11to12BudgetFreeze(unittest.TestCase):
    def row(self, s, **kw):
        return {"policy_id": "srgc_current", "seed": s, "status": "triggered", "continuation_gpu_seconds": 100.0,
                "cold_start_gpu_seconds": None, "run_id": f"r{s}", "switch_update": 50, **kw}

    def test_complete_cold_start_scope_refused(self):
        from switching.freeze import freeze_budget
        with self.assertRaises(ValueError):
            freeze_budget([self.row(5, cold_start_gpu_seconds=150.0), self.row(6, cold_start_gpu_seconds=150.0)])
        self.assertEqual(freeze_budget([self.row(5), self.row(6)])["scope"], "continuation_algorithm (cold-start incomplete)")

    def test_unobserved_or_absent_seed_refused(self):
        from switching.freeze import freeze_budget
        with self.assertRaises(ValueError):
            freeze_budget([self.row(5), self.row(6), {"policy_id": "srgc_current", "seed": 9, "status": "unavailable"}])
        with self.assertRaises(ValueError):
            freeze_budget([self.row(5), self.row(6)], expected_seeds=[5, 6, 7])


class SCI13Report(unittest.TestCase):
    def test_significance_blocked_and_one_seed_sentence(self):
        from switching.analyze import paired
        from switching.report import guard, primary_sentence
        for text in ("SR-GC switching is statistically significant (p < 0.05).",
                     "SR-GC significantly outperforms tuned fixed.", "The difference is significant across seeds."):
            with self.assertRaises(ValueError):
                guard(text, costs_partial=True, any_loss=True)
        c = paired({10: .5, 11: None}, {10: .4, 11: .4}, [10, 11])
        self.assertIn("not computed (n=1)", primary_sentence(c, budget="B", unit="GPU-h", scope_components="x",
                                                             missing_summary="[11]", costs_partial=True))


class SCI14FreezeOrder(unittest.TestCase):
    def test_freeze_after_evaluations_refused(self):
        from switching import cluster
        d = tmpdir()
        atomic_write_json(d / "olmo_math" / "evaluations" / "x.json", {"reward_fraction": .5})
        os.environ["V8_ALLOW_ANY_ROOT"] = "1"
        try:
            with self.assertRaises(SystemExit):
                cluster.cmd_freeze_budget(SimpleNamespace(cohort="olmo_math", v8_root=str(d)))
        finally:
            os.environ.pop("V8_ALLOW_ANY_ROOT", None)


class SCI15LosoInterval(unittest.TestCase):
    def test_naive_interval_reported_with_caveat(self):
        from switching.analyze import loso_contrast
        rows = [{"policy_id": f"fixed_sr_t{t:03d}", "seed": s, "status": "triggered", "reward_fraction": v + s / 1000}
                for s in (5, 6, 7, 8, 9) for t, v in ((50, .40), (100, .41))]
        rows += [{"policy_id": "srgc_current", "seed": s, "status": "triggered", "reward_fraction": .42 + s / 500}
                 for s in (5, 6, 7, 8, 9)]
        c = loso_contrast(rows, [5, 6, 7, 8, 9], taus=[50, 100])["primary_contrast"]
        self.assertIsNone(c["ci_low"])                       # not the primary interval
        self.assertEqual(c["naive_paired_t"]["df"], 4)
        self.assertIsNotNone(c["naive_paired_t"]["ci_low"])
        self.assertNotIn("naive_iid_t_not_valid", c)


class SCI16InflightLineage(unittest.TestCase):
    def test_discarded_attempt_never_wins(self):
        from switching.matrix import monitored_observations
        d = tmpdir()
        rd = d / "runs" / "traj"

        def sig(u, v):
            return {"completed_update": u, "signal_name": "srgc_d", "signal_definition_version": SIG, "value": v,
                    "valid": True}
        write_jsonl(rd / "attempts" / "f000" / "updates.jsonl",
                    [{"completed_updates": 26, "signals": [sig(25, -1.0)]}, {"completed_updates": 51, "signals": [sig(50, -1.0)]}])
        write_jsonl(rd / "attempts" / "0aaa" / "updates.jsonl", [{"completed_updates": 51, "signals": [sig(50, 1.0)]}])
        atomic_write_json(rd / "manifest.json", {"attempts": [{"attempt_id": "f000", "start_update": 25},
                                                             {"attempt_id": "0aaa", "start_update": 40}]})
        self.assertEqual(monitored_observations(d, "traj"), [(25, -1.0), (50, 1.0)])
        atomic_write_json(rd / "manifest.json", {"attempts": []})
        with self.assertRaises(ValueError):  # no lineage information: a conflict is an error, not file order
            monitored_observations(d, "traj")


@unittest.skipUnless(HAS_CODE, "code repository not available")
class SCI17Bidirectional(unittest.TestCase):
    def test_v8_engine_refuses_bidirectional(self):
        from switching.methods import method
        from switching.policies import build_policy
        srgc, PE, data, cfg, backend, prefix = toy_setup()
        m = method("switch_repeat_external")
        with self.assertRaises(ValueError):
            PE(backend(), data["candidate_ids"], data["ranking_validation_ids"], data["cached_rewards"], method=m,
               policy=build_policy(m["policy"]), config=cfg)


class SCI18SameTauAndObjective(unittest.TestCase):
    def test_destination_contrast_requires_equal_tau(self):
        from switching.analyze import secondary
        rows = [{"policy_id": "srgc_current", "seed": s, "status": "ran", "reward_fraction": .5, "switch_update": 50}
                for s in (10, 11)]
        rows += [{"policy_id": "srgc_uniform_destination", "seed": 10, "status": "ran", "reward_fraction": .4,
                  "switch_update": 50},
                 {"policy_id": "srgc_uniform_destination", "seed": 11, "status": "ran", "reward_fraction": .4,
                  "switch_update": 75}]
        r = {x["label"]: x for x in secondary(rows, [10, 11])}["srgc_current - srgc_uniform_destination"]
        self.assertEqual((r["n_training_seeds"], r["tau_mismatch_seeds"]), (1, [11]))

    def test_tuning_records_its_objective(self):
        from switching.tune import select_dev
        rows = [{"role": "development", "tau": 50, "reward_fraction": .4}]
        self.assertIn("matched_update", select_dev(rows, candidates=[50], key="tau", tie_tolerance_pp=.1)["objective_endpoint"])


class PlateauStatsIncomplete(unittest.TestCase):
    def test_missing_training_stats_is_censored_not_not_triggered(self):
        from switching.bank import SeedBank
        d, traj = SCI05CadenceCost().bank_root()
        (d / "runs" / traj / "training_stats.jsonl").unlink()
        rows = {r["policy_id"]: r for r in SeedBank(d, CFG, "c", 5, "rebuttal_bank", evals={}).policy_rows()}
        self.assertEqual(rows["plateau_sr_eps0005"]["status"], "censored")


if __name__ == "__main__":
    unittest.main()
