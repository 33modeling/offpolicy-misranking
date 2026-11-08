import unittest

from helpers import HAS_CODE, tmpdir
from switching.budget import common_target_cost, select_endpoint, should_continue
from switching.ledger import (Ledger, component_totals, lineage_events, physical_gpu_seconds, union_seconds,
                              views)

S = 1_000_000_000


def ev(eid, gpus, start, end, **kw):
    return {"run_id": "r", "attempt_id": kw.pop("attempt", "a"), "event_id": eid, "gpu_ids": gpus,
            "start_monotonic_ns": start * S, "end_monotonic_ns": end * S, "gpu_seconds": kw.pop("gs", None),
            "measurement_status": kw.pop("status", "measured"), **kw}


class Ledgers(unittest.TestCase):
    def test_plan_example_union(self):
        events = [ev("A", ["g0", "g1"], 0, 10, gs=20.0), ev("B", ["g0", "g1"], 5, 15, gs=20.0)]
        r = physical_gpu_seconds(events)
        self.assertEqual(r["gpu_seconds"], 30.0)  # not 40, not 60
        self.assertTrue(r["complete"])

    def test_dedupe_event_ids(self):
        e = ev("A", ["g0"], 0, 10, gs=10.0)
        self.assertEqual(physical_gpu_seconds([e, dict(e)])["gpu_seconds"], 10.0)

    def test_codex_unknown_cost_is_incomplete(self):
        r = physical_gpu_seconds([{"run_id": "r", "attempt_id": "a", "event_id": "x", "gpu_seconds": None,
                                   "measurement_status": "measured", "gpu_ids": ["h:0"]}])
        self.assertFalse(r["complete"])
        self.assertIsNone(r["gpu_seconds"])
        r = physical_gpu_seconds([dict(ev("i", [], 0, 1, status="interrupted"), end_monotonic_ns=None)])
        self.assertFalse(r["complete"])

    def test_codex_amortized_unknown(self):
        v = views({"training": 10.0, "on_selection": 0.0}, "srgc", amortize_cache_over=4)
        self.assertIsNone(v["declared_amortized"])
        self.assertEqual(v["declared_amortized_lower_bound"], 10.0)
        self.assertIsNone(v["cold_start_algorithm"])
        v = views({"training": 10.0}, "srgc", prefix_gpu_seconds=2.0, cache_gpu_seconds=8.0, amortize_cache_over=4)
        self.assertEqual(v["declared_amortized"], 14.0)
        self.assertEqual(v["cold_start_algorithm"], 20.0)

    def test_on_only_excludes_unneeded_cache_and_sr_comparison(self):
        v = views({"training": 10.0, "on_selection": 5.0, "sr_comparison": 3.0}, "on_continued", prefix_gpu_seconds=1.0)
        self.assertEqual(v["continuation_algorithm"], 15.0)
        self.assertEqual(v["cold_start_algorithm"], 16.0)
        v = views({"training": 10.0, "on_selection": 5.0, "sr_comparison": 3.0}, "srgc", prefix_gpu_seconds=1.0,
                  cache_gpu_seconds=2.0)
        self.assertEqual(v["cold_start_algorithm"], 21.0)

    def test_component_window_and_trigger_check(self):
        rows = [{"run_id": "r", "attempt_id": "a", "event_id": f"s{u}", "source_phase": "selection", "component": "on_selection",
                 "completed_update": u, "gpu_seconds": 1.0} for u in (25, 50, 75)] + \
               [{"run_id": "r", "attempt_id": "a", "event_id": f"t{u}", "source_phase": "training", "component": "training",
                 "completed_update": u, "gpu_seconds": 0.1} for u in range(25, 80)]
        tot, _ = component_totals(rows, max_update=50)
        self.assertAlmostEqual(tot["on_selection"], 1.0)  # refresh at 25 only
        self.assertAlmostEqual(tot["training"], 2.5)
        tot, _ = component_totals(rows, max_update=50, include_selection_at=50)
        self.assertAlmostEqual(tot["on_selection"], 2.0)  # + the triggering check at 50

    def test_lineage_excludes_redone_work(self):
        rows = [{"event_id": "1", "attempt_id": "a1", "completed_update": 30}, {"event_id": "2", "attempt_id": "a1", "completed_update": 61},
                {"event_id": "3", "attempt_id": "a2", "completed_update": 61}, {"event_id": "4", "attempt_id": "a1", "completed_update": None}]
        keep = lineage_events(rows, [("a1", 25), ("a2", 60)])
        self.assertEqual([r["event_id"] for r in keep], ["1", "3", "4"])

    def test_union_helper(self):
        self.assertEqual(union_seconds([(0, 10), (5, 15), (20, 25)]), 20)

    @unittest.skipUnless(HAS_CODE, "code repository not available")
    def test_meter_bridge_multi_gpu_and_stage_mapping(self):
        from switching import coderepo
        coderepo.activate()
        from srgc_rebuttal.timing import CostMeter
        d = tmpdir()
        led = Ledger(d / "e.jsonl", run_id="r", attempt_id="a", gpu_ids=["h:0", "h:1", "h:2", "h:3"])
        clock = iter([0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0]).__next__
        meter = CostMeter(rank=0, local_gpu_count=4, record=led.on_meter_event, clock=clock)
        meter.begin_phase("selection", 25, 4)
        with meter.section("sr_comparison"), meter.stage("scoring"):
            pass
        with meter.stage("cosine_ranking"):
            pass
        meter.end_phase()
        rows = led.log.rows()
        parent = [r for r in rows if r["attribution"] == "interval"][0]
        self.assertEqual(parent["gpu_count"], 4)
        comps = {r["component"] for r in rows if r["attribution"] == "exclusive_stage"}
        self.assertIn("sr_comparison", comps)
        self.assertIn("on_selection", comps)
        tot, _ = component_totals(rows)
        self.assertAlmostEqual(sum(tot.values()), parent["gpu_seconds"])

    def test_interrupted_phase_is_unknown(self):
        d = tmpdir()
        led = Ledger(d / "e.jsonl", run_id="r", attempt_id="a")
        led.on_meter_event({"id": "p1", "phase": "training", "checkpoint": 30, "gpu_count": 4, "state": "started"})
        led.close_interrupted()
        rows = led.log.rows()
        self.assertEqual(rows[0]["measurement_status"], "interrupted")
        self.assertFalse(physical_gpu_seconds(rows)["complete"])
        _, missing = component_totals(rows)
        self.assertEqual(len(missing), 1)


class Budget(unittest.TestCase):
    snaps = [{"completed_update": 25, "cumulative_cost": 5}, {"completed_update": 50, "cumulative_cost": 10},
             {"completed_update": 75, "cumulative_cost": 15}]

    def test_latest_within_budget(self):
        r = select_endpoint(self.snaps, 12)
        self.assertEqual((r["status"], r["endpoint"]["completed_update"], r["slack"]), ("selected", 50, 2))
        self.assertEqual(r["overshoot"]["completed_update"], 75)
        self.assertEqual(select_endpoint(self.snaps, 100)["endpoint"]["completed_update"], 75)

    def test_infeasible_only_when_known_over(self):
        self.assertEqual(select_endpoint(self.snaps, 1)["status"], "infeasible_budget")

    def test_codex_unknown_later_cost_is_undetermined(self):
        r = select_endpoint([{"completed_update": 25, "cumulative_cost": 10}, {"completed_update": 50, "cumulative_cost": None}], 20)
        self.assertEqual(r["status"], "undetermined")
        r = select_endpoint([{"completed_update": 25, "cumulative_cost": None}], 20)
        self.assertEqual(r["status"], "undetermined")
        # unknown sandwiched before a known in-budget later snapshot cannot change the choice
        r = select_endpoint([{"completed_update": 25, "cumulative_cost": 10}, {"completed_update": 50, "cumulative_cost": None},
                             {"completed_update": 75, "cumulative_cost": 15}], 20)
        self.assertEqual(r["endpoint"]["completed_update"], 75)
        # unknown after the first known overshoot does not matter either
        r = select_endpoint([{"completed_update": 25, "cumulative_cost": 10}, {"completed_update": 50, "cumulative_cost": 30},
                             {"completed_update": 75, "cumulative_cost": None}], 20)
        self.assertEqual(r["endpoint"]["completed_update"], 25)

    def test_decreasing_cost_rejected(self):
        with self.assertRaises(ValueError):
            select_endpoint([{"completed_update": 25, "cumulative_cost": 10}, {"completed_update": 50, "cumulative_cost": 5}], 20)

    def test_not_frozen(self):
        self.assertEqual(select_endpoint(self.snaps, None)["status"], "budget_not_frozen")

    def test_common_target(self):
        pts = [(25, 1.0, 0.30), (50, 2.0, 0.34), (75, 3.0, 0.36)]
        self.assertEqual(common_target_cost(pts, 0.35), {"status": "reached", "first_update": 75, "interval": [2.0, 3.0]})
        self.assertEqual(common_target_cost(pts, 0.5)["status"], "not_reached")
        self.assertEqual(common_target_cost([(25, 1.0, None)], 0.5)["status"], "unknown")

    def test_should_continue_cheap_arm_past_horizon(self):
        self.assertTrue(should_continue(100, horizon=275, hard_cap=400, budget_cap=10, cumulative_cost=None, next_update_cost=None))
        self.assertTrue(should_continue(275, horizon=275, hard_cap=400, budget_cap=10, cumulative_cost=5, next_update_cost=1))
        self.assertFalse(should_continue(275, horizon=275, hard_cap=400, budget_cap=10, cumulative_cost=9.5, next_update_cost=1))
        self.assertFalse(should_continue(400, horizon=275, hard_cap=400, budget_cap=1e9, cumulative_cost=5, next_update_cost=1))
        self.assertFalse(should_continue(275, horizon=275, hard_cap=400, budget_cap=10, cumulative_cost=None, next_update_cost=1))


if __name__ == "__main__":
    unittest.main()


class UnknownPrefixCost(unittest.TestCase):
    def test_missing_receipts_are_unknown_not_zero(self):
        from switching import smoke
        from switching.cluster import prefix_cost_from_receipts
        d = tmpdir()
        self.assertIsNone(smoke.prefix_cost(d, 5))
        (d / "locator.json").write_text('{"seeds": {"5": {"folder": "%s"}}}' % (d / "nofolder"))
        self.assertIsNone(prefix_cost_from_receipts(d, 5))
