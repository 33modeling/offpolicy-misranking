import os
import unittest
from pathlib import Path

os.environ.setdefault("V8_FSYNC", "0")
from helpers import HAS_CODE, tmpdir  # noqa: E402
from switching import snapshots  # noqa: E402
from switching.common import read_json, read_jsonl  # noqa: E402


class Hashing(unittest.TestCase):
    def test_codex_tree_hash_collision(self):
        self.assertNotEqual(snapshots.tree_hash([1, 23]), snapshots.tree_hash([12, 3]))
        self.assertNotEqual(snapshots.tree_hash(["a", "bc"]), snapshots.tree_hash(["ab", "c"]))
        self.assertNotEqual(snapshots.tree_hash({"a": [1]}), snapshots.tree_hash({"a": 1}))
        self.assertNotEqual(snapshots.tree_hash((1, 2)), snapshots.tree_hash([1, 2]))
        self.assertEqual(snapshots.tree_hash({"b": 1, "a": 2}), snapshots.tree_hash({"a": 2, "b": 1}))

    def test_codex_immutable_snapshot_protects_rng_and_controller(self):
        import numpy as np
        d = tmpdir()
        state = {"backend": {"weights": np.zeros(3)}, "step": 5, "v8": {"policy": {"history": [-1.0]}}}
        payload = {"engine": state, "rng": {"numpy": ("MT", np.arange(3))}, "saved_at": 1.0}
        snapshots.save(d / "s.pt", payload, immutable=True)
        snapshots.save(d / "s.pt", dict(payload, saved_at=2.0), immutable=True)  # timing only: idempotent
        for changed in (dict(payload, rng={"numpy": ("MT", np.arange(1, 4))}),
                        dict(payload, engine=dict(state, v8={"policy": {"history": [-2.0]}}))):
            with self.assertRaises(ValueError):
                snapshots.save(d / "s.pt", changed, immutable=True)
        # sibling equivalence ignores controller state but not learner or RNG
        a = snapshots.branch_identity(payload)
        b = snapshots.branch_identity(dict(payload, engine=dict(state, v8={"policy": {"history": [9.0]}})))
        self.assertEqual(a["branch_hash"], b["branch_hash"])
        c = snapshots.branch_identity(dict(payload, rng={"numpy": ("MT", np.arange(1, 4))}))
        self.assertNotEqual(a["branch_hash"], c["branch_hash"])


@unittest.skipUnless(HAS_CODE, "code repository not available")
class RunnerLifecycle(unittest.TestCase):
    def setUp(self):
        from switching import backends, coderepo, matrix as mx, smoke, spec
        from switching.engine import make_engine_class
        self.srgc = coderepo.srgc()
        self.PE = make_engine_class(self.srgc)
        self.feat, self.ans, self.data = backends.toy_problem(0)
        self.cfg = spec.resolve("olmo_math", role="smoke", extra={"cohort_id": "t", "seeds": {"training": [5]},
                                "training": {"matched_update_endpoint": 125}})
        self.out = tmpdir()
        self.smoke, self.mx = smoke, mx
        self.ctx = smoke.toy_ctx(self.out, 5, self.srgc, self.PE, self.feat, self.ans, self.data, 125)
        smoke.make_prefix(self.ctx, 5, 25, self.srgc)
        self.loc = smoke.Locator(self.out, self.cfg, 25)

    def traj(self, out=None, stop=None):
        from switching.runner import Interrupted, execute_train
        out = out or self.out
        ctx = self.ctx if out == self.out else self.smoke.toy_ctx(out, 5, self.srgc, self.PE, self.feat, self.ans,
                                                                  self.data, 125)
        if out != self.out:
            self.smoke.make_prefix(ctx, 5, 25, self.srgc)
        loc = self.smoke.Locator(out, self.cfg, 25)
        t = self.mx.build_stage("bank-trajectory", self.cfg, loc, out, seeds=[5])[0][0]
        if stop:
            t["debug_stop_after_update"] = stop
            with self.assertRaises(Interrupted):
                execute_train(t, ctx)
            t.pop("debug_stop_after_update")
        return t, execute_train(t, ctx), ctx

    def test_interrupt_resume_equals_uninterrupted_and_is_idempotent(self):
        from switching.runner import execute_train
        t, res, ctx = self.traj(stop=60)
        self.assertEqual(res["completed_updates"], 125)
        other = tmpdir()
        t2, _, _ = self.traj(out=other)
        a = {r["completed_update"]: r["identity"]["state_hash"] for r in read_jsonl(self.out / "runs" / t["run_id"] / "snapshots.jsonl")}
        b = {r["completed_update"]: r["identity"]["state_hash"] for r in read_jsonl(other / "runs" / t2["run_id"] / "snapshots.jsonl")}
        self.assertEqual(a, b)                                    # resume reproduces the uninterrupted lineage
        man = read_json(self.out / "runs" / t["run_id"] / "manifest.json")
        self.assertEqual([x["start"] for x in man["attempts"]], ["fork", "resume@60"])
        self.assertEqual(execute_train(t, ctx)["status"], "skipped_complete")
        hist = read_jsonl(self.out / "runs" / t["run_id"] / "history.jsonl")
        self.assertEqual(len(hist), len({r["completed_updates"] for r in hist}))  # no duplicated update records
        # a safe-boundary stop saves latest first, so nothing is redone: lineage keeps every event
        from switching.runner import run_attempts, run_events
        self.assertEqual([s for _, s in run_attempts(self.out / "runs" / t["run_id"])], [25, 60])
        self.assertEqual(len(run_events(self.out / "runs" / t["run_id"])),
                         len(run_events(self.out / "runs" / t["run_id"], lineage=False)))

    def test_config_change_blocks_resume(self):
        from switching.runner import ConfigChanged, execute_train
        t, _, ctx = self.traj()
        bad = dict(t, config_hash="0" * 64)
        with self.assertRaises(ConfigChanged):
            execute_train(bad, ctx)

    def test_sibling_branches_start_from_identical_state(self):
        from switching.runner import execute_train
        t, _, ctx = self.traj()
        parent = self.mx.parent_from_snapshot(self.out, t["run_id"], 50)
        ids = []
        for mid in ("bank_sr_t050", "bank_uniform_t050"):
            bt = self.mx.train_task(self.cfg, role="rebuttal_bank", seed=5, method_id=mid, parent=parent, horizon=125,
                                    stage="test", interval=25)
            execute_train(bt, ctx)
            ids.append(read_json(self.out / "runs" / bt["run_id"] / "branch_start.json"))
        self.assertEqual(ids[0]["parent_identity"], ids[1]["parent_identity"])
        self.assertEqual(ids[0]["parent_branch_identity"]["branch_hash"], ids[1]["parent_branch_identity"]["branch_hash"])
        snap = {r["completed_update"]: r for r in read_jsonl(self.out / "runs" / t["run_id"] / "snapshots.jsonl")}[50]
        self.assertEqual(snap["identity"]["state_hash"], ids[0]["parent_identity"]["state_hash"])
        self.assertEqual(ids[0]["first_update_of_branch"], 51)

    def test_retained_snapshots_and_training_rewards_every_25(self):
        """User rule 2026-10-28: rewards + optimizer/RNG checkpoints at least every 25 updates; no eval in training."""
        t, _, _ = self.traj()
        rdir = self.out / "runs" / t["run_id"]
        snaps = sorted(r["completed_update"] for r in read_jsonl(rdir / "snapshots.jsonl"))
        self.assertEqual(snaps, [50, 75, 100, 125])
        payload = snapshots.load(rdir / "snapshots" / "step-000050.pt")
        self.assertIn("rng", payload)
        self.assertIn("m", payload["engine"]["backend"])  # optimizer moments (toy AdamW)
        stats = {r["completed_update"] for r in read_jsonl(rdir / "training_stats.jsonl")}
        self.assertTrue(set(range(1, 126)) <= stats)
        self.assertFalse((self.out / "evaluations").exists())  # training wrote no held-out evaluation


if __name__ == "__main__":
    unittest.main()
