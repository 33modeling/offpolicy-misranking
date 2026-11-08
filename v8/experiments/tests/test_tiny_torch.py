"""Real TorchBackend (tiny random OLMo-3 + LoRA, CPU): branch identity with true AdamW/RNG state."""
import importlib.util
import os
import unittest

os.environ.setdefault("V8_FSYNC", "0")
from helpers import HAS_CODE, coderepo, tmpdir  # noqa: E402

HAS_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft"))


@unittest.skipUnless(HAS_CODE and HAS_TORCH, "requires the code repo and CPU torch/transformers/peft")
class TinyOlmo(unittest.TestCase):
    def setUp(self):
        from switching import backends, snapshots
        from switching.engine import make_engine_class, fork_arm_for
        from switching.methods import method
        from switching.policies import build_policy
        self.snapshots, self.method, self.build_policy, self.fork = snapshots, method, build_policy, fork_arm_for
        self.srgc = coderepo.srgc()
        self.PE = make_engine_class(self.srgc)
        self.records, data = backends.tiny_problem()
        self.cfg = self.srgc.Config(seed=5, scoring_prompts=8, training_prompts=2, responses=4, candidate_group_size=2,
                                    projection_dim=16, selection_interval=5, check_interval=5, first_check=5)
        self.be = backends.tiny_backend(self.records)
        cache = backends.initial_cache(self.be, data["candidate_ids"], responses=4, seed=5, srgc=self.srgc)
        self.data = dict(data, cached_rewards=cache)
        base = self.srgc.Engine(self.be, data["candidate_ids"], data["ranking_validation_ids"], cache, arm="on_policy",
                                config=self.cfg)
        base.run_until(5)
        self.prefix = base.state_dict()

    def engine(self, mid, state):
        m = self.method(mid, interval=5, prefix=5)
        e = self.PE(self.be, self.data["candidate_ids"], self.data["ranking_validation_ids"], self.data["cached_rewards"],
                    method=m, policy=self.build_policy(m["policy"]), config=self.cfg, monitor=m["monitor"])
        e.load_state_dict(state, fork_arm=self.fork(m))
        return e

    def test_switch_parity_and_branch_identity(self):
        base = self.srgc.Engine(self.be, self.data["candidate_ids"], self.data["ranking_validation_ids"],
                                self.data["cached_rewards"], arm="switch", config=self.cfg)
        base.load_state_dict(self.prefix, fork_arm="switch")
        base.run_until(20)
        v = self.engine("srgc_current", self.prefix)
        v.run_until(20)
        self.assertEqual([r["train_ids"] for r in base.history], [r["train_ids"] for r in v.history])
        self.assertEqual(self.snapshots.state_identity(base.state_dict()), self.snapshots.state_identity(v.state_dict()))
        traj = self.engine("on_monitored", self.prefix)
        traj.run_until(10)
        parent = traj.state_dict()
        ident = self.snapshots.state_identity(parent)
        a, b = self.engine("bank_sr_t010", parent), self.engine("bank_uniform_t010", parent)
        self.assertEqual(self.snapshots.state_identity(a.state_dict())["optimizer_hash"], ident["optimizer_hash"])
        self.assertEqual(self.snapshots.state_identity(b.state_dict())["model_hash"], ident["model_hash"])
        a.run_until(12)
        self.assertNotEqual(self.snapshots.state_identity(a.state_dict())["model_hash"], ident["model_hash"])
        # the parent snapshot itself is untouched by training a branch (deep copies)
        self.assertEqual(self.snapshots.state_identity(parent), ident)


if __name__ == "__main__":
    unittest.main()
