import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import srgc_resumable_rollouts as resumable


class FakeOOM(RuntimeError):
    pass


class FakeBackend:
    """Stands in for TorchBackend: sequential rollouts, gathered scoring/evaluation."""

    device = SimpleNamespace(type="cpu")

    def __init__(self, fail_first=()):
        self.calls, self.fail_first = [], set(fail_first)
        self.rank, self.world = 0, 1

    def _rollout(self, prompt_id, responses, seed):
        self.calls.append(prompt_id)
        if prompt_id in self.fail_first:
            self.fail_first.discard(prompt_id)
            raise FakeOOM("CUDA out of memory")
        length = 3 + len(prompt_id)
        return ([np.arange(length) + i for i in range(responses)],
                np.array([float(i % 2) for i in range(responses)]), 2)

    def score_gradients(self, ids, *, responses, group_size, seed):
        return {i: np.full(4, float(sum(_to_int(s) for s in self._rollout(i, responses, seed)[0]))) for i in ids}

    def evaluate(self, ids, *, seed, responses=8):
        return {i: float(self._rollout(i, responses, seed)[1].mean()) for i in ids}


def _to_int(sequence):
    return int(np.asarray(sequence).sum())


class ResumableRolloutTest(unittest.TestCase):
    def setUp(self):
        self._oom = resumable._oom_types
        resumable._oom_types = lambda: (FakeOOM,)

    def tearDown(self):
        resumable._oom_types = self._oom

    def test_rollouts_persist_per_prompt_and_a_restart_reuses_them(self):
        with tempfile.TemporaryDirectory() as folder:
            Backend = resumable.make_resumable(FakeBackend, Path(folder) / "cache")
            first = Backend()
            first._resumable_dir = first._resumable("score", 7)
            first._rollout("p1", 8, 7)
            first._rollout("p2", 8, 7)
            saved = sorted(p.name for p in (Path(folder) / "cache" / "score-7").iterdir())
            self.assertEqual(len(saved), 2)
            # A new attempt (new process) scores the same block: p1/p2 are loaded, p3 is generated.
            second = Backend()
            result = second.score_gradients(["p1", "p2", "p3"], responses=8, group_size=4, seed=7)
            self.assertEqual(second.calls, ["p3"])
            self.assertEqual((second.resumed_rollouts, second.generated_rollouts), (2, 1))
            reference = FakeBackend().score_gradients(["p1", "p2", "p3"], responses=8, group_size=4, seed=7)
            for key in reference:
                np.testing.assert_array_equal(result[key], reference[key])
            self.assertFalse((Path(folder) / "cache" / "score-7").exists())  # block complete -> cache removed

    def test_evaluation_resumes_and_phases_do_not_share_files(self):
        with tempfile.TemporaryDirectory() as folder:
            Backend = resumable.make_resumable(FakeBackend, Path(folder) / "cache")
            backend = Backend()
            backend._resumable_dir = backend._resumable("eval", 9)
            backend._rollout("q", 8, 9)
            self.assertTrue((Path(folder) / "cache" / "eval-9").exists())
            self.assertFalse((Path(folder) / "cache" / "score-9").exists())
            fresh = Backend()
            self.assertEqual(fresh.evaluate(["q", "r"], seed=9), {"q": 0.5, "r": 0.5})
            self.assertEqual(fresh.calls, ["r"])
            self.assertFalse((Path(folder) / "cache" / "eval-9").exists())

    def test_one_oom_is_retried_and_a_second_one_propagates(self):
        with tempfile.TemporaryDirectory() as folder:
            Backend = resumable.make_resumable(FakeBackend, Path(folder) / "cache")
            backend = Backend(fail_first={"p1"})
            backend._resumable_dir = backend._resumable("score", 1)
            sequences, rewards, start = backend._rollout("p1", 8, 1)
            self.assertEqual(backend.calls, ["p1", "p1"])
            self.assertEqual(len(sequences), 8)
            stubborn = Backend(fail_first={"p2"})
            original = stubborn._rollout

            def always_oom(self, prompt_id, responses, seed):
                self.calls.append(prompt_id)
                raise FakeOOM("CUDA out of memory")
            FakeBackend._rollout, saved = always_oom, FakeBackend._rollout
            try:
                stubborn._resumable_dir = stubborn._resumable("score", 2)
                with self.assertRaises(FakeOOM):
                    stubborn._rollout("p2", 8, 2)
            finally:
                FakeBackend._rollout = saved

    def test_outside_a_resumable_phase_nothing_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            Backend = resumable.make_resumable(FakeBackend, Path(folder) / "cache")
            backend = Backend()
            backend._rollout("t", 8, 3)  # e.g. a training rollout
            self.assertFalse((Path(folder) / "cache").exists())

    def test_install_replaces_the_module_class_once(self):
        try:
            from srgc_rebuttal import torch_backend
        except ImportError:
            self.skipTest("torch backend unavailable")
        original = torch_backend.TorchBackend
        try:
            patched = resumable.install("/tmp/x")
            self.assertTrue(issubclass(patched, original))
            self.assertEqual(patched.rollout_cache_root, Path("/tmp/x"))
            again = resumable.install("/tmp/y")
            self.assertIs(again, patched)
            self.assertEqual(patched.rollout_cache_root, Path("/tmp/y"))
        finally:
            torch_backend.TorchBackend = original


if __name__ == "__main__":
    unittest.main()
