"""CUDA control-flow tests use mock devices; they do not claim GPU execution."""

from contextlib import nullcontext
import importlib.util
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "requires optional PyTorch")
class RolloutRngTests(unittest.TestCase):
    def test_async_generation_failure_is_not_replaced_by_cuda_restore(self):
        import torch
        from srgc_rebuttal.torch_state import rollout_rng
        cpu = torch.get_rng_state().clone()
        error = RuntimeError("first asynchronous generation failure")
        device = torch.device("cuda", 2)
        with patch.object(torch.cuda, "get_rng_state", return_value=object()) as get, \
                patch.object(torch.cuda, "device", return_value=nullcontext()) as select, \
                patch.object(torch.cuda, "manual_seed") as seed, \
                patch.object(torch.cuda, "manual_seed_all") as seed_all, \
                patch.object(torch.cuda, "synchronize", side_effect=error) as synchronize, \
                patch.object(torch.cuda, "set_rng_state", side_effect=RuntimeError("restore failure")):
            with self.assertRaises(RuntimeError) as raised:
                with rollout_rng(device, 19):
                    torch.rand(3)
        self.assertIs(raised.exception, error)
        get.assert_called_once_with(device)
        select.assert_called_once_with(device)
        seed.assert_called_once_with(19)
        seed_all.assert_not_called()
        synchronize.assert_called_once_with(device)
        torch.testing.assert_close(torch.get_rng_state(), cpu)
        self.assertIn("restore failure", " ".join(error.__notes__))

    def test_restore_failure_after_success_is_not_swallowed(self):
        import torch
        from srgc_rebuttal.torch_state import rollout_rng
        with patch.object(torch.random, "set_rng_state", side_effect=RuntimeError("restore failure")):
            with self.assertRaisesRegex(RuntimeError, "restore failure"):
                with rollout_rng(torch.device("cpu"), 19):
                    pass

    def test_snapshot_has_independent_host_storage_and_restores_optimizer(self):
        import torch
        from srgc_rebuttal.torch_state import cpu_snapshot
        parameter = torch.nn.Parameter(torch.tensor([2.]))
        optimizer = torch.optim.AdamW([parameter])
        parameter.sum().backward()
        optimizer.step()
        snapshot = cpu_snapshot(optimizer.state_dict())
        source = optimizer.state[parameter]["exp_avg"]
        stored = snapshot["state"][0]["exp_avg"]
        self.assertEqual(stored.device.type, "cpu")
        self.assertNotEqual(stored.data_ptr(), source.data_ptr())
        expected = stored.clone()
        source.zero_()
        torch.testing.assert_close(stored, expected)
        optimizer.load_state_dict(cpu_snapshot(snapshot))
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], expected)
