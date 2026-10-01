import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.srgc_sr_refresh import continue_updates, extra_checkpoint_policy, make_engine
from srgc_rebuttal.cost_ledger import PhaseLedger
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.tests.test_sr_refresh import EvalToyBackend
from srgc_rebuttal.timing import CostMeter
from srgc_rebuttal.toy_backend import make_problem


class ExtraCheckpointTests(unittest.TestCase):
    def test_new_arm_inherits_prefix_kernel_despite_environment_default(self):
        for kernel in ("eager", "sdpa", "flash_attention_2"):
            with self.subTest(kernel=kernel), patch.dict("os.environ", {"SRGC_ATTENTION": "sdpa"}):
                policy = extra_checkpoint_policy({"checkpoint_policy": {"attention": kernel}}, resuming=False)
                self.assertEqual(policy["attention"], kernel)
                self.assertEqual(policy["attention_source"], "shared-prefix")
                self.assertEqual(policy["interval_updates"], 1)

    def test_legacy_extra_resume_keeps_eager_not_prefix_or_shell_sdpa(self):
        self.assertEqual(extra_checkpoint_policy({}, resuming=True)["attention"], "eager")
        prefix = {"checkpoint_policy": {"attention": "sdpa"}}
        self.assertEqual(extra_checkpoint_policy(prefix, resuming=False, previous_run={})["attention"], "eager")
        previous = {"checkpoint_policy": {"attention": "flash_attention_2"}}
        self.assertEqual(extra_checkpoint_policy(prefix, resuming=False, previous_run=previous)["attention"],
                         "flash_attention_2")
        self.assertEqual(extra_checkpoint_policy(prefix, resuming=True, previous_run=previous)["attention"], "sdpa")
        with self.assertRaisesRegex(ValueError, "unsupported saved attention"):
            extra_checkpoint_policy({"checkpoint_policy": {"attention": "invalid"}}, resuming=True)

    def test_all_extra_arms_save_and_resume_every_update(self):
        features, answers, candidates, validation, _, cache = make_problem(3)
        data = {"candidate_ids": candidates, "ranking_validation_ids": validation, "cached_rewards": cache}
        config = Config(seed=3, projection_dim=64)

        def backend(folder):
            value = EvalToyBackend(features, answers, projection_dim=64, seed=3)
            value.cost_meter = CostMeter(record=PhaseLedger(folder / "costs").record)
            return value

        arms = ("sr_refresh", "sr_refresh-pool", "sr_hold", "switch_repeat", "switch_fixed25",
                "switch_fixed100", "direction_removed", "direction_magnitude", "direction_replaced",
                "replicate1-random", "replicate1-sr", "replicate1-on_policy", "replicate1-switch",
                "replicate1-switch_fixed25", "replicate2-switch_fixed100")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = Engine(backend(root), candidates, validation, cache, arm="on_policy", config=config)
            prefix.run_until(25)
            shared = prefix.state_dict()
            policy = extra_checkpoint_policy(shared, resuming=False)
            for arm in arms:
                with self.subTest(arm=arm), contextlib.redirect_stdout(io.StringIO()):
                    folder = root / arm
                    folder.mkdir()
                    current, label = make_engine(arm, backend(folder), data, config)
                    current.load_state_dict(shared, fork_arm=label)
                    for end in (26, 27, 28):
                        continue_updates(current, folder, arm, arm, end, {}, policy)
                        state = torch.load(folder / f"{arm}-latest.pt", weights_only=False)
                        self.assertEqual(state["step"], end)
                        self.assertEqual(state["checkpoint_policy"], policy)
                        progress = json.loads((folder / f"{arm}-progress.json").read_text())
                        self.assertEqual(progress["step"], end)
                        self.assertEqual(len(progress["history"]), end - 25)
                        resumed, _ = make_engine(arm, backend(folder), data, config)
                        resumed.load_state_dict(state)
                        self.assertEqual(resumed.active_selection, current.active_selection)
                        current = resumed
                    uninterrupted, _ = make_engine(arm, backend(folder), data, config)
                    uninterrupted.load_state_dict(shared, fork_arm=label)
                    uninterrupted.run_until(29)
                    actual = current.update()
                    expected = uninterrupted.history[-1]
                    self.assertEqual(actual["train_ids"], expected["train_ids"])
                    self.assertEqual(actual["metrics"], expected["metrics"])
                    self.assertEqual(current.switched_at, uninterrupted.switched_at)

    def test_failed_save_leaves_previous_checkpoint_and_progress(self):
        from srgc_rebuttal.tests.test_step_checkpoints import make_engine as fixture_engine
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(directory)
            current = fixture_engine(folder, task="sr", step=25, adapted=False)
            policy = extra_checkpoint_policy({}, resuming=False)
            continue_updates(current, folder, "sr", "replicate1-sr", 26, {}, policy)
            before = (folder / "sr-latest.pt").read_bytes()
            def fail(state, handle):
                handle.write(b"partial")
                raise OSError("storage full")
            with patch.object(torch, "save", side_effect=fail), self.assertRaisesRegex(Exception, "storage full"):
                continue_updates(current, folder, "sr", "replicate1-sr", 27, {}, policy)
            self.assertEqual((folder / "sr-latest.pt").read_bytes(), before)
            self.assertEqual(json.loads((folder / "sr-progress.json").read_text())["step"], 26)
            self.assertFalse((folder / "sr-latest.tmp").exists())


if __name__ == "__main__":
    unittest.main()
