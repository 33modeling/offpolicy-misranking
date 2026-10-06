"""Four CPU ranks, real tiny OLMo/LoRA; not an H100 performance measurement."""

import copy
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from srgc_rebuttal.distributed import primary
from srgc_rebuttal.srgc import Config
from srgc_rebuttal.torch_backend import TorchBackend
from srgc_research.design import Condition
from srgc_research.study import Diagnostic, NestedEngine
from srgc_research.tests import test_research
from srgc_research.tests.test_execution import attach_meter, manifest_for
from srgc_research.worker import execute


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    folder = Path(primary(lambda: tempfile.mkdtemp(prefix="srgc-research-four-rank-")))
    try:
        data = test_research.data.__wrapped__()
        b = test_research.study_backend()
        b.records = data["records"]
        initial = b.state_dict()
        manifest = manifest_for(data)
        with patch.object(TorchBackend, "_rollout", side_effect=b.generate_test_rollout):
            for condition in (Condition("cache", "cache", updates=0), Condition("n01-features", "features", updates=0),
                              Condition("test-lesser", "trajectory", arm="lesser", updates=2),
                              Condition("test-arcus", "trajectory", arm="arcus_adapted", updates=2)):
                b.load_state_dict(initial)
                attach_meter(b, folder, condition)
                result = execute(folder, condition, manifest, copy.deepcopy(data), b)
                replicas = [None] * dist.get_world_size()
                dist.all_gather_object(replicas, result)
                assert all(r == replicas[0] for r in replicas)
                checkpoint = torch.load(folder / condition.key / "state-latest.pt", map_location="cpu", weights_only=False)
                assert checkpoint["seed"] == 5
            b.load_state_dict(initial)
            config = Config(seed=5, projection_dim=16)
            engine = NestedEngine(b, data["candidate_ids"], data["ranking_validation_ids"],
                                  data["cached_rewards"], config=config, arm="on_policy")
            condition = Condition("n02-stage-0", "diagnostic", arm="n02", stage=0, updates=25)
            attach_meter(b, folder, condition)
            study = Diagnostic(b, data, config, condition, engine.state_dict())
            while not study.done:
                study.advance()
                saved = copy.deepcopy(study.state_dict())
                study.load_state_dict(saved)
            test_research.equal_tree(initial, b.state_dict())
            norms = [None] * dist.get_world_size()
            dist.all_gather_object(norms, study.result())
            assert all(r == norms[0] for r in norms)
        primary(lambda: print("PASS: 4-rank feature scoring, rollout replay, GRPO/Adam audit, checkpoint and endpoint recovery", flush=True))
    finally:
        primary(lambda: shutil.rmtree(folder))
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
