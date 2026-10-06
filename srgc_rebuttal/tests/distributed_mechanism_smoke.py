"""CPU four-rank integration check; synthetic models, never experiment evidence."""

from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist

from scripts.srgc_stage_mechanism import ARM, StageStudy
from scripts.srgc_step_checkpoints import save_checkpoint
from srgc_rebuttal.distributed import primary
from srgc_rebuttal.srgc import Config
from srgc_rebuttal.tests.test_torch_backend import ModelBackendTests


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    folder = Path(primary(lambda: tempfile.mkdtemp(prefix="srgc-mechanism-four-rank-")))
    try:
        fixture = ModelBackendTests()
        fixture.setUp()
        backend = fixture.backend
        backend._rollout = fixture.deterministic_rollout
        initial = backend.state_dict()
        data = dict(candidate_ids=[f"p{i}" for i in range(4)], ranking_validation_ids=["p4"],
                    evaluation_ids=["p5"], cached_rewards={f"p{i}": [0., 1.] * 4 for i in range(4)})
        config = Config(seed=5, scoring_prompts=4, training_prompts=4, projection_dim=16)
        study = StageStudy(backend, data, config, stages=(0,), horizon=1)
        while study.phase != "evaluate_branch":
            study.advance()
        save_checkpoint(study, folder, ARM)
        state = torch.load(folder / f"{ARM}-latest.pt", map_location="cpu", weights_only=False)
        resumed = StageStudy(backend, data, config, stages=(0,), horizon=1)
        resumed.load_state_dict(state)
        while not resumed.done:
            resumed.advance()
        assert len(resumed.rows) == len(resumed.training_records) == 6
        actual = backend.state_dict()
        for name, value in initial["trainable"].items():
            torch.testing.assert_close(actual["trainable"][name], value, rtol=0, atol=0)
        assert actual["optimizer"]["state"] == initial["optimizer"]["state"]
        rows = [None] * dist.get_world_size()
        dist.all_gather_object(rows, resumed.rows)
        assert all(value == rows[0] for value in rows)
        primary(lambda: print("PASS: four-rank scoring, six branches, checkpoint resume and carrier restoration", flush=True))
    finally:
        primary(lambda: shutil.rmtree(folder))
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
