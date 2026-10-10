"""Four CPU/Gloo ranks: actual gradients, resume and shared failure handling."""

import shutil
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from srgc_rebuttal.distributed import primary
from srgc_rebuttal.torch_backend import TorchBackend
from srgc_research.information import InformationStudy
from srgc_research.information_report import read_measurement, write_report
from srgc_research.tests.test_information import setup_measurement
from srgc_research.tests.test_research import equal_tree, study_backend


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    folder = Path(primary(lambda: tempfile.mkdtemp(prefix="srgc-information-four-rank-")))
    try:
        backend = study_backend()
        data, config, identity = primary(lambda: setup_measurement(folder, backend))
        backend.records = data["records"]
        initial = backend.state_dict()
        with patch.object(TorchBackend, "_rollout", side_effect=backend.generate_test_rollout):
            value = InformationStudy(backend, data, config, 0, folder, identity, probe_prompts=1)
            result = value.run()
        replicas = [None] * dist.get_world_size()
        dist.all_gather_object(replicas, result)
        assert all(r == replicas[0] for r in replicas)
        equal_tree(initial, backend.state_dict())
        read_measurement(folder)
        with patch.object(TorchBackend, "_rollout", side_effect=AssertionError("resume regenerated")), \
                patch.object(backend, "train", side_effect=AssertionError("resume retrained")):
            assert value.run() == result
        # A rank-zero disk failure must be broadcast, without hanging peers.
        from srgc_research import information
        with patch.object(information, "atomic_torch", side_effect=OSError("injected disk failure")):
            try:
                value.publish("unused", {}, {})
            except RuntimeError as exc:
                assert "injected disk failure" in str(exc)
            else:
                raise AssertionError("disk failure was hidden")
        primary(lambda: write_report(folder.with_name(folder.name + "-report"), folders=[folder]))
        # Same frozen responses on one process should reproduce the all-reduced update.
        measured = torch.load(folder / "on_policy.pt", map_location="cpu", weights_only=False)
        receipt = read_measurement(folder)[1]["on_policy"]
        records = {rid: ([torch.tensor(s["sequence_ids"]) for s in row["samples"]],
                         [s["reward"] for s in row["samples"]], row["samples"][0]["prompt_tokens"])
                   for rid, row in receipt["responses"].items()}
        records = {rid: records[rid] for rid in receipt["selected_ids"]}

        def compare_serial():
            old_rank, old_world = backend.rank, backend.world
            try:
                backend.rank, backend.world = 0, 1
                with backend.replaying(records):
                    metrics = backend.train(list(records), responses=8, objective="grpo", seed=29)
                assert abs(metrics["gradient_norm"] - receipt["metrics"]["gradient_norm"]) < 1e-6
                for name, weights in backend.state_dict()["trainable"].items():
                    torch.testing.assert_close(weights, measured["backend_after"]["trainable"][name], rtol=1e-5, atol=1e-7)
            finally:
                backend.rank, backend.world = old_rank, old_world
                backend.load_state_dict(initial)

        primary(compare_serial)
        # Global gradient-vector values must agree on every rank as well.
        norm = float(measured["vectors"]["gradient_before_clip"].norm())
        dist.all_gather_object(replicas, norm)
        assert all(n == replicas[0] for n in replicas)
        primary(lambda: print("PASS: 4-rank/serial update agreement, selection, exact responses, GRPO gradients, AdamW deltas, restore, resume, report and disk-failure propagation", flush=True))
    finally:
        primary(lambda: shutil.rmtree(folder))
        primary(lambda: shutil.rmtree(folder.with_name(folder.name + "-report"), ignore_errors=True))
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
