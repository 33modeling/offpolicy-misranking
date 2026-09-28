import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from srgc_pair_inputs import default_plan  # noqa: E402
from srgc_seed_order import seed_first_queue  # noqa: E402
from srgc_shared_storage import route_plan  # noqa: E402
from srgc_rebuttal import cluster  # noqa: E402
from srgc_rebuttal.plan import digest, input_path, load_plan  # noqa: E402
from srgc_rebuttal.runtime import atomic_json  # noqa: E402


def source_run(base, seed, code):
    run = base / f"family-s{seed}" / f"run-s{seed}-d0"
    run.mkdir(parents=True)
    offset = seed * 10_000

    def item(tag, i):
        if code:
            return {"question": f"Write a Python function for {tag}{offset + i}.\n\nassert f{offset + i}(1) == 1",
                    "answer": f"assert f{offset + i}(1) == 1"}
        return {"question": f"{tag}{offset + i}: what is {i}+1?", "answer": f"${i + 1}$"}

    (run / "prompts.json").write_text(json.dumps({"train": [item("T", i) for i in range(400)],
                                                  "val": [item("V", i) for i in range(100)]}))
    with (run / "rollouts_behavior_train.jsonl").open("w") as handle:
        for i in range(400):
            for j in range(8):
                handle.write(json.dumps({"prompt_idx": i, "rollout_idx": j, "reward": float((i + j) % 3 == 0)}) + "\n")
    (run / "run_config.json").write_text(json.dumps(
        {"prompt_format": "olmo_rlzero_code" if code else "olmo_rlzero_math", "behavior_k": 8}))
    return run


def evaluation(code, count=300):
    if code:
        test = [{"question": f"Write a Python function for eval {i}.\n\nassert e{i}(2) == 2",
                 "answer": f"assert e{i}(2) == 2"} for i in range(count)]
    else:
        test = [{"question": f"E{i}: what is {i}+3?", "answer": f"${i + 3}$"} for i in range(count)]
    return {"test": test, "provenance": {"dataset": "synthetic"}}


class PairSeedReuseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        base = Path(self.directory.name)
        self.checkout = base / "checkout"
        (self.checkout / "srgc_rebuttal").mkdir(parents=True)
        for folder in ("experiments", "inputs"):
            source = ROOT / "srgc_rebuttal" / folder
            (self.checkout / "srgc_rebuttal" / folder).mkdir()
            for path in source.glob("*.json"):
                (self.checkout / "srgc_rebuttal" / folder / path.name).write_bytes(path.read_bytes())
        self.group = base / "group"
        self.work = self.group / "u/offpolicy-misranking"
        self.environment = {"GROUP_VOLUME": str(self.group), "OM_WORK": str(self.work)}
        self.patch = patch.dict(os.environ, self.environment)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.directory.cleanup()

    def sources(self, dataset, count=300):
        code = dataset == "mbpp"
        manifest = {"sources": {str(s): {"path": str(source_run(self.work / f"matrix-{dataset}", s, code))}
                                for s in (3, 4)}, "evaluation": evaluation(code, count)}
        if code:
            path = self.work / "runs/selection-switch-mbpp-v1/switch.json"
        else:
            path = self.work / "runs/selector-pair-v1/branches/on_policy/switch.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(manifest))

    def plan(self, dataset, writing=True):
        with contextlib.redirect_stderr(io.StringIO()):
            return default_plan(self.checkout, dataset, os.environ, writing=writing)

    def test_without_sources_the_prepared_plans_are_unchanged(self):
        self.assertEqual(self.plan("math").name, "additional_seeds.json")
        self.assertEqual(self.plan("mbpp").name, "mbpp_seeds.json")

    def test_short_mbpp_evaluation_falls_back_without_partial_bundles(self):
        self.sources("mbpp", count=120)
        self.assertEqual(self.plan("mbpp").name, "mbpp_seeds.json")
        self.assertFalse(list((self.checkout / "srgc_rebuttal/inputs").glob("mbpp-pair-seed-*.json")))

    def test_seed_three_four_caches_skip_cache_tasks_and_run_seed_by_seed(self):
        for dataset in ("math", "mbpp"):
            with self.subTest(dataset=dataset):
                self.sources(dataset)
                self.assertNotIn("pair", self.plan(dataset, writing=False).name)
                plan = self.plan(dataset)
                self.assertIn("pair_seeds", plan.name)
                with contextlib.redirect_stderr(io.StringIO()):
                    target = route_plan(plan, writing=True, start_or_continue=True)
                spec = load_plan(target)
                for seed in spec["seeds"]:
                    bundle = json.loads(input_path(target, spec, seed).read_text())
                    self.assertEqual(len(bundle["cached_rewards"]), 400)
                    self.assertEqual(bundle["provenance"]["reused_from_seed"], 3 if seed % 2 else 4)
                self.assertEqual(self.plan(dataset, writing=False), plan)
                ran = []

                def child(command, log, environment, **kwargs):
                    ran.append(log.stem)
                    queue = cluster.TaskQueue(target)
                    queue.verify()
                    seed, arm = log.stem.split(".")
                    seed = int(seed.removeprefix("seed-"))
                    folder = queue.root / f"seed-{seed}"
                    folder.mkdir(parents=True, exist_ok=True)
                    if arm == "prefix":
                        (folder / "prefix.pt").write_bytes(b"synthetic")
                        atomic_json(folder / "prefix-ready.json", {**queue.identities[seed], "completed_updates": 25,
                                                                  "checkpoint_sha256": digest(folder / "prefix.pt")})
                    else:
                        atomic_json(folder / f"{arm}-endpoint.json",
                                    {**queue.identities[seed], "arm": arm, "total_updates": 275})
                    return 0

                args = SimpleNamespace(plan=target, node_lock_root=self.group / "locks", worker_id=f"test{dataset}",
                                       retry_failed=True, max_attempts=3, retry_delay=0, heartbeat_seconds=.01,
                                       poll_seconds=.01, stall_seconds=1800)
                with seed_first_queue(), \
                        patch.object(cluster, "gpu_identity", return_value=("0,1,2,3", ["a", "b", "c", "d"])), \
                        patch.object(cluster, "device_leases", return_value=contextlib.nullcontext(())), \
                        patch.object(cluster, "admit", return_value={}), \
                        patch.object(cluster, "run_child", side_effect=child), \
                        patch.object(cluster, "publish_reports"), contextlib.redirect_stdout(io.StringIO()):
                    cluster.worker(args)
                self.assertEqual(len(ran), 25)
                self.assertFalse([task for task in ran if task.endswith(".cache")])
                self.assertEqual(ran[:6], ["seed-5.prefix", "seed-5.on_policy", "seed-5.switch", "seed-5.sr",
                                           "seed-5.random", "seed-6.prefix"])
                self.assertEqual(cluster.TaskQueue.__name__, "TaskQueue")


if __name__ == "__main__":
    unittest.main()
