import copy
import json
from unittest.mock import patch

import pytest

from srgc_rebuttal.runtime import atomic_json
from srgc_research.cli import run_queue, status
from srgc_research.design import Condition
from srgc_research.storage import identity, validate_endpoint


def test_timeout_is_bounded_and_does_not_cancel_other_tasks(tmp_path):
    stalled = Condition("stalled", "trajectory", arm="random", updates=0)
    healthy = Condition("healthy", "trajectory", arm="random", updates=0)
    completed, calls = set(), []

    def launch(folder, condition, manifest, handle):
        calls.append(condition.key)
        if condition == stalled:
            raise TimeoutError("no task progress")
        completed.add(condition.key)
        return 0

    with patch("srgc_research.cli.tasks", return_value=[stalled, healthy]), \
            patch("srgc_research.cli.complete", side_effect=lambda f, m, c: c.key in completed), \
            patch("srgc_research.cli.launch", side_effect=launch), patch("srgc_research.cli.time.sleep"):
        assert run_queue([(tmp_path, {})], "n01", poll=0, retry_delay=0) == 2
    assert calls.count("stalled") == 3 and calls.count("healthy") == 1
    receipt = json.loads((tmp_path / "stalled/queue.json").read_text())
    assert receipt["status"] == "failed" and receipt["exit_code"] == 124
    assert "no task progress" in receipt["error"]


@pytest.mark.parametrize("text", ["{", "null", "[]"])
def test_status_isolates_a_corrupt_manifest(tmp_path, text):
    path = tmp_path / "math/seed-5/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    rows = status(tmp_path, ("math",), "n01")
    assert all(r["status"] == "invalid" and r["error"] for r in rows if r["seed"] == 5)
    assert all(r["status"] == "not-started" for r in rows if r["seed"] != 5)


@pytest.mark.parametrize("filename", ["progress.json", "queue.json"])
@pytest.mark.parametrize("value", [None, [], "invalid"])
def test_status_isolates_malformed_task_records(tmp_path, filename, value):
    folder = tmp_path / "math/seed-5"
    condition = Condition("random", "trajectory", arm="random", updates=0)
    atomic_json(folder / "manifest.json", {"dataset": "math", "seed": 5})
    atomic_json(folder / condition.key / filename, value)
    with patch("srgc_research.cli.tasks", return_value=[condition]):
        rows = status(tmp_path, ("math",), "n01")
    assert rows[0]["status"] == "invalid" and filename in rows[0]["error"]
    assert all(r["status"] == "not-started" for r in rows[1:])


def test_endpoint_rejects_nonfinite_aggregate_reward():
    condition = Condition("random", "trajectory", arm="random", updates=0)
    manifest = {k: "test" for k in ("protocol", "dataset", "input_sha256", "source_plan_sha256", "implementation_sha256")}
    manifest.update(seed=5, evaluation_ids=["x"])
    value = {**identity(manifest, condition), "status": "complete", "cost_measurement_complete": True,
             "cost_receipts": {"complete": True, "known_gpu_seconds": {"selection_gpu_seconds": 0., "training_gpu_seconds": 0.},
                               "total_gpu_seconds": 0., "unfinished_phases": []},
             "result": {"history": [], "curve": [{"update": 0, "reward": .5, "per_question_reward": {"x": .5},
                         "binary_samples": {"x": [0, 1]}, "sampling": {"responses": 2}}]}}
    validate_endpoint(value, manifest, condition)
    changed = copy.deepcopy(value)
    changed["result"]["curve"][0]["reward"] = float("nan")
    with pytest.raises(ValueError, match="reward"):
        validate_endpoint(changed, manifest, condition)


def test_status_rejects_manifest_from_a_different_seed(tmp_path):
    atomic_json(tmp_path / "math/seed-5/manifest.json", {"dataset": "mbpp", "seed": 6})
    rows = status(tmp_path, ("math",), "n01")
    assert all(r["status"] == "invalid" for r in rows if r["seed"] == 5)
    from srgc_research.report import collect
    report = collect(tmp_path, ("math",), "n01")
    assert report["errors"] and not report["source_results"] and not report["performance_summary"]
