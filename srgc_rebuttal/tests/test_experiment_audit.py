import copy
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from scripts import srgc_extra_status as status
from scripts import srgc_stage_report as stage_report
from scripts import srgc_support_report as support
from scripts.srgc_resumable_rollouts import ResumableRolloutMixin
from srgc_rebuttal.tests import test_stage_mechanism

complete_endpoint = test_stage_mechanism.complete_endpoint


def support_row(seed, attention="eager", implementation="runtime", **rewards):
    return dict(dataset="math", seed=seed, arms={
        arm: dict(reward_percent=reward, implementation_sha256=implementation,
                  checkpoint_policy={"attention": attention})
        for arm, reward in rewards.items()})


def test_support_never_pools_runtime_or_attention_generations():
    rows = [support_row(5, on_policy=40, direction_removed=30),
            support_row(6, "sdpa", on_policy=30, direction_removed=40),
            support_row(7, implementation="older", on_policy=99, direction_removed=0)]
    comparisons = [r for r in support.summarize(rows) if r["left"] == "on_policy"]
    assert len(comparisons) == 3
    assert all(r["n"] == 1 for r in comparisons)


def test_support_excludes_unverified_and_cross_kernel_pairs():
    bad = support_row(5, on_policy=40, direction_removed=30)
    bad["arms"]["direction_removed"]["checkpoint_policy"]["attention"] = "sdpa"
    unknown = dict(dataset="math", seed=6, arms={
        "on_policy": {"reward_percent": 90}, "direction_removed": {"reward_percent": 0}})
    comparisons = support.summarize([bad, unknown])
    assert all(r["n"] == 0 for r in comparisons)


def test_support_rejects_duplicate_seed_rows():
    row = support_row(5, on_policy=40, direction_removed=30)
    with pytest.raises(ValueError, match="duplicate"):
        support.summarize([row, row])


@pytest.mark.parametrize("field", ["score_correlation", "top4_overlap_fraction", "selected_diagnostics"])
def test_mechanism_recomputes_reported_diagnostics(complete_endpoint, field):
    data, identity, original = complete_endpoint
    value = copy.deepcopy(original)
    measurement = value["measurements"][0]
    if field == "selected_diagnostics":
        measurement[field]["on_policy"]["independent_dot"] += 100
    else:
        measurement[field] = 100
    with pytest.raises(ValueError):
        stage_report.validate_endpoint(value, identity, "prefix", data)


def test_status_retains_later_tasks_when_one_plan_is_invalid():
    from scripts.srgc_replicate_worker import Task
    broken = Task("math", 5, 0, "sr_hold", Path("/missing/plan.json"))
    healthy = Task("math", 6, 0, "sr_hold", Path("/other/plan.json"))
    def inspect(task):
        if task is broken:
            raise ValueError("bad seed plan")
        return dict(task=task.key, state="waiting", seed=6)
    output = io.StringIO()
    with patch.object(status, "tasks_for", return_value=[broken, healthy]), \
            patch.object(status, "inspect", side_effect=inspect), contextlib.redirect_stdout(output):
        rc = status.main(["--dataset", "math", "--scope", "support", "--json"])
    value = json.loads(output.getvalue())
    assert rc == 1
    assert any(row["seed"] == 6 for row in value["rows"])
    assert "bad seed plan" in " ".join(value["errors"])


@pytest.mark.parametrize("bad", ["nan_reward", "reward_count", "negative_start", "bad_sequence"])
def test_corrupt_rollout_arrays_are_rejected(tmp_path, bad):
    path = tmp_path / "cached.npz"
    arrays = dict(count=np.int64(2), rewards=np.array([0., 1.]), start=np.int64(1),
                  seq0=np.array([1, 2]), seq1=np.array([1, 3]))
    if bad == "nan_reward":
        arrays["rewards"][0] = np.nan
    elif bad == "reward_count":
        arrays["rewards"] = np.array([0.])
    elif bad == "negative_start":
        arrays["start"] = np.int64(-1)
    else:
        arrays["seq0"] = np.array([[1, 2]])
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        ResumableRolloutMixin()._load_rollout(path)


@pytest.mark.parametrize("bad", [[], {"attempt": -1}, {"attempt": True}, {"attempt": 1, "finished": "later"},
                                  {"attempt": 1, "finished": float("nan")}])
def test_invalid_queue_receipt_does_not_hide_healthy_tasks(bad):
    from scripts import srgc_replicate_worker as worker
    from srgc_rebuttal.tests.test_extra_arm_launch import ExtraArmLaunchTests
    from srgc_rebuttal.runtime import atomic_json
    fixture = ExtraArmLaunchTests()
    fixture.setUp()
    try:
        plan = fixture.plan()
        fixture.prefix(plan)
        broken, healthy = [worker.Task("math", 5, 0, arm, plan) for arm in ("sr_hold", "switch_repeat")]
        broken.receipt.parent.mkdir(parents=True)
        broken.receipt.write_text(json.dumps(bad))
        before = broken.receipt.read_bytes()
        def run(task, handle):
            assert task == healthy
            atomic_json(task.out / f"{task.arm}-endpoint.json", {"synthetic": True})
            return 0
        counts, code = worker.sweep([broken, healthy], runner=run)
        assert counts["failed"] == 1 and code == 0
        assert broken.receipt.read_bytes() == before
    finally:
        fixture.doCleanups()
