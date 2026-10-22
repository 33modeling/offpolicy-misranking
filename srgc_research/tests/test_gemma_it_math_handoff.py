"""MATH cache completion releases MATH training without waiting for MBPP."""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from srgc_rebuttal import cluster
from srgc_rebuttal.cluster_queue import Task, TaskQueue
from srgc_rebuttal.plan import digest, input_path
from srgc_rebuttal.runtime import atomic_json, lease
from srgc_research.dispatch import gemma_it_run, resume_drain
from srgc_research.dispatch.gemma4_it import adapter, entry
from srgc_research.dispatch.gemma4_it.resume import GemmaITResumeFirst
from srgc_research.tests.test_model_results import make_cohort


@pytest.fixture
def math_scope(tmp_path, monkeypatch):
    cohort = make_cohort("gemma4_it", tmp_path)
    for key, value in cohort.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    monkeypatch.setenv("SRGC_CACHE_ONLY", "1")
    monkeypatch.setenv("SRGC_CACHE_DATASET", "math")
    monkeypatch.setenv("SRGC_RESUME_ONLY", "0")
    with adapter.runtime_adapter():
        queues = [TaskQueue(cohort.plans[name]) for name in ("math", "mbpp")]
        for queue in queues:
            queue.bind()
        yield cohort, queues


def finish_math_caches(queue):
    """Synthetic rewards for CPU dispatcher verification, never experiment results."""
    from srgc_research.dispatch.gemma4_it.protocol import sampling

    for seed in queue.plan["seeds"]:
        path = input_path(queue.plan_path, queue.plan, seed)
        data = json.loads(path.read_text())
        data["cached_rewards"] = {key: [0, 1] * 4 for key in data["candidate_ids"]}
        data["provenance"]["cache"] = {
            **{key: queue.plan[key] for key in (
                "model", "model_revision", "responses", "max_new_tokens", "verifier")},
            "cache_seed": seed, "attention": "eager",
            "gemma_extension": queue.plan["extension"], "sampling": sampling(),
        }
        atomic_json(path, data)
        atomic_json(path.with_suffix(".cache") / "cost-summary.json",
                    {"bundle_sha256": digest(path), "complete": False,
                     "total_gpu_seconds": None, "scope": "synthetic CPU queue test"})
    queue.verify()


def test_math_scope_excludes_mbpp_from_claims_and_pending_work(math_scope):
    _, (math, mbpp) = math_scope
    manager = GemmaITResumeFirst([math, mbpp])
    assert resume_drain.phase_queues([math, mbpp]) == [math]
    assert not manager.dataset_allowed(mbpp)
    with manager.claim(mbpp, retry_failed=True, retry_delay=0) as task:
        assert task is None
    pending = resume_drain.phase_pending(manager, set(manager.tasks))
    assert pending == {(str(math.plan_path.resolve()), Task(seed, "cache").key)
                       for seed in math.plan["seeds"]}


def test_completed_math_cache_pass_returns_even_while_mbpp_cache_is_owned(math_scope, monkeypatch):
    cohort, (math, mbpp) = math_scope
    finish_math_caches(math)
    assert not mbpp.complete(Task(5, "cache"))
    monkeypatch.setattr(cluster, "child_environment",
                        lambda: pytest.fail("MATH cache completion waited on MBPP"))
    monkeypatch.setattr(cluster, "publish_reports", lambda queue: 0)
    updates = []
    worker = SimpleNamespace(write_worker=lambda queues, worker_id, state, **kw: updates.append(state))
    with lease(mbpp.directory / "leases" / "seed-5.cache.lock"):
        resume_drain.worker_loop(
            worker, GemmaITResumeFirst, "gemma4-12b-it-{dataset}.json", "SRGC_GEMMA_PLANS",
            "GEMMA IT", list(cohort.plans.values()), SimpleNamespace(),
            cohort.root, cohort.root, lambda *a, **kw: pytest.fail("unneeded GPU admission"))
    assert updates[-1] == "complete"


def test_math_prefix_is_claimable_with_every_mbpp_cache_missing(math_scope, monkeypatch):
    _, (math, mbpp) = math_scope
    finish_math_caches(math)
    monkeypatch.setenv("SRGC_CACHE_ONLY", "0")
    manager = GemmaITResumeFirst([math, mbpp])
    with manager.claim(math, retry_failed=True, retry_delay=0) as task:
        assert task == Task(5, "prefix")
    assert not any(mbpp.complete(Task(seed, "cache")) for seed in mbpp.plan["seeds"])


def test_public_math_handoff_restricts_cache_then_enters_math_training(math_scope, monkeypatch):
    cohort, _ = math_scope
    monkeypatch.delenv("SRGC_CACHE_ONLY")
    monkeypatch.delenv("SRGC_CACHE_DATASET")
    monkeypatch.setenv("SRGC_GEMMA_CACHE_FIRST", "1")
    stages = []
    with patch.object(entry, "main", side_effect=lambda args: stages.append(
            (os.environ.get("SRGC_CACHE_ONLY"), os.environ.get("SRGC_CACHE_DATASET"))) or 0):
        assert gemma_it_run.main(["all", "--root", str(cohort.root)]) == 0
    assert stages == [("1", "math"), (None, None)]


def test_math_cache_validation_does_not_touch_mbpp_preflight(math_scope):
    cohort, _ = math_scope
    checked = []
    original = adapter.validate_extension
    with patch.object(adapter, "validate_extension", side_effect=lambda path: checked.append(path) or original(path)):
        gemma_it_run.check_saved_preflight(SimpleNamespace(dataset="all", root=cohort.root))
    assert checked == [cohort.plans["math"]]


def test_math_admission_does_not_launch_mbpp_diagnostics(math_scope):
    cohort, _ = math_scope
    environment = {**os.environ, "SRGC_GEMMA_PLAN": str(cohort.plans["math"]),
                   "SRGC_GEMMA_PLANS": json.dumps([str(path) for path in cohort.plans.values()])}
    before = dict(environment)
    seen = []
    def admission(base, root, env, child, **kwargs):
        seen.extend(json.loads(env["SRGC_GEMMA_PLANS"]))
        return {"passed": True}
    assert gemma_it_run.scoped_admission(admission)(None, cohort.root, environment, None) == {"passed": True}
    assert seen == [str(cohort.plans["math"])]
    assert environment == before
