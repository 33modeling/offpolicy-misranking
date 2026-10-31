"""Dispatch readiness and public handoff: synthetic CPU data, no GPU work."""
import json
import os
from unittest.mock import patch

import pytest

from srgc_rebuttal.cluster_queue import Task
from srgc_rebuttal.runtime import lease
from srgc_research.dispatch import gemma_it_run
from srgc_research.dispatch.gemma4_it import entry
from srgc_research.dispatch.model_chain import YIELD
from srgc_research.tests.test_gemma_mbpp_early import (
    EarlyGemma,
    drain,
)
from srgc_research.tests.test_gemma_mbpp_early import (
    gemma as gemma,  # noqa: PLC0414 -- fixture registration
)
from srgc_research.tests.test_gemma_mbpp_early import (
    it_math as it_math,  # noqa: PLC0414 -- fixture registration
)
from srgc_research.tests.test_resume_drain import run_worker


def test_resume_only_admission_can_bootstrap_an_unrun_mbpp_panel(it_math):
    cohort, _ = it_math
    environment = {**cohort.env, "SRGC_DATASET_PHASE": "math", "SRGC_RESUME_ONLY": "1",
                   "SRGC_CACHE_ONLY": "0"}
    assert gemma_it_run.early_preflight_plan(list(cohort.plans.values()), environment) == cohort.plans["mbpp"]


def test_resume_only_admission_really_includes_the_unrun_mbpp_panel(it_math):
    cohort, _ = it_math
    environment = {**cohort.env, "SRGC_DATASET_PHASE": "math", "SRGC_RESUME_ONLY": "1",
                   "SRGC_CACHE_ONLY": "0", "SRGC_GEMMA_PLAN": str(cohort.plans["math"]),
                   "SRGC_GEMMA_PLANS": json.dumps([str(p) for p in cohort.plans.values()])}
    seen = []

    def admission(base, root, env, child, **kwargs):
        seen.extend(json.loads(env["SRGC_GEMMA_PLANS"]))
        return {"passed": True}

    gemma_it_run.scoped_admission(admission)(None, cohort.root, environment, None)
    assert seen == [str(cohort.plans["math"]), str(cohort.plans["mbpp"])]


def test_resume_only_worker_with_no_math_backlog_runs_early_mbpp_only(gemma, monkeypatch, tmp_path):
    monkeypatch.setenv("SRGC_RESUME_ONLY", "1")
    run_worker(gemma, monkeypatch, tmp_path, selected=gemma.queues)
    assert gemma.jobs == [("mbpp", "seed-9.prefix")]
    assert not gemma.queues[0].complete(Task(5, "prefix"))


def test_resume_only_drain_with_no_math_backlog_runs_early_mbpp_only(gemma, monkeypatch):
    monkeypatch.setenv("SRGC_RESUME_ONLY", "1")
    assert drain(gemma) is None
    assert gemma.jobs == [("mbpp", "seed-9.prefix")]
    assert not gemma.queues[0].complete(Task(5, "prefix"))


def test_idle_worker_bootstraps_mbpp_when_math_is_owned_and_panel_is_unrun(gemma, monkeypatch, tmp_path):
    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    monkeypatch.setenv("SRGC_RESUME_ONLY", "1")
    monkeypatch.setattr(EarlyGemma, "panel", None)
    math, _ = gemma.queues
    calls = []

    def admission(*args, **kwargs):
        calls.append("panel")
        EarlyGemma.panel = "passed"
        return {"passed": True}

    with lease(math.directory / "leases/seed-5.prefix.lock"):
        run_worker(gemma, monkeypatch, tmp_path, admission=admission, selected=gemma.queues)
    assert calls == ["panel"]
    assert gemma.jobs == [("mbpp", "seed-9.prefix")]


@pytest.mark.parametrize("yield_style", ["return", "raise"])
@pytest.mark.parametrize("fallback_code", [0, YIELD])
def test_public_cache_wait_enters_mbpp_path_without_completing_math(it_math, monkeypatch, yield_style, fallback_code):
    cohort, _ = it_math
    monkeypatch.setenv("SRGC_GEMMA_CACHE_FIRST", "1")
    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    seen = []

    def child(args):
        seen.append((os.environ.get("SRGC_CACHE_ONLY"), os.environ.get("SRGC_RESUME_ONLY")))
        if len(seen) == 1:
            if yield_style == "raise":
                raise SystemExit(YIELD)
            return YIELD
        return fallback_code

    with patch.object(entry, "main", side_effect=child):
        code = gemma_it_run.main(["all", "--root", str(cohort.root)])
    assert code == YIELD
    assert seen == [("1", "0"), ("0", "1")]


@pytest.mark.parametrize("code", [1, 77])
def test_cache_failure_or_pause_never_opens_mbpp_fallback(it_math, monkeypatch, code):
    cohort, _ = it_math
    monkeypatch.setenv("SRGC_GEMMA_CACHE_FIRST", "1")
    with patch.object(entry, "main", return_value=code) as child:
        assert gemma_it_run.main(["all", "--root", str(cohort.root)]) == code
    assert child.call_count == 1


@pytest.mark.parametrize("enabled,expected", [("1", 20), ("0", 0)])
def test_forty_contenders_can_claim_all_twenty_ready_mbpp_arms_while_three_math_tasks_run(
        it_math, monkeypatch, enabled, expected):
    from contextlib import ExitStack

    from srgc_rebuttal.tests.test_cluster import finish_fake
    from srgc_research.tests.test_gemma_it_math_handoff import finish_math_caches
    from srgc_research.tests.test_gemma_mbpp_early import write_panel

    cohort, (math, mbpp) = it_math
    monkeypatch.setenv("SRGC_GEMMA_MBPP_EARLY", enabled)
    monkeypatch.setenv("SRGC_RESUME_ONLY", "1")
    finish_math_caches(math)
    # MBPP uses the pinned code verifier as well as the shared cache protocol.
    from srgc_rebuttal.plan import digest, input_path
    from srgc_rebuttal.runtime import atomic_json
    from srgc_research.dispatch.gemma4_it.protocol import CODE_VERSION, sampling

    for seed in mbpp.plan["seeds"]:
        path = input_path(mbpp.plan_path, mbpp.plan, seed)
        bundle = json.loads(path.read_text())
        bundle["cached_rewards"] = {key: [0, 1] * 4 for key in bundle["candidate_ids"]}
        bundle["provenance"]["cache"] = {
            **{key: mbpp.plan[key] for key in (
                "model", "model_revision", "responses", "max_new_tokens", "verifier")},
            "cache_seed": seed, "attention": "eager", "gemma_extension": mbpp.plan["extension"],
            "sampling": sampling(), "code_verifier_version": CODE_VERSION,
        }
        atomic_json(path, bundle)
        atomic_json(path.with_suffix(".cache") / "cost-summary.json",
                    {"bundle_sha256": digest(path), "complete": False})
    mbpp.verify()
    write_panel(cohort, "mbpp", correct=2)
    remaining = [Task(seed, "switch") for seed in (5, 6, 7)]
    for queue in (math, mbpp):
        for seed in queue.plan["seeds"]:
            finish_fake(queue, Task(seed, "prefix"))
            if queue is math:
                for arm in queue.plan["arms"]:
                    task = Task(seed, arm)
                    if task not in remaining:
                        finish_fake(queue, task)
    immutable = {str(p): p.read_bytes() for queue in (math, mbpp)
                 for p in queue.root.glob("seed-*/prefix.pt")}
    manager = gemma_it_run.GemmaITEarlyMBPP([math, mbpp])
    with ExitStack() as locks:
        for task in remaining:
            locks.enter_context(lease(math.directory / "leases" / (task.key + ".lock")))
        assert not manager.math_complete()
        jobs = [locks.enter_context(manager.claim(mbpp, retry_failed=True, retry_delay=0,
                                                  worker_id="cpu-contender-" + str(i))) for i in range(40)]
        claimed = [task.key for task in jobs if task is not None]
        assert len(claimed) == expected
        assert len(set(claimed)) == expected
        assert all(not name.endswith((".cache", ".prefix")) for name in claimed)
    from pathlib import Path

    assert all(Path(name).read_bytes() == content for name, content in immutable.items())
