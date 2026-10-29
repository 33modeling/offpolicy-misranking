"""Gemma MBPP cache/prefix start while MATH work remains; later MBPP arms keep the barrier."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from srgc_rebuttal import cluster
from srgc_rebuttal.cluster_queue import Task, TaskQueue
from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.runtime import atomic_json, lease, run_root
from srgc_rebuttal.tests.test_cluster import finish_fake, write_inputs
from srgc_research.dispatch import gemma_it_run, resume_drain
from srgc_research.dispatch.gemma4 import worker as gemma_worker
from srgc_research.dispatch.gemma4_it import adapter, entry, preflight
from srgc_research.dispatch.model_chain import YIELD
from srgc_research.dispatch.ready_dispatch import GemmaResumeFirst
from srgc_research.tests.test_gemma_it_math_handoff import finish_math_caches
from srgc_research.tests.test_model_results import make_cohort
from srgc_research.tests.test_qwen_resume import abandoned
from srgc_research.tests.test_resume_drain import run_worker


class EarlyGemma(GemmaResumeFirst):
    early_arms = ("cache", "prefix")
    panel = "passed"

    def early_state(self, queue):
        return type(self).panel


@pytest.fixture
def gemma(tmp_path, monkeypatch):
    parent = tmp_path / "experiments"
    parent.mkdir()
    queues = []
    for dataset in ("math", "mbpp"):
        folder = tmp_path / dataset
        folder.mkdir()
        plan = json.loads(write_inputs(folder).read_text())
        plan.update(dataset="math_train" if dataset == "math" else "mbpp", output_root=str(folder / "runs"),
                    input_pattern=str(folder / "inputs-{seed}.json"))
        if dataset == "mbpp":
            plan["verifier"] = "srgc_rebuttal.verifiers:code_reward"
        path = parent / f"gemma4-12b-pt-{dataset}.json"
        atomic_json(path, plan)
        queue = TaskQueue(path)
        queue.bind()
        queues.append(queue)
    math, mbpp = queues
    math.tasks = [Task(5, "prefix"), Task(5, "on_policy"), Task(5, "sr")]
    mbpp.tasks = [Task(9, "prefix"), Task(9, "sr")]
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    for key in ("SRGC_CACHE_ONLY", "SRGC_CACHE_DATASET", "SRGC_RESUME_ONLY", "SRGC_MODEL_CHAIN_CHILD"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(EarlyGemma, "panel", "passed")
    options = SimpleNamespace(max_attempts=3, retry_delay=0, heartbeat_seconds=.01,
                              stall_seconds=30, poll_seconds=.01, retry_failed=False)
    return SimpleNamespace(queues=queues, jobs=[], worker=gemma_worker, factory=EarlyGemma, options=options,
                           pattern="gemma4-12b-pt-{dataset}.json", env_key="SRGC_GEMMA_PLANS")


def drain(model, child=None):
    """The public drain with fake children; synthetic endpoints, not experiment results."""
    def complete(command, log, environment, **kwargs):
        queue, task = command
        model.jobs.append(("math" if queue.plan["dataset"] == "math_train" else "mbpp", task.key))
        code = child(queue, task) if child else 0
        if not code:
            finish_fake(queue, task)
        return code

    with patch.object(cluster, "gpu_identity"), \
            patch.object(cluster, "task_command", side_effect=lambda q, t: (q, t)), \
            patch.object(cluster, "run_child", side_effect=complete), \
            patch.object(cluster, "publish_reports"), \
            resume_drain.resume_worker(model.worker, model.factory, pattern=model.pattern,
                                      env_key=model.env_key, label="GEMMA"):
        # Both queues are passed so the cohort keeps these narrowed task lists.
        return model.worker.drain(model.queues, model.options, {}, (), "test", lambda *a, **kw: None)


def test_math_phase_starts_mbpp_prefix_before_fresh_math_arms_and_never_mbpp_arms(gemma):
    math, mbpp = gemma.queues
    assert drain(gemma) is None
    assert gemma.jobs == [("math", "seed-5.prefix"), ("mbpp", "seed-9.prefix"),
                          ("math", "seed-5.on_policy"), ("math", "seed-5.sr")]
    assert all(math.complete(task) for task in math.tasks)
    assert mbpp.complete(Task(9, "prefix")) and not mbpp.complete(Task(9, "sr"))


def test_math_errors_stay_ahead_of_new_mbpp_stages(gemma):
    math, _ = gemma.queues
    abandoned(math, Task(5, "sr"), state="failed")
    drain(gemma)
    assert gemma.jobs == [("math", "seed-5.sr"), ("mbpp", "seed-9.prefix"), ("math", "seed-5.on_policy")]


def test_failed_mbpp_prefix_retries_only_after_math_work(gemma):
    _, mbpp = gemma.queues
    abandoned(mbpp, Task(9, "prefix"), state="failed")
    drain(gemma)
    assert gemma.jobs == [("math", "seed-5.prefix"), ("math", "seed-5.on_policy"),
                          ("math", "seed-5.sr"), ("mbpp", "seed-9.prefix")]


def test_chain_child_keeps_math_running_after_an_mbpp_prefix_error(gemma, monkeypatch):
    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    failures = []

    def child(queue, task):
        if queue.plan["dataset"] == "mbpp" and not failures:
            failures.append(task.key)
            return 1
        return 0

    assert drain(gemma, child) is None
    assert failures == ["seed-9.prefix"]
    assert gemma.jobs == [("math", "seed-5.prefix"), ("mbpp", "seed-9.prefix"), ("math", "seed-5.on_policy"),
                          ("math", "seed-5.sr"), ("mbpp", "seed-9.prefix")]


@pytest.mark.parametrize("panel", (None, "paused"))
def test_mbpp_stages_need_a_passed_panel(gemma, monkeypatch, panel):
    monkeypatch.setattr(EarlyGemma, "panel", panel)
    drain(gemma)
    assert gemma.jobs == [("math", "seed-5.prefix"), ("math", "seed-5.on_policy"), ("math", "seed-5.sr")]


def test_opt_out_models_keep_the_full_math_barrier(gemma, monkeypatch):
    monkeypatch.setattr(EarlyGemma, "early_arms", ())
    drain(gemma)
    assert gemma.jobs == [("math", "seed-5.prefix"), ("math", "seed-5.on_policy"), ("math", "seed-5.sr")]


@pytest.mark.parametrize("phase", ("mbpp", None, "math"))
def test_mbpp_arm_keeps_the_math_barrier(gemma, monkeypatch, phase):
    math, mbpp = gemma.queues
    finish_fake(mbpp, Task(9, "prefix"))
    if phase is None:
        monkeypatch.delenv("SRGC_DATASET_PHASE")
    else:
        monkeypatch.setenv("SRGC_DATASET_PHASE", phase)
    manager = EarlyGemma(gemma.queues)
    assert not manager.task_allowed(mbpp, Task(9, "sr"))
    with manager.claim(mbpp, retry_failed=True, retry_delay=0) as task:
        assert task is None
    for task in math.tasks:
        finish_fake(math, task)
    # A MATH-phase worker never takes MBPP arms; the MBPP phase does.
    assert manager.task_allowed(mbpp, Task(9, "sr")) is (phase != "math")
    with manager.claim(mbpp, retry_failed=True, retry_delay=0) as task:
        assert task == (None if phase == "math" else Task(9, "sr"))


@pytest.mark.parametrize("panel,admitted", ((None, True), ("passed", True), ("paused", False)))
def test_chain_child_admits_for_mbpp_prefix_only_while_its_panel_can_pass(gemma, monkeypatch, tmp_path,
                                                                          panel, admitted):
    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    math, _ = gemma.queues
    finish_fake(math, Task(5, "prefix"))
    monkeypatch.setattr(EarlyGemma, "panel", panel)
    calls = []

    def admission(*args, **kwargs):
        calls.append(kwargs["plan"]["dataset"])
        EarlyGemma.panel = "passed"  # The MBPP panel that this admission ran passed.
        return {"passed": True}

    with lease(math.directory / "leases" / "seed-5.on_policy.lock"), \
            lease(math.directory / "leases" / "seed-5.sr.lock"), \
            pytest.raises(SystemExit) as stopped:
        run_worker(gemma, monkeypatch, tmp_path, admission=admission, selected=gemma.queues)
    assert stopped.value.code == YIELD
    assert calls == (["math_train"] if admitted else [])
    assert gemma.jobs == ([("mbpp", "seed-9.prefix")] if admitted else [])


@pytest.fixture
def it_math(tmp_path, monkeypatch):
    cohort = make_cohort("gemma4_it", tmp_path)
    for key, value in cohort.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SRGC_DATASET_PHASE", "math")
    monkeypatch.setenv("SRGC_RESUME_ONLY", "0")
    for key in ("SRGC_CACHE_ONLY", "SRGC_CACHE_DATASET", "SRGC_GEMMA_MBPP_EARLY", "SRGC_MODEL_CHAIN_CHILD"):
        monkeypatch.delenv(key, raising=False)
    gemma_it_run.PANEL_STATES.clear()
    with adapter.runtime_adapter():
        queues = [TaskQueue(cohort.plans[name]) for name in ("math", "mbpp")]
        for queue in queues:
            queue.bind()
        yield cohort, queues
    gemma_it_run.PANEL_STATES.clear()


def write_panel(cohort, dataset, correct):
    """Synthetic fixed-panel receipts for CPU dispatch checks, never model results."""
    path = cohort.plans[dataset]
    plan = load_plan(path)
    data = json.loads(input_path(path, plan, plan["seeds"][0]).read_text())
    protocol = preflight.identity(path, plan, data)
    directory = preflight.folder(path, plan)
    rows = []
    for candidate in protocol["candidate_ids"]:
        row = {"protocol": protocol, "id": candidate,
               "rewards": [1] * correct + [0] * (plan["responses"] - correct),
               "responses": ["synthetic dispatch fixture"] * plan["responses"],
               "token_ids": [[9, 106]] * plan["responses"]}
        rows.append(row)
        atomic_json(directory / f"{hashlib.sha256(candidate.encode()).hexdigest()}.json", row)
    atomic_json(directory / "summary.json", preflight.summarize(protocol, rows))


def test_it_mbpp_cache_then_prefix_claimable_while_math_prefixes_remain(it_math):
    cohort, (math, mbpp) = it_math
    finish_math_caches(math)
    write_panel(cohort, "mbpp", correct=2)
    manager = gemma_it_run.GemmaITEarlyMBPP([math, mbpp])
    assert manager.dataset_allowed(mbpp)
    with manager.claim(mbpp, retry_failed=True, retry_delay=0) as task:
        assert task == Task(5, "cache")
    finish_math_caches(mbpp)
    with manager.claim(mbpp, retry_failed=True, retry_delay=0) as task:
        assert task == Task(5, "prefix")
    assert not manager.task_allowed(mbpp, Task(5, "on_policy"))
    with manager.claim(math, retry_failed=True, retry_delay=0) as task:
        assert task == Task(5, "prefix")


@pytest.mark.parametrize("correct,state", ((None, None), (0, "paused")))
def test_it_mbpp_stays_closed_until_its_panel_passes(it_math, correct, state):
    cohort, (math, mbpp) = it_math
    finish_math_caches(math)
    if correct is not None:
        write_panel(cohort, "mbpp", correct=correct)
    manager = gemma_it_run.GemmaITEarlyMBPP([math, mbpp])
    assert manager.early_state(mbpp) == state
    assert manager.dataset_allowed(mbpp) is (state is None)
    assert manager.task_allowed(mbpp, Task(5, "cache"), admitted=False) is (state is None)
    with manager.claim(mbpp, retry_failed=True, retry_delay=0) as task:
        assert task is None


@pytest.mark.parametrize("setting", ("cache-only", "opt-out"))
def test_it_cache_only_stage_and_opt_out_keep_mbpp_behind_math(it_math, monkeypatch, setting):
    cohort, (math, mbpp) = it_math
    write_panel(cohort, "mbpp", correct=2)
    if setting == "cache-only":
        monkeypatch.setenv("SRGC_CACHE_ONLY", "1")
        monkeypatch.setenv("SRGC_CACHE_DATASET", "math")
    else:
        monkeypatch.setenv("SRGC_GEMMA_MBPP_EARLY", "0")
    manager = gemma_it_run.GemmaITEarlyMBPP([math, mbpp])
    assert not manager.dataset_allowed(mbpp)
    with manager.claim(mbpp, retry_failed=True, retry_delay=0) as task:
        assert task is None


def admit_plans(cohort, admission, run_child=lambda *a, **kw: 0):
    environment = {"SRGC_DATASET_PHASE": "math", "SRGC_GEMMA_PLAN": str(cohort.plans["math"]),
                   "SRGC_GEMMA_PLANS": json.dumps([str(path) for path in cohort.plans.values()])}
    before = dict(environment)
    result = gemma_it_run.scoped_admission(admission)(None, cohort.root, environment, run_child)
    assert environment == before
    return result


def test_math_training_admission_adds_only_an_unrun_mbpp_panel(it_math):
    cohort, (_, mbpp) = it_math
    seen = []

    def admission(base, root, env, child, **kwargs):
        seen.append(json.loads(env["SRGC_GEMMA_PLANS"]))
        return {"passed": True}

    math_plan, mbpp_plan = str(cohort.plans["math"]), str(cohort.plans["mbpp"])
    assert admit_plans(cohort, admission) == {"passed": True}
    write_panel(cohort, "mbpp", correct=2)
    admit_plans(cohort, admission)
    atomic_json(run_root(cohort.plans["mbpp"], load_plan(cohort.plans["mbpp"])) / ".queue" / "stop.json",
                {"immediate": False, "requested": 0})
    gemma_it_run.PANEL_STATES.clear()
    (preflight.folder(cohort.plans["mbpp"], load_plan(cohort.plans["mbpp"])) / "summary.json").unlink()
    admit_plans(cohort, admission)
    assert seen == [[math_plan, mbpp_plan], [math_plan], [math_plan]]


def test_failed_mbpp_panel_readmits_math_alone(it_math, capsys):
    cohort, _ = it_math
    seen = []

    def admission(base, root, env, child, **kwargs):
        plans = json.loads(env["SRGC_GEMMA_PLANS"])
        seen.append(plans)
        for plan in plans:
            log = Path(root) / f"gemma-preflight-{Path(plan).stem}.log"
            child(["preflight", plan], log, env)
            if load_plan(Path(plan))["dataset"] == "mbpp":
                raise RuntimeError(f"Gemma init preflight failed; full cache/training paused: {log}")
        return {"passed": True}

    assert admit_plans(cohort, admission) == {"passed": True}
    assert seen == [[str(cohort.plans["math"]), str(cohort.plans["mbpp"])], [str(cohort.plans["math"])]]
    assert "MATH continues and MBPP cache/prefix stay closed" in capsys.readouterr().out


def test_math_panel_failure_still_stops_admission(it_math):
    cohort, _ = it_math

    def admission(base, root, env, child, **kwargs):
        log = Path(root) / f"gemma-preflight-{cohort.plans['math'].stem}.log"
        child(["preflight"], log, env)
        raise RuntimeError(f"Gemma init preflight failed; full cache/training paused: {log}")

    with pytest.raises(RuntimeError, match="init preflight failed"):
        admit_plans(cohort, admission)


def test_saved_failed_math_panel_still_stops_before_gpu(it_math):
    cohort, _ = it_math
    write_panel(cohort, "math", correct=0)
    with pytest.raises(ValueError, match="preflight failed in the saved math panel"):
        gemma_it_run.check_saved_preflight(SimpleNamespace(dataset="all", root=cohort.root))
    write_panel(cohort, "math", correct=2)
    assert gemma_it_run.check_saved_preflight(SimpleNamespace(dataset="all", root=cohort.root)) is None


def test_public_it_run_dispatches_with_early_mbpp_claims(it_math, monkeypatch):
    cohort, _ = it_math
    factories = []
    original = resume_drain.resume_worker

    def recording(worker, factory, **kwargs):
        factories.append(factory)
        return original(worker, factory, **kwargs)

    monkeypatch.setattr(resume_drain, "resume_worker", recording)
    with patch.object(entry, "main", return_value=0):
        assert gemma_it_run.main(["all", "--root", str(cohort.root)]) == 0
    assert factories and set(factories) == {gemma_it_run.GemmaITEarlyMBPP}
