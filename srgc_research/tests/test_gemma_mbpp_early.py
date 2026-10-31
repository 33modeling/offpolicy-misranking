"""Gemma MBPP cache/prefix start while MATH work remains; later MBPP arms keep the barrier."""

import hashlib
import json
from contextlib import ExitStack, nullcontext
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


def test_math_phase_starts_mbpp_prefix_first_and_never_mbpp_arms(gemma):
    math, mbpp = gemma.queues
    assert drain(gemma) is None
    assert gemma.jobs == [("mbpp", "seed-9.prefix"), ("math", "seed-5.prefix"),
                          ("math", "seed-5.on_policy"), ("math", "seed-5.sr")]
    assert all(math.complete(task) for task in math.tasks)
    assert mbpp.complete(Task(9, "prefix")) and not mbpp.complete(Task(9, "sr"))


def test_new_mbpp_stages_go_ahead_of_math_errors(gemma):
    math, _ = gemma.queues
    abandoned(math, Task(5, "sr"), state="failed")
    drain(gemma)
    assert gemma.jobs == [("mbpp", "seed-9.prefix"), ("math", "seed-5.sr"), ("math", "seed-5.on_policy")]


def test_failed_mbpp_prefix_retries_before_math_work(gemma):
    _, mbpp = gemma.queues
    abandoned(mbpp, Task(9, "prefix"), state="failed")
    drain(gemma)
    assert gemma.jobs == [("mbpp", "seed-9.prefix"), ("math", "seed-5.prefix"),
                          ("math", "seed-5.on_policy"), ("math", "seed-5.sr")]


def test_failed_mbpp_prefix_in_retry_delay_lets_math_run(gemma):
    _, mbpp = gemma.queues
    abandoned(mbpp, Task(9, "prefix"), state="failed")
    gemma.options.retry_delay = 3600
    drain(gemma)
    assert gemma.jobs[:3] == [("math", "seed-5.prefix"), ("math", "seed-5.on_policy"), ("math", "seed-5.sr")]


def test_unclaimable_math_backlog_does_not_idle_a_gpu_with_mbpp_prefix(gemma, monkeypatch):
    # The old gate counted every unowned MATH backlog task, even one waiting
    # out its retry delay, and so held fresh MBPP stages: the GPU stayed idle.
    math, mbpp = gemma.queues
    finish_fake(math, Task(5, "prefix"))
    abandoned(math, Task(5, "sr"), state="failed")
    manager = EarlyGemma(gemma.queues)
    with manager.claim(math, retry_failed=True, retry_delay=3600) as task:
        assert task is None
    with manager.claim(mbpp, retry_failed=True, retry_delay=3600) as task:
        assert task == Task(9, "prefix")


@pytest.mark.parametrize("resume_only", ("0", "1"))
def test_chain_child_with_blocked_math_backlog_runs_mbpp_prefix(gemma, monkeypatch, resume_only):
    math, mbpp = gemma.queues
    finish_fake(math, Task(5, "prefix"))
    abandoned(math, Task(5, "sr"), state="failed")
    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    monkeypatch.setenv("SRGC_RESUME_ONLY", resume_only)
    manager = EarlyGemma(gemma.queues)
    pending = resume_drain.phase_pending(manager, manager.pending())
    assert pending
    args = SimpleNamespace(retry_delay=3600)
    assert resume_drain.runnable(manager, [math], pending, args)
    with manager.claim(mbpp, retry_failed=True, retry_delay=3600) as task:
        assert task == Task(9, "prefix")


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
    assert gemma.jobs == [("mbpp", "seed-9.prefix"), ("mbpp", "seed-9.prefix"), ("math", "seed-5.prefix"),
                          ("math", "seed-5.on_policy"), ("math", "seed-5.sr")]


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
    assert manager.task_allowed(mbpp, Task(5, "on_policy"))
    assert not mbpp.ready(Task(5, "on_policy"))  # Its own prefix is still unfinished.
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


@pytest.mark.parametrize("panel,runnable", ((None, True), ("paused", False), ("passed", True)))
def test_resume_only_child_can_verify_unrun_mbpp_panel_before_claim(gemma, monkeypatch, panel, runnable):
    # Admission may verify an unrun panel, while claiming always needs a pass.
    math, mbpp = gemma.queues
    finish_fake(math, Task(5, "prefix"))
    abandoned(math, Task(5, "sr"), state="failed")
    monkeypatch.setattr(EarlyGemma, "panel", panel)
    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    monkeypatch.setenv("SRGC_RESUME_ONLY", "1")
    manager = EarlyGemma(gemma.queues)
    pending = resume_drain.phase_pending(manager, manager.pending())
    args = SimpleNamespace(retry_delay=3600)
    assert resume_drain.runnable(manager, [math], pending, args) is runnable
    with manager.claim(mbpp, retry_failed=True, retry_delay=3600) as task:
        assert task == (Task(9, "prefix") if panel == "passed" else None)


@pytest.mark.parametrize("resume_only", ("0", "1"))
@pytest.mark.parametrize("math_state", ("unfinished", "complete_but_owned"))
def test_requested_mbpp_admission_obeys_the_claims_math_barrier(gemma, monkeypatch, resume_only, math_state):
    math, mbpp = gemma.queues
    abandoned(mbpp, Task(9, "sr"), state="failed")
    if math_state == "complete_but_owned":
        for task in math.tasks:
            finish_fake(math, task)
    monkeypatch.setenv("SRGC_DATASET_PHASE", "mbpp")
    monkeypatch.setenv("SRGC_RESUME_ONLY", resume_only)
    lock = (lease(math.directory / "leases" / "seed-5.sr.lock")
            if math_state == "complete_but_owned" else nullcontext())
    with lock:
        manager = EarlyGemma(gemma.queues)
        pending = resume_drain.phase_pending(manager, manager.pending())
        assert not manager.math_complete()
        with manager.claim(mbpp, retry_failed=True, retry_delay=0) as task:
            assert task is None
        assert not resume_drain.runnable(manager, [mbpp], pending, SimpleNamespace(retry_delay=0))


@pytest.mark.parametrize("resume_only", ("0", "1"))
def test_blocked_requested_mbpp_yields_before_gpu_admission(gemma, monkeypatch, tmp_path, resume_only):
    _, mbpp = gemma.queues
    abandoned(mbpp, Task(9, "sr"), state="failed")
    checkpoint = mbpp.root / "seed-9/prefix.pt"
    before = checkpoint.read_bytes()
    monkeypatch.setenv("SRGC_DATASET_PHASE", "mbpp")
    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    monkeypatch.setenv("SRGC_RESUME_ONLY", resume_only)
    with pytest.raises(SystemExit) as stopped:
        run_worker(gemma, monkeypatch, tmp_path, selected=gemma.queues,
                   probe=lambda: pytest.fail("blocked MBPP child requested GPUs"),
                   admission=lambda *a, **kw: pytest.fail("blocked MBPP child requested admission"))
    assert stopped.value.code == YIELD
    assert gemma.jobs == []
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("resume_only", ("0", "1"))
@pytest.mark.parametrize("panel", (None, "paused", "passed"))
def test_requested_early_mbpp_panel_matches_admission_mode(gemma, monkeypatch, resume_only, panel):
    _, mbpp = gemma.queues
    abandoned(mbpp, Task(9, "prefix"), state="failed")
    monkeypatch.setenv("SRGC_DATASET_PHASE", "mbpp")
    monkeypatch.setenv("SRGC_RESUME_ONLY", resume_only)
    monkeypatch.setattr(EarlyGemma, "panel", panel)
    manager = EarlyGemma(gemma.queues)
    pending = resume_drain.phase_pending(manager, manager.pending())
    expected = panel in {None, "passed"}
    assert resume_drain.runnable(manager, [mbpp], pending, SimpleNamespace(retry_delay=0)) is expected


class LateGemma(EarlyGemma):
    late_mbpp = True


def test_math_phase_runs_mbpp_arms_after_its_math_work(gemma):
    # 2026-10-31: MBPP prefix finished and its arms waited behind one MATH arm.
    math, mbpp = gemma.queues
    gemma.factory = LateGemma
    assert drain(gemma) is None
    assert gemma.jobs == [("mbpp", "seed-9.prefix"), ("math", "seed-5.prefix"), ("math", "seed-5.on_policy"),
                          ("math", "seed-5.sr"), ("mbpp", "seed-9.sr")]
    assert all(queue.complete(task) for queue in (math, mbpp) for task in queue.tasks)


@pytest.mark.parametrize("resume_only", ("0", "1"))
def test_gpu_with_math_owned_elsewhere_takes_an_mbpp_arm(gemma, monkeypatch, resume_only):
    math, mbpp = gemma.queues
    for task in (Task(5, "prefix"), Task(5, "on_policy")):
        finish_fake(math, task)
    finish_fake(mbpp, Task(9, "prefix"))
    monkeypatch.setenv("SRGC_MODEL_CHAIN_CHILD", "1")
    manager = LateGemma(gemma.queues)
    # Another node runs the last MATH arm.
    with manager.claim(math, retry_failed=True, retry_delay=0) as running:
        assert running == Task(5, "sr")
        monkeypatch.setenv("SRGC_RESUME_ONLY", resume_only)
        pending = resume_drain.phase_pending(manager, manager.pending())
        assert resume_drain.runnable(manager, [math], pending, SimpleNamespace(retry_delay=0))
        order = resume_drain.claim_order(manager, [math, mbpp], [math], [mbpp])
        assert order[-1] == (mbpp, "late")
        assert [t for t in mbpp.tasks if resume_drain.in_scope(manager, mbpp, t, "late", pending)] == [Task(9, "sr")]
        with manager.claim(mbpp, retry_failed=True, retry_delay=0) as task:
            assert task == Task(9, "sr")


@pytest.mark.parametrize("panel", (None, "paused"))
def test_mbpp_arms_still_need_a_passed_panel(gemma, monkeypatch, panel):
    _, mbpp = gemma.queues
    finish_fake(mbpp, Task(9, "prefix"))
    monkeypatch.setattr(EarlyGemma, "panel", panel)
    manager = LateGemma(gemma.queues)
    assert not manager.task_allowed(mbpp, Task(9, "sr"))
    assert not resume_drain.late_sibling(manager, mbpp, Task(9, "sr"))


@pytest.mark.parametrize("setting,expected", (("1", True), ("0", False)))
def test_gemma_it_opts_in_to_mbpp_arms_with_early_start(monkeypatch, setting, expected):
    monkeypatch.setenv("SRGC_GEMMA_MBPP_EARLY", setting)
    manager = gemma_it_run.GemmaITEarlyMBPP.__new__(gemma_it_run.GemmaITEarlyMBPP)
    assert manager.late_mbpp is expected


def test_direct_idle_worker_spaces_out_claim_scans_and_keeps_its_heartbeat(gemma, monkeypatch, capsys):
    # Without the chain every idle worker stays resident; 2026-10-31 status
    # stopped updating while all of them rescanned the shared queues each 10 s.
    import time as clock

    math, mbpp = gemma.queues
    finish_fake(mbpp, Task(9, "prefix"))
    finish_fake(mbpp, Task(9, "sr"))
    gemma.options.poll_seconds = 10
    now, slept, heartbeats = [1000.0], [], []

    class Stop(Exception):
        pass

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds
        if sum(slept) > 200:
            raise Stop()

    monkeypatch.setattr(clock, "sleep", sleep)
    monkeypatch.setattr(clock, "monotonic", lambda: now[0])
    monkeypatch.setattr(resume_drain.random, "uniform", lambda low, high: 0)
    original = gemma_worker.write_worker

    def write_worker(queues, worker_id, state, **details):
        heartbeats.append(state)
        return original(queues, worker_id, state, **details)

    monkeypatch.setattr(gemma_worker, "write_worker", write_worker)
    with ExitStack() as locks:
        for task in math.tasks:
            locks.enter_context(lease(math.directory / "leases" / f"{task.key}.lock"))
        with pytest.raises(Stop):
            drain(gemma)
    scans = capsys.readouterr().out.count("IDLE GEMMA")
    assert scans == 4  # at t=0, 60, 120 and 180 s, not every 10 s
    assert max(slept) <= resume_drain.IDLE_HEARTBEAT_SECONDS
    assert heartbeats.count("idle") >= 18  # refreshed every 10 s while waiting
