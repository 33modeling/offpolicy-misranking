"""Real prepared cohorts, two-dataset endpoints, immutable input and one output."""

import hashlib
import importlib
import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import srgc_qwen35 as qwen
from srgc_rebuttal import reports
from srgc_rebuttal.plan import digest, input_path, load_plan
from srgc_rebuttal.runtime import atomic_json, run_root
from srgc_research.dispatch import model_results as export
from srgc_research.dispatch import resume_drain
from srgc_research.tests.test_gemma4 import prepared as gemma_prepared
from srgc_research.tests.test_llama31 import prepared as llama_prepared
from srgc_research.tests.test_qwen_status import ChatTokenizer

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(params=("qwen35", "gemma4", "llama31"))
def cohort(request, tmp_path):
    model = request.param
    work = tmp_path / "work with spaces"
    root = work / "srgc-rebuttal" / model
    names = export.CONFIGS[model]
    adapter = importlib.import_module(names[0])
    plans = {}
    for name in ("math", "mbpp"):
        if model == "qwen35":
            filename = "additional_seeds.json" if name == "math" else "mbpp_seeds.json"
            plans[name] = qwen.prepare(name, REPO / "srgc_rebuttal/experiments" / filename, root, ChatTokenizer())
        else:
            prepare = gemma_prepared if model == "gemma4" else llama_prepared
            plans[name] = prepare(root, name)
    env_key = {"qwen35": "SRGC_QWEN_ROOT", "gemma4": "SRGC_GEMMA_ROOT", "llama31": "SRGC_LLAMA_ROOT"}[model]
    env = {**os.environ, "GROUP_VOLUME": str(tmp_path), "OM_WORK": str(work), env_key: str(root)}
    return SimpleNamespace(model=model, work=work, root=root, plans=plans, env=env, adapter=adapter)


def finish(cohort, dataset, seed=5, arm="on_policy", reward=.5):
    path = cohort.plans[dataset]
    plan = load_plan(path)
    bundle_path = input_path(path, plan, seed)
    bundle = json.loads(bundle_path.read_text())
    bundle["cached_rewards"] = {rid: [0, 1] * 4 for rid in bundle["candidate_ids"]}
    bundle["provenance"]["cache"] = {"model": cohort.adapter.MODEL, "model_revision": cohort.adapter.REVISION,
        "responses": 8, "max_new_tokens": plan["max_new_tokens"], "cache_seed": seed,
        "verifier": plan["verifier"], "attention": "eager"}
    atomic_json(bundle_path, bundle)
    identity = {"seed": seed, "plan_sha256": digest(path), "input_sha256": digest(bundle_path),
        "implementation_sha256": hashlib.sha256((cohort.adapter.engine_digest() + cohort.adapter.adapter_digest()).encode()).hexdigest()}
    folder = run_root(path, plan) / f"seed-{seed}"
    checkpoint = folder / "prefix.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic immutable checkpoint")
    atomic_json(folder / "run.json", {**identity, "status": "running"})
    prefix = {**identity, "completed_updates": plan["shared_prefix_updates"], "checkpoint_sha256": digest(checkpoint)}
    atomic_json(folder / "prefix-ready.json", prefix)
    endpoint = {**identity, "arm": arm, "total_updates": plan["total_updates"],
        "shared_prefix_updates": plan["shared_prefix_updates"], "prefix_checkpoint_sha256": prefix["checkpoint_sha256"],
        "reward": reward, "per_question_reward": {rid: reward for rid in bundle["evaluation_ids"]}, "switched_at": None}
    atomic_json(folder / f"{arm}-endpoint.json", endpoint)
    return folder


def saved(cohort):
    return {p: p.read_bytes() for p in cohort.root.rglob("*") if p.is_file()}


def read_result(cohort):
    return json.loads((cohort.work / "results" / f"{cohort.model}-results.json").read_text())


def test_both_datasets_yield_one_compact_file_with_outcomes_and_costs(cohort):
    for name in ("math", "mbpp"):
        finish(cohort, name)
    before = saved(cohort)
    with patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}):
        assert export.export(cohort.model, environment=cohort.env) == 0
    result = read_result(cohort)
    assert result["format"] == "compact"
    assert set(result["datasets"]) == {"math", "mbpp"}
    assert result["coverage"] == {"completed_continuations": 2, "planned_continuations": 40}
    assert not result["complete"]
    for name in ("math", "mbpp"):
        report = result["datasets"][name]
        assert len(report["endpoints"]) == 1 and len(report["costs"]) == 5
        assert len(report["endpoints"][0]["endpoint"]["per_question_reward"]) == 300
        assert report["requested_dataset"] == name
        assert "workers" not in report
        assert all("progress" not in task and "log" not in task for task in report["tasks"])
        assert report["arm_statistics"]["on_policy"]["mean_reward_percent"] is None
    assert saved(cohort) == before
    assert [p.name for p in (cohort.work / "results").iterdir()] == [f"{cohort.model}-results.json"]
    assert not list(cohort.work.rglob("*.txt")) and not list(cohort.work.rglob("*.csv"))


def test_raw_endpoint_arrays_are_excluded_without_changing_rewards(cohort):
    for name in ("math", "mbpp"):
        folder = finish(cohort, name)
        path = folder / "on_policy-endpoint.json"
        endpoint = json.loads(path.read_text())
        endpoint.update(history=[{"sequence_ids": list(range(20000)), "logps": [-.4] * 20000}] * 8,
                        raw_responses=["raw completion " * 10000] * 4,
                        checks=[{"step": 25, "d": -.01}], selection_steps=[25])
        atomic_json(path, endpoint)
    before = saved(cohort)
    assert export.export(cohort.model, environment=cohort.env) == 0
    result = read_result(cohort)
    text = (cohort.work / "results" / f"{cohort.model}-results.json").read_text()
    assert len(text.encode()) < 150000
    for name in ("math", "mbpp"):
        endpoint = result["datasets"][name]["endpoints"][0]["endpoint"]
        assert endpoint["reward"] == .5 and len(endpoint["per_question_reward"]) == 300
        assert endpoint["checks"] == [{"step": 25, "d": -.01}] and endpoint["selection_steps"] == [25]
    assert "sequence_ids" not in text and "logps" not in text and "raw_responses" not in text
    assert saved(cohort) == before
    assert export.compact(result) == result


def test_corrupt_endpoint_is_not_exported_as_a_valid_reward(cohort):
    folder = finish(cohort, "math")
    path = folder / "on_policy-endpoint.json"
    endpoint = json.loads(path.read_text())
    endpoint["reward"] = .8
    atomic_json(path, endpoint)
    assert export.export(cohort.model, environment=cohort.env) == 1
    result = read_result(cohort)
    assert result["datasets"]["math"]["endpoints"] == [] and result["errors"]


def test_completed_export_retains_null_costs_when_not_measured(cohort):
    finish(cohort, "math")
    assert export.export(cohort.model, "math", environment=cohort.env) == 0
    result = read_result(cohort)
    row = next(row for row in result["datasets"]["math"]["tasks"] if row["task"] == "seed-5.on_policy")
    assert row["status"] == "complete" and not row["cost_complete"]
    assert row["selection_training_preparation_gpu_seconds"] is None


def test_endpoint_replacement_during_snapshot_is_reported_and_excluded(cohort):
    folder = finish(cohort, "math")
    path = folder / "on_policy-endpoint.json"
    original = reports.snapshot

    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        if result["dataset"] in {"math", "math_train"}:
            data = json.loads(path.read_text())
            atomic_json(path, {**data, "changed_during_export": True})
        return result
    with patch.object(reports, "snapshot", side_effect=change):
        assert export.export(cohort.model, environment=cohort.env) == 1
    result = read_result(cohort)
    assert result["datasets"]["math"]["endpoints"] == []
    assert any("changed during" in error for error in result["errors"])


def test_public_shell_results_uses_cpu_and_does_not_need_model_interpreter(cohort):
    script = {"qwen35": "run_srgc_qwen35.sh", "gemma4": "run_srgc_gemma4.sh", "llama31": "run_srgc_llama31.sh"}[cohort.model]
    env = {**cohort.env, "QWEN_PYTHON": "/missing", "GEMMA_PYTHON": "/missing", "LLAMA_PYTHON": "/missing"}
    for args in (("all", "results"), ("results",)):
        result = subprocess.run(["sh", str(REPO / "scripts" / script), *args], cwd="/tmp", env=env,
                                capture_output=True, text=True, timeout=30, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "RESULT:" in result.stdout
    assert [p.name for p in (cohort.work / "results").iterdir()] == [f"{cohort.model}-results.json"]


def test_missing_dataset_still_produces_one_partial_file(cohort):
    cohort.plans["mbpp"].unlink()
    assert export.export(cohort.model, environment=cohort.env) == 1
    result = read_result(cohort)
    assert set(result["datasets"]) == {"math", "mbpp"}
    assert not result["datasets"]["mbpp"]["prepared"] and result["errors"]


def test_automatic_publication_uses_the_same_artifact(cohort):
    from srgc_rebuttal import cluster

    queue = SimpleNamespace(plan_path=cohort.plans["math"])
    worker = SimpleNamespace(drain=lambda *a: None)
    previous = cluster.publish_reports
    with patch.dict(os.environ, cohort.env, clear=True), \
            resume_drain.resume_worker(worker, None, pattern="unused", env_key="unused", label="unused", result_model=cohort.model):
        assert cluster.publish_reports(queue) == 0
    assert cluster.publish_reports is previous
    assert [p.name for p in (cohort.work / "results").iterdir()] == [f"{cohort.model}-results.json"]


def test_export_failure_cannot_fail_successful_training(cohort, capsys):
    queue = SimpleNamespace(plan_path=cohort.plans["math"])
    with patch.object(export, "export", side_effect=OSError("export disk unavailable")):
        assert export.publish(cohort.model, queue) == 1
    assert "RESULT export failed" in capsys.readouterr().out


def test_full_matrix_completes_in_one_file_without_pooling_dataset_rewards(cohort):
    for name, reward in (("math", .3), ("mbpp", .7)):
        plan = load_plan(cohort.plans[name])
        for seed in plan["seeds"]:
            for arm in plan["arms"]:
                finish(cohort, name, seed, arm, reward)
    assert export.export(cohort.model, environment=cohort.env) == 0
    result = read_result(cohort)
    assert result["complete"] and result["coverage"]["completed_continuations"] == 40
    for name, reward in (("math", 30), ("mbpp", 70)):
        report = result["datasets"][name]
        assert report["complete"] and len(report["endpoints"]) == 20
        assert all(stats["mean_reward_percent"] == pytest.approx(reward) for stats in report["arm_statistics"].values())
    assert [p.name for p in (cohort.work / "results").iterdir()] == [f"{cohort.model}-results.json"]
    assert (cohort.work / "results" / f"{cohort.model}-results.json").stat().st_size < 1000000


def test_export_inside_live_adapter_does_not_double_hash_scientific_identity(cohort):
    finish(cohort, "math")
    with cohort.adapter.runtime_adapter():
        assert export.export(cohort.model, environment=cohort.env) == 0
    result = read_result(cohort)
    assert not any("code changed since queue initialization" in warning for warning in result["warnings"])


def test_two_nodes_serialize_snapshot_and_publish_to_the_same_file(cohort):
    context = multiprocessing.get_context("fork")
    event, trace = context.Event(), context.Queue()
    original = export._export

    def guarded(*args, **kwargs):
        trace.put(("start", os.getpid()))
        time.sleep(.05)
        result = original(*args, **kwargs)
        trace.put(("end", os.getpid()))
        return result

    def worker():
        event.wait(10)
        with patch.object(export, "_export", side_effect=guarded):
            assert export.export(cohort.model, environment=cohort.env) == 0

    nodes = [context.Process(target=worker) for _ in range(2)]
    try:
        for node in nodes:
            node.start()
        event.set()
        for node in nodes:
            node.join(15)
            assert node.exitcode == 0
        events = [trace.get(timeout=2) for _ in range(4)]
        assert [event[0] for event in events] == ["start", "end", "start", "end"]
        assert events[0][1] == events[1][1] and events[2][1] == events[3][1]
        assert set(read_result(cohort)["datasets"]) == {"math", "mbpp"}
        assert [p.name for p in (cohort.work / "results").iterdir()] == [f"{cohort.model}-results.json"]
    finally:
        for node in nodes:
            if node.is_alive():
                node.kill()
                node.join()
