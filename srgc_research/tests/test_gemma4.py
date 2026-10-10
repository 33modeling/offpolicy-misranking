"""Gemma integration: cohort identity, offline weights, resume, and live CPU model."""

import json
import multiprocessing
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from srgc_rebuttal import cluster_queue, existing_runtime
from srgc_rebuttal.cluster_queue import Task, TaskQueue
from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.runtime import atomic_json, code_digest
from srgc_research.dispatch.gemma4 import adapter, cli, storage, worker
from srgc_research.dispatch.gemma4.resume import resume_first_worker
from srgc_research.dispatch.gemma4.status import show

REPO = Path(__file__).resolve().parents[2]
MODEL_SHA = "a" * 64


def prepared(root, dataset="math"):
    filename = "additional_seeds.json" if dataset == "math" else "mbpp_seeds.json"
    return adapter.prepare(
        dataset,
        REPO / "srgc_rebuttal/experiments" / filename,
        root,
        None,
        snapshot_sha256=MODEL_SHA,
    )


@pytest.mark.parametrize("dataset", ["math", "mbpp"])
def test_same_splits_four_arms_and_budget_with_model_specific_cache(tmp_path, dataset):
    path = prepared(tmp_path, dataset)
    plan = adapter.validate_extension(path)
    source = (
        REPO
        / "srgc_rebuttal/experiments"
        / ("additional_seeds.json" if dataset == "math" else "mbpp_seeds.json")
    )
    original = load_plan(source)
    for key in (
        "seeds",
        "arms",
        "total_updates",
        "shared_prefix_updates",
        "responses",
        "world_size",
        "selection_interval",
        "training_prompts",
        "projection_seed",
    ):
        assert plan[key] == original[key]
    assert plan["model"] == "google/gemma-4-12B"
    assert plan["model_snapshot_sha256"] == MODEL_SHA
    assert plan["lora_targets"] == ["q_proj", "v_proj", "k_proj"]
    assert plan["model_initialization"] == "pretrained"
    for seed in plan["seeds"]:
        bundle = json.loads(input_path(path, plan, seed).read_text())
        prior = json.loads(input_path(source, original, seed).read_text())
        assert bundle["cached_rewards"] == {}
        for key in (
            "candidate_ids",
            "validation_pool_ids",
            "ranking_validation_ids",
            "evaluation_ids",
        ):
            assert bundle[key] == prior[key]
        rid = bundle["candidate_ids"][0]
        assert prior["records"][rid]["prompt"] == bundle["records"][rid]["prompt"]
        assert bundle["records"][rid]["answer"] == prior["records"][rid]["answer"]
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    assert prepared(tmp_path, dataset) == path
    assert all(p.read_bytes() == value for p, value in before.items())


def test_wrong_model_or_changed_adapter_cannot_resume(tmp_path):
    path = prepared(tmp_path)
    before = path.read_bytes()
    with patch.object(adapter, "adapter_digest", return_value="b" * 64):
        with pytest.raises(ValueError, match="adapter differs"):
            adapter.validate_extension(path)
        assert (
            adapter.validate_extension(path, read_only=True)["model"] == adapter.MODEL
        )
    assert path.read_bytes() == before
    spec = json.loads(before)
    spec["model"] = "Qwen/Qwen3.5-9B"
    atomic_json(path, spec)
    with pytest.raises(ValueError, match="adapter differs"):
        adapter.validate_extension(path, read_only=True)


def test_missing_plan_with_existing_run_never_replaces_inputs(tmp_path):
    artifact = tmp_path / "runs/math/prefix-latest.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"preserved")
    with patch.object(cli, "resolve_snapshot") as resolve:
        with pytest.raises(ValueError, match="restore the original plan"):
            cli.prepare_missing(("math",), tmp_path, {})
        resolve.assert_not_called()
    assert artifact.read_bytes() == b"preserved"
    assert not (tmp_path / "inputs").exists()


def test_runtime_hook_routes_all_stages_to_gemma_and_restores_olmo(tmp_path):
    path = prepared(tmp_path)
    original_digest, original_loader, original_queue = (
        code_digest(),
        existing_runtime.load_model,
        cluster_queue.TaskQueue,
    )
    with adapter.runtime_adapter():
        queue = cluster_queue.TaskQueue(path)
        assert len(queue.tasks) == 30
        assert queue.protocol["implementation_sha256"] != original_digest
        for arm in ("cache", "prefix", "random", "sr", "on_policy", "switch"):
            command = adapter.task_command(queue, Task(5, arm))
            assert "--nproc_per_node=4" in command and "--module" in command
            assert "srgc_research.dispatch.gemma4.rank" in command
            assert not any("qwen35" in arg for arg in command)
            if arm != "cache":
                assert command[-1] == "--resume"
    assert code_digest() == original_digest
    assert existing_runtime.load_model is original_loader
    assert cluster_queue.TaskQueue is original_queue


def test_status_is_gpu_free_and_read_only(tmp_path, capsys):
    root = tmp_path / "work/srgc-rebuttal/gemma4-12b-pt-v1"
    for dataset in ("math", "mbpp"):
        prepared(root, dataset)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}):
        assert (
            show(
                "all",
                root,
                {"GROUP_VOLUME": str(tmp_path), "OM_WORK": str(tmp_path / "work")},
            )
            == 0
        )
    text = capsys.readouterr().out
    assert "Gemma-4-12B-PT" in text and "0/60 completed" in text
    assert "MATH" in text and "MBPP" in text and "On-policy" in text
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert not list(tmp_path.rglob("*.html"))


def test_rank_cache_is_separate_and_stays_in_group_volume(tmp_path):
    env = {
        "GROUP_VOLUME": str(tmp_path),
        "LOCAL_RANK": "2",
        "WORLD_SIZE": "4",
        "LOCAL_WORLD_SIZE": "4",
    }
    storage.setup_storage(tmp_path / "work/gemma4-12b-pt-v1", env)
    assert "rank-2" in env["TRITON_CACHE_DIR"]
    assert "gemma-runtime-cache" in env["TRITON_CACHE_DIR"]
    assert all(
        Path(env[key]).is_relative_to(tmp_path)
        for key in ("TMPDIR", "HF_HOME", "TORCH_HOME")
    )


def test_saved_inputs_are_reused_without_preparation_or_download(tmp_path):
    path = prepared(tmp_path)
    env = {}
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    with (
        patch.object(cli, "resolve_snapshot", return_value=(tmp_path, MODEL_SHA)),
        patch.object(adapter, "runtime_packages"),
        patch("srgc_rebuttal.cluster.run_child") as child,
    ):
        assert cli.prepare_missing(("math",), tmp_path, env) == [path]
        child.assert_not_called()
    assert all(p.read_bytes() == data for p, data in before.items())


def test_admission_failure_is_recorded_before_any_training(tmp_path):
    root = tmp_path / "admission"
    original = lambda *args, **kwargs: {"allocated_gpu_seconds": 2.0}
    with pytest.raises(RuntimeError, match="Gemma generation/backward"):
        cli.admit_with_smoke(
            original, root, {"SRGC_GEMMA_PLAN": "/test/plan"}, lambda *args, **kwargs: 1
        )
    receipt = json.loads((root / "gemma-admission.json").read_text())
    assert receipt["gemma_model_smoke"] == "failed"
    assert receipt["allocated_gpu_seconds"] >= 2


def test_one_command_and_status_shortcut_use_shared_python(tmp_path):
    python = tmp_path / "work/.venv-cu126/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    python.chmod(0o755)
    env = {**os.environ, "OM_WORK": str(tmp_path / "work")}
    for args, expected in [
        ([], ["all"]),
        (["math"], ["math"]),
        (["status"], ["all", "status"]),
    ]:
        result = subprocess.run(
            ["sh", str(REPO / "scripts/run_srgc_gemma4.sh"), *args],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(result.stdout) == [
            "-m",
            "srgc_research.dispatch.gemma4.entry",
            *expected,
        ]


def test_explicit_python_failure_never_uses_another_venv(tmp_path):
    result = subprocess.run(
        ["sh", str(REPO / "scripts/run_srgc_gemma4.sh")],
        env={**os.environ, "GEMMA_PYTHON": str(tmp_path / "absent")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "Python not found" in result.stderr


@pytest.mark.parametrize("action", ["run", "resume"])
def test_cli_starts_worker_and_resume_clears_stop_before_restarting(tmp_path, action):
    from scripts import srgc_process_guard as guard

    paths = [prepared(tmp_path, name) for name in ("math", "mbpp")]
    resets = []

    def start(plans, args, *unused):
        assert plans == paths and args.poll_seconds == 10
        assert "srgc_research.dispatch.gemma4.cli" in guard.OWNER_MARKERS
        assert guard.TARGET_MARKERS == ("srgc_research.dispatch.gemma4.rank",)
        table = {
            90001: (
                1,
                os.getuid(),
                "python -m srgc_research.dispatch.gemma4.rank --plan test",
            ),
            90002: (1, os.getuid(), "python scripts/srgc_qwen35_rank.py --plan other"),
        }
        assert guard.orphan_pids(table=table) == [90001]

    with (
        patch.object(cli, "setup_storage", return_value=(tmp_path, tmp_path)),
        patch.object(cli, "prepare_missing", return_value=paths),
        patch("srgc_log_format.uniform_log", side_effect=nullcontext),
        patch(
            "srgc_checkpoint_backup.automatic_backup",
            side_effect=lambda *args: nullcontext(),
        ),
        patch.object(guard, "process_guard", side_effect=lambda *args: nullcontext()),
        patch(
            "srgc_rebuttal.cluster.main",
            side_effect=lambda: resets.append(sys.argv[1:]),
        ) as reset,
        patch.object(worker, "worker", side_effect=start) as launch,
    ):
        assert cli.main(["all", action, "--root", str(tmp_path)]) == 0
        launch.assert_called_once()
        assert reset.call_count == (2 if action == "resume" else 0)
    if action == "resume":
        assert resets == [["resume", "--plan", str(path)] for path in paths]


def claim_parallel(path, barrier, output):
    with adapter.runtime_adapter():
        queue = cluster_queue.TaskQueue(Path(path))
        queue.bind()
        barrier.wait(10)
        with queue.claim() as task:
            output.put(task.key)
            barrier.wait(10)


def test_three_nodes_never_claim_the_same_gemma_cache(tmp_path):
    path = prepared(tmp_path)
    context = multiprocessing.get_context("spawn")
    barrier, output = context.Barrier(3), context.Queue()
    processes = [
        context.Process(target=claim_parallel, args=(str(path), barrier, output))
        for _ in range(3)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(15)
        if process.is_alive():
            process.kill()
            process.join()
        assert process.exitcode == 0
    jobs = [output.get(timeout=3) for _ in processes]
    assert len(set(jobs)) == 3 and all(job.endswith(".cache") for job in jobs)


def test_math_only_worker_respects_mbpp_resume_backlog(tmp_path):
    from srgc_rebuttal.tests.test_cluster import write_inputs
    from srgc_research.tests.test_qwen_resume import abandoned

    directory = tmp_path / "experiments"
    directory.mkdir()
    source = write_inputs(directory)
    queues = []
    for name in ("math", "mbpp"):
        path = directory / f"gemma4-12b-pt-{name}.json"
        plan = json.loads(source.read_text())
        plan["output_root"] = f"../runs/{name}"
        atomic_json(path, plan)
        queue = TaskQueue(path)
        queue.bind()
        queues.append(queue)
    abandoned(queues[1], Task(9, "prefix"))

    def drain(current, args, *unused):
        assert args.retry_failed
        with current[0].claim() as task:
            assert task is None  # MBPP resume blocks fresh MATH work.

    with (
        patch.object(worker, "drain", side_effect=drain) as original,
        resume_first_worker(),
    ):
        worker.drain(
            [queues[0]],
            SimpleNamespace(retry_failed=False),
            {},
            (),
            "test",
            lambda *args: None,
        )
    assert original.call_count == 1



def test_initial_candidate_accuracy_is_separate_from_held_out_scores(tmp_path, capsys):
    root = tmp_path / 'work/srgc-rebuttal/gemma4-12b-pt-v1'
    plan_path = prepared(root)
    plan = adapter.validate_extension(plan_path)
    path = input_path(plan_path, plan, 5)
    data = json.loads(path.read_text())
    data['cached_rewards'] = {candidate: [1] * (7 if index < 160 else 6) + [0] * (1 if index < 160 else 2)
                              for index, candidate in enumerate(data['candidate_ids'])}
    data['provenance']['cache'] = {key: plan[key] for key in ('model','model_revision','responses','max_new_tokens','verifier','attention')}
    data['provenance']['cache']['cache_seed'] = 5
    atomic_json(path, data)
    env = {'GROUP_VOLUME': str(tmp_path), 'OM_WORK': str(tmp_path/'work')}
    assert show('math', root, env, as_json=True) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['datasets'][0]['initial_candidate_accuracy'] == [{'seed':5, 'responses':3200, 'success_percent':80.0}]
    assert show('math', root, env) == 0
    text = capsys.readouterr().out
    assert 'initial candidate success' in text and 'seed 5 80.0%' in text
