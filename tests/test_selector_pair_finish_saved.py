"""Evaluate saved final policies without changing the original Pair experiment."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
import selector_pair_finish_saved as finish
import test_mbpp_budget_recovery as legacy


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root = tmp_path / "original"
    _, old_directory, c, _ = legacy.fixture(root / "prefixes", monkeypatch)
    seed = c["config"]["seed"]
    monkeypatch.setattr(finish, "TARGETS", {seed: (100, "random_full")})
    directory = finish.source_directory(root, seed)
    directory.parent.mkdir(parents=True)
    for p in old_directory.parent.iterdir():
        dest = directory.parent / p.name
        if p.is_dir():
            shutil.copytree(p, dest)
        else:
            shutil.copyfile(p, dest)
    monkeypatch.setattr(finish.base, "verify", lambda _: c)
    core = finish.core
    core.atomic_json(directory.parent / "evaluation.json", c["evaluation"])
    core.atomic_json(directory.parent.parent.parent / "net_protocol.json", legacy.protocol())
    core.atomic_json(directory.parent.parent.parent / "suite.json", {"eval_timeout": 5})
    core.atomic_json(root / "branches/on_policy/switch.json", {"curve": {"points": 1, "k": 8}})
    # The old recovery plan pins a missing checkpoint, and a separate result
    # remains. Neither may be rewritten to point at the newer final policy.
    core.atomic_json(directory / "budget-recovery/plan.json", {
        "runner_sha256": "old-runtime", "points": [{"adapter": str(directory / "policy/checkpoint-000145")}]})
    core.atomic_json(directory / "budget-recovery/result.json", {"old_measurement": "preserve"})
    (directory / ".task.lock").write_text("existing lock contents")
    (directory / ".cost.lock").write_text("existing cost lock contents")
    expected = finish.saved.checkpoint_contract(directory.parent, c, directory.name)
    archived = directory / "policy/curve-checkpoints/step-125"
    archived.mkdir(parents=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copyfile(directory / "policy" / name, archived / name)
    core.atomic_json(archived / "checkpoint_state.json", {
        **expected, "completed_steps": 125,
        "adapter_sha256": finish.base.digest(archived / "adapter_model.safetensors")})
    return root, tmp_path / "new-final-evaluation", directory, seed, c


@pytest.fixture
def fake_gpu(monkeypatch):
    import additive_experiment  # import real dependencies before substituting GPU calls
    calls = []
    def collect(model, tokenizer, prompts, k, tokens, temperature, path, *, idx_offset, sampling_seed_base):
        rows = [{"prompt_idx": idx_offset+i, "rollout_idx": j, "reward": float(j % 2)}
                for i in range(len(prompts)) for j in range(k)]
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    monkeypatch.setitem(sys.modules, "rollout", SimpleNamespace(
        load_policy=lambda *args: (None, None), collect_rollouts=collect))
    original = finish.base.meter
    def meter(target, phase, gpu, *, commands, env, timeout, ledger):
        assert ledger == "reporting" and phase.startswith("evaluate-")
        def act():
            for command, device in commands:
                assert "train" not in command
                def opt(name): return command[command.index(name)+1]
                index, shard = int(opt("--point")), int(opt("--shard"))
                calls.append((index, shard))
                finish.evaluate(Path(opt("--root")), Path(opt("--output")), int(opt("--seed")), index, shard)
        return original(target, phase, gpu, action=act, ledger=ledger)
    monkeypatch.setattr(finish.base, "meter", meter)
    return calls


def test_finishes_new_policy_not_stale_plan_and_preserves_every_source_byte(experiment, fake_gpu):
    root, output, directory, seed, _ = experiment
    before = snapshot(root)
    result = finish.finish(root, output, seed, list("0123"))
    assert result["completed_steps"] == 150
    assert result["evaluation_complete"] and not result["canonical_complete"]
    assert result["training_performed"] is False
    assert [p["step"] for p in result["points"]] == [100, 125, 150]
    assert len(fake_gpu) == 12
    assert snapshot(root) == before
    assert not (directory / "result.json").exists()
    assert result["new_evaluation_cost"]["ledgers"]["deployment"]["gpu_seconds"] == 0
    assert result["new_evaluation_cost"]["ledgers"]["reporting"]["gpu_seconds"] > 0
    saved = snapshot(output)
    assert finish.finish(root, output, seed, list("0123")) == result
    assert len(fake_gpu) == 12 and snapshot(output) == saved


def test_reuses_sealed_original_shard_without_writing_source(experiment, fake_gpu):
    root, output, directory, seed, c = experiment
    plan = finish.make_plan(root, seed)
    item = plan["points"][-1]
    source = directory / "evaluation/shard-0.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("".join(json.dumps({"prompt_idx": 0, "rollout_idx": j, "reward": 0.5})+"\n"
                              for j in range(item["k"])))
    finish.core.atomic_json(source.with_suffix(".done.json"), {"binding": {
        "experiment_sha256": finish.base.digest(directory.parent / "contract.json"),
        "adapter_sha256": item["hashes"]["adapter_model.safetensors"], "arm": directory.name,
        "policy_manifest_sha256": item["hashes"]["policy_train.json"], "shard": 0},
        "sha256": finish.base.digest(source)})
    before = snapshot(root)
    finish.finish(root, output, seed, list("0123"))
    assert (2, 0) not in fake_gpu and len(fake_gpu) == 11
    assert snapshot(root) == before


@pytest.mark.parametrize("name", ["adapter_model.safetensors", "optimizer.pt", "grpo_stats.jsonl"])
def test_damaged_final_policy_never_reaches_gpu(experiment, fake_gpu, name):
    root, output, directory, seed, _ = experiment
    (directory / "policy" / name).write_text("damaged")
    before = snapshot(root)
    with pytest.raises(ValueError):
        finish.finish(root, output, seed, list("0123"))
    assert not fake_gpu and snapshot(root) == before


def test_output_inside_source_or_symlink_is_rejected_without_writes(experiment):
    root, output, _, _, _ = experiment
    before = snapshot(root)
    with pytest.raises(ValueError):
        finish.checked_output(root, root / "new-results")
    output.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        finish.checked_output(root, output)
    assert snapshot(root) == before


def test_active_original_worker_is_not_stopped_or_bypassed(experiment, fake_gpu):
    root, output, directory, seed, _ = experiment
    before = snapshot(root)
    with (directory / ".task.lock").open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            finish.finish(root, output, seed, list("0123"))
    assert not fake_gpu and snapshot(root) == before


def test_interrupted_evaluation_resumes_only_missing_shards(experiment, fake_gpu, monkeypatch):
    root, output, _, seed, _ = experiment
    original = finish.evaluate
    def fail_once(root, output, seed, point, shard):
        if (point, shard) == (0, 2):
            raise RuntimeError("preemption")
        return original(root, output, seed, point, shard)
    monkeypatch.setattr(finish, "evaluate", fail_once)
    before = snapshot(root)
    with pytest.raises(RuntimeError, match="preemption"):
        finish.finish(root, output, seed, list("0123"))
    assert fake_gpu == [(0, 0), (0, 1), (0, 2)]
    fake_gpu.clear()
    monkeypatch.setattr(finish, "evaluate", original)
    result = finish.finish(root, output, seed, list("0123"))
    assert result["evaluation_complete"] and len(fake_gpu) == 10
    assert (0, 0) not in fake_gpu and (0, 1) not in fake_gpu
    assert snapshot(root) == before


def test_changed_input_cannot_replace_new_frozen_plan(experiment, fake_gpu):
    root, output, directory, seed, _ = experiment
    finish.finish(root, output, seed, list("0123"))
    saved_output = snapshot(output)
    old = directory / "budget-recovery/result.json"
    old.write_text('{"concurrent": "change"}')
    before = snapshot(root)
    with pytest.raises(ValueError, match="frozen contract changed"):
        finish.finish(root, output, seed, list("0123"))
    assert snapshot(output) == saved_output and snapshot(root) == before


def test_embedded_runner_matches_source():
    shell = finish.HERE.with_name("finish_selector_pair_two.sh").read_text()
    source = shell.split("<<'PAIR_FINISH_PYTHON'\n", 1)[1].split("\nPAIR_FINISH_PYTHON\n", 1)[0]+"\n"
    assert source == finish.HERE.read_text()


def test_second_finisher_cannot_share_an_active_output(experiment, fake_gpu):
    root, output, _, seed, _ = experiment
    target = output / f"seed-{seed}"
    target.mkdir(parents=True)
    with (target / ".finish.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = snapshot(root)
        with pytest.raises(BlockingIOError):
            finish.finish(root, output, seed, list("0123"))
    assert snapshot(root) == before and not fake_gpu


@pytest.mark.parametrize("code", [0, 17])
def test_standalone_launch_disables_source_cleanup_and_preserves_status(tmp_path, code):
    repo = tmp_path / "existing checkout"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts/mbpp_budget_recovery.py").write_text("# runtime marker\n")
    (repo / "scripts/_e5_node.sh").write_text(
        "e5_cleanup_lock_helpers() { return 91; }\n"
        "e5_recover_pair_gpu() { return 92; }\n"
        "e5_acquire_node() { e5_cleanup_lock_helpers; e5_recover_pair_gpu; }\n")
    commands = tmp_path / "bin"
    commands.mkdir()
    python = commands / "python"
    python.write_text('#!/bin/bash\nif [[ "$1" == src/bootstrap_math_verify.py ]]; then\n'
                      '  echo /tmp/existing-verifier; exit 0\nfi\n'
                      'printf "test evaluation: %s\\n" "$*"\n'
                      'exit "${TEST_EXIT_CODE}"\n')
    nvidia = commands / "nvidia-smi"
    nvidia.write_text('#!/bin/bash\nprintf "0\\n0\\n0\\n0\\n"\n')
    for p in (python, nvidia):
        p.chmod(0o755)
    before = snapshot(repo)
    env = {**os.environ, "PATH": f"{commands}:{os.environ['PATH']}", "PAIR_PYTHON": str(python),
           "PAIR_FINISH_REPO": str(repo), "OM_WORK": str(tmp_path / "work"),
           "CUDA_VISIBLE_DEVICES": "0,1,2,3", "TEST_EXIT_CODE": str(code)}
    result = subprocess.run(["bash", str(finish.HERE.with_name("finish_selector_pair_two.sh")),
                             "run", "--seed", "4"], cwd=repo, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == code, result.stdout+result.stderr
    assert "test evaluation:" in result.stdout and "--seed 4" in result.stdout
    report = Path(next(s.removeprefix("[log] ") for s in result.stdout.splitlines() if s.startswith("[log] /")))
    try:
        assert "test evaluation:" in report.read_text()
    finally:
        report.unlink()
    assert snapshot(repo) == before
