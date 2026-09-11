import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from test_evidence_downstream import source_point
from test_low_order_backend import TinyLoRA

import low_order_experiment as le

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path, monkeypatch):
    run, test = source_point(tmp_path, drift=100)
    config = le.ed.read(run / "run_config.json")
    config.update(behavior_k=8, val_k=8, proj_dim=4, grad_layers=1, clip_cap=10., top_p=1.)
    le.ed.atomic_json(run / "run_config.json", config)
    torch.save(torch.ones(8, 4), run / "val_groups.pt")
    source = le.ed.read(run / "prompts.json")
    for field, name in (("train", "rollouts_behavior_train"), ("val", "rollouts_fresh_val")):
        data = [{"prompt_idx": i, "rollout_idx": j, "input_ids": [1, 2 if j%2 else 3],
                 "resp_start": 1, "resp_end": 2, "reward": j%2} for i in range(len(source[field])) for j in range(8)]
        (run / f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in data))
    monkeypatch.setattr(le, "validate_generation_contract", lambda *a: {"fixture": True})
    monkeypatch.setattr(le.ae, "validate_generation_contract", lambda *a: {"fixture": True})
    root = tmp_path / "low-order"
    return run, test, root


def prepare(tmp_path, monkeypatch):
    run, test, root = fixture(tmp_path, monkeypatch)
    le.prepare(root, [run], evaluation=test, derivative="autograd", geometry="identity", random_extra_steps=7)
    out, = le.entries(root)
    return run, test, root, out


def score_all(out, monkeypatch):
    loads = []
    def load(*args):
        loads.append(1)
        return TinyLoRA(), None
    monkeypatch.setattr(le.backend, "load_current", load)
    for shard in range(4):
        le.validation_worker(out, shard)
    le.merge_direction(out)
    for shard in range(4):
        le.score_worker(out, shard)
    return loads


def test_prepare_is_frozen_and_preserves_source(tmp_path, monkeypatch):
    run, test, root = fixture(tmp_path, monkeypatch)
    before = {str(p): le.ed.digest(p) for p in run.rglob("*") if p.is_file()}
    first = le.prepare(root, [run], evaluation=test, derivative="autograd", geometry="identity")
    assert first == le.prepare(root, [run], evaluation=test, derivative="autograd", geometry="identity")
    assert before == {str(p): le.ed.digest(p) for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="contract changed"):
        le.prepare(root, [run], evaluation=test, geometry="identity")


def test_source_restrictions_and_separate_outputs(tmp_path, monkeypatch):
    run, test, root = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="input location"):
        le.prepare(run / "bad", [run], evaluation=test)
    with pytest.raises(ValueError, match="contain a source"):
        le.prepare(tmp_path, [run], evaluation=test)
    config = le.ed.read(run / "run_config.json")
    config["behavior_k"] = 4
    le.ed.atomic_json(run / "run_config.json", config)
    with pytest.raises(ValueError, match="K=8"):
        le.prepare(root, [run], evaluation=test)
    assert not root.exists()


def test_cpu_scoring_resume_merge_and_training_contracts(tmp_path, monkeypatch):
    run, _, root, out = prepare(tmp_path, monkeypatch)
    before = {str(p): le.ed.digest(p) for p in run.rglob("*") if p.is_file()}
    loads = score_all(out, monkeypatch)
    count = len(loads)
    for shard in range(4):
        le.validation_worker(out, shard)
        le.score_worker(out, shard)
    assert len(loads) == count, "resume must not load GPU models for completed shards"
    le.merge(out)
    le.merge(out)
    le.verify_selection(out)
    selected = le.ed.read(out / "selection.json")["selected"]
    assert set(selected) == set(le.ARMS)
    assert all(len(v) == 4 for v in selected.values())
    for arm in ("random", "pair_u2", "low_order"):
        folder = out / "training" / arm
        c = le.ed.read(folder / "experiment.json")
        assert c["steps"] == (107 if arm == "random" else 100)
        args = le.ed.train_args(le.ed.read(run / "run_config.json"), run, folder, arm, c["steps"])
        assert args[args.index("--target-steps")+1] == ("207" if arm == "random" else "200")
        assert args[args.index("--resume-adapter")+1] == str(run / "policy_step_100")
        assert len(le.ed.read(folder / "subsets" / f"subset-{arm}.json")["train"]) == 4
    assert before == {str(p): le.ed.digest(p) for p in run.rglob("*") if p.is_file()}
    le.status(root)
    assert le.ed.read(root / "results.json")["matched_total_cost"] is False


def test_partial_results_do_not_publish_selection(tmp_path, monkeypatch):
    _, _, _, out = prepare(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="incomplete"):
        le.merge(out)
    assert not (out / "selection.json").exists()


def test_interruption_reuses_completed_scores_and_beta_cache(tmp_path, monkeypatch):
    _, _, _, out = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(le.backend, "load_current", lambda *a: (TinyLoRA(), None))
    for s in range(4):
        le.validation_worker(out, s)
    le.merge_direction(out)
    original = le.backend.exact_directional
    calls = 0
    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated GPU failure")
        return original(*args)
    monkeypatch.setattr(le.backend, "exact_directional", interrupted)
    with pytest.raises(RuntimeError, match="GPU failure"):
        le.score_worker(out, 0)
    assert (out / "scores/p0.json").exists()
    assert (out / "cache/beta-4.pt").exists()
    assert not (out / "scores/p4.json").exists()
    monkeypatch.setattr(le.backend, "exact_directional", original)
    le.score_worker(out, 0)
    assert not le.ed.read(out / "scores/p4.json")["cost"]["beta_evaluated_now"]


def test_changed_source_or_direction_rejected(tmp_path, monkeypatch):
    run, _, _, out = prepare(tmp_path, monkeypatch)
    score_all(out, monkeypatch)
    (out / "direction.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="invalid completed score"):
        le.completed(out, 0, le.ed.digest(out / "experiment.json"))
    (run / "rollouts_fresh_val.jsonl").write_text("")
    with pytest.raises(ValueError, match="source input changed"):
        le.verify(out)


def test_training_contract_mutation_is_rejected(tmp_path, monkeypatch):
    _, _, _, out = prepare(tmp_path, monkeypatch)
    score_all(out, monkeypatch)
    le.merge(out)
    path = out / "training/random/experiment.json"
    contract = le.ed.read(path)
    contract["steps"] += 1
    le.ed.atomic_json(path, contract)
    with pytest.raises(ValueError, match="frozen training input"):
        le.verify_selection(out)


def test_validation_shard_substitution_is_rejected(tmp_path, monkeypatch):
    _, _, _, out = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(le.backend, "load_current", lambda *a: (TinyLoRA(), None))
    for shard in range(4):
        le.validation_worker(out, shard)
    (out / "validation/part-1.pt").write_bytes((out / "validation/part-0.pt").read_bytes())
    with pytest.raises(ValueError, match="shard identities"):
        le.merge_direction(out)


def test_failed_training_arm_yields_to_other_arms(tmp_path, monkeypatch):
    _, _, root, out = prepare(tmp_path, monkeypatch)
    score_all(out, monkeypatch)
    le.merge(out)
    monkeypatch.setenv("OM_NODE_LOCK_HELD", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    attempted = []
    def phase(root, out, arm, stage, *args):
        attempted.append((arm, stage))
        return int(arm == "random" and stage == "train")
    monkeypatch.setattr(le, "phase", phase)
    monkeypatch.setattr(le.ed, "arm_policy", lambda *a: Path("unused-test-policy"))
    assert le.work(root, "train") == 1
    assert ("pair_u2", "train") in attempted
    assert ("low_order", "train") in attempted
    assert ("low_order", "evaluate") in attempted
    assert (out / "error-random.json").exists()


def test_failed_subprocess_terminates_other_worker_promptly(tmp_path):
    root = tmp_path / "suite"
    out = root / "point"
    out.mkdir(parents=True)
    commands = [[sys.executable, "-c", "raise SystemExit(7)"],
                [sys.executable, "-c", "import time; time.sleep(60)"]]
    start = time.monotonic()
    assert le.phase(root, out, "arm", "score", commands, ["0", "1"], dict(os.environ)) == 1
    assert time.monotonic()-start < 8
    cost = [json.loads(line) for line in (root / "cost.jsonl").read_text().splitlines()]
    assert cost[-1]["exit_code"] == 1
    assert cost[-1]["allocated_gpu_seconds"] > 0


def test_launcher_cpu_modes_are_available_without_cluster(tmp_path):
    env = {**os.environ, "OM_WORK": str(tmp_path), "LOW_ORDER_PYTHON": sys.executable,
           "CUDA_VISIBLE_DEVICES": "", "GROUP_VOLUME": str(tmp_path / "absent")}
    result = subprocess.run(["bash", "scripts/run_low_order.sh", "plan"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0
    assert "K=8, G=8" in result.stdout
    assert not (tmp_path / "runs").exists()
    result = subprocess.run(["bash", "scripts/run_low_order.sh", "status"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0
    assert "not prepared" in result.stdout
