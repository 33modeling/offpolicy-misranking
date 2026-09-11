import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_evidence_downstream import source_point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import additive_experiment as ae


def fixture(tmp_path, monkeypatch):
    run, _ = source_point(tmp_path)
    config = ae.ed.read(run / "run_config.json")
    config.update(proj_dim=4, grad_layers=1, clip_cap=10.)
    ae.ed.atomic_json(run / "run_config.json", config)
    torch.save(torch.ones(8, 4), run / "val_groups.pt")
    rows = [{"prompt_idx": i, "rollout_idx": j, "input_ids": [1, 2, 3],
             "resp_start": 1, "resp_end": 3, "reward": j % 2} for i in range(40) for j in range(2)]
    (run / "rollouts_behavior_train.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    monkeypatch.setattr(ae, "validate_generation_contract", lambda *args: {"fixture": True})
    root = tmp_path / "additive"
    return run, root


def fake_model_runtime(monkeypatch):
    loaded = []
    monkeypatch.setenv("OM_NODE_LOCK_HELD", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    def load(model, adapter):
        loaded.append(adapter)
        return {"adapter": adapter}, None
    monkeypatch.setattr(ae, "load_policy", load)
    monkeypatch.setattr(ae, "grad_params", lambda *args: [])
    def logps(model, rows, **kw):
        return [torch.full((r["input_ids"].numel()-r["resp_start"],), -1. if model["adapter"] is None else -.9) for r in rows]
    monkeypatch.setattr(ae, "sequence_logprobs_batch", logps)
    def grad(model, params, rows, weights, spec, **kw):
        g = torch.zeros(spec.dim)
        for j, w in enumerate(weights):
            g[j % spec.dim] += w.sum()
        return g
    monkeypatch.setattr(ae, "prompt_gradient", grad)
    return loaded


def test_prepare_freezes_all_sources_and_leaves_them_unchanged(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    before = {str(p): ae.ed.digest(p) for p in run.rglob("*") if p.is_file()}
    first = ae.prepare(root, [run])
    assert first == ae.prepare(root, [run])
    assert before == {str(p): ae.ed.digest(p) for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="contract changed"):
        ae.prepare(root, [run], micro_batch=2)
    out, = ae.suite_entries(root)
    assert ae.verify_point(out)["no_generation"]
    (run / "scores_offpolicy.json").write_text("{}")
    with pytest.raises(ValueError, match="source input changed"):
        ae.verify_point(out)


def test_output_must_be_separate_and_preparation_failure_is_not_partial(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="refusing"):
        ae.prepare(run / "additive", [run])
    with pytest.raises(ValueError, match="contain a source"):
        ae.prepare(tmp_path, [run])
    with pytest.raises(ValueError, match="not complete"):
        ae.prepare(root, [run, tmp_path / "missing"])
    assert not root.exists()


def test_worker_resume_shards_merge_and_saved_gradient_integrity(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    ae.prepare(root, [run])
    out, = ae.suite_entries(root)
    loaded = fake_model_runtime(monkeypatch)
    ae.worker(out, 0, 4)
    assert len(loaded) == 2
    ae.worker(out, 0, 4)
    assert len(loaded) == 2, "completed shard must not load either model"
    with pytest.raises(ValueError, match="incomplete"):
        ae.merge(out)
    for shard in (1, 2, 3):
        ae.worker(out, shard, 4)
    first = ae.merge(out)
    assert first == ae.merge(out)
    report = ae.ed.read(out / "comparison.json")
    assert {row["method"] for row in report["rows"]} >= {"gadd", "tay2_terminal", "g00", "g11", "random"}
    assert report["cost"]["end_to_end_complete"] is False
    for method in ae.METHODS:
        subset = ae.ed.read(out / "subsets" / f"subset-{method}.json")
        assert len(subset["train"]) == 4 and len(subset["selected_idx"]) == 4
    (out / "scores/p0.pt").write_bytes(b"broken")
    with pytest.raises(ValueError, match="gradients changed"):
        ae.merge(out)


def test_interruption_reuses_beta_cache_not_half_scores(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    ae.prepare(root, [run])
    out, = ae.suite_entries(root)
    loaded = fake_model_runtime(monkeypatch)
    original = ae.prompt_gradient
    calls = 0
    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated CUDA failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(ae, "prompt_gradient", fail)
    with pytest.raises(RuntimeError, match="CUDA"):
        ae.worker(out, 0, 4)
    assert (out / "scores/p0.json").exists()
    assert not (out / "scores/p4.json").exists()
    monkeypatch.setattr(ae, "prompt_gradient", original)
    ae.worker(out, 0, 4)
    assert len(loaded) == 3 and loaded[-1] is not None


def test_wrong_or_invalid_cache_is_rejected(tmp_path):
    path = tmp_path / "cache.pt"
    rows = [{"input_ids": torch.ones(3), "resp_start": 1}]
    assert ae.cached_logps(path, "bound", rows) is None
    torch.save({"binding": "other", "logps": [torch.ones(2)]}, path)
    with pytest.raises(ValueError, match="binding"):
        ae.cached_logps(path, "bound", rows)
    torch.save({"binding": "bound", "logps": [torch.ones(3)]}, path)
    with pytest.raises(ValueError, match="invalid behavior"):
        ae.cached_logps(path, "bound", rows)


def test_worker_fails_before_model_loading_without_gpu_admission(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    ae.prepare(root, [run])
    out, = ae.suite_entries(root)
    monkeypatch.delenv("OM_NODE_LOCK_HELD", raising=False)
    with pytest.raises(ValueError, match="admitted GPU"):
        ae.worker(out, 0, 4)


def test_preflight_catches_bad_validation_before_any_gpu_work(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    torch.save(torch.ones(7, 4), run / "val_groups.pt")
    with pytest.raises(ValueError, match="multiple of four"):
        ae.prepare(root, [run])
    assert not root.exists()


def test_preflight_rejects_old_scores_without_ranking_split(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    halves = ae.ed.read(run / "scores_splithalf.json")
    for value in halves.values():
        value.pop("r", None)
    ae.ed.atomic_json(run / "scores_splithalf.json", halves)
    with pytest.raises(ValueError, match="matched R"):
        ae.prepare(root, [run])
    assert not root.exists()


def test_coordinator_skips_busy_point_and_continues_after_error(tmp_path, monkeypatch):
    outs = []
    for index in range(3):
        out = tmp_path / f"point{index}"
        out.mkdir()
        ae.ed.atomic_json(out / "experiment.json", {"seed": index})
        outs.append(out)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("OM_NODE_LOCK_HELD", "1")
    monkeypatch.setattr(ae, "suite_entries", lambda _: iter(outs))
    monkeypatch.setattr(ae, "verify_point", lambda _: {"n": 4, "config": {}})
    monkeypatch.setattr(ae, "completed_prompt", lambda *args: None)
    monkeypatch.setattr(ae.signal, "signal", lambda *args: None)
    calls = []
    def phase(root, out, *args):
        calls.append(out.name)
        return int(out == outs[0])
    monkeypatch.setattr(ae, "phase", phase)
    monkeypatch.setattr(ae, "merge", lambda _: {})
    with ae.lease(outs[1] / ".work.lock"):
        assert ae.work(tmp_path) == 1
    assert calls == ["point0", "point2"]


def test_shell_cpu_modes_do_not_create_work_or_require_source(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts/run_additive.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    work = tmp_path / "absent"
    env = dict(os.environ, OM_WORK=str(work), ADDITIVE_PYTHON=sys.executable)
    for mode in ("plan", "check", "status", "live"):
        result = subprocess.run(["bash", str(script), mode], env=env, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
    assert not work.exists()
