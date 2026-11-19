"""X5 measurement: toy (ToyProbe) and tiny OLMo-3 + LoRA (TorchProbe, real autograd scoring)."""

from pathlib import Path

import pytest

from switching import backends, coderepo, snapshots
from switching.common import read_json, read_jsonl
from switching.engine import make_engine_class
from switching.runner import Context

from v9lib.measure import CHUNK, execute_measure


def _task(path, sha, update, out_dir, seed=5):
    return {"kind": "measure", "task_id": f"x5.test.s{seed}.u{update}", "cohort_id": "test", "training_seed": seed,
            "completed_update": update, "snapshot_path": str(path), "snapshot_sha256": sha,
            "source": {"kind": "test"}, "responses_half": 4, "validation_responses": 8, "out_dir": out_dir,
            "beta": {"policy": "test"}}


def _toy(tmp_path):
    srgc = coderepo.srgc()
    feat, ans, data = backends.toy_problem(0)
    ctx = Context(srgc=srgc, engine_class=make_engine_class(srgc), backend=None, data=data,
                  config=srgc.Config(seed=5, projection_dim=64), out_root=tmp_path, measurement_status="simulated_cpu")
    ctx.backend = backends.toy_backend(feat, ans, seed=5, record=ctx.on_meter_event)
    ctx.meter = ctx.backend.cost_meter
    e = srgc.Engine(ctx.backend, data["candidate_ids"], data["ranking_validation_ids"], data["cached_rewards"],
                    arm="on_policy", config=ctx.config)
    e.run_until(30)
    sha = snapshots.save(tmp_path / "snap.pt", e.state_dict(), immutable=False)
    from srgc_rebuttal.toy_backend import ToyBackend
    beta = ToyBackend(feat, ans, seed=0).weights
    return ctx, data, sha, beta


def test_toy_measure_complete_and_resume_is_identical(tmp_path):
    ctx, data, sha, beta = _toy(tmp_path)
    task = _task(tmp_path / "snap.pt", sha, 30, "x5/full")
    r = execute_measure(task, ctx, beta=beta)
    assert r["status"] == "measured"
    s = read_json(tmp_path / "x5/full/summary.json")
    assert s["n_prompts"] == 400 and s["tokens_total"] == 400 * 8
    for k in ("rho_on_split_half_pearson", "sigma_on", "rho_sr_split_half_pearson", "abs_p_shift_mean", "kl_seq_mean"):
        assert s[k] is not None
    full = {r["prompt_id"]: r for r in read_jsonl(tmp_path / "x5/full/rows.jsonl")}
    # interrupted after 3 chunks: same rows, resumed from chunk 3
    part = tmp_path / "x5/part"
    part.mkdir(parents=True)
    import json
    (part / "validation_vector.json").write_text((tmp_path / "x5/full/validation_vector.json").read_text())
    keep = [r for r in read_jsonl(tmp_path / "x5/full/rows.jsonl") if r["chunk"] < 3]
    (part / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    calls = []
    orig = ctx.backend.score_gradients
    ctx.backend.score_gradients = lambda ids, **kw: (calls.append(len(ids)), orig(ids, **kw))[1]
    execute_measure(_task(tmp_path / "snap.pt", sha, 30, "x5/part"), ctx, beta=beta)
    assert len(calls) == 2 * (400 // CHUNK - 3)  # finished chunks are not recomputed
    resumed = {r["prompt_id"]: r for r in read_jsonl(part / "rows.jsonl")}
    assert resumed.keys() == full.keys()
    assert all(resumed[k] == full[k] for k in full)
    assert execute_measure(task, ctx, beta=beta)["status"] == "skipped_complete"


def test_toy_measure_rejects_wrong_update(tmp_path):
    ctx, data, sha, beta = _toy(tmp_path)
    with pytest.raises(ValueError, match="snapshot holds update"):
        execute_measure(_task(tmp_path / "snap.pt", sha, 25, "x5/bad"), ctx, beta=beta)


torch = pytest.importorskip("torch")
pytest.importorskip("peft")


def _tiny(tmp_path, updates):
    srgc = coderepo.srgc()
    records, data = backends.tiny_problem(n_candidates=40, n_val=4, n_eval=2)
    ctx = Context(srgc=srgc, engine_class=make_engine_class(srgc), backend=None, data=None,
                  config=srgc.Config(seed=5, scoring_prompts=8, training_prompts=2, responses=8, candidate_group_size=4,
                                     projection_dim=16, selection_interval=5, check_interval=5, first_check=5),
                  out_root=tmp_path, measurement_status="measured_cpu")
    ctx.backend = backends.tiny_backend(records, record=ctx.on_meter_event)
    ctx.meter = ctx.backend.cost_meter
    cache = backends.initial_cache(ctx.backend, data["candidate_ids"], responses=8, seed=5, srgc=srgc)
    ctx.data = dict(data, cached_rewards=cache)
    e = srgc.Engine(ctx.backend, data["candidate_ids"], data["ranking_validation_ids"], cache, arm="on_policy",
                    config=ctx.config)
    e.run_until(updates)
    sha = snapshots.save(tmp_path / f"snap{updates}.pt", e.state_dict(), immutable=False)
    return ctx, sha


def test_tiny_olmo_kl_is_zero_at_the_cache_policy_and_positive_after_training(tmp_path):
    ctx, sha0 = _tiny(tmp_path, 0)
    execute_measure(_task(tmp_path / "snap0.pt", sha0, 0, "x5/u0"), ctx)
    s0 = read_json(tmp_path / "x5/u0/summary.json")
    # LoRA B starts at zero: pi_0 == beta exactly, every log ratio is 0
    assert s0["kl_seq_mean"] == pytest.approx(0.0, abs=1e-9) and s0["kl_token_mean"] == pytest.approx(0.0, abs=1e-9)
    assert s0["n_prompts"] == 40 and s0["tokens_total"] > 0
    ctx, sha = _tiny(tmp_path, 20)
    execute_measure(_task(tmp_path / "snap20.pt", sha, 20, "x5/u20"), ctx)
    s = read_json(tmp_path / "x5/u20/summary.json")
    assert s["kl_seq_mean"] != 0.0
    rows = read_jsonl(tmp_path / "x5/u20/rows.jsonl")
    assert all(len(r["kl_seq_per_response"]) == 8 for r in rows)
    # adapters are enabled again after the beta pass
    assert not any(getattr(m, "disable_adapters", False) for m in ctx.backend.model.modules() if hasattr(m, "lora_A"))
