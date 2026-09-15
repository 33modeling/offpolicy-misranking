import copy
import json

import numpy as np
import pytest

import mopps
import selection_gate as core


def sampler(arm="mopps", pool=100):
    return mopps.Sampler(mopps.specification(arm, 3, 25, pool))


def advance(s, step):
    proposal = s.begin(step)
    rewards = [[int((i+j+step) % 3 == 0) for j in range(8)] for i in range(4)]
    return {"step": step, "online_selection": s.finish(proposal, rewards),
            "reward_mean": float(np.mean(rewards)), "groups": 4, "samples": 32}


def test_registered_math_variant():
    s = sampler()
    assert s.config == mopps.Config(1., 1., .5, 1., 16)
    assert mopps.PAPER["venue"] == "KDD 2026"
    assert mopps.PAPER["commit"] == "4110c5ab40fc9a2b9d09c70989b87c9e74bfdd17"
    assert (s.alpha == 1.).all() and (s.beta == 1.).all()
    assert len(s.begin(26)["candidates"]) == 64


@pytest.mark.parametrize("seed", range(8))
def test_selection_matches_upstream_beta_draw_and_topk(seed):
    s = sampler()
    s.alpha = np.arange(1, 101, dtype=float)
    s.beta = s.alpha[::-1].copy()
    candidates = np.random.RandomState(seed).permutation(100)[:64]
    predicted = np.random.RandomState(seed).beta(s.alpha[candidates], s.beta[candidates])
    expected = candidates[np.argsort((predicted-.5)**2)[:4]].tolist()
    actual = s.select(candidates, np.random.RandomState(seed))
    assert actual["selected"] == expected
    assert actual["predicted"] == predicted.tolist()


def test_online_counts_update_once_per_group_and_only_selected_prompts():
    s = sampler()
    row = advance(s, 26)
    evidence = row["online_selection"]
    successes = np.array(evidence["rewards"]).sum(axis=1)
    assert s.alpha[evidence["selected"]].tolist() == (1+successes).tolist()
    assert s.beta[evidence["selected"]].tolist() == (1+8-successes).tolist()
    unused = list(set(range(100))-set(evidence["selected"]))
    assert (s.alpha[unused] == 1.).all() and (s.beta[unused] == 1.).all()
    with pytest.raises(ValueError, match="preceding posterior"):
        s.finish({k: evidence[k] for k in ("step", "candidates", "selected", "predicted")}, evidence["rewards"])


@pytest.mark.parametrize("arm", mopps.ARMS)
def test_restart_restores_posterior_and_next_selections_exactly(arm):
    s = sampler(arm)
    rows = [advance(s, step) for step in range(26, 36)]
    restored = sampler(arm).replay(json.loads(json.dumps(rows)), 35)
    assert restored.state() == s.state()
    assert restored.begin(36) == s.begin(36)
    for step in range(36, 42):
        assert advance(restored, step) == advance(s, step)


@pytest.mark.parametrize("mutation", ["rewards", "selected", "predicted", "state", "step", "missing", "metric"])
def test_replay_rejects_corrupt_or_incomplete_history(mutation):
    s = sampler()
    rows = [advance(s, step) for step in range(26, 29)]
    evidence = rows[0]["online_selection"]
    if mutation == "rewards": evidence["rewards"][0][0] = 1-evidence["rewards"][0][0]
    elif mutation == "selected": evidence["selected"].reverse()
    elif mutation == "predicted": evidence["predicted"][0] += .001
    elif mutation == "state": evidence["state_sha256"] = "bad"
    elif mutation == "step": rows[0]["step"] += 1
    elif mutation == "missing": rows.pop()
    else: rows[0]["reward_mean"] += .1
    with pytest.raises(ValueError):
        sampler().replay(rows, 28)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1, .1, 2.])
def test_nonbinary_or_invalid_rewards_do_not_change_posterior(bad):
    s = sampler()
    before = copy.deepcopy(s.state())
    rewards = np.zeros((4, 8))
    rewards[0, 0] = bad
    with pytest.raises(ValueError, match="binary reward"):
        s.finish(s.begin(26), rewards)
    assert s.state() == before


def test_random_control_matches_candidates_but_never_uses_rewards():
    s, r = sampler(), sampler("random_online")
    for step in range(26, 36):
        assert s.begin(step)["candidates"] == r.begin(step)["candidates"]
        advance(s, step)
        advance(r, step)
    assert (r.alpha == 1.).all() and (r.beta == 1.).all()
    assert r.begin(36)["predicted"] is None


@pytest.mark.parametrize("pool", [4, 7, 63, 64, 100])
def test_batch_cardinality_and_candidate_cap(pool):
    proposal = sampler(pool=pool).begin(26)
    assert len(proposal["selected"]) == len(set(proposal["selected"])) == 4
    assert len(proposal["candidates"]) == min(pool, 64)
    assert set(proposal["selected"]) <= set(proposal["candidates"]) <= set(range(pool))


def test_selector_rng_does_not_change_global_numpy_state():
    np.random.seed(123)
    expected = np.random.RandomState(123).random_sample()
    advance(sampler(), 26)
    assert np.random.random_sample() == expected


def test_specification_cannot_import_cached_or_future_rewards():
    spec = mopps.specification("mopps", 3, 25, 100)
    for key, value in (("cached_rewards", [1]), ("initialization", "warm"), ("config", {**spec["config"], "target": .9})):
        with pytest.raises(ValueError):
            mopps.Sampler({**spec, key: value})


def test_published_posterior_is_bound_to_actual_policy_history(tmp_path):
    import selection_gate_gpu as base
    s = sampler()
    rows = [advance(s, step) for step in range(26, 29)]
    core.atomic_json(tmp_path / "selector_state.json", s.state())
    (tmp_path / "grpo_stats.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    core.atomic_json(tmp_path / "policy_train.json", {"online_selection": s.spec, "completed_steps": 28,
        "selector_state_sha256": base.digest(tmp_path / "selector_state.json")})
    assert mopps.validate_policy_evidence(tmp_path, s.spec).state() == s.state()
    core.atomic_json(tmp_path / "selector_state.json", sampler().state())
    with pytest.raises(ValueError, match="receipt"):
        mopps.validate_policy_evidence(tmp_path, s.spec)


def test_shared_training_helpers_preserve_loss_generation_and_cache_behavior():
    import train_mopps_grpo as added
    import train_selection_gate_grpo as original
    for name in ("_sample_group", "_response_logps_batch", "checked_optimizer_step", "clipped_grpo_loss",
                 "standardized_group_advantages", "_save_checkpoint", "load_model"):
        assert getattr(added, name) is getattr(original, name)


def test_feedback_moves_to_collective_device_before_gather(monkeypatch):
    import torch
    import train_mopps_grpo as driver
    seen = []
    def gather(groups, tensor):
        seen.append(tensor.device)
        for i, group in enumerate(groups):
            group.copy_(tensor+i)
    monkeypatch.setattr(driver.dist, "all_gather", gather)
    groups = driver.gather_feedback(torch.zeros(8), torch.device("cpu"), 4)
    assert seen == [torch.device("cpu")]
    assert groups == [[float(i)]*8 for i in range(4)]


@pytest.mark.parametrize("arm", mopps.ARMS)
def test_real_tiny_policy_update_checkpoint_and_posterior_resume(tmp_path, monkeypatch, arm):
    import torch
    from types import SimpleNamespace
    import train_mopps_grpo as driver
    import selection_gate_budget as budget
    from test_logit_chunking import _tiny_olmo3

    monkeypatch.setattr(driver, "_distributed_setup", lambda _: (0, "cpu", 1))
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", lambda model, **kwargs: model)
    monkeypatch.setattr(driver, "load_model", lambda *args, **kwargs: (_tiny_olmo3(), SimpleNamespace(eos_token_id=0)))
    monkeypatch.setattr(driver, "_lora_targets", lambda: ["q_proj", "v_proj"])
    monkeypatch.setattr(driver, "chat_ids", lambda *args: torch.tensor([1, 2, 3]))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: 0.)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *args: 0.)
    monkeypatch.setattr(driver, "_sample_group", lambda *args: (
        [torch.tensor([1, 2, 3, 4+i, 12+i]) for i in range(8)], torch.tensor([0., 1.]*4)))
    monkeypatch.setattr(driver, "SAMPLING", {"top_p": 1.})
    prompts = tmp_path / "prompts.json"
    selector = tmp_path / "selector.json"
    core.atomic_json(prompts, {"train": [{"question": str(i), "answer": "1"} for i in range(8)]})
    spec = mopps.specification(arm, 3, 0, 8, batch_size=1)
    core.atomic_json(selector, spec)
    args = SimpleNamespace(model=str(tmp_path / "base"), prompts=str(prompts), selector_config=str(selector),
        output=str(tmp_path / "policy"), seed=3, objective="grpo", expected_world_size=1, group_size=8,
        clip_epsilon=.2, learning_rate=1e-5, epochs_per_batch=1, max_grad_norm=1., advantage_epsilon=1e-4,
        lora_rank=4, lora_alpha=8, checkpoint_every=1, target_steps=3, start_step=0,
        resume_adapter=None, resume_optimizer=None, logprob_micro_batch=8, max_new_tokens=16,
        disable_gradient_checkpointing=False, wall_budget_deadline=1e12, budget_save_reserve=30.)
    save = driver._save_checkpoint
    def interrupt(*args, **kwargs):
        save(*args, **kwargs)
        raise RuntimeError("test stop after durable checkpoint")
    monkeypatch.setattr(driver, "_save_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="durable checkpoint"):
        driver.train(args)
    assert (tmp_path / "policy/checkpoint-000001/checkpoint_state.json").exists()
    monkeypatch.setattr(driver, "_save_checkpoint", save)
    checks = 0
    def stop(*args):
        nonlocal checks
        checks += 1
        return checks > 2
    monkeypatch.setattr(budget, "stop_before_step", stop)
    driver.train(args)
    policy_path = tmp_path / "policy"
    restored = mopps.validate_policy_evidence(policy_path, spec)
    assert restored.completed_step == 2
    assert core.read(policy_path / "budget_stop.json")["stop_reason"] == "no_block_fits"
    rows = [json.loads(line) for line in (policy_path / "grpo_stats.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert all(row["grad_norm"] > 0 for row in rows)
    expected = mopps.Sampler(spec)
    for step in (1, 2):
        expected.finish(expected.begin(step), [[0., 1.]*4])
    assert restored.state() == expected.state()
