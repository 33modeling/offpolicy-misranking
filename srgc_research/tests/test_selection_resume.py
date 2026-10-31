import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from srgc_research.dispatch import selection_resume

ROOT = Path(__file__).resolve().parents[2]


def fresh_mixin_module():
    """A private copy, so patching never leaks into other tests' mixin."""
    spec = importlib.util.spec_from_file_location("fresh_resumable", ROOT / "scripts/srgc_resumable_rollouts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selection_resume.patch_mixin(module.ResumableRolloutMixin)
    return module


class Interrupted(RuntimeError):
    pass


class FakeBackend:
    """TorchBackend stand-in: rollouts, gathered scoring vectors and a train step."""

    device = SimpleNamespace(type="cpu")

    def __init__(self, weight=1.0, fail_on=()):
        self.rank, self.world = 0, 1
        self.train_parameters = [("lora_A", np.full(3, weight, dtype=np.float32))]
        self.generated, self.scored, self.trained = [], [], 0
        self.fail_on = set(fail_on)

    def _rollout(self, prompt_id, responses, seed):
        if prompt_id in self.fail_on:
            raise Interrupted(prompt_id)
        self.generated.append(prompt_id)
        length = 3 + len(prompt_id)
        return ([np.arange(length) + seed % 7 + i for i in range(responses)],
                np.array([float(i % 2) for i in range(responses)]), 2)

    def score_gradients(self, ids, *, responses, group_size, seed):
        result = {}
        for prompt_id in ids:
            sequences, _, _ = self._rollout(prompt_id, responses, seed)
            self.scored.append(prompt_id)
            result[prompt_id] = np.full(4, float(sum(int(np.asarray(s).sum()) for s in sequences)))
        return result

    def train(self, ids, **kwargs):
        self.trained += 1
        return {"sample_reward": 0.5}


CANDIDATES = [f"c{i}" for i in range(6)]
VALIDATION = [f"v{i}" for i in range(4)]


def backend_class(module, folder):
    return module.make_resumable(FakeBackend, folder / "rollout-cache" / "switch")


def test_restart_after_the_candidate_block_reuses_it_without_generation(tmp_path, capsys):
    module = fresh_mixin_module()
    Backend = backend_class(module, tmp_path)
    reference = FakeBackend().score_gradients(CANDIDATES, responses=8, group_size=4, seed=11)

    first = Backend(fail_on={"v2"})
    first.score_gradients(CANDIDATES, responses=8, group_size=4, seed=11)
    with pytest.raises(Interrupted):
        first.score_gradients(VALIDATION, responses=8, group_size=8, seed=12)
    assert (tmp_path / "rollout-cache/switch/score-11" / selection_resume.RESULT).is_file()

    second = Backend()
    candidates = second.score_gradients(CANDIDATES, responses=8, group_size=4, seed=11)
    assert second.generated == [] and second.scored == []
    assert second.resumed_rollouts == len(CANDIDATES)  # replayed for the raw trace
    for key in reference:
        np.testing.assert_array_equal(candidates[key], reference[key])
    second.score_gradients(VALIDATION, responses=8, group_size=8, seed=12)
    # Validation prompts finished before the interruption are not generated again.
    assert second.generated == ["v2", "v3"]
    assert "reused 6 saved gradients" in capsys.readouterr().out


def test_other_weights_never_reuse_a_saved_block(tmp_path):
    module = fresh_mixin_module()
    Backend = backend_class(module, tmp_path)
    Backend(weight=1.0).score_gradients(CANDIDATES, responses=8, group_size=4, seed=11)
    moved = Backend(weight=2.0)
    moved.score_gradients(CANDIDATES, responses=8, group_size=4, seed=11)
    assert moved.generated == CANDIDATES
    directory = tmp_path / "rollout-cache/switch/score-11"
    assert (directory / selection_resume.POLICY).read_text() == selection_resume.policy_fingerprint(moved)


def test_a_changed_request_or_group_size_is_scored_again(tmp_path):
    module = fresh_mixin_module()
    Backend = backend_class(module, tmp_path)
    Backend().score_gradients(CANDIDATES, responses=8, group_size=4, seed=11)
    again = Backend()
    again.score_gradients(CANDIDATES, responses=8, group_size=8, seed=11)
    assert again.scored == CANDIDATES and again.generated == []  # rollouts still reused per prompt


def test_training_keeps_the_current_block_and_drops_it_after_the_weights_move(tmp_path):
    module = fresh_mixin_module()
    Backend = backend_class(module, tmp_path)
    backend = Backend()
    backend.score_gradients(CANDIDATES, responses=8, group_size=4, seed=11)
    backend.score_gradients(VALIDATION, responses=8, group_size=8, seed=12)
    cache = tmp_path / "rollout-cache/switch"
    backend.train(["c0"])
    assert sorted(p.name for p in cache.iterdir()) == ["score-11", "score-12"]
    backend.train_parameters = [("lora_A", np.full(3, 1.5, dtype=np.float32))]  # the optimizer step
    backend.train(["c1"])
    assert list(cache.iterdir()) == []
    assert backend.trained == 2


def test_a_legacy_block_without_a_policy_keeps_its_rollouts(tmp_path):
    module = fresh_mixin_module()
    Backend = backend_class(module, tmp_path)
    legacy = Backend()
    legacy._resumable_dir = legacy._resumable("score", 11)
    legacy._rollout("c0", 8, 11)  # written by the pinned code before this patch
    legacy._resumable_dir = None
    resumed = Backend()
    resumed.score_gradients(CANDIDATES, responses=8, group_size=4, seed=11)
    assert resumed.generated == CANDIDATES[1:]
    resumed.train(["c0"])  # same weights: still kept
    assert (tmp_path / "rollout-cache/switch/score-11" / selection_resume.RESULT).is_file()


def test_replayed_rollouts_reach_the_trace_capture(tmp_path):
    from srgc_research.dispatch.rollout_records import capture

    module = fresh_mixin_module()
    Backend = backend_class(module, tmp_path)
    Backend().score_gradients(CANDIDATES, responses=8, group_size=4, seed=11)
    resumed = Backend()
    with capture(resumed) as groups:
        resumed.score_gradients(CANDIDATES, responses=8, group_size=4, seed=11)
    assert [(phase, prompt) for phase, prompt, *_ in groups] == [("selection", p) for p in CANDIDATES]
    assert resumed.generated == []


def test_without_a_cache_root_the_pinned_path_is_used(tmp_path):
    module = fresh_mixin_module()

    class Backend(module.ResumableRolloutMixin, FakeBackend):
        pass

    backend = Backend()
    backend.score_gradients(CANDIDATES, responses=8, group_size=4, seed=11)
    backend.train(["c0"])
    assert backend.generated == CANDIDATES and not list(tmp_path.iterdir())


def test_install_patches_both_import_names_and_only_once():
    from scripts import srgc_resumable_rollouts as module

    mixin = module.ResumableRolloutMixin
    original = mixin.__dict__["score_gradients"]
    previous = sys.modules.get("srgc_resumable_rollouts")
    try:
        assert selection_resume.install() == [mixin]
        assert sys.modules["srgc_resumable_rollouts"] is module
        patched = mixin.__dict__["score_gradients"]
        selection_resume.install()
        assert mixin.__dict__["score_gradients"] is patched is not original
    finally:
        mixin.score_gradients = original
        for name in ("train", "selection_resume_installed"):
            if name in mixin.__dict__:
                delattr(mixin, name)
        if previous is None:
            sys.modules.pop("srgc_resumable_rollouts", None)
        else:
            sys.modules["srgc_resumable_rollouts"] = previous


def test_rank_wrapper_installs_for_gemma_only(monkeypatch):
    from srgc_research.dispatch import rank_runtime

    calls = []
    monkeypatch.setattr(selection_resume, "install", lambda: calls.append(True))
    monkeypatch.setattr(selection_resume, "extend_collective_timeout", lambda: calls.append("timeout"))
    monkeypatch.setattr(rank_runtime.runpy, "run_module", lambda *a, **k: None)
    fake = SimpleNamespace(record=lambda function: function)
    monkeypatch.setitem(sys.modules, "torch.distributed.elastic.multiprocessing.errors", fake)
    for model, expected in (("gemma4_it", 2), ("gemma4", 4), ("qwen35", 4), ("llama31", 4)):
        monkeypatch.setattr(sys, "argv", ["rank_runtime", "--model", model, "--entry-module",
                                          "srgc_research.dispatch.reward_curve_rank", "--", "--model", model])
        rank_runtime.main()
        assert len(calls) == expected


class EngineBackend:
    """Deterministic policy whose scores and updates depend on its weights."""

    device = SimpleNamespace(type="cpu")
    gpu_count = 0

    def __init__(self, fail=None):
        self.rank, self.world = 0, 1
        self.weights = np.zeros(4, dtype=np.float32)
        self.train_parameters = [("lora", self.weights)]
        self.generated, self.fail = [], fail

    def synchronize(self):
        pass

    def _rollout(self, prompt_id, responses, seed):
        if self.fail == ("rollout", prompt_id):
            raise Interrupted(prompt_id)
        self.generated.append(prompt_id)
        rng = np.random.default_rng(int(hashlib_digest((prompt_id, seed, float(self.weights.sum()))), 16))
        return ([rng.integers(1, 50, size=6) for _ in range(responses)],
                rng.integers(0, 2, size=responses).astype(float), 2)

    def score_gradients(self, ids, *, responses, group_size, seed):
        result = {}
        for prompt_id in ids:
            sequences, rewards, _ = self._rollout(prompt_id, responses, seed)
            base = np.array([float(np.sum(s)) for s in sequences[:4]]) + rewards[:4] + self.weights
            result[prompt_id] = base / (1 + np.linalg.norm(base))
        return result

    def train(self, ids, *, responses, objective, seed):
        if self.fail == ("train", None):
            raise Interrupted("train")
        self.weights += np.float32(len(ids) * 0.25)
        return {"sample_reward": 0.5, "gradient_norm": 1.0}

    def state_dict(self):
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state):
        self.weights[:] = state["weights"]


def test_switch_check_restarts_reach_the_next_update_with_identical_decisions(tmp_path):
    from srgc_rebuttal import srgc

    module = fresh_mixin_module()
    Backend = module.make_resumable(EngineBackend, tmp_path / "rollout-cache" / "switch")
    candidates = [f"c{i}" for i in range(12)]
    validation = [f"v{i}" for i in range(3)]
    cached = {c: [float((i + j) % 3 == 0) for j in range(8)] for i, c in enumerate(candidates)}
    config = srgc.Config(seed=5, scoring_prompts=4, training_prompts=2, responses=8, candidate_group_size=4,
                         projection_dim=4, selection_interval=2, check_interval=2, first_check=2)

    def engine(backend, state=None):
        made = srgc.Engine(backend, candidates, validation, cached, arm="switch", config=config)
        if state is not None:
            made.load_state_dict(state)
        return made

    reference = engine(EngineBackend())
    reference.run_until(3)
    expected = reference.history[2]

    prefix = engine(Backend())
    prefix.run_until(2)
    checkpoint = prefix.state_dict()  # the update-2 resume point, before the switch check
    attempts = []
    for fail in (("rollout", "v1"), ("train", None), None):
        backend = Backend(fail=fail)
        attempt = engine(backend, checkpoint)
        attempts.append(backend)
        if fail is None:
            record = attempt.update()
        else:
            with pytest.raises(Interrupted):
                attempt.update()
    union = set(expected["on_ids"]) | set(expected["sr_ids"])
    assert expected["d"] is not None and len(union) > config.scoring_prompts
    assert union <= set(attempts[0].generated)
    # Neither restart generated a candidate/SR prompt or a finished validation prompt again.
    assert not union & set(attempts[1].generated) and not union & set(attempts[2].generated)
    assert set(attempts[1].generated) == {"v1", "v2"} and attempts[2].generated == []
    for key in ("d", "switched", "on_ids", "sr_ids", "ranking_scores", "selected_on_ids", "train_ids"):
        assert record[key] == expected[key], key
    assert attempt.step == 3
    assert list((tmp_path / "rollout-cache" / "switch").iterdir()) != []
    attempt.backend.train(["c0"], responses=8, objective="grpo", seed=1)  # next update: weights moved
    assert list((tmp_path / "rollout-cache" / "switch").iterdir()) == []


def hashlib_digest(value):
    import hashlib
    return hashlib.sha256(repr(value).encode()).hexdigest()[:8]


def test_gemma_ranks_wait_longer_than_the_pinned_nccl_limit(monkeypatch):
    from datetime import timedelta

    seen = []
    fake = SimpleNamespace(init_process_group=lambda *a, **k: seen.append(k.get("timeout")))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(distributed=fake))
    monkeypatch.setitem(sys.modules, "torch.distributed", fake)
    selection_resume.extend_collective_timeout()
    selection_resume.extend_collective_timeout()  # idempotent
    fake.init_process_group("nccl", timeout=timedelta(seconds=1800))
    fake.init_process_group("nccl")
    fake.init_process_group("nccl", timeout=timedelta(hours=12))
    assert seen == [timedelta(hours=6), timedelta(hours=6), timedelta(hours=12)]


def test_running_task_shows_why_its_previous_attempt_ended(tmp_path):
    from srgc_research.dispatch import qwen_status
    from srgc_research.dispatch.rank_failures import previous_attempt_line

    log = tmp_path / "seed-5.switch.log"
    log.write_text("SRGC CHILD START model=gemma4_it\n"
                   "[rank2]: RuntimeError: Watchdog caught collective operation timeout: "
                   "WorkNCCL(OpType=ALLGATHER, Timeout(ms)=1800000) ran for 1800321 milliseconds\n"
                   "Root Cause (first observed failure):\n  exitcode  : -6 (pid: 7)\n"
                   "SRGC CHILD START model=gemma4_it\nGEMMA generation progress\n")
    line = previous_attempt_line(log)
    assert line.startswith("[rank2]: RuntimeError: Watchdog caught collective operation timeout")
    text = qwen_status.progress({"arm": "switch", "attempt": 3, "log": str(log)})
    assert "attempt 3 | previous attempt ended: [rank2]: RuntimeError: Watchdog" in text
    assert "previous attempt" not in qwen_status.progress({"arm": "switch", "attempt": 1, "log": str(log)})
    single = tmp_path / "single.log"
    single.write_text("SRGC CHILD START model=gemma4_it\n")
    assert previous_attempt_line(single) is None
    killed = tmp_path / "killed.log"
    killed.write_text("SRGC CHILD START a\nGEMMA generation\nTIMEOUT: no progress for 1800s\nSRGC CHILD START b\n")
    assert previous_attempt_line(killed) == "TIMEOUT: no progress for 1800s"
