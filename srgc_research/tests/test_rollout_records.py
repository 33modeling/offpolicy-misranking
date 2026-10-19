"""Actual returned groups survive cache cleanup and failed/resumed updates."""

import gzip
import json
from unittest.mock import patch

import numpy as np
import pytest

from scripts.srgc_resumable_rollouts import make_resumable
from scripts.srgc_step_checkpoints import checkpoint_engine
from srgc_rebuttal.runtime import atomic_json
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.tests.test_reference import ScriptedBackend
from srgc_research.dispatch import rollout_records
from srgc_research.dispatch.reward_curve_results import collect
from srgc_research.dispatch.reward_curves import recording_engine
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 - pytest fixture registration
)
from srgc_research.tests.test_model_results import finish, read_result, saved


class RawBackend(ScriptedBackend):
    def __init__(self):
        super().__init__()
        torch = pytest.importorskip("torch")
        from srgc_rebuttal.timing import torch_meter
        self.model = torch.nn.Linear(1, 1)
        self.device = torch.device("cpu")
        self.cost_meter = torch_meter(cuda=False)
        self.rollout_calls = []

    def _rollout(self, prompt_id, responses, seed):
        self.rollout_calls.append((prompt_id, responses, seed))
        return [np.array([99, 98, seed % 97, i]) for i in range(responses)], np.array([0., 1.] * 4), 2

    def _gather(self, local):
        return local

    def score_gradients(self, ids, **kwargs):
        for rid in ids:
            self._rollout(rid, kwargs["responses"], kwargs["seed"])
        return super().score_gradients(ids, **kwargs)

    def train(self, ids, **kwargs):
        for rid in ids:
            self._rollout(rid, kwargs["responses"], kwargs["seed"])
        return super().train(ids, **kwargs)

    def evaluate(self, ids, *, seed, responses):
        return {rid: .5 for rid in ids}


def make(folder, *, measured=True, arm="on_policy"):
    ids = [f"p{i}" for i in range(400)]
    cache = {rid: [0, 1] * 4 for rid in ids}
    data = {"evaluation_ids": ["held"], "cached_rewards": cache}
    plan = {"model": "cpu-fixture", "dataset": "math", "shared_prefix_updates": 25,
            "total_updates": 275, "training_prompts": 4}
    identity = {"seed": 3, "plan_sha256": "a" * 64, "input_sha256": "b" * 64, "implementation_sha256": "c" * 64}
    atomic_json(folder / "run.json", identity)
    backend = make_resumable(RawBackend, folder / "rollout-cache")()
    base = recording_engine(Engine, folder, arm, folder / "plan.json", plan, data) if measured else Engine
    base = checkpoint_engine(base, folder, arm, {}, total_updates=275)
    engine = base(backend, ids, ["reference"], cache, arm=arm, config=Config(projection_dim=2), step=25)
    return engine, plan, data, identity


def read(path):
    if not path.exists():
        return None
    return rollout_records.read(path) if path.suffix == ".gz" else json.loads(path.read_text())


def test_no_extra_generation_and_successful_groups_survive_cleanup_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    plain, *_ = make(tmp_path / "plain", measured=False)
    plain.update()
    folder = tmp_path / "recorded"
    engine, plan, data, identity = make(folder)
    engine.update()
    assert engine.backend.rollout_calls == plain.backend.rollout_calls
    assert engine.backend.parameter == plain.backend.parameter
    assert not list((folder / "rollout-cache").rglob("*.npz"))
    path = folder / "reward-curves/on_policy/rollouts/step-26.json.gz"
    trace = read(path)
    assert 'timing_receipt_ids' not in trace['history_record']
    assert len(trace["groups"]) == 45  # 40 candidates, one reference, four applied prompts.
    assert {g["phase"] for g in trace["groups"]} == {"selection", "diagnostic", "training"}
    for group in trace["groups"]:
        assert len(group["responses"]) == 8
        assert all(len(r["token_ids"]) == 2 for r in group["responses"])
        assert group["cached_success_rate"] == .5 if group["phase"] != "diagnostic" else "cached_success_rate" not in group
    rows = collect(folder, "on_policy", plan, identity, data["evaluation_ids"], read)
    assert len(rows["rollout_groups"]) == 45
    assert rows["missing_training_rollout_steps"] == []
    assert rows["rollout_sources"][0]["step"] == 26
    assert "token_ids" not in json.dumps(rows)
    assert all(r["completion_tokens"] == 2 for g in rows["rollout_groups"] for r in g["responses"])
    state = torch.load(folder / "on_policy-latest.pt", weights_only=False)
    assert set(state['history'][-1]['timing_receipt_ids']) == {'selection', 'training'}
    restored, *_ = make(folder)
    restored.load_state_dict(state)
    restored.update()
    assert len(read(folder / "reward-curves/on_policy/rollouts/step-27.json.gz")["groups"]) == 4
    assert len(collect(folder, "on_policy", plan, identity, data["evaluation_ids"], read)["training_rewards"]) == 2


def test_export_failure_cannot_advance_resume_checkpoint(tmp_path):
    pytest.importorskip("torch")
    engine, *_ = make(tmp_path)
    with patch.object(rollout_records, "publish", side_effect=OSError("disk full")), pytest.raises(OSError, match="disk full"):
        engine.update()
    assert not (tmp_path / "on_policy-latest.pt").exists()


def test_retry_after_trace_write_reuses_identical_responses_despite_timing_changes(tmp_path):
    pytest.importorskip("torch")
    engine, *_ = make(tmp_path)
    with patch.object(engine, "save_training_rewards", side_effect=OSError("interrupted scalar write")), pytest.raises(OSError):
        engine.update()
    path = tmp_path / "reward-curves/on_policy/rollouts/step-26.json.gz"
    assert path.exists() and not (tmp_path / "on_policy-latest.pt").exists()
    original = path.read_bytes()
    restored, *_ = make(tmp_path)
    restored.update()
    assert path.read_bytes() == original
    assert (tmp_path / "on_policy-latest.pt").exists()


@pytest.mark.parametrize("damage", ["reward", "tokens", "batch", "protocol"])
def test_corrupt_durable_groups_are_rejected(tmp_path, damage):
    pytest.importorskip("torch")
    engine, plan, data, identity = make(tmp_path)
    engine.update()
    path = tmp_path / "reward-curves/on_policy/rollouts/step-26.json.gz"
    trace = read(path)
    if damage == "reward":
        trace["groups"][0]["responses"][0]["reward"] = .2
    elif damage == "tokens":
        trace["groups"][0]["responses"][0]["token_ids"] = []
    elif damage == "batch":
        trace["history_record"]["train_ids"] = ["different"]
    else:
        trace["protocol_sha256"] = "d" * 64
    path.unlink()
    rollout_records.publish(path, trace)
    with pytest.raises(ValueError):
        collect(tmp_path, "on_policy", plan, identity, data["evaluation_ids"], read)


def test_atomic_archive_reuse_and_conflicting_retry(tmp_path):
    path = tmp_path / "step-1.json.gz"
    rollout_records.publish(path, {"reward": 1})
    original = path.read_bytes()
    rollout_records.publish(path, {"reward": 1})
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="differs"):
        rollout_records.publish(path, {"reward": 0})
    assert path.read_bytes() == original


def export_records(cohort, arm="on_policy", *, complete=True):
    from srgc_rebuttal.plan import input_path, load_plan
    from srgc_rebuttal.runtime import run_root
    from srgc_research.dispatch import reward_curves
    plan = load_plan(cohort.plans["math"])
    # A running arm has no endpoint; it shares the seed folder of a finished one.
    folder = (finish(cohort, "math", arm=arm) if complete
              else run_root(cohort.plans["math"], plan) / "seed-5")
    data = json.loads(input_path(cohort.plans["math"], plan, 5).read_text())
    identity = read(folder / "run.json")
    identity.pop("status")
    protocol = {**identity, "schema": reward_curves.SCHEMA, "arm": arm, "extension_sha256": "e" * 64,
                "interval_updates": 25, "responses_per_prompt": 8, "question_count": len(data["evaluation_ids"]),
                "evaluation_ids_sha256": reward_curves.fingerprint(data["evaluation_ids"]),
                "stream": "reporting-evaluation", "training_batch_reward_is_held_out": False}
    ids = data["candidate_ids"][:4]
    history = {"checkpoint": 25, "completed_updates": 26, "selector": arm,
               "train_ids": ids, "metrics": {"sample_reward": .5}}
    atomic_json(folder / f"reward-curves/{arm}/protocol.json", protocol)
    atomic_json(folder / f"reward-curves/{arm}/training.json", {**protocol, "step": 26,
                "points": reward_curves.training_points([history], start=25, stop=26)})

    def trace(step):
        record = {**history, "checkpoint": step - 1, "completed_updates": step}
        return {"schema": rollout_records.SCHEMA, "model": plan["model"], "dataset": plan["dataset"],
                "seed": 5, "arm": arm, "step": step, "protocol_sha256": reward_curves.fingerprint(protocol),
                "identity": {k: identity[k] for k in ("plan_sha256", "input_sha256", "implementation_sha256")},
                "history_record": record,
                "groups": [{"arm": arm, "phase": "training", "update": step, "prompt_id": rid,
                            "responses": [{"reward": i % 2, "token_ids": [10, 20, i]} for i in range(8)]}
                           for rid in ids]}
    rollout_records.publish(folder / f"reward-curves/{arm}/rollouts/step-26.json.gz", trace(26))
    if not complete:
        return folder, trace
    return folder


def test_full_export_retains_rewards_and_lengths_for_each_model(cohort):
    from srgc_research.dispatch import model_results

    export_records(cohort)
    assert model_results.export(cohort.model, "math", environment=cohort.env, full=True) == 0
    artifact = json.loads((cohort.work / "results" / f"{cohort.model}-full-results.json").read_text())
    curves = artifact["datasets"]["math"]["reward_curves"]
    row = next(c for c in curves if c["seed"] == 5 and c["arm"] == "on_policy")
    assert len(row["rollout_groups"]) == 4 and len(row["rollout_sources"]) == 1
    assert all(r["completion_tokens"] == 3 for g in row["rollout_groups"] for r in g["responses"])
    assert "token_ids" not in json.dumps(row)


@pytest.mark.parametrize('damage', ('truncated_gzip', 'invalid_deflate', 'trace_array',
                                    'training_array', 'group_array', 'history_array'))
def test_bad_reward_archive_preserves_endpoints_and_other_arms_in_partial_export(cohort, damage):
    from srgc_research.dispatch import model_results

    folder = export_records(cohort)
    finish(cohort, 'math', arm='sr', reward=.625)
    assert model_results.export(cohort.model, 'math', environment=cohort.env) == 0
    path = folder / 'reward-curves/on_policy/rollouts/step-26.json.gz'
    if damage == 'truncated_gzip':
        path.write_bytes(path.read_bytes()[:-4])
    elif damage == 'invalid_deflate':
        payload = path.read_bytes()
        path.write_bytes(payload[:10] + b'\x06' + payload[11:])
    elif damage == 'trace_array':
        path.write_bytes(gzip.compress(b'[]'))
    elif damage == 'training_array':
        atomic_json(folder / 'reward-curves/on_policy/training.json', [])
    else:
        trace = read(path)
        if damage == 'group_array':
            trace['groups'][0] = []
        else:
            trace['history_record'] = []
        path.write_bytes(gzip.compress(json.dumps(trace).encode()))
    before = saved(cohort)
    assert model_results.export(cohort.model, 'math', environment=cohort.env) == 1
    result = read_result(cohort)
    snapshot = result['datasets']['math']
    assert not result['complete'] and result['errors']
    assert any('invalid reward record' in error for error in result['errors'])
    assert {row['arm']: row['endpoint']['reward'] for row in snapshot['endpoints']} == {
        'on_policy': .5, 'sr': .625}
    assert all(row['status'] == 'complete' for row in snapshot['tasks']
               if row['seed'] == 5 and row['arm'] in {'on_policy', 'sr'})
    sr = next(row for row in snapshot['reward_curves'] if row['seed'] == 5 and row['arm'] == 'sr')
    assert sr['held_out_rewards'][0]['reward'] == .625
    assert saved(cohort) == before


def test_running_arm_can_advance_during_export_without_errors(cohort):
    from srgc_rebuttal import reports
    from srgc_research.dispatch import model_results
    from srgc_research.tests.test_restart_costs import event
    from srgc_research.tests.test_submission_time import progress

    export_records(cohort)
    folder, trace = export_records(cohort, "random", complete=False)
    progress(folder, "random", step=26)
    event(folder, "random", "training-25", 25, 4.)
    original = reports.snapshot

    def training_continues(*args, **kwargs):
        result = original(*args, **kwargs)
        if result["dataset"] in {"math", "math_train"}:
            # The running update saves its responses before its scalar reward,
            # and its new phase receipt appears after the file inventory.
            rollout_records.publish(folder / "reward-curves/random/rollouts/step-27.json.gz", trace(27))
            event(folder, "random", "training-26", 26, 4., state="started")
        return result
    with patch.object(reports, "snapshot", side_effect=training_continues):
        assert model_results.export(cohort.model, "math", environment=cohort.env, full=True) == 0
    artifact = json.loads((cohort.work / "results" / f"{cohort.model}-full-results.json").read_text())
    assert artifact["errors"] == []
    snapshot = artifact["datasets"]["math"]
    curve = next(c for c in snapshot["reward_curves"] if c["seed"] == 5 and c["arm"] == "random")
    assert [p["step"] for p in curve["training_rewards"] if not p.get("shared_prefix")] == [26]
    assert [s["step"] for s in curve["rollout_sources"]] == [26]
    timing = next(t for t in snapshot["tasks"] if t["seed"] == 5 and t["arm"] == "random")["submission_timing"]
    assert timing["restart_accounting"]["unfinished_receipts"] == 1
    assert timing["restart_accounting"]["removed_gpu_seconds"] is None


def test_completed_arm_records_must_not_change_during_export(cohort):
    from srgc_rebuttal import reports
    from srgc_research.dispatch import model_results

    folder = export_records(cohort)
    path = folder / "reward-curves/on_policy/training.json"
    original = reports.snapshot

    def rewrite(*args, **kwargs):
        result = original(*args, **kwargs)
        if result["dataset"] in {"math", "math_train"}:
            atomic_json(path, json.loads(path.read_text()))
        return result
    with patch.object(reports, "snapshot", side_effect=rewrite):
        assert model_results.export(cohort.model, "math", environment=cohort.env) == 1
    assert any("seed-5.on_policy reward curve" in error and "changed during result inspection" in error
               for error in read_result(cohort)["errors"])


def distributed_writer(rank, folder, damage=None):
    from datetime import timedelta
    from pathlib import Path
    from types import SimpleNamespace

    import torch.distributed as dist

    from srgc_rebuttal.torch_backend import TorchBackend

    folder = Path(folder)
    dist.init_process_group("gloo", init_method=f"file://{folder / 'rendezvous'}",
                            rank=rank, world_size=4, timeout=timedelta(seconds=20))
    try:
        class Backend(RawBackend):
            _gather = TorchBackend._gather
        backend = Backend()
        backend.world = 4
        engine = SimpleNamespace(backend=backend, step=26, config=Config(), switched_at=None)
        record = {"checkpoint": 25, "completed_updates": 26, "selector": "sr",
                  "train_ids": [f"p{i}" for i in range(4)], "metrics": {"sample_reward": .5}}
        with rollout_records.capture(backend) as groups:
            backend.train([f"p{rank}"], responses=8, seed=17)
        plan = {"dataset": "math", "model": "cpu-distributed-fixture", "total_updates": 275, "shared_prefix_updates": 25}
        protocol = {"plan_sha256": "a" * 64, "input_sha256": "b" * 64, "implementation_sha256": "c" * 64}
        if rank == 2 and damage == "tokens":
            groups[0][2][0][-1] = -1
        elif rank == 2 and damage == "responses":
            phase, prompt, sequences, rewards, start = groups[0]
            groups[0] = (phase, prompt, sequences[:-1], rewards, start)
        try:
            rollout_records.save(engine, groups, folder, "sr", plan, {}, protocol, record)
        except Exception as error:
            if damage is None:
                raise
            atomic_json(folder / f"rank-{rank}-error.json", {
                "type": type(error).__name__, "error": str(error)})
        else:
            if damage is not None:
                raise AssertionError("damaged rank published a successful rollout record")
    finally:
        dist.destroy_process_group()


def test_four_rank_export_keeps_all_training_prompts(tmp_path):
    torch = pytest.importorskip("torch")
    torch.multiprocessing.start_processes(distributed_writer, args=(str(tmp_path),),
                                         nprocs=4, join=True, start_method="fork")
    trace = rollout_records.read(tmp_path / "reward-curves/sr/rollouts/step-26.json.gz")
    assert {g["prompt_id"] for g in trace["groups"]} == {f"p{i}" for i in range(4)}
    assert all(len(g["responses"]) == 8 for g in trace["groups"])


@pytest.mark.parametrize("damage, message", [
    ("tokens", "response-only integer tokens"),
    ("responses", "invalid rollout rewards"),
])
def test_rank_local_trace_failure_reaches_every_rank_before_publication(tmp_path, damage, message):
    """A rank-local export error must not become a peer collective failure."""
    torch = pytest.importorskip("torch")
    torch.multiprocessing.start_processes(distributed_writer, args=(str(tmp_path), damage),
                                         nprocs=4, join=True, start_method="fork")
    errors = [json.loads((tmp_path / f"rank-{rank}-error.json").read_text()) for rank in range(4)]
    assert all(row["type"] == "ValueError" and message in row["error"] for row in errors), errors
    assert not (tmp_path / "reward-curves/sr/rollouts/step-26.json.gz").exists()
    from srgc_rebuttal.cost_ledger import PhaseLedger

    cost = PhaseLedger(tmp_path / "reward-curves/sr/rollout-costs").totals()
    assert not cost["complete"] and cost["total_gpu_seconds"] is None
    assert len(cost["unfinished_phases"]) == 1


def test_fatal_runtime_error_is_preserved_without_another_collective(tmp_path):
    """A poisoned CUDA context must not mask its first error in a gather."""
    from types import SimpleNamespace

    backend = RawBackend()
    backend.world = 2
    engine = SimpleNamespace(backend=backend, step=26, config=Config(), switched_at=None)
    with rollout_records.capture(backend) as groups:
        backend.train(["p0"], responses=8, seed=17)
    record = {"checkpoint": 25, "completed_updates": 26, "selector": "sr", "train_ids": ["p0"]}
    plan = {"dataset": "math", "model": "cpu-fault-fixture", "total_updates": 275, "shared_prefix_updates": 25}
    protocol = {"plan_sha256": "a" * 64, "input_sha256": "b" * 64, "implementation_sha256": "c" * 64}
    original = RuntimeError("CUDA error: unspecified launch failure")
    with patch.object(rollout_records, "as_tokens", side_effect=original), \
            patch.object(backend, "_gather") as gather:
        with pytest.raises(RuntimeError) as caught:
            rollout_records.save(engine, groups, tmp_path, "sr", plan, {}, protocol, record)
    assert caught.value is original
    gather.assert_not_called()
