"""Fresh four-arm execution with real contracts/meters and a CPU GPU-backend stand-in."""
import copy
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import selector_pair_cost_measure as measure
import selector_pair_cost_worker as timing_worker
from selector_pair_cost_worker import Timings
from test_selector_pair import allocation

import evidence_downstream as ed
import selection_gate as core
import selection_gate_gpu as base


@pytest.fixture
def study(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from fake_trainer import parse, write_policy
    root, output, run = tmp_path / "pair", tmp_path / "measurement", tmp_path / "source"
    model = tmp_path / "model"
    core.atomic_json(model / "config.json", {"model_type": "fixture"})
    pools = {split: [{"question": f"{split}{i}", "answer": str(i)} for i in range(n)]
             for split, n in (("train", 20), ("val", 8))}
    core.atomic_json(run / "prompts.json", pools)
    cache = run / "rollouts_behavior_train.jsonl"
    cache.write_text("".join(json.dumps({"prompt_idx": i, "rollout_idx": j,
                                         "reward": int(i in (2, 3) and j < 4)}) + "\n"
                             for i in range(20) for j in range(8)))
    args = parse(["--model", str(model), "--prompts", str(run / "prompts.json"),
                  "--output", str(run / "policy_step_25"), "--target-steps", "25", "--seed", "3",
                  "--epochs-per-batch", "1"])
    write_policy(args)
    config = {"model": str(model), "seed": 3, "drift": 25, "max_new_tokens": args.max_new_tokens,
              "prompt_format": "olmo_rlzero_math", "grpo_gradient_checkpointing": True, "temperature": 1.,
              "fresh_k": 32, "micro_group": 4, "behavior_k": 8, "val_k": 8, "proj_dim": 2,
              "topk_frac": .1, "grad_layers": 1,
              **{field: getattr(args, flag.replace("-", "_")) for flag, field in ed.TRAIN_FLAGS.items()}}
    contract = {"source_run": str(run), "config": config, "n": 20, "scope": {"gpu_type": "H100"},
                "evaluation": {"val": [{"question": f"eval{i}", "answer": str(i)} for i in range(5)],
                               "provenance": {"dataset": "fixture", "revision": "v1", "split": "test"}},
                "eval_k": 8, "eval_seed": 123}
    core.atomic_json(root / "pair.json", {"protocol_id": "fixture"})
    core.atomic_json(root / "branches/on_policy/states/s3-t25/points/view-25/contract.json", contract)
    options = SimpleNamespace(root=root, output=output, seed=[3], end_step=100, interval=25,
                              selection_interval=25, replicates=1, max_gpu_hours=160.)
    plan = measure.make_plan(options)
    plan["hardware"] = {"gpu_type": "H100"}
    calls = []

    def gpu(command):
        if "--nproc_per_node=4" in command:
            index = command.index(str(measure.REPO / "src/train_policy_grpo.py"))
            write_policy(parse(command[index + 1:]))
            calls.append(("train", command))
            return
        directory = Path(command[command.index("--root") + 1])
        stage = command[command.index("--stage") + 1]
        kind = command[command.index("--kind") + 1]
        shard = int(command[command.index("--shard") + 1])
        calls.append((kind, stage, shard, directory))
        if kind == "evaluation":
            c = core.read(directory / "evaluation.json")
            n, k = len(c["questions"]), c["k"]
            path = directory / f"evaluation-{shard}.jsonl"
            path.write_text("".join(json.dumps({"prompt_idx": i, "rollout_idx": j, "reward": int(j < 4)}) + "\n"
                                    for i in range(n * shard // 4, n * (shard + 1) // 4) for j in range(k)))
            core.atomic_json(directory / f"evaluation-{shard}.done.json", {
                "evaluation_sha256": base.digest(directory / "evaluation.json"),
                "shard": shard, "sha256": base.digest(path)})
        else:
            if kind == "ranking":
                c = core.read(directory / "scoring.json")
                ids = list(range(4 if stage == "validation" else 20))
                values = {i: [1., 0.] if stage == "validation" else float(20 - i) for i in ids}
                binding = {"contract_sha256": base.digest(directory / "scoring.json")}
            else:
                c = core.read(directory / "reference.json")
                ids = measure.reference_score.indices(c, pools, stage)
                values = {i: ([-1., 0.] if i in c["sets"]["on_policy"] else [1., 0.])
                          if stage == "candidate-a" else [1., 0.] for i in ids}
                binding = {"reference_sha256": base.digest(directory / "reference.json")}
            indices = ids[len(ids) * shard // 4:len(ids) * (shard + 1) // 4]
            payload = directory / f"{stage}-{shard}.json"
            core.atomic_json(payload, {str(i): values[i] for i in indices})
            core.atomic_json(directory / f"{stage}-{shard}.done.json", {
                **binding, "stage": stage, "shard": shard, "sha256": base.digest(payload)})
        core.atomic_json(directory / f"timing-{stage}-{shard}.json", {
            "stage": stage, "shard": shard, "gpus": 1,
            "seconds": {"model_setup": 1., "response_generation": 2., "gradient_computation": 3.}})

    real_meter = base.meter

    def meter(directory, name, gpu_type, *, commands=None, action=None, **kwargs):
        def operation():
            if commands:
                for command, device in commands:
                    assert device
                    gpu(command)
            else:
                return action()
        return real_meter(directory, name, gpu_type, action=operation, **kwargs)

    monkeypatch.setattr(base, "meter", meter)
    monkeypatch.setattr(measure.pair_worker, "environment", lambda _: {})
    return options, plan, calls


def test_prospective_four_arm_run_no_source_writes_and_idempotent(study):
    args, plan, calls = study
    before = {path: path.read_bytes() for path in args.root.parent.rglob("*") if path.is_file()}
    for unit, source, replica, arm in measure.units(plan):
        measure.run_unit(unit, arm, plan, source, ["0", "1", "2", "3"])
    count = len(calls)
    for unit, source, replica, arm in measure.units(plan):
        measure.run_unit(unit, arm, plan, source, ["0", "1", "2", "3"])
    assert len(calls) == count
    assert all(path.read_bytes() == content for path, content in before.items())
    report = measure.report(plan)
    assert report["complete"] and len(report["rows"]) == 4
    rows = {r["arm"]: r for r in report["rows"]}
    assert rows["switch"]["switch_step"] == 50
    assert all(r["update_timer_gpu_seconds"] == 75 * 70 * 4 for r in rows.values())
    assert rows["on_policy"]["ranking_steps"] == [25, 50, 75]
    assert rows["switch"]["ranking_steps"] == [25, 50]
    assert rows["switch"]["check_steps"] == [25, 50]
    assert rows["on_policy"]["check_steps"] == []
    for arm, row in rows.items():
        assert row["operating_gpu_seconds"] == sum(row["gpu_seconds"][k] for k in measure.CATEGORIES[:-1])
        assert row["gpu_seconds"]["online_check"] > 0 if arm == "switch" else row["gpu_seconds"]["online_check"] == 0
        assert row["reward"] == .5
        assert row["gpu_seconds"]["evaluation"] > 0
    ranks = [call for call in calls if call[0] == "ranking"]
    assert len(ranks) == 40  # On-policy: three rankings; Switch: two before its trigger
    assert len({call[3] for call in ranks}) == 5
    check_calls = [call for call in calls if call[0] == "check"]
    assert {c[1] for c in check_calls} == {"candidate-a", "validation-a"}
    later = next(c[3] for c in check_calls if "check-50" in str(c[3]))
    assert "s3-r1-switch/phases/train-25-50" in core.read(later / "reference.json")["parent"]
    switch_commands = [c[1] for c in calls if c[0] == "train" and "s3-r1-switch" in " ".join(c[1])]
    assert "subset-on_policy-step-25.json" in " ".join(switch_commands[0])
    assert "subset-cached.json" in " ".join(switch_commands[1])
    assert rows["on_policy"]["function_gpu_seconds_successful_scoring_only"]["gradient_computation"] == 72.
    assert rows["switch"]["scoring_function_gpu_seconds_by_category"]["selection"]["gradient_computation"] == 48.
    assert rows["switch"]["scoring_function_gpu_seconds_by_category"]["online_check"]["gradient_computation"] == 48.
    csv_rows = list(csv.DictReader(io.StringIO(measure.render_csv(report))))
    assert all(None not in row and None not in row.values() for row in csv_rows)
    switch_row = next(row for row in csv_rows if row["arm"] == "switch")
    assert switch_row["ranking_steps"] == "25;50" and switch_row["ranking_count"] == "2"
    assert switch_row["check_count"] == "2"


def test_no_switch_keeps_ranking_and_training_to_endpoint(study, monkeypatch):
    _, plan, _ = study
    monkeypatch.setattr(measure, "should_switch", lambda _: False)
    unit, source, _, arm = next(u for u in measure.units(plan) if u[3] == "switch")
    result = measure.run_unit(unit, arm, plan, source, ["0", "1", "2", "3"])
    assert result["switch_step"] is None
    assert result["ranking_steps"] == [25, 50, 75]
    assert [c["step"] for c in result["checks"]] == [25, 50, 75]
    assert sum(b["updates"] for b in result["blocks"]) == 75


def test_switch_and_on_policy_rerank_the_current_model(study):
    _, plan, calls = study
    for unit, source, _, arm in measure.units(plan):
        if arm in ("switch", "on_policy"):
            measure.run_unit(unit, arm, plan, source, ["0", "1", "2", "3"])
    later_ranks = {c[3] for c in calls if c[0] == "ranking" and "on-ranking-50" in str(c[3])}
    assert len(later_ranks) == 2
    for directory in later_ranks:
        c = core.read(directory / "scoring.json")
        assert "train-25-50" in c["parent"] and c["config"]["drift"] == 50


def test_training_boundaries_cover_selection_and_checks_without_gaps():
    plan = {"end_step": 35, "interval": 4, "selection_interval": 3}
    assert measure.block_boundaries(plan) == [25, 28, 29, 31, 33, 34, 35]
    plan["selection_interval"] = 1
    assert measure.block_boundaries(plan) == list(range(25, 36))


@pytest.mark.parametrize("values,expected", [([], False), ([-1], False), ([-1, -2], True),
    ([-2, 1, -2], True), ([-1, 10, -1], False), ([-1, 0, -1], True), ([1, -1, 1], False)])
def test_switch_rule(values, expected):
    assert measure.should_switch(values) is expected


@pytest.mark.parametrize("bad", [True, None, float("nan"), float("inf")])
def test_bad_contrast_not_default_switch(bad):
    with pytest.raises(ValueError, match="finite"):
        measure.should_switch([-1., bad])


def test_costs_include_failed_attempts_and_keep_missing_unknown(tmp_path):
    unit = tmp_path / "unit"
    for name, category, duration, code in (("score", "selection", 2., 1), ("train", "training", 3., 0),
                                            ("eval", "evaluation", 7., 0)):
        core.atomic_json(unit / f"phases/{name}/category.json", {"category": category})
        for event in allocation(name, name, 0., duration, rc=code):
            base.journal(unit / f"phases/{name}/attempt-0001/cost.jsonl", event)
    costs = measure.costs(unit)
    assert costs["gpu_seconds"] == {"selection": 8., "online_check": 0., "training": 12., "evaluation": 28.}
    assert costs["failed_events"] == 1
    base.journal(unit / "phases/train/attempt-0002/cost.jsonl", allocation("open", "train", 10, 20)[0])
    assert measure.costs(unit)["unknown"]
    with pytest.raises(ValueError, match="unclosed"):
        measure.phase(unit, "next", "selection", {}, {}, [], lambda *_: pytest.fail("must not run"))


def test_failed_phase_retries_in_fresh_directory_without_erasing_cost(study):
    _, plan, _ = study
    unit, source, _, _ = next(measure.units(plan))
    attempts = []
    def job(directory, paid):
        attempts.append(directory)
        def action():
            if len(attempts) == 1:
                raise RuntimeError("interrupted")
            core.atomic_json(directory / "value.json", {"ok": True})
        paid("action", action=action)
        return {"ok": True}, [directory / "value.json"]
    with pytest.raises(RuntimeError, match="interrupted"):
        measure.phase(unit, "test", "selection", plan, source, [], job)
    assert measure.phase(unit, "test", "selection", plan, source, [], job) == {"ok": True}
    assert len(set(attempts)) == 2
    assert measure.costs(unit)["failed_events"] == 1
    (attempts[1] / "value.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        measure.phase(unit, "test", "selection", plan, source, [], job)


def test_source_code_and_optimizer_are_bound(study):
    _, plan, _ = study
    measure.validate_plan(plan)
    altered = copy.deepcopy(plan)
    altered["code_sha256"]["scripts/selector_pair_cost_measure.py"] = "wrong"
    with pytest.raises(ValueError, match="code changed"):
        measure.validate_plan(altered)
    parent = Path(plan["sources"][0]["parent"])
    (parent / "optimizer.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        measure.validate_plan(plan)


@pytest.mark.parametrize("path", ["source", "source/nested", "."])
def test_output_cannot_overlap_source(tmp_path, path):
    with pytest.raises(ValueError, match="separate"):
        measure.separated(tmp_path / path, [tmp_path / "source"])


def test_partial_report_never_presents_zero_as_complete(study):
    _, plan, _ = study
    data = measure.report(plan)
    assert not data["complete"]
    assert all(r["operating_gpu_seconds"] is None and r["reward"] is None for r in data["rows"])
    assert "unknown,unknown,unknown,unknown" in measure.render_csv(data)


def test_deleted_completed_phase_cannot_be_reported_as_free(study):
    _, plan, _ = study
    unit, source, _, arm = next(measure.units(plan))
    measure.run_unit(unit, arm, plan, source, ["0", "1", "2", "3"])
    (unit / "phases/random-selection/success.json").unlink()
    with pytest.raises(FileNotFoundError):
        measure.report(plan)


def test_zero_budget_cannot_launch_more_work(study):
    _, plan, _ = study
    unit, source, _, arm = next(measure.units(plan))
    plan["max_gpu_hours_per_arm"] = 0
    with pytest.raises(ValueError, match="cap exhausted"):
        measure.choose_subset(unit, arm, plan, source, ["0", "1", "2", "3"])
    assert not list(unit.rglob("cost.jsonl"))


def test_plan_rejects_changed_parent_protocol_and_nonpositive_interval(study):
    args, _, _ = study
    args.interval = 0
    with pytest.raises(ValueError, match="positive interval"):
        measure.make_plan(args)
    args.interval = 25
    path = args.root / "branches/on_policy/states/s3-t25/points/view-25/contract.json"
    c = core.read(path)
    c["config"]["grpo_learning_rate"] *= 2
    core.atomic_json(path, c)
    with pytest.raises(ValueError, match="common parent"):
        measure.make_plan(args)


def test_hardware_snapshot_rejects_duplicate_or_mixed_gpus(monkeypatch):
    monkeypatch.setattr(measure.subprocess, "check_output", lambda *a, **k: "u,H100,1,80000\n" * 4)
    with pytest.raises(ValueError, match="matching GPUs"):
        measure.hardware_inventory(["0", "1", "2", "3"])
    monkeypatch.setattr(measure.subprocess, "check_output", lambda *a, **k: "".join(
        f"u{i},H100,1,80000\n" for i in range(4)))
    assert measure.hardware_inventory(["0", "1", "2", "3"])["gpu_type"] == "H100"


def test_timing_synchronizes_and_preserves_return_and_exceptions():
    sync = []
    times = iter([1., 3., 5., 9.])
    timings = Timings(lambda: sync.append(True), clock=lambda: next(times))
    assert timings.wrap("gradient_computation", lambda x: x + 1)(2) == 3
    def fail():
        raise RuntimeError("failure")
    with pytest.raises(RuntimeError):
        timings.wrap("gradient_computation", fail)()
    assert timings.seconds == {"gradient_computation": 6.}
    assert timings.calls == {"gradient_computation": 2}
    assert len(sync) == 4


def test_worker_profiles_existing_functions_and_restores_them(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import grads
    import rollout
    functions = {}
    for module, name in ((rollout, "load_policy"), (rollout, "collect_rollouts"), (grads, "prompt_gradient")):
        functions[name] = lambda: "measured"
        monkeypatch.setattr(module, name, functions[name])
    def work(*args):
        assert rollout.load_policy() == "measured"
        assert rollout.collect_rollouts() == "measured"
        assert grads.prompt_gradient() == "measured"
    monkeypatch.setattr(measure.scoring, "worker", work)
    timing_worker.worker(tmp_path, "ranking", "validation", 0)
    data = core.read(tmp_path / "timing-validation-0.json")
    assert data["calls"] == {"model_setup": 1, "response_generation": 1, "gradient_computation": 1}
    assert all(value >= 0 for value in data["seconds"].values())
    assert grads.prompt_gradient is functions["prompt_gradient"]
    assert rollout.collect_rollouts is functions["collect_rollouts"]
    with pytest.raises(ValueError, match="one reference"):
        timing_worker.worker(tmp_path, "check", "candidate-b", 0)
    assert not (tmp_path / "timing-candidate-b-0.json").exists()


def test_launcher_default_is_plan_and_never_recovers_old_pair():
    shell = (measure.REPO / "scripts/run_selector_pair_cost_measure.sh").read_text()
    assert "MODE=${1:-plan}" in shell
    assert "e5_recover_pair_gpu() { return 0; }" in shell
    assert "E5_FORCE=0" in shell
    assert "OM_NODE_LOCK_HELD=1" in shell
    assert 'exec "$PY"' not in shell
