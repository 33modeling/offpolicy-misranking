"""Regression assertions for the September 13 review, using temporary data."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import benchmark_eval as be
import evidence_downstream as ed
import gate_decision as gd
import gain_vs_reliability as gv
from test_benchmark_eval import _datasets, _fake_shards
from test_evidence_downstream import source_point
from test_gate_arm import _train


def make_pilot(tmp_path, monkeypatch):
    run, evaluation = source_point(tmp_path, drift=400)
    rule = tmp_path / "rule.json"
    rule.write_text(json.dumps(gd.default_rule(pilot_size=40, seed=1)))
    monkeypatch.setenv("E5_GATE_RULE", str(rule))
    out = tmp_path / "out"
    ed.prepare(run, out, evaluation, 100, 8, ["random", "passrate_beta", "gate_passrate"])
    _train(out / "subsets/train-gate_passrate-pilot.args", rho=.7)
    return out


def test_identical_pilot_replays_do_not_inflate_sample_count(tmp_path, monkeypatch):
    out = make_pilot(tmp_path, monkeypatch)
    log = out / "gate_passrate/pilot/reliability_log.rank0.jsonl"
    rows = log.read_text().splitlines(keepends=True)
    before = ed.gate_decide(out, "gate_passrate")
    log.write_text("".join(rows + rows[5:8]))
    assert ed.gate_decide(out, "gate_passrate") == before
    assert before["pilot_pairs"] == 40


@pytest.mark.parametrize("problem", ["missing_rank", "missing_row", "wrong_prompt", "conflicting_replay"])
def test_invalid_pilot_log_cannot_publish_decision(tmp_path, monkeypatch, problem):
    out = make_pilot(tmp_path, monkeypatch)
    log = out / "gate_passrate/pilot/reliability_log.rank3.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    if problem == "missing_rank":
        log.unlink()
    else:
        if problem == "missing_row":
            rows.pop()
        elif problem == "wrong_prompt":
            rows[0]["prompt_index"] += 1
        else:
            rows.append({**rows[0], "pass_a": .99})
        log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        ed.gate_decide(out, "gate_passrate")
    assert not (out / "gate_passrate/decision.json").exists()


def test_unknown_cost_falls_back_under_zero_budget():
    decision = gd.decide_signal({i: (float(i), float(i)) for i in range(40)}, "fresh",
                                gd.default_rule(pilot_size=20, budget_seconds=0), pool_size=400)
    assert decision["decision"] == "random" and decision["reason"] == "unknown_cost"


def test_extending_benchmarks_preserves_completed_shard_bindings(tmp_path):
    run, evaluation = source_point(tmp_path)
    out = tmp_path / "out"
    ed.prepare(run, out, evaluation, 100, 8, ["random"])
    datasets = _datasets(tmp_path)
    be.prepare(out, datasets, ["aime24"], count=8, eval_k=4)
    frozen_hash = ed.digest(out / "benchmarks.json")
    _fake_shards(out, "before", .4)
    before, seconds = be.set_means(out, "before", "aime24")
    be.prepare(out, datasets, ["aime24", "gsm8k"], count=8, eval_k=4)
    after, _ = be.set_means(out, "before", "aime24")
    assert (before == after).all()
    assert ed.digest(out / "benchmarks.json") == frozen_hash
    assert be.sets_of(out) == ["aime24", "gsm8k"]
    _fake_shards(out, "before", .4)
    assert len(be.set_means(out, "before", "gsm8k")[0]) == 8
    assert be.prepare(out, datasets, ["gsm8k", "aime24"], count=8, eval_k=4)["sets"] == be.benchmark_contract(out)["sets"]


def test_changed_source_answer_fails_before_benchmark_freeze(tmp_path):
    run, evaluation = source_point(tmp_path)
    out = tmp_path / "out"
    ed.prepare(run, out, evaluation, 100, 8, ["random"])
    datasets = _datasets(tmp_path)
    target = datasets / "aime24.jsonl"
    target.write_text(target.read_text().replace('"answer": "0"', '"answer": "999"'))
    with pytest.raises(ValueError, match="manifest"):
        be.prepare(out, datasets, ["aime24"], count=8, eval_k=4)


def test_cached_topk_constant_does_not_recompute(tmp_path, monkeypatch):
    run = tmp_path / "point"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({"dataset": "math500", "seed": 0, "drift": 0}))
    monkeypatch.setattr(gv, "signal_halves", lambda *args: {i: (float(i), float(i)) for i in range(40)})
    with patch.object(gv, "topk_constant", return_value=1.7) as constant:
        assert len(gv.point_rows(run, .1, {(40, 4): 1.7})) == 6
    constant.assert_not_called()


def test_stale_merge_rejects_conflicting_parameters(tmp_path):
    import stale_splithalf as ss
    from test_stale_splithalf import fake_point, tiny_model
    run = fake_point(tmp_path, n=4)
    model = tiny_model()
    for shard in range(2):
        ss.compute_shard(run, shard, 2, loader=lambda *args: (model, None), log=lambda *args: None)
    path = run / "scores_stale_splithalf.shard1.json"
    part = json.loads(path.read_text())
    part["parameters"]["clip_cap"] = 999.
    path.write_text(json.dumps(part))
    with pytest.raises(ValueError, match="parameters"):
        ss.merge(run, 2)


@pytest.mark.parametrize("script", ["run_cost_accounting.sh", "run_gain_law.sh", "run_gate_decision.sh"])
@pytest.mark.parametrize("failure", [False, True])
def test_cpu_export_shell_preserves_analysis_exit_status(tmp_path, script, failure):
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(root / "scripts" / script, scripts / script)
    (scripts / "setup_env.sh").write_text('export OM_WORK="$TEST_WORK" VENV_DIR="$TEST_VENV"\n')
    source = tmp_path / "src"
    source.mkdir()
    for module in ("cost_accounting", "gain_vs_reliability", "gate_decision"):
        (source / f"{module}.py").write_text(f'print("analysis fixture")\nraise SystemExit({7 if failure else 0})\n')
    (source / "gain_law_simulation.py").write_text('print("synthetic fixture")\n')
    config = tmp_path / "config"
    config.mkdir()
    (config / "gate_rule.json").write_text(json.dumps({"r_min": .25, "confidence": .9}))
    point = tmp_path / "matrix/family-math500-s0/test-s0-math500-d0"
    point.mkdir(parents=True)
    (point / "DONE").touch()
    (point / "DONE").write_text("complete\n")
    env = {**os.environ, "TEST_WORK": str(tmp_path / "work"), "TEST_VENV": str(Path(sys.executable).parent.parent),
           "OM_OLMO3_ROOT": str(tmp_path / "matrix"), "OM_OLMO3_MODEL_TAG": "test", "E5_SEEDS": "0"}
    result = subprocess.run(["bash", str(scripts / script)], cwd=tmp_path, env=env,
                            text=True, capture_output=True, timeout=20)
    assert (result.returncode != 0) is failure, result.stdout + result.stderr
    exports = list((tmp_path / "work/exports").glob("*.txt"))
    assert len(exports) == 1 and "analysis fixture" in exports[0].read_text()
