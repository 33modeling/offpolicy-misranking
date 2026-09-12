"""CPU regression tests for the v3-derived, corrected v2 screening experiment."""

import json
import os
import math
import random
import statistics
import time
import subprocess
import sys
from pathlib import Path

import pytest

import light_selection_gate as light
import light_selection_gate_gpu as gpu
import selection_gate as core
import selection_gate_gpu as base


def cache(tmp_path, rows=None):
    path = tmp_path / "cache.jsonl"
    if rows is None:
        rows = [{"prompt_idx": i, "rollout_idx": j, "reward": int(j < i)}
                for i in range(8) for j in range(8)]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    return path


def measure(path, **kwargs):
    return light.measure(path, prompts=8, responses=8, k=2, seed=1, cache_step=0, target_step=100, **kwargs)


def test_exact_non_gaussian_counterexample_invalidates_universal_ceiling():
    r = light.counterexamples()["non_gaussian"]
    assert r["top_noisy_gain"] == .125 and r["oracle_gain"] == .25
    assert r["decoded_gain"] == .25
    assert r["top_noisy_gain"]/r["oracle_gain"] > r["sqrt_rho"]


@pytest.mark.parametrize("rho", [.04, .25, .81])
def test_gaussian_model_score_advantage_matches_conditional_formula(rho):
    rng = random.Random(20260912)
    gains, oracles = [], []
    for _ in range(10000):
        theta = [rng.gauss(0, 1) for _ in range(8)]
        noisy = [v+rng.gauss(0, math.sqrt((1-rho)/rho)) for v in theta]
        chosen = sorted(range(8), key=lambda i: noisy[i])[-2:]
        baseline = statistics.fmean(theta)
        gains.append(statistics.fmean(theta[i] for i in chosen)-baseline)
        oracles.append(statistics.fmean(sorted(theta)[-2:])-baseline)
    assert abs(statistics.fmean(gains)/statistics.fmean(oracles)-math.sqrt(rho)) < .025


def test_perfect_passrate_reliability_does_not_imply_grpo_activity():
    r = light.counterexamples()["perfect_passrate_zero_activity"]
    assert r["half_passrate_correlation"] == 1
    assert r["half_score_correlation"] is None and r["mixed_fraction"] == 0
    assert r["score_spread"] == 0


def test_actual_score_not_passrate_and_no_invalid_spearman_brown(tmp_path):
    r = measure(cache(tmp_path))
    assert r["half_score_correlation"] != r["half_passrate_correlation"]
    assert r["spearman_brown"] is None
    assert r["cache_step"] == 0 and r["target_step"] == 100
    assert len(r["selected_indices"]) == 2 and r["full_pool_coverage"]
    assert r["wall_seconds"] > 0 and r["cpu_seconds"] > 0


@pytest.mark.parametrize("kind", ["duplicate", "missing", "fractional", "index", "truncated", "boolean"])
def test_bad_cache_is_rejected(tmp_path, kind):
    path = cache(tmp_path)
    rows = [json.loads(v) for v in path.read_text().splitlines()]
    if kind == "duplicate": rows.append(rows[0])
    if kind == "missing": rows.pop()
    if kind == "fractional": rows[0]["reward"] = .5
    if kind == "index": rows[0]["prompt_idx"] = 8
    if kind == "boolean": rows[0]["reward"] = True
    path = cache(tmp_path, rows)
    if kind == "truncated": path.write_text(path.read_text()+'{"prompt_idx":')
    with pytest.raises((ValueError, TypeError)):
        measure(path)


def test_byte_and_wall_caps_are_enforced(tmp_path):
    path = cache(tmp_path)
    with pytest.raises(ValueError, match="byte cap"):
        measure(path, max_bytes=10)
    with pytest.raises(TimeoutError):
        measure(path, wall_cap=1e-12)


def report():
    return {"full_pool_coverage": True, "target_step": 100, "cache_step": 0,
            "half_score_correlation": .5, "mixed_fraction": .5,
            "mixed_uplift": .2, "score_spread": .1}


def test_screen_is_exploratory_and_charges_measurement():
    rule = light.default_rule()
    r = report()
    decision = light.choose(r, rule, measured_gpu_seconds=1, budget_gpu_seconds=1000)
    assert decision["action"] == "select" and "not a benchmark" in decision["claim"]
    assert light.choose(r, rule, measured_gpu_seconds=11, budget_gpu_seconds=1000)["action"] == "random"
    r["target_step"] = 401
    assert "cache_age_out_of_scope" in light.choose(r, rule, measured_gpu_seconds=1, budget_gpu_seconds=1000)["reasons"]


@pytest.mark.parametrize("field,value", [("half_score_correlation", None), ("half_score_correlation", -.5),
                                         ("mixed_fraction", 0), ("score_spread", 0), ("mixed_uplift", 0)])
def test_no_signal_falls_back_to_random(field, value):
    r = report(); r[field] = value
    assert light.choose(r, light.default_rule(), measured_gpu_seconds=0, budget_gpu_seconds=1)["action"] == "random"


def test_nonlinear_full_score_is_not_average_half_score():
    score = lambda xs: -abs(sum(xs)/len(xs)-.5)
    assert score([0, 0, 1, 1]) != (score([0, 0])+score([1, 1]))/2


def test_frozen_rule_requires_development_ids():
    rule = light.default_rule(); rule["status"] = "development_frozen"
    with pytest.raises(ValueError, match="trajectory IDs"):
        light.validate_rule(rule)


def test_freeze_uses_development_trajectory_ids_and_never_fits_a_tree():
    results = {"schema": light.SCHEMA, "role": "development", "excluded": [],
               "rule": light.default_rule(), "points": [{"trajectory_id": "m:seed-0"}, {"trajectory_id": "m:seed-1"}]}
    rule = light.freeze(results)
    assert rule["status"] == "development_frozen"
    assert rule["development_trajectory_ids"] == ["m:seed-0", "m:seed-1"]
    for key in ("min_half_score_correlation", "min_mixed_uplift", "max_measurement_fraction"):
        assert rule[key] == results["rule"][key]
    results["role"] = "test"
    with pytest.raises(ValueError, match="development"):
        light.freeze(results)


def test_incomplete_development_summary_cannot_freeze_a_rule():
    with pytest.raises(ValueError):
        light.freeze({"schema": light.SCHEMA, "role": "development", "points": [], "excluded": []})


def test_initial_reuses_decision_without_any_measurement(tmp_path, monkeypatch):
    rule = light.default_rule()
    value = {"action": "random", "rule_sha256": core.fingerprint(rule)}
    core.atomic_json(tmp_path / "gated/decision.json", value)
    monkeypatch.setattr(base, "meter", lambda *a, **k: pytest.fail("second measurement"))
    assert gpu.initial(tmp_path, {}, rule, [], {}) == value


def test_failed_measurement_is_charged_and_never_repeated(tmp_path, monkeypatch):
    rule = light.default_rule()
    core.atomic_json(tmp_path / "contract.json", {"budget_gpu_seconds": 1000, "scope": {"gpu_type": "H100"}})
    calls = []
    real = base.meter
    def fail(directory, name, kind, **kwargs):
        calls.append(name)
        def action(): raise ValueError("invalid cache")
        return real(directory, name, kind, action=action, ledger="deployment")
    monkeypatch.setattr(base, "meter", fail)
    first = gpu.initial(tmp_path, {"measurement_wall_seconds": 30}, rule, [], {})
    assert first["action"] == "random" and first["measurement_gpu_seconds"] > 0
    assert gpu.initial(tmp_path, {}, rule, [], {}) == first
    assert calls == ["measurement"]


def test_random_baseline_never_measures(tmp_path, monkeypatch):
    source = tmp_path / "source"
    core.atomic_json(source / "prompts.json", {"train": list(range(40)), "val": []})
    out = tmp_path / "out"
    c = {"source_run": str(source), "n": 40, "scope": {"gpu_type": "H100"},
         "config": {"seed": 0, "drift": 100, "topk_frac": .1}, "budget_gpu_seconds": 1000, "max_steps": 1000}
    core.atomic_json(out / "contract.json", c)
    monkeypatch.setattr(gpu, "initial", lambda *a: pytest.fail("random measured"))
    monkeypatch.setattr(base, "verify", lambda *a: c)
    monkeypatch.setattr(base, "policy", lambda *a: source)
    monkeypatch.setattr(base, "rewards", lambda *a: {"0": .5})
    core.atomic_json(out / "random_full/policy/budget_stop.json", {"completed_steps": 101, "stop_reason": "no_block_fits"})
    for i in range(4): core.atomic_json(out / f"random_full/evaluation/shard-{i}.done.json", {})
    gpu.run_arm(out, {"eval_timeout": 5}, light.default_rule(), "random_full", ["0", "1", "2", "3"], {})
    result = gpu.validate_result(out, "random_full")
    assert result["matched_budget"] and result["action"] == "random"
    paid = base.spent(out / "random_full")
    gpu.run_arm(out, {}, light.default_rule(), "random_full", [], {})
    assert base.spent(out / "random_full") == paid


def test_sigterm_reaps_child_and_closes_cost_ledger(tmp_path):
    child_pid = tmp_path / "child.pid"
    child = "import os,time; from pathlib import Path; Path("+repr(str(child_pid))+").write_text(str(os.getpid())); time.sleep(60)"
    program = ("import sys; from pathlib import Path; import light_selection_gate_gpu as g; "
               "g.install_signal_handlers(); g.base.meter(Path("+repr(str(tmp_path))+"), 'train', 'H100', "
               "commands=[([sys.executable, '-c', "+repr(child)+"], '')], timeout=60, devices=0)")
    process = subprocess.Popen([sys.executable, "-c", program], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic()+10
        while not child_pid.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert child_pid.exists(), "worker did not start"
        pid = int(child_pid.read_text())
        process.terminate()
        process.wait(timeout=10)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert base.cost(tmp_path)["complete"]
        assert base.cost(tmp_path)["ledgers"]["research"]["failed_events"] == 1
    finally:
        if process.poll() is None:
            process.kill(); process.wait()


@pytest.mark.parametrize("mode", ["plan", "status"])
def test_shell_cpu_modes_without_source_or_gpu(tmp_path, mode):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "LIGHT_GATE_PYTHON": sys.executable,
           "LIGHT_GATE_ROOT": str(tmp_path / "new")}
    r = subprocess.run(["bash", "scripts/run_light_gate.sh", mode], cwd=root, env=env,
                       capture_output=True, text=True, timeout=15)
    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "new").exists()


def test_launcher_waits_for_both_preparation_artifacts(tmp_path):
    script = (Path(__file__).resolve().parents[1] / "scripts/run_light_gate.sh").read_text()
    start = script.index('if [ "$MODE" = prepare ]')
    condition = script[start:script.index("; then", start)]
    command = condition+'; then echo prepare; else echo run; fi'
    env = {**os.environ, "MODE": "run", "OUT_ROOT": str(tmp_path)}
    def action():
        return subprocess.check_output(["bash", "-c", command], env=env, text=True).strip()
    assert action() == "prepare"
    core.atomic_json(tmp_path / "suite.json", {})
    assert action() == "prepare", "suite.json alone is not a ready light-gate suite"
    core.atomic_json(tmp_path / "light_protocol.json", {})
    assert action() == "run"


def test_four_workers_claim_ten_arms_without_duplicate_training(tmp_path):
    entries = []
    for seed in range(5):
        name = f"seed-{seed}"
        out = tmp_path / "points" / name
        core.atomic_json(out / "contract.json", {"config": {"seed": seed}, "scope": {"gpu_type": "H100"}})
        entries.append({"name": name, "sha256": base.digest(out / "contract.json")})
    core.atomic_json(tmp_path / "suite.json", {"schema": base.SCHEMA, "points": entries})
    core.atomic_json(tmp_path / "light_protocol.json", {"schema": light.SCHEMA,
                     "arms": list(gpu.ARMS), "rule": light.default_rule()})
    program = """
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import light_selection_gate_gpu as gpu
sys.modules['additive_experiment'] = SimpleNamespace(model_environment=lambda c: {})
gpu.subprocess.check_output = lambda *a, **kw: 'H100\\n'*4
def run(out, suite, rule, arm, devices, env):
    marker = out / arm / 'mock-trained.json'
    if marker.exists():
        return
    time.sleep(.1)
    with marker.open('x') as handle:
        json.dump({'worker': os.getpid()}, handle)
gpu.run_arm = run
gpu.status = lambda root: None
raise SystemExit(gpu.work(Path(sys.argv[1])))
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3", "OM_NODE_LOCK_HELD": "1"}
    processes = []
    try:
        for _ in range(4):
            processes.append(subprocess.Popen([sys.executable, "-c", program, str(tmp_path)],
                             env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))
        for process in processes:
            _, error = process.communicate(timeout=15)
            assert process.returncode == 0, error
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill(); process.wait()
    markers = list(tmp_path.glob("points/*/*/mock-trained.json"))
    assert len(markers) == 10
    assert not list(tmp_path.glob("points/*/*/failure.json"))
