"""CPU contracts for the offline random-fallback gate (src/gate_decision.py)."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from test_additional_experiments import _write_run

import gate_decision as gd

ROOT = Path(__file__).resolve().parents[1]


def test_required_pilot_size_follows_the_fisher_formula():
    z = gd.z_score(0.90)
    assert z == pytest.approx(1.6449, abs=1e-3)
    n = gd.required_pilot_size(0.6, 0.3, 0.90)
    assert n == pytest.approx(3 + (z / (math.atanh(0.6) - math.atanh(0.3))) ** 2)
    assert 20 < n < 23
    assert gd.required_pilot_size(0.1, 0.25, 0.90) > 100  # rejecting a weak signal is expensive
    assert math.isinf(gd.required_pilot_size(0.25, 0.25, 0.90))
    with pytest.raises(ValueError):
        gd.required_pilot_size(1.0, 0.25, 0.90)


def test_fisher_bounds_bracket_the_estimate_and_need_four_pairs():
    lo, hi = gd.fisher_bounds(0.5, 40, 0.90)
    assert lo < 0.5 < hi and lo == pytest.approx(0.26, abs=0.02)
    assert all(math.isnan(v) for v in gd.fisher_bounds(0.5, 3, 0.90))
    lo1, hi1 = gd.fisher_bounds(0.999999999, 40, 0.90)  # clamped, finite
    assert math.isfinite(lo1) and math.isfinite(hi1)


def test_behavior_halves_split_by_response_index(tmp_path):
    run = _write_run(tmp_path, n=6, reward_rate=0.5)
    halves = gd.behavior_halves(run)
    assert set(halves) == set(range(6))
    rows = [json.loads(l) for l in (run / "rollouts_behavior_train.jsonl").read_text().splitlines()]
    first = [r["reward"] for r in rows if r["prompt_idx"] == 0 and r["rollout_idx"] < 4]
    assert halves[0][0] == pytest.approx(sum(first) / 4)
    with (run / "rollouts_behavior_train.jsonl").open("a") as handle:
        handle.write(json.dumps({"prompt_idx": 0, "rollout_idx": 0, "reward": 1.0}) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        gd.behavior_halves(run)


def test_pilot_is_deterministic_and_bounded_by_the_pool():
    ids = list(range(100))
    assert gd.pilot_indices(ids, 200, 1) == ids
    a, b = gd.pilot_indices(ids, 20, 1), gd.pilot_indices(ids, 20, 1)
    assert a == b and len(a) == 20 and a == sorted(a)
    assert gd.pilot_indices(ids, 20, 2) != a


def test_decision_retains_a_reliable_fresh_score_and_records_reasons(tmp_path):
    run = _write_run(tmp_path, n=40, seed=3)
    rule = gd.default_rule(pilot_size=40, r_min=0.25, confidence=0.90, seed=1)
    report = gd.decide(run, rule, ("fresh", "difficulty", "g11"))
    rows = {r["signal"]: r for r in report["rows"]}
    fresh = rows["fresh"]
    assert fresh["decision"] == "retain" and fresh["reason"] == "reliable" and fresh["r_half"] > 0.9
    assert fresh["pilot_pairs"] == 40 and fresh["pilot_cost_seconds"] is None  # cost not measured
    assert rows["difficulty"]["decision_steps"] == 10 and rows["difficulty"]["remaining_scoring_seconds"] == 0.0
    assert rows["difficulty"]["reason"] in ("weak", "unresolved", "invalid")
    assert report["skipped"][0]["signal"] == "g11" and "stale_splithalf" in report["skipped"][0]["reason"]
    assert not report["rewards_available"]


def test_topk_overlap_has_the_right_chance_level():
    halves = {i: (float(i), float(i)) for i in range(40)}
    exact = gd.topk_overlap(halves)
    assert exact == {"k": 4, "n": 40, "overlap": 1.0, "chance": 0.1}
    reversed_halves = {i: (float(i), float(-i)) for i in range(40)}
    assert gd.topk_overlap(reversed_halves)["overlap"] == 0.0
    record = gd.decide_signal(halves, "difficulty", gd.default_rule(pilot_size=40))
    assert record["pool_topk_overlap"] == 1.0 and record["pool_topk_k"] == 4


def test_constant_halves_are_invalid_and_fall_back():
    halves = {i: (0.0, 0.0) for i in range(30)}
    record = gd.decide_signal(halves, "difficulty", gd.default_rule(pilot_size=30))
    assert record["decision"] == "random" and record["reason"] == "invalid" and record["valid"] is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_score_cannot_be_clamped_to_perfect_reliability(bad):
    halves = {i: (float(i), float(i)) for i in range(40)}
    halves[0] = (bad, 0.0)
    record = gd.decide_signal(halves, "fresh", gd.default_rule())
    assert record["decision"] == "random" and record["reason"] == "invalid"
    assert record["r_half"] is None and record["pool_topk_overlap"] is None
    json.dumps(record, allow_nan=False)


def test_incomplete_pilot_falls_back_instead_of_shrinking_the_frozen_sample():
    halves = {i: (float(i), float(i)) for i in range(4)}
    record = gd.decide_signal(halves, "fresh", gd.default_rule(), pool_size=400)
    assert record["decision"] == "random" and record["reason"] == "invalid"
    assert record["pilot_pairs"] == 4 and record["pool_topk_overlap"] is None
    # A genuinely smaller full pool is different from missing pilot pairs.
    assert gd.decide_signal(halves, "fresh", gd.default_rule())["valid"]


def test_threshold_equality_and_empty_pilots_have_serializable_decisions(monkeypatch):
    rule = gd.default_rule()
    halves = {i: (float(i), float(i)) for i in range(40)}
    monkeypatch.setattr(gd, "pearson", lambda a, b: rule["r_min"])
    record = gd.decide_signal(halves, "fresh", rule)
    assert record["decision"] == "random" and record["required_pairs_at_r"] is None
    json.dumps(record, allow_nan=False)
    empty = gd.decide_signal({}, "fresh", rule, pool_size=400)
    assert empty["decision"] == "random" and not empty["valid"]
    json.dumps(empty, allow_nan=False)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), True])
def test_rule_rejects_nonfinite_or_boolean_costs(bad):
    for field, value in (("scoring_budget_seconds", bad), ("cost_per_prompt_seconds", {"fresh": bad})):
        with pytest.raises(ValueError, match="finite"):
            gd.validate_rule({**gd.default_rule(), field: value})


def test_budget_check_precedes_the_reliability_check():
    halves = {i: (i / 10.0, i / 10.0 + 0.01 * (-1) ** i) for i in range(40)}
    rule = gd.default_rule(pilot_size=20, r_min=0.25, seed=5, budget_seconds=100.0)
    rule["cost_per_prompt_seconds"] = {"fresh": 10.0}
    record = gd.decide_signal(halves, "fresh", rule, pool_size=40)
    assert record["remaining_scoring_seconds"] == 200.0 and record["pilot_cost_seconds"] == 200.0
    assert record["decision"] == "random" and record["reason"] == "over_budget"
    rule["scoring_budget_seconds"] = 500.0
    assert gd.decide_signal(halves, "fresh", rule, pool_size=40)["reason"] == "reliable"


def _e5_dir(tmp_path):
    seed_dir = tmp_path / "e5"
    seed_dir.mkdir()
    rows = [{"selector": "random", "reward_after": "0.40", "difference_vs_random": "0", "random_lower": "0", "random_upper": "0"},
            {"selector": "passrate_beta", "reward_after": "0.43", "difference_vs_random": "0.03", "random_lower": "0.01", "random_upper": "0.05"},
            {"selector": "fresh_r", "reward_after": "0.39", "difference_vs_random": "-0.01", "random_lower": "-0.03", "random_upper": "0.01"}]
    with (seed_dir / "downstream_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return seed_dir


def test_mapping_to_e5_rewards_charges_forgone_reward(tmp_path):
    rewards = gd.e5_rewards(_e5_dir(tmp_path))
    retained = {"selector": "passrate_beta", "decision": "retain"}
    fallback = {"selector": "passrate_beta", "decision": "random"}
    assert gd.map_to_e5(retained, rewards)["forgone_reward"] == pytest.approx(0.0)
    mapped = gd.map_to_e5(fallback, rewards)
    assert mapped["reward_decision"] == 0.40 and mapped["forgone_reward"] == pytest.approx(0.03)
    assert mapped["selector_vs_random_lower"] == 0.01
    assert gd.map_to_e5({"selector": None, "decision": "random"}, rewards)["reward_selector"] is None


def test_cli_freezes_a_rule_once_and_writes_decisions(tmp_path):
    run = _write_run(tmp_path, n=40, seed=4)
    seed_dir = _e5_dir(tmp_path)
    rule = tmp_path / "rule.json"
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    cmd = [sys.executable, str(ROOT / "src/gate_decision.py")]
    first = subprocess.run(cmd + ["write-rule", "--out", str(rule), "--pilot-size", "20"], capture_output=True, text=True, env=env)
    assert first.returncode == 0 and "frozen" in first.stdout
    frozen = rule.read_text()
    second = subprocess.run(cmd + ["write-rule", "--out", str(rule), "--pilot-size", "99"], capture_output=True, text=True, env=env)
    assert second.returncode == 0 and "unchanged" in second.stdout and rule.read_text() == frozen
    result = subprocess.run(cmd + ["decide", "--run", str(run), "--rule", str(rule), "--e5", str(seed_dir),
                                   "--signals", "fresh", "difficulty"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (seed_dir / "gate_decision.json").is_file() and (seed_dir / "gate_decision.csv").is_file()
    report = json.loads((seed_dir / "gate_decision.json").read_text())
    assert report["rewards_available"] and {r["signal"] for r in report["rows"]} == {"fresh", "difficulty"}
    fresh = next(r for r in report["rows"] if r["signal"] == "fresh")
    assert fresh["pilot_pairs"] == 20 and fresh["reward_selector"] == 0.39
    table = subprocess.run(cmd + ["table", "--r-min", "0.25"], capture_output=True, text=True, env=env)
    assert table.returncode == 0 and "rho=0.60 retain" in table.stdout


def test_rule_validation_rejects_bad_fields():
    for bad in ({"schema": "x"}, {**gd.default_rule(), "pilot_size": 3}, {**gd.default_rule(), "r_min": 1.0},
                {**gd.default_rule(), "confidence": 1.0}, {**gd.default_rule(), "cost_per_prompt_seconds": {"fresh": -1}}):
        with pytest.raises(ValueError):
            gd.validate_rule(bad)
