import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import net_gain_gate as net
import selection_gate as core
from test_selection_gate_study import fixture as legacy_fixture

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    data = legacy_fixture()
    data.update(schema=net.SCHEMA, label_protocol="v3_measured")
    data["measurement_config"] = {"recent_window": 20, "max_measurement_fraction": .01, "measurement_wall_seconds": 30.}
    for point in data["points"]:
        point["features"].update(decision_step=100, cache_age_steps=100, recent_available=1.,
                                 recent_reward=.5, recent_active_fraction=.5, recent_gpu_seconds_per_update=40.)
    return data


def cache(path, groups):
    path.write_text("".join(json.dumps({"prompt_idx": i, "rollout_idx": j, "reward": r})+"\n"
                            for i, group in enumerate(groups) for j, r in enumerate(group)))


def stats(path, steps):
    path.write_text("".join(json.dumps({"step": i, "reward_mean": .1*i, "groups": 4,
                                        "nonzero_advantage_groups": 2, "step_seconds": i+1})+"\n" for i in steps))


def test_full_pool_and_current_checkpoint_state_are_different_inputs(tmp_path):
    c, s = tmp_path / "cache", tmp_path / "stats"
    groups = [[0]*4, [0, 1, 0, 1], [1]*4, [0, 0, 0, 1]]
    cache(c, groups)
    stats(s, [1, 2])
    a = net.measure(c, stats=s, step=2, prompts=4, responses=4)
    stats(s, [1, 2, 3])
    b = net.measure(c, stats=s, step=3, prompts=4, responses=4)
    assert a["source_sha256"] == b["source_sha256"]
    assert a["features"]["success_rate"] == b["features"]["success_rate"]
    assert a["features"]["recent_reward"] != b["features"]["recent_reward"]
    assert a["features"]["recent_gpu_seconds_per_update"] == 10.
    assert a["full_pool_coverage"] and a["schedule"] == net.SCHEDULE
    from light_selection_gate import summarize
    assert a["difficulty_indices"] == summarize(groups, k=1, seed=0)["selected_indices"]
    net.validate_features(a["features"])


@pytest.mark.parametrize("steps", [[1], [1, 2, 3], [1, 1, 2], [2, 1], []])
def test_missing_future_duplicate_or_unordered_current_state_is_rejected(tmp_path, steps):
    s = tmp_path / "stats"
    stats(s, steps)
    with pytest.raises(ValueError):
        net.state_features(s, 2, 0)


def test_base_checkpoint_does_not_get_later_rewards(tmp_path):
    values, digest = net.state_features(None, 0, 0)
    assert values["recent_available"] == 0 and digest is None
    with pytest.raises(ValueError, match="later"):
        net.state_features(tmp_path / "later", 0, 0)
    with pytest.raises(ValueError, match="newer"):
        net.state_features(None, 0, 1)


@pytest.mark.parametrize("kind", ["duplicate", "missing", "bool", "nonbinary", "nan"])
def test_bad_caches_cannot_drive_decisions(tmp_path, kind):
    c = tmp_path / "cache"
    cache(c, [[0, 0, 1, 1], [0, 1, 0, 1]])
    rows = c.read_text().splitlines()
    if kind == "duplicate":
        rows.append(rows[0])
    elif kind == "missing":
        rows.pop()
    else:
        r = json.loads(rows[0])
        r["reward"] = {"bool": True, "nonbinary": .5, "nan": float("nan")}[kind]
        rows[0] = json.dumps(r)
    c.write_text("\n".join(rows))
    with pytest.raises(ValueError):
        net.measure(c, stats=None, step=0, prompts=2, responses=4)


def test_resource_caps_and_zero_update_cost(tmp_path):
    c = tmp_path / "cache"
    cache(c, [[0]*4, [1]*4])
    with pytest.raises(TimeoutError):
        net.measure(c, stats=None, step=0, prompts=2, responses=4, wall_cap=1e-12)
    features = fixture()["points"][0]["features"]
    features["recent_gpu_seconds_per_update"] = 0
    with pytest.raises(ValueError, match="positive"):
        net.validate_features(features)


def test_fit_uses_net_gain_not_raw_selected_vs_shortened_random():
    data = fixture()
    for p in data["points"]:
        p["branches"]["selection_reduced"]["rewards"] = {"q0": .495, "q1": .495}
    model = net.fit(data)
    decision = net.choose(model, data["points"][0]["features"])
    assert decision["action"] == "random"
    assert decision["prediction"] == pytest.approx(-.005)


def test_paid_fallback_and_trajectory_aggregation():
    data = fixture()
    report = net.analyze(data, net.fit(data))
    held_out = [r for r in report["rows"] if r["role"] == "test"]
    assert [r["action"] for r in held_out] == ["select", "random"]
    assert held_out[1]["gate_minus_random"] == pytest.approx(-.01)
    assert all(abs(r["accounting_residual"]) < 1e-12 for r in report["rows"])
    assert report["by_role"]["test"]["trajectories"] == 2
    assert report["horizon"]["optimal_switch_step"] is None
    assert report["evaluation_kind"].startswith("offline") and not report["certified"]


def test_held_out_rewards_do_not_change_fitted_model():
    data = fixture()
    model = net.fit(data)
    for p in data["points"]:
        if p["role"] != "development":
            p["branches"]["selection_reduced"]["rewards"] = {"q0": 0., "q1": 1.}
    assert net.fit(data) == model


def test_changed_measurement_schedule_cannot_reuse_a_fitted_model():
    data = fixture()
    model = net.fit(data)
    data["measurement_config"]["recent_window"] = 10
    with pytest.raises(ValueError, match="configuration differs"):
        net.analyze(data, model)


@pytest.mark.parametrize("kind", ["trajectory", "parent", "budget", "selector", "gpu", "synthetic"])
def test_no_scope_or_split_leakage(kind):
    data = fixture()
    model = net.fit(data)
    p = copy.deepcopy(data["points"][-1])
    parent = (p["branches"]["random_full"]["parent_sha256"], p["branches"]["random_full"]["optimizer_sha256"])
    if kind == "trajectory":
        p["trajectory_id"] = "train-0"
    elif kind == "parent":
        parent = ("parent-train-0", "optimizer-train-0")
    elif kind == "budget":
        p["budget_gpu_seconds"] += 1
    elif kind == "selector":
        p["scope"]["selector"] = "other"
    elif kind == "gpu":
        p["scope"]["gpu_type"] = "other"
    with pytest.raises(ValueError):
        net.check_scope(model, p["scope"], p["budget_gpu_seconds"], trajectory=p["trajectory_id"],
                        parent=parent, role="test", observed=kind == "synthetic")


def test_no_extrapolation_no_post_training_features_no_repeated_seeds():
    data = fixture()
    model = net.fit(data)
    f = copy.deepcopy(data["points"][0]["features"])
    f["decision_step"] = 400
    assert net.choose(model, f)["reason"] == "outside_development_support"
    f["heldout_test_reward"] = .99
    with pytest.raises(ValueError, match="unapproved"):
        net.choose(model, f)
    for p in data["points"]:
        if p["role"] == "development":
            p["trajectory_id"] = "same-seed"
    with pytest.raises(ValueError, match="three independent"):
        net.fit(data)


def test_legacy_labels_cannot_be_relabelled_new_test_or_equal_update():
    data = fixture()
    data["label_protocol"] = "legacy_replay_development_only"
    with pytest.raises(ValueError, match="held-out"):
        net.validate_study(data)
    data = fixture()
    data["points"][0]["branches"]["random_full"]["stop_reason"] = "updates_completed"
    with pytest.raises(ValueError, match="equal-update"):
        net.validate_study(data)


def test_core_import_is_gpu_free():
    result = subprocess.run([sys.executable, "-c",
        "import sys, net_gain_gate; assert not {'torch', 'transformers'} & sys.modules.keys()"],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_cpu_shell_fits_analyzes_and_preserves_frozen_files(tmp_path):
    data, model, report = [tmp_path / f"{name}.json" for name in ("study", "model", "report")]
    core.atomic_json(data, fixture())
    env = {**os.environ, "NET_GATE_PYTHON": sys.executable, "NET_GATE_ROOT": str(tmp_path / "unused")}
    def run(*args):
        return subprocess.run(["bash", "scripts/run_net_gain_gate.sh", *args], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=30)
    for mode in ("plan", "status"):
        r = run(mode)
        assert r.returncode == 0, r.stderr
    assert not (tmp_path / "unused").exists()
    r = run("fit", "--study", str(data), "--out", str(model))
    assert r.returncode == 0, r.stderr
    r = run("analyze", "--study", str(data), "--model", str(model), "--out", str(report))
    assert r.returncode == 0, r.stderr
    assert core.read(report)["horizon"]["optimal_switch_step"] is None
    assert run("fit", "--study", str(data), "--out", str(model)).returncode != 0
    assert core.read(data) == fixture()
