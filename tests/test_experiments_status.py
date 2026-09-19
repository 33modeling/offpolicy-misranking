import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import selection_gate as core

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


combined = load("experiments_status")
sys.path.insert(0, str(ROOT / "tests"))
from test_mopps_comparison_status import fixture as mopps_fixture  # noqa: E402
from test_selection_switch_status import four_nodes  # noqa: E402


def test_progress_shows_budget_block_instead_of_hiding_it(tmp_path):
    progress = load("experiments_progress")
    data = {"tasks": [{"kind": "branch", "status": "BUDGET", "seed": 3, "step": 100,
                       "arm": "selection_full", "reason": "allocation exhausted; saved work preserved"}]}
    text = '\n'.join(progress.render_root(tmp_path, data, width=120, kind="switch"))
    assert 'BUDGET 1' in text and 'BUDGET s3/t100' in text and 'saved work preserved' in text


def test_progress_distinguishes_saved_training_from_new_ready_work(tmp_path):
    progress = load('experiments_progress')
    data = {'training_published': 1, 'tasks': [
        {'kind': 'branch', 'status': status, 'seed': 3, 'step': 25, 'arm': arm,
         'reason': 'saved work; do not restart from parent'}
        for status, arm in [('EVAL', 'random_full'), ('RESUME', 'random_reduced'),
                            ('REVIEW', 'selection_full'), ('SAVING', 'selection_reduced')]
    ]}
    text = '\n'.join(progress.render_root(tmp_path, data, width=120, kind='switch'))
    for status in ('EVAL', 'RESUME', 'REVIEW', 'SAVING'):
        assert f'{status} 1' in text
    assert 'TRAINED 1' in text
    assert 'REVIEW s3/t25' in text
    assert 'READY 4' not in text


def test_random_progress_counts_are_separate_from_selectors_and_archive_history(tmp_path):
    progress = load('experiments_progress')
    tasks = [{'kind': 'branch', 'arm': arm, 'status': 'DONE', 'seed': 3, 'step': 25,
              'archived_work': 'discarded/old/policy'}
             for arm in ('random_full', 'random_reduced')]
    tasks.append({'kind': 'branch', 'arm': 'selection_full', 'status': 'READY', 'seed': 3, 'step': 25})
    text = '\n'.join(progress.render_root(tmp_path / 'selection-switch-v1', {'tasks': tasks},
                                          width=80, kind='switch'))
    assert text.startswith('on-policy:')
    assert 'RF DONE 1/1' in text and 'RR DONE 1/1' in text
    assert 'HISTORY 2' in text
    assert all(len(line) <= 80 for line in text.splitlines())


def test_one_screen_shows_both_experiments_and_this_node_once(tmp_path):
    switch_root, mopps_root = tmp_path / "switch", tmp_path / "mopps"
    now = time.time()
    four_nodes(switch_root, now)
    mopps_fixture(mopps_root, switch_root, now)
    data = combined.snapshot(switch_root, mopps_root, now=now)
    # The one screen is exactly the two views, nothing recomputed differently.
    assert data["selection_switch"] == combined.switch_status.snapshot(switch_root, now=now)
    assert data["mopps_comparison"] == combined.mopps_status.snapshot(mopps_root, now=now)
    assert data["selection_switch"]["active_nodes"] >= 1 and data["mopps_comparison"]["prepared"] is True
    output = combined.render(data, width=120)
    first, second = output.splitlines()[:2]
    assert first.startswith("EXPERIMENTS  ") and second.startswith("NODES  ") and " live  |  " in second
    assert output.index("NODES (state then") < output.index("SELECTION SWITCH") < output.index("MOPPS COMPARISON")
    assert output.count("NODES (state then") == 1
    assert output.count("IDLE  ") == 1 and output.index("IDLE  ") < output.index("NODES (state then")
    assert "CONTINUATIONS" in output and "PARENT PREFIXES" in output
    assert output.count("THIS NODE GPUS") == 1
    assert output.index("THIS NODE GPUS") > output.index("MOPPS COMPARISON")
    assert all(len(line) <= 120 for line in output.splitlines())


def test_nodes_training_a_sibling_root_are_counted_and_labelled(tmp_path):
    """A node running long or difficulty is silent on its console for hours; the
    combined view must still show it as RUN from that root's own heartbeat."""
    from test_selection_switch_status import point, prepared, running
    switch_root, mopps_root = tmp_path / "selection-switch-v1", tmp_path / "mopps-comparison-v1"
    now = time.time()
    four_nodes(switch_root, now)
    mopps_fixture(mopps_root, switch_root, now)
    long_root = tmp_path / "selection-switch-long-v1"
    prepared(long_root)
    running(point(long_root) / "random_reduced", "node-long", now=now, phase="train")
    data = combined.snapshot(switch_root, mopps_root, now=now)
    hosts = {node["host"]: node for node in data["nodes"]}
    assert hosts["node-long"]["state"] == "RUN" and hosts["node-long"]["phase"] == "train"
    assert hosts["node-long"]["task"].endswith("long: random_reduced")
    assert "node-long" in combined.render(data, width=120)
    assert data["selection_switch"] == combined.switch_status.snapshot(switch_root, now=now)


def test_unprepared_mopps_root_does_not_hide_the_switch(tmp_path):
    switch_root = tmp_path / "switch"
    four_nodes(switch_root, time.time())
    data = combined.snapshot(switch_root, tmp_path / "absent")
    output = combined.render(data)
    assert "Dev 1/18" in output and "NOT PREPARED" in output and output.count("THIS NODE GPUS") == 1


def test_both_launchers_status_show_one_screen_and_stay_read_only(tmp_path):
    switch_root, mopps_root = tmp_path / "switch", tmp_path / "mopps"
    now = time.time()
    four_nodes(switch_root, now)
    mopps_fixture(mopps_root, switch_root, now)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    env = {**os.environ, "SWITCH_ROOT": str(switch_root), "MOPPS_ROOT": str(mopps_root),
           "SWITCH_PYTHON": sys.executable, "OM_WORK": str(tmp_path / "absent-work")}
    for launcher in ("run_selection_switch.sh", "run_mopps_comparison.sh", "run_experiments.sh"):
        result = subprocess.run(["bash", f"scripts/{launcher}", "status"], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (launcher, result.stderr)
        assert "SELECTION SWITCH" in result.stdout and "MOPPS COMPARISON" in result.stdout, launcher
        assert result.stdout.count("THIS NODE GPUS") == 1, launcher
    result = subprocess.run(["bash", "scripts/run_mopps_comparison.sh", "status", "--json"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert set(payload) == {"updated", "nodes", "node_summary", "selection_switch", "mopps_comparison", "other_experiments"}
    single = subprocess.run(["bash", "scripts/run_selection_switch.sh", "status"], cwd=ROOT,
                            env={**env, "EXPERIMENTS_COMBINED": "0"}, capture_output=True, text=True, timeout=30)
    assert single.returncode == 0 and "MOPPS COMPARISON" not in single.stdout
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not list(tmp_path.rglob("*.lock")) and not list(tmp_path.rglob("*-runtime.json"))


def test_why_from_any_launcher_writes_one_report_for_both_experiments(tmp_path):
    # Real layout: both roots under WORK/runs, next to the node launcher's runs/experiments/logs.
    switch_root, mopps_root = tmp_path / "work/runs/selection-switch-v1", tmp_path / "work/runs/mopps-comparison-v1"
    now = time.time()
    four_nodes(switch_root, now)
    mopps_fixture(mopps_root, switch_root, now)
    (tmp_path / "work/runs/experiments/logs").mkdir(parents=True)
    (tmp_path / "work/runs/experiments/logs/console.node-9_.log").write_text(
        "[node-launcher-start] host=node-9\n[holding] node retained (switch rc=75 node busy: lock held or GPUs occupied | mopps rc=0 nothing left to claim); next pass in 60s\n")
    # A node that never held (and an empty keepalive log) must not mark the report incomplete.
    (tmp_path / "work/runs/experiments/logs/console.node-8_.log").write_text("[node-launcher-start] host=node-8\n[pass 1] selection switch\n")
    (tmp_path / "work/runs/experiments/logs/keepalive.node-8_.log").write_text("")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    env = {**os.environ, "SWITCH_ROOT": str(switch_root), "MOPPS_ROOT": str(mopps_root),
           "SWITCH_PYTHON": sys.executable, "OM_WORK": str(tmp_path / "work")}
    for launcher in ("run_selection_switch.sh", "run_mopps_comparison.sh", "run_experiments.sh"):
        result = subprocess.run(["bash", f"scripts/{launcher}", "why"], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, (launcher, result.stdout + result.stderr)
        path = Path(result.stdout.strip().splitlines()[-1].removeprefix("[saved] "))
        assert path.parent == tmp_path / "work/reports/experiments", launcher
        report = path.read_text()
        assert report.startswith("EXPERIMENTS WHY")
        assert "NODES  " in report and "HOLD" in report
        assert "######## run_selection_switch.sh why ########" in report
        assert "######## run_mopps_comparison.sh why ########" in report
        assert "SELECTION SWITCH EXPERIMENT" in report and "MOPPS COMPARISON" in report
        assert "===== states/s3-t50/random_online/failure.json =====" in report
        assert "NODE LAUNCHER LOG" in report and "lock held or GPUs occupied" in report
        assert "incomplete" not in report
        assert report.count("[pass 1] selection switch") >= 2
    assert {path: path.read_bytes() for path in before} == before


def test_progress_screen_lists_every_root_with_running_updates_and_failures(tmp_path):
    from test_selection_switch_status import completed_prefix, point, prefix, prepared, running
    import importlib.util
    spec = importlib.util.spec_from_file_location("experiments_progress", ROOT / "scripts/experiments_progress.py")
    progress = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(progress)
    work = tmp_path / "work"
    now = time.time()
    switch_root, mopps_root = work / "runs/selection-switch-v1", work / "runs/mopps-comparison-v1"
    four_nodes(switch_root, now)
    mopps_fixture(mopps_root, switch_root, now)
    difficulty = work / "runs/selection-switch-difficulty-v1"
    prepared(difficulty)
    completed_prefix(difficulty, 1, 50)
    running(point(difficulty, 1, 50) / "random_reduced", "run1-wss-3-gab12", now=now, phase="train")
    (point(difficulty, 1, 50) / "random_reduced/policy").mkdir(parents=True)
    (point(difficulty, 1, 50) / "random_reduced/policy/grpo_stats.jsonl").write_text('{"step": 87}\n')
    core.atomic_json(point(difficulty, 1, 50) / "selection_reduced/failure.json", {"error": "train worker failed: [1]", "time": now})
    text = progress.render(work, width=80, now=now)
    lines = text.splitlines()
    assert lines[0].startswith("PROGRESS  ") and all(len(line) <= 80 for line in lines)
    assert any(line.startswith("NODES TRAINING NOW  ") for line in lines)
    assert text.index("\non-policy:") < text.index("\ndifficulty:") < text.index("\nMoPPS:")
    assert "  RUN  s1/t50 random_reduced    train       37u" in text and "run1-wss-3-gab12" in text
    assert "  FAIL s1/t50 selection_reduced train worker failed" in text
    assert "difficulty: DONE 0/" in text and "gate WAIT (dev 0/18)" in text
    # The node list at the bottom names every node: state, identity, phase, and what it does.
    assert "\nNODES  " in text and text.index("\nNODES  ") > text.index("\nMoPPS:")
    node_lines = text[text.index("\nNODES  "):].splitlines()[1:]
    assert any(l.startswith("  RUN    run1-wss-3-gab12") and "s1/t50 difficulty: random_r" in l for l in node_lines), node_lines
    assert any(l.startswith("  RUN    node-1") for l in node_lines)


def mbpp_results(work, count=21):
    from test_selection_switch_status import completed_prefix, point, prepared, published
    import selection_switch as rule
    root = work / "runs/selection-switch-mbpp-v1"
    prepared(root)
    remaining = count
    for seed in (*rule.DEV_SEEDS, *rule.TEST_SEEDS):
        for step in rule.STEPS:
            completed_prefix(root, seed, step)
            for arm in rule.DEV_ARMS if seed in rule.DEV_SEEDS else rule.TEST_ARMS:
                if remaining:
                    published(point(root, seed, step) / arm)
                    remaining -= 1
    return root


def test_generic_status_keeps_twenty_one_saved_mbpp_results_visible_under_default_math_view(tmp_path):
    from test_selection_switch_status import prepared
    work = tmp_path / "work"
    fresh = mbpp_results(work)
    primary = work / "runs/selection-switch-v1"
    quality = work / "runs/selection-switch-mbpp-quality-v1"
    prepared(primary)
    prepared(quality)
    before = {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}
    env = {**os.environ, "OM_WORK": str(work), "SWITCH_PYTHON": sys.executable}
    for key in ("SWITCH_ROOT", "MOPPS_ROOT", "EXPERIMENTS_MBPP_SUITE", "OUT_ROOT"):
        env.pop(key, None)
    result = subprocess.run(["bash", "scripts/run_experiments.sh", "status", "--json"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["selection_switch"]["root"] == str(primary)
    by_root = {item["root"]: item for item in data["other_experiments"]}
    assert by_root[str(fresh)]["branch_counts"]["DONE"] == 21
    assert by_root[str(fresh)]["training_published"] == 21
    assert by_root[str(quality)]["training_published"] == 0
    text = combined.render(data, width=120)
    assert "selection-switch-mbpp-v1: DONE 21/48  TRAINED 21" in text
    assert "selection-switch-mbpp-quality-v1: DONE 0/48  TRAINED 0" in text
    assert before == {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}


def test_mbpp_status_routes_fresh_saved_results_not_generic_math_root(tmp_path):
    work = tmp_path / "work"
    fresh = mbpp_results(work)
    env = {**os.environ, "OM_WORK": str(work), "SWITCH_PYTHON": sys.executable,
           "SWITCH_ROOT": str(work / "runs/wrong-math-root")}
    for key in ("SWITCH_MBPP_ROOT", "SWITCH_MBPP_QUALITY_ROOT", "SWITCH_MBPP_DIFFICULTY_ROOT"):
        env.pop(key, None)
    result = subprocess.run(["bash", "scripts/run_mbpp_experiments.sh", "status", "fresh"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "MBPP EXPERIMENTS" in result.stdout and "21/48" in result.stdout
    assert "On-policy · 선택비용 포함" in result.stdout and "CURRENT RUN 0" in result.stdout
    assert fresh.exists()
    assert "wrong-math-root" not in result.stdout and "MOPPS COMPARISON" not in result.stdout
