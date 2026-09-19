"""MBPP orchestration and input contracts; never launches a GPU process."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mbpp_preflight", ROOT / "scripts/check_mbpp_experiments.py")
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def inputs(tmp_path):
    pool = tmp_path / "mbpp.jsonl"
    rows = [{"text": f"Return integer {i}.", "test_list": [f"assert f() == {i}"]} for i in range(20)]
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows))
    items = preflight.mbpp_items(rows)
    manifest = tmp_path / "dataset_manifest.json"
    write_json(manifest, {"source_repository": "google-research-datasets/mbpp",
                          "source_revision": "pinned-revision", "sha256": preflight.ed.digest(pool)})
    matrix = tmp_path / "matrix"
    for seed in range(5):
        run = matrix / f"family-mbpp-s{seed}" / f"model-s{seed}-mbpp-d0"
        write_json(run / "run_config.json", {"dataset": "mbpp", "seed": seed, "drift": 0})
        write_json(run / "prompts.json", {"train": [items[2 * seed]], "val": [items[2 * seed + 1]]})
        (run / "DONE").write_text("complete")
    reference = matrix / "family-mbpp-s0/model-s0-mbpp-d100/policy_step_100/grpo_stats.jsonl"
    write_json(reference, {"step_seconds": 72.0})
    return SimpleNamespace(pool=pool, manifest=manifest, matrix=matrix,
                           fresh_root=tmp_path / "fresh", quality_root=tmp_path / "quality",
                           difficulty_root=tmp_path / "difficulty", long_root=tmp_path / "long", suite="all")


def test_preflight_is_read_only_and_excludes_every_seed(inputs, capsys):
    frozen_fresh(inputs)
    for seed in range(5):
        for step in (25, 50, 100):
            write_json(inputs.fresh_root / f"prefixes/seed-{seed}/prefix-{step}.json", {})
    before = {str(p): p.read_bytes() for p in inputs.pool.parent.rglob("*") if p.is_file()}
    preflight.check(inputs)
    assert "10 disjoint questions" in capsys.readouterr().out
    assert before == {str(p): p.read_bytes() for p in inputs.pool.parent.rglob("*") if p.is_file()}


def test_preflight_rejects_too_few_independent_questions(inputs):
    lines = inputs.pool.read_text().splitlines()[:12]
    inputs.pool.write_text("\n".join(lines) + "\n")
    p = preflight.ed.read(inputs.manifest)
    p["sha256"] = preflight.ed.digest(inputs.pool)
    write_json(inputs.manifest, p)
    with pytest.raises(ValueError, match="only 2 MBPP questions remain"):
        preflight.check(inputs)


@pytest.mark.parametrize("problem", ["hash", "missing-source", "wrong-dataset", "leak", "answer", "timing"])
def test_preflight_fails_closed(inputs, problem):
    run = inputs.matrix / "family-mbpp-s4/model-s4-mbpp-d0"
    if problem == "hash":
        inputs.pool.write_text(inputs.pool.read_text() + "\n")
    elif problem == "missing-source":
        (run / "DONE").unlink()
    elif problem == "wrong-dataset":
        write_json(run / "run_config.json", {"dataset": "math500", "seed": 4, "drift": 0})
    elif problem in ("leak", "answer"):
        prompts = preflight.ed.read(run / "prompts.json")
        if problem == "leak":
            prompts["val"] = prompts["train"]
        else:
            prompts["train"][0]["answer"] = "2"
        write_json(run / "prompts.json", prompts)
    else:
        write_json(inputs.matrix / "family-mbpp-s0/model-s0-mbpp-d100/policy_step_100/grpo_stats.jsonl",
                   {"step_seconds": -1})
    with pytest.raises(ValueError):
        preflight.check(inputs)


def frozen_fresh(inputs):
    items = preflight.mbpp_items([json.loads(line) for line in inputs.pool.read_text().splitlines()])
    p = {"dataset": "mbpp", "selector": "fresh_r", "accounting": "budget", "gate": "final",
         "evaluation": {"test": items[10:], "provenance": {
             "dataset": "google-research-datasets/mbpp", "revision": "pinned-revision", "split": "full"}}}
    write_json(inputs.fresh_root / "switch.json", p)
    return p


@pytest.mark.parametrize("suite,selector,accounting,gate", [
    ("difficulty", "difficulty", "budget", "convergence"),
    ("long", "fresh_r", "budget", "final"),
    ("quality", "fresh_r", "matched", "convergence"),
    ("all", "fresh_r", "matched", "convergence"),
])
def test_variant_requires_all_shared_prefixes(inputs, suite, selector, accounting, gate):
    inputs.suite = suite
    with pytest.raises(ValueError, match="shared MBPP prefixes/evaluation are not prepared"):
        preflight.check(inputs)
    p = frozen_fresh(inputs)
    with pytest.raises(ValueError, match="shared on-policy prefix not ready"):
        preflight.check(inputs)
    for seed in range(5):
        for step in (25, 50, 100):
            write_json(inputs.fresh_root / f"prefixes/seed-{seed}/prefix-{step}.json", {})
    preflight.check(inputs)
    p.update(selector=selector, accounting=accounting, gate=gate,
             prefix_source={"root": str(inputs.fresh_root)})
    if suite == "long":
        p["budget_gpu_seconds"] = 87120
    root = getattr(inputs, f"{'quality' if suite == 'all' else suite}_root")
    write_json(root / "switch.json", p)
    preflight.check(inputs)
    p["prefix_source"]["root"] = "/some/math/root"
    write_json(root / "switch.json", p)
    with pytest.raises(ValueError, match="reuse the MBPP on-policy"):
        preflight.check(inputs)


@pytest.mark.parametrize("problem", ["math-root", "wrong-evaluation", "small-test"])
def test_frozen_root_cannot_be_relabelled_mbpp(inputs, problem):
    p = frozen_fresh(inputs)
    if problem == "math-root":
        p["dataset"] = "math500"
    elif problem == "wrong-evaluation":
        p["evaluation"]["test"][0]["answer"] = "3"
    else:
        p["evaluation"]["test"] = p["evaluation"]["test"][:3]
    write_json(inputs.fresh_root / "switch.json", p)
    with pytest.raises(ValueError):
        preflight.check(inputs)


@pytest.fixture
def launcher(tmp_path):
    repo = tmp_path / "repo with spaces"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("run_mbpp_experiments.sh", "_mbpp_experiments.sh", "mbpp_failure_summary.py", "selection_switch_errors.py", "_status_summary.py"):
        shutil.copy(ROOT / "scripts" / name, scripts)
    (scripts / "setup_env.sh").write_text('echo "preflight must not source setup_env" >&2\nexit 99\n')
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    (scripts / "check_mbpp_experiments.py").write_text(
        'import json, os, sys\nassert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
        'open(os.environ["CHECK_LOG"], "a").write("checked\\n")\n'
        'open(os.environ["CHECK_ARGS"], "a").write(json.dumps(sys.argv[1:]) + "\\n")\n')
    (scripts / "check_mbpp_storage.sh").write_text(
        '#!/usr/bin/env bash\n"$TEST_PYTHON" - "$@" <<\'PY\'\n'
        'import json, os, sys\n'
        'assert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
        'with open(os.environ["AUDIT_LOG"], "a") as f:\n'
        '    f.write(json.dumps(sys.argv[1:]) + "\\n")\n'
        'sys.exit(int(os.environ.get("AUDIT_EXIT", "0")))\nPY\n')
    (scripts / "run_selection_switch.sh").write_text(
        '#!/usr/bin/env bash\n"$TEST_PYTHON" - "$@" <<\'PY\'\n'
        'import json, os, sys\n'
        'with open(os.environ["CALLS"], "a") as f:\n'
        '    f.write(json.dumps({"args": sys.argv[1:], "env": dict(os.environ)}) + "\\n")\n'
        'sys.exit(int(os.environ.get("FAKE_EXIT", "0")))\nPY\n')
    shutil.copy(scripts / "run_selection_switch.sh", scripts / "run_experiments.sh")
    (scripts / "mbpp_status.py").write_text(
        'import argparse, json, os, sys\n'
        'p = argparse.ArgumentParser()\n'
        'p.add_argument("--root", action="append", required=True)\n'
        'p.add_argument("--retained-root", action="append", default=[])\n'
        'p.add_argument("--all", action="store_true")\n'
        'a = p.parse_args()\n'
        'assert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
        'print("MBPP EXPERIMENTS")\n'
        'with open(os.environ["CALLS"], "a") as f:\n'
        '    for root in a.root:\n'
        '        f.write(json.dumps({"args": ["status"] + (["--all"] if a.all else []), '
        '"root": root, "dashboard_pid": os.getpid(), "dashboard_argv": sys.argv[1:], '
        '"env": dict(os.environ)}) + "\\n")\n'
        '        if os.environ.get("FAKE_EXIT", "0") != "0": print("ERROR " + root)\n'
        'sys.exit(int(os.environ.get("FAKE_EXIT", "0") != "0"))\n')
    env = {**os.environ, "OM_WORK": str(tmp_path / "work"), "VENV_DIR": str(venv),
           "TEST_PYTHON": sys.executable, "CALLS": str(tmp_path / "calls.jsonl"),
           "AUDIT_LOG": str(tmp_path / "storage-audits.jsonl"),
           "CHECK_LOG": str(tmp_path / "checks"), "CHECK_ARGS": str(tmp_path / "check-args.jsonl"),
           "SWITCH_ROOT": "/wrong/math",
           "SWITCH_PREFIX_SOURCE": "/wrong/math", "SWITCH_DATASET": "math500",
           "SWITCH_ONLY_SEEDS": "3,4", "SWITCH_ONLY_ARMS": "random_full",
           "OM_NODE_LOCK_HELD": "1", "SWITCH_BUDGET_GPU_SECONDS": "29040"}
    def run(*args, **overrides):
        return subprocess.run(["bash", str(scripts / "run_mbpp_experiments.sh"), *args],
                              cwd=tmp_path, env={**env, **overrides}, text=True, capture_output=True, timeout=10, check=False)
    return run, env


@pytest.mark.parametrize("mode", ["run", "restart", "stop", "progress"])
def test_lifecycle_is_delegated_without_input_preflight_after_required_storage_audit(launcher, mode):
    run, env = launcher
    result = run(mode, EXPERIMENTS_KEEPALIVE="1", EXPERIMENTS_WATCHDOG="1", EXPERIMENTS_AUTO_PULL="1")
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]
    assert len(calls) == 1 and calls[0]["args"] == [mode]
    e = calls[0]["env"]
    assert e["EXPERIMENTS_MBPP_SUITE"] == "all"
    assert e["SWITCH_ROOT"] == e["SWITCH_MBPP_QUALITY_ROOT"]
    assert e["EXPERIMENTS_HELP_SIBLINGS"] == "1" and e["EXPERIMENTS_SKIP_MOPPS"] == "1"
    assert all(e[key] == "1" for key in ("EXPERIMENTS_KEEPALIVE", "EXPERIMENTS_WATCHDOG", "EXPERIMENTS_AUTO_PULL"))
    assert all(key not in e for key in ("SWITCH_PREFIX_SOURCE", "SWITCH_ONLY_SEEDS", "SWITCH_ONLY_ARMS", "OM_NODE_LOCK_HELD", "SWITCH_BUDGET_GPU_SECONDS"))
    assert not Path(env["CHECK_LOG"]).exists()
    audit = Path(env["AUDIT_LOG"])
    if mode in ("run", "restart"):
        assert json.loads(audit.read_text()) == ["all"]
    else:
        assert not audit.exists()


@pytest.mark.parametrize("mode", ["run", "restart"])
@pytest.mark.parametrize("code", ["1", "2", "127"])
def test_storage_audit_failure_blocks_start_and_restart_before_controller_or_input_preflight(launcher, mode, code):
    run, env = launcher
    result = run(mode, "fresh", AUDIT_EXIT=code)
    assert result.returncode == 2, result.stderr
    assert 'no controller was started or stopped' in result.stderr
    assert json.loads(Path(env["AUDIT_LOG"]).read_text()) == ["fresh"]
    assert not Path(env["CALLS"]).exists()
    assert not Path(env["CHECK_LOG"]).exists()


@pytest.mark.parametrize("mode", ["stop", "plan", "check", "status", "progress", "results", "saved", "why"])
def test_non_start_modes_bypass_storage_audit(launcher, mode):
    run, env = launcher
    result = run(mode, AUDIT_EXIT="2")
    assert result.returncode == 0, result.stderr
    assert not Path(env["AUDIT_LOG"]).exists()


@pytest.mark.parametrize("mode", ["status", "results"])
def test_readers_do_not_check_or_start_training(launcher, mode):
    run, env = launcher
    result = run(mode)
    assert result.returncode == 0, result.stderr
    assert not Path(env["CHECK_LOG"]).exists()
    assert all(json.loads(line)["args"] == [mode] for line in Path(env["CALLS"]).read_text().splitlines())


@pytest.mark.parametrize("suite,expected", [
    ("all", ["selection-switch-mbpp-quality-v1"]),
    ("fresh", ["selection-switch-mbpp-v1"]),
    ("quality", ["selection-switch-mbpp-quality-v1"]),
    ("difficulty", ["selection-switch-mbpp-difficulty-v1"]),
    ("long", ["selection-switch-mbpp-long-v1"]),
])
def test_status_routes_exact_mbpp_suite_roots_despite_inherited_math_environment(launcher, suite, expected):
    run, env = launcher
    result = run("status", suite, OUT_ROOT="/wrong/old-math", SWITCH_ROOT="/wrong/math",
                 EXPERIMENTS_COMBINED="1", SWITCH_RUNTIME_REPO="/wrong/old-code")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]
    assert [call["root"] for call in calls] == [str(Path(env["OM_WORK"]) / "runs" / name) for name in expected]
    assert len({call["dashboard_pid"] for call in calls}) == 1
    assert result.stdout.count("MBPP EXPERIMENTS") == 1
    assert all(call["args"] == ["status"] for call in calls)
    for call in calls:
        passed = call["env"]
        assert passed["EXPERIMENTS_COMBINED"] == "0"
        assert passed["EXPERIMENTS_SKIP_MOPPS"] == "1"
        assert all(key not in passed for key in ("OUT_ROOT", "SWITCH_ROOT", "SWITCH_ONLY_ARMS", "SWITCH_ONLY_SEEDS", "SWITCH_RUNTIME_REPO"))
    assert not Path(env["AUDIT_LOG"]).exists()
    assert not Path(env["CHECK_LOG"]).exists()


def test_status_keeps_existing_custom_suite_roots_without_renaming_on_policy_storage(launcher):
    run, env = launcher
    paths = {"SWITCH_MBPP_QUALITY_ROOT": "/existing/saved-quality"}
    result = run("status", **paths)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]
    assert [call["root"] for call in calls] == list(paths.values())
    assert len({call["dashboard_pid"] for call in calls}) == 1
    assert result.stdout.count("MBPP EXPERIMENTS") == 1
    assert "[mbpp:fresh]" not in result.stdout
    assert "selector=fresh_r" not in result.stdout


def test_status_failure_for_a_root_does_not_hide_other_suite_roots(launcher):
    run, env = launcher
    result = run("status", FAKE_EXIT="2")
    assert result.returncode == 1
    calls = [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]
    assert len(calls) == 1
    assert len({call["root"] for call in calls}) == 1
    assert result.stdout.count("ERROR ") == 1
    assert not Path(env["AUDIT_LOG"]).exists()


def test_why_writes_one_small_report_without_training_or_full_exports(launcher):
    run, env = launcher
    result = run('why')
    assert result.returncode == 0, result.stderr
    assert not Path(env['CHECK_LOG']).exists()
    assert not Path(env['CALLS']).exists()
    reports = list((Path(env['OM_WORK']) / 'reports/selection-switch').glob('*.txt'))
    assert len(reports) == 1 and reports[0].stat().st_size <= 16 * 1024
    assert result.stdout.count('[saved]') == 1
    assert 'selection-switch-mbpp-quality-v1' in reports[0].read_text()
    assert 'selection-switch-mbpp-long-v1' not in reports[0].read_text()


def test_saved_command_is_read_only_small_and_does_not_start_controller(launcher):
    run, env = launcher
    result = run('saved')
    assert result.returncode == 0, result.stderr
    assert 'MBPP SAVED-WORK AUDIT' in result.stdout
    assert len(result.stdout.encode()) <= 4096
    assert not Path(env['CALLS']).exists() and not Path(env['CHECK_LOG']).exists()
    assert not (Path(env['OM_WORK']) / 'reports').exists()


def test_default_hold_is_short_and_polls_without_extra_environment_variables(launcher):
    run, env = launcher
    result = run('run', MBPP_HOLD_SECONDS='', EXPERIMENTS_HOLD_SECONDS='', EXPERIMENTS_HOLD_POLL_SECONDS='')
    assert result.returncode == 0, result.stderr
    passed = json.loads(Path(env['CALLS']).read_text())['env']
    assert passed['EXPERIMENTS_HOLD_SECONDS'] == '15'
    assert passed['EXPERIMENTS_HOLD_POLL_SECONDS'] == '5'


@pytest.mark.parametrize("code", ["1", "75", "78", "130", "143"])
def test_wrapper_preserves_the_node_controllers_exit_status(launcher, code):
    run, env = launcher
    result = run(FAKE_EXIT=code)
    assert result.returncode == int(code)
    assert len(Path(env["CALLS"]).read_text().splitlines()) == 1


def test_plan_and_check_do_not_launch(launcher):
    run, env = launcher
    assert run("plan").returncode == 0
    assert not Path(env["CHECK_LOG"]).exists()
    assert run("check").returncode == 0
    assert Path(env["CHECK_LOG"]).read_text() == "checked\n"
    assert not Path(env["CALLS"]).exists()


def test_long_check_receives_its_root_and_explicit_suite(launcher):
    run, env = launcher
    result = run("check", "long")
    assert result.returncode == 0, result.stdout + result.stderr
    arguments = json.loads(Path(env["CHECK_ARGS"]).read_text().splitlines()[-1])
    assert arguments[arguments.index("--long-root") + 1] == str(Path(env["OM_WORK"]) / "runs/selection-switch-mbpp-long-v1")
    assert arguments[arguments.index("--suite") + 1] == "long"
    assert not Path(env["CALLS"]).exists()


def test_default_preflight_does_not_reinterpret_or_modify_legacy_variants(inputs):
    frozen_fresh(inputs)
    for seed in range(5):
        for step in (25, 50, 100):
            write_json(inputs.fresh_root / f"prefixes/seed-{seed}/prefix-{step}.json", {})
    for root in (inputs.difficulty_root, inputs.long_root):
        write_json(root / "switch.json", {"dataset": "historical-do-not-relabel"})
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in inputs.pool.parent.rglob("*") if path.is_file()}
    preflight.check(inputs)
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in inputs.pool.parent.rglob("*") if path.is_file()}


def test_default_preflight_checks_existing_quality_without_changing_its_budget(inputs):
    protocol = frozen_fresh(inputs)
    for seed in range(5):
        for step in (25, 50, 100):
            write_json(inputs.fresh_root / f"prefixes/seed-{seed}/prefix-{step}.json", {})
    protocol.update(accounting="matched", gate="convergence", budget_gpu_seconds=12345,
                    prefix_source={"root": str(inputs.fresh_root)})
    path = inputs.quality_root / "switch.json"
    write_json(path, protocol)
    before = path.read_bytes(), path.stat().st_mtime_ns
    preflight.check(inputs)
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)
    protocol["accounting"] = "budget"
    write_json(path, protocol)
    before = path.read_bytes(), path.stat().st_mtime_ns
    with pytest.raises(ValueError, match="different protocol"):
        preflight.check(inputs)
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)


@pytest.mark.parametrize("cap", [0, 29040, 87121, True, "87120"])
def test_long_preflight_rejects_wrong_frozen_cap_without_rewriting_it(inputs, cap):
    inputs.suite = "long"
    protocol = frozen_fresh(inputs)
    for seed in range(5):
        for step in (25, 50, 100):
            write_json(inputs.fresh_root / f"prefixes/seed-{seed}/prefix-{step}.json", {})
    protocol.update(prefix_source={"root": str(inputs.fresh_root)}, budget_gpu_seconds=cap)
    write_json(inputs.long_root / "switch.json", protocol)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in inputs.pool.parent.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="expected MATH long cap 87120"):
        preflight.check(inputs)
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in inputs.pool.parent.rglob("*") if path.is_file()}


@pytest.mark.parametrize("name", ["selection-switch-mbpp-v1", "selection-switch-mbpp-difficulty-v1", "selection-switch-mbpp-long-v1"])
def test_status_retains_existing_variants_outside_the_default_quality_root(launcher, name):
    run, env = launcher
    root = Path(env["OM_WORK"]) / "runs" / name
    marker = root / "policy/checkpoint-000150/optimizer.pt"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"preserved historical optimizer")
    before = marker.read_bytes(), marker.stat().st_mtime_ns
    result = run("status")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]
    assert len(calls) == 1 and calls[0]["root"].endswith("selection-switch-mbpp-quality-v1")
    arguments = calls[0]["dashboard_argv"]
    assert arguments[arguments.index("--retained-root") + 1] == str(root)
    assert before == (marker.read_bytes(), marker.stat().st_mtime_ns)


def test_default_results_reads_existing_variants_without_dispatching_training(launcher):
    run, env = launcher
    root_names = ("selection-switch-mbpp-quality-v1", "selection-switch-mbpp-v1",
                  "selection-switch-mbpp-difficulty-v1", "selection-switch-mbpp-long-v1")
    for name in root_names[1:]:
        write_json(Path(env["OM_WORK"]) / "runs" / name / "switch.json", {"keep": name})
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in Path(env["OM_WORK"]).rglob("*") if path.is_file()}
    result = run("results")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]
    assert [Path(call["env"]["SWITCH_ROOT"]).name for call in calls] == list(root_names)
    assert all(call["args"] == ["results"] for call in calls)
    assert not Path(env["CHECK_LOG"]).exists()
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in Path(env["OM_WORK"]).rglob("*") if path.is_file()}


@pytest.mark.parametrize("suite", ["fresh", "difficulty", "long", "quality"])
def test_mbpp_profile_budget_is_local_and_math_variant_settings_match(launcher, suite):
    _, env = launcher
    repo = Path(env["CHECK_LOG"]).parent / "repo with spaces"
    fresh = Path(env["OM_WORK"]) / "runs/selection-switch-mbpp-v1"
    write_json(fresh / "switch.json", {})
    for seed in range(5):
        for step in (25, 50, 100):
            write_json(fresh / f"prefixes/seed-{seed}/prefix-{step}.json", {})
    script = ('source scripts/_mbpp_experiments.sh\nmbpp_queue_init\n'
              'inner() { bash "$@"; }\nmbpp_queue_run "$1"\n')
    result = subprocess.run(["bash", "-c", script, "profile-probe", suite], cwd=repo,
                            env={**env, "EXPERIMENTS_MBPP_SUITE": suite, "SWITCH_BUDGET_GPU_SECONDS": "999999"},
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    passed = json.loads(Path(env["CALLS"]).read_text().splitlines()[-1])["env"]
    assert passed["SWITCH_DATASET"] == "mbpp"
    assert passed.get("SWITCH_BUDGET_GPU_SECONDS") == ("87120" if suite == "long" else None)
    expected = {"fresh": ("fresh_r", "budget", "final"),
                "difficulty": ("difficulty", "budget", "convergence"),
                "long": ("fresh_r", "budget", "final"),
                "quality": ("fresh_r", "matched", "convergence")}[suite]
    assert tuple(passed[key] for key in ("SWITCH_SELECTOR", "SWITCH_ACCOUNTING", "SWITCH_GATE")) == expected
    assert passed["SWITCH_PREFIX_SOURCE"] == ("" if suite == "fresh" else str(fresh))
    if suite not in ("difficulty", "long"):
        return
    wrapper = f"run_switch_{suite}.sh"
    shutil.copy(ROOT / "scripts" / wrapper, repo / "scripts" / wrapper)
    math_env = {key: value for key, value in env.items()
                if not key.startswith("SWITCH_") and key != "EXPERIMENTS_MBPP_SUITE"}
    result = subprocess.run(["bash", f"scripts/{wrapper}", "status"], cwd=repo, env=math_env,
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    math = json.loads(Path(env["CALLS"]).read_text().splitlines()[-1])["env"]
    assert tuple(math.get(key, default) for key, default in (
        ("SWITCH_SELECTOR", "fresh_r"), ("SWITCH_ACCOUNTING", "budget"), ("SWITCH_GATE", "final"))) == expected
    assert math.get("SWITCH_BUDGET_GPU_SECONDS") == passed.get("SWITCH_BUDGET_GPU_SECONDS")


def test_plan_distinguishes_selector_from_accounting_and_preserves_legacy_keys(launcher):
    from _status_summary import MBPP_SUITE_LABELS

    run, env = launcher
    result = run("plan")
    assert result.returncode == 0, result.stderr
    assert "1 condition(s), 48 continuation branches" in result.stdout
    assert "matched accounting, convergence gate" in result.stdout
    assert "recorded on reporting" in result.stdout
    assert "existing frozen training allocation is retained" in result.stdout
    expected = {"fresh": ("On-policy", "선택비용 포함", "최종 보상 기준"),
                "quality": ("On-policy", "선택비용 별도", "비용 보정 학습 효율 기준"),
                "difficulty": ("Difficulty", "선택비용 포함", "비용 보정 학습 효율 기준"),
                "long": ("On-policy", "선택비용 포함", "최종 보상 기준")}
    for key, (selector, accounting, gate) in expected.items():
        header = f"[mbpp:{MBPP_SUITE_LABELS[key]}] selector={selector} accounting={accounting} gate={gate}"
        assert (header in result.stdout) is (key == "quality")
        selected = run("plan", key)
        assert selected.returncode == 0 and header in selected.stdout
        assert "1 condition(s), 48 continuation branches" in selected.stdout
    assert "[mbpp:quality]" not in result.stdout and "selector=fresh_r" not in result.stdout
    assert not Path(env["CALLS"]).exists() and not Path(env["AUDIT_LOG"]).exists()


@pytest.mark.parametrize("args,overrides", [
    (("bogus",), {}), (("run", "bogus"), {}), (("run", "all", "extra"), {}),
    (("plan",), {"MBPP_HOLD_SECONDS": "0"}),
    (("plan",), {"SWITCH_MBPP_ROOT": "/tmp/same", "SWITCH_MBPP_QUALITY_ROOT": "/tmp/same"}),
    (("plan",), {"SWITCH_MBPP_ROOT": "/tmp/parent", "SWITCH_MBPP_QUALITY_ROOT": "/tmp/parent/child"}),
    (("plan",), {"SWITCH_MBPP_ROOT": "/tmp/same", "SWITCH_MBPP_LONG_ROOT": "/tmp/same"}),
    (("plan",), {"SWITCH_MBPP_ROOT": "/tmp/parent", "SWITCH_MBPP_LONG_ROOT": "/tmp/parent/child"}),
])
def test_bad_arguments_and_roots_fail_before_launch(launcher, args, overrides):
    run, env = launcher
    assert run(*args, **overrides).returncode == 2
    assert not Path(env["CALLS"]).exists()


def test_bash_syntax():
    for name in ("run_mbpp_experiments.sh", "_mbpp_experiments.sh", "run_experiments.sh"):
        subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], check=True)
