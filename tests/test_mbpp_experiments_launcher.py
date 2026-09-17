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
                           difficulty_root=tmp_path / "difficulty", suite="all")


def test_preflight_is_read_only_and_excludes_every_seed(inputs, capsys):
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


def test_variant_requires_all_shared_prefixes(inputs):
    inputs.suite = "difficulty"
    with pytest.raises(ValueError, match="fresh suite first"):
        preflight.check(inputs)
    p = frozen_fresh(inputs)
    with pytest.raises(ValueError, match="fresh prefix not ready"):
        preflight.check(inputs)
    for seed in range(5):
        for step in (25, 50, 100):
            write_json(inputs.fresh_root / f"prefixes/seed-{seed}/prefix-{step}.json", {})
    preflight.check(inputs)
    p.update(selector="difficulty", gate="convergence", prefix_source={"root": str(inputs.fresh_root)})
    write_json(inputs.difficulty_root / "switch.json", p)
    preflight.check(inputs)
    p["prefix_source"]["root"] = "/some/math/root"
    write_json(inputs.difficulty_root / "switch.json", p)
    with pytest.raises(ValueError, match="reuse the MBPP fresh"):
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
    shutil.copy(ROOT / "scripts/run_mbpp_experiments.sh", scripts)
    (scripts / "setup_env.sh").write_text('echo "preflight must not source setup_env" >&2\nexit 99\n')
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    (scripts / "check_mbpp_experiments.py").write_text(
        'import os\nassert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
        'open(os.environ["CHECK_LOG"], "a").write("checked\\n")\n')
    (scripts / "run_selection_switch.sh").write_text(
        '#!/usr/bin/env bash\n"$TEST_PYTHON" - "$@" <<\'PY\'\n'
        'import json, os, sys\n'
        'with open(os.environ["CALLS"], "a") as f:\n'
        '    f.write(json.dumps({"args": sys.argv[1:], "env": dict(os.environ)}) + "\\n")\n'
        'sys.exit(int(os.environ.get("FAKE_EXIT", "0")))\nPY\n')
    env = {**os.environ, "OM_WORK": str(tmp_path / "work"), "VENV_DIR": str(venv),
           "TEST_PYTHON": sys.executable, "CALLS": str(tmp_path / "calls.jsonl"),
           "CHECK_LOG": str(tmp_path / "checks"), "SWITCH_ROOT": "/wrong/math",
           "SWITCH_PREFIX_SOURCE": "/wrong/math", "SWITCH_DATASET": "math500",
           "SWITCH_ONLY_SEEDS": "3,4", "SWITCH_ONLY_ARMS": "random_full",
           "OM_NODE_LOCK_HELD": "1", "SWITCH_BUDGET_GPU_SECONDS": "29040"}
    def run(*args, **overrides):
        return subprocess.run(["bash", str(scripts / "run_mbpp_experiments.sh"), *args],
                              cwd=tmp_path, env={**env, **overrides}, text=True, capture_output=True, timeout=10)
    return run, env


def test_one_command_routes_all_suites_and_clears_math_settings(launcher):
    run, env = launcher
    result = run()
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]
    assert len(calls) == 3
    for call, selector, accounting, gate in zip(calls, ("fresh_r", "fresh_r", "difficulty"),
                                               ("budget", "matched", "budget"), ("final", "convergence", "convergence")):
        e = call["env"]
        assert call["args"] == ["run"]
        assert (e["SWITCH_DATASET"], e["SWITCH_SELECTOR"], e["SWITCH_ACCOUNTING"], e["SWITCH_GATE"]) == (
            "mbpp", selector, accounting, gate)
        assert all(key not in e for key in ("SWITCH_ONLY_SEEDS", "SWITCH_ONLY_ARMS", "OM_NODE_LOCK_HELD", "SWITCH_BUDGET_GPU_SECONDS"))
        assert e["SWITCH_HOLD_SECONDS"] == "600" and e["SWITCH_FOREGROUND"] == "1"
        assert e["EXPERIMENTS_COMBINED"] == "0" and e["EXPERIMENTS_SKIP_MOPPS"] == "1"
    fresh = calls[0]["env"]["SWITCH_ROOT"]
    assert calls[0]["env"]["SWITCH_PREFIX_SOURCE"] == ""
    assert all(c["env"]["SWITCH_PREFIX_SOURCE"] == fresh for c in calls[1:])
    assert len({c["env"]["SWITCH_ROOT"] for c in calls}) == 3


@pytest.mark.parametrize("mode", ["status", "results", "why"])
def test_readers_do_not_check_or_start_training(launcher, mode):
    run, env = launcher
    result = run(mode)
    assert result.returncode == 0, result.stderr
    assert not Path(env["CHECK_LOG"]).exists()
    assert all(json.loads(line)["args"] == [mode] for line in Path(env["CALLS"]).read_text().splitlines())


@pytest.mark.parametrize("code", ["1", "75", "78", "130", "143"])
def test_run_failure_never_starts_next_suite(launcher, code):
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


@pytest.mark.parametrize("args,overrides", [
    (("bogus",), {}), (("run", "bogus"), {}), (("run", "all", "extra"), {}),
    (("plan",), {"MBPP_HOLD_SECONDS": "0"}),
    (("plan",), {"SWITCH_MBPP_ROOT": "/tmp/same", "SWITCH_MBPP_QUALITY_ROOT": "/tmp/same"}),
    (("plan",), {"SWITCH_MBPP_ROOT": "/tmp/parent", "SWITCH_MBPP_QUALITY_ROOT": "/tmp/parent/child"}),
])
def test_bad_arguments_and_roots_fail_before_launch(launcher, args, overrides):
    run, env = launcher
    assert run(*args, **overrides).returncode == 2
    assert not Path(env["CALLS"]).exists()


def test_bash_syntax():
    subprocess.run(["bash", "-n", str(ROOT / "scripts/run_mbpp_experiments.sh")], check=True)
