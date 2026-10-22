"""IT status bootstraps absent plans and preserves already recorded work."""

import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from srgc_rebuttal.plan import input_path, load_plan
from srgc_rebuttal.runtime import atomic_json, lease
from srgc_research.dispatch import model_status
from srgc_research.dispatch import gemma_it_bootstrap as bootstrap
from srgc_research.dispatch.gemma4_it import preflight
from srgc_research.tests.test_model_results import make_cohort, saved


def run_status(cohort, capsys, *options):
    with patch.dict(os.environ, cohort.env, clear=True), \
            patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}):
        code = model_status.main(["gemma4_it", "--root", str(cohort.root), *options])
    return code, capsys.readouterr()


def test_status_skips_rollout_tree_and_announces_root_before_loading(tmp_path, capsys):
    cohort = make_cohort("gemma4_it", tmp_path)
    # A rollout archive is unrelated to status metadata and can be huge on NFS.
    archive = cohort.root / "rollout-archive"
    archive.symlink_to(tmp_path / "other-storage", target_is_directory=True)
    before = saved(cohort)
    original = model_status.load_reports

    def loading(*args):
        assert "Reading Gemma-4-12B-IT" in capsys.readouterr().err
        return original(*args)

    with patch.object(model_status, "load_reports", side_effect=loading), \
            patch("os.walk", side_effect=AssertionError("status walked the experiment tree")):
        code, output = run_status(cohort, capsys)
    assert code == 0
    assert "MATH" in output.out and "MBPP" in output.out
    assert "PREFLIGHT NOT STARTED" in output.out
    assert saved(cohort) == before


def test_missing_it_plans_show_preparing_and_separate_historical_pt_path(tmp_path, capsys):
    cohort = make_cohort("gemma4_it", tmp_path)
    for path in cohort.plans.values():
        path.unlink()
    old_root = cohort.work / "srgc-rebuttal/gemma4-12b-pt-v1"
    old = old_root / "experiments/gemma4-12b-pt-math.json"
    atomic_json(old, {"model": "google/gemma-4-12B", "historical_rewards": [0, 1]})
    old_bytes = old.read_bytes()
    with lease(cohort.root / ".start-prepare.lock"):
        code, output = run_status(cohort, capsys)
    assert code == 0  # Initial preparation is a state, not a corrupt experiment.
    assert "MATH PREPARING" in output.out and "MBPP PREPARING" in output.out
    assert "IT experiment plan has not been created" in output.out
    assert "No such file" not in output.out and "READ ERRORS" not in output.out
    assert f"Historical PT records: {old_root}" in output.out
    assert "model_status gemma4 --root" in output.out
    assert "initial candidate success" not in output.out
    assert old.read_bytes() == old_bytes


@pytest.mark.parametrize("action", ["status", "results"])
def test_public_inspection_automatically_prepares_a_fully_empty_it_root(tmp_path, action):
    import subprocess
    repo = Path(__file__).resolve().parents[2]
    root = tmp_path / "work/srgc-rebuttal/gemma4-12b-it-v1"
    interpreter = tmp_path / "preparation python"
    interpreter.write_text(f"#!{sys.executable}\n" + '''
import os, sys
from pathlib import Path
from srgc_research.dispatch.gemma4_it import adapter
from srgc_research.tests.test_model_results import GemmaITTokenizer
assert sys.argv[1:5] == ['-m', 'srgc_research.dispatch.gemma_it_run', 'all', 'prepare'], sys.argv
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert os.environ['SRGC_MODEL_CHAIN'] == '0'
assert os.environ['SRGC_GEMMA_IT_PROTOCOL'] == '1'
assert os.environ['SRGC_GEMMA_LEGACY_PROTOCOL'] == '0'
root = Path(sys.argv[sys.argv.index('--root') + 1])
repo = Path(adapter.REPO)
for name, source in [('math','additional_seeds.json'), ('mbpp','mbpp_seeds.json')]:
    adapter.prepare(name, repo / 'srgc_rebuttal/experiments' / source, root,
                    GemmaITTokenizer(), snapshot_sha256='a' * 64)
print('real cohort preparation completed; no generation or training')
''')
    interpreter.chmod(0o755)
    environment = {**os.environ, "GROUP_VOLUME": str(tmp_path), "OM_WORK": str(tmp_path / "work"),
                   "SRGC_GEMMA_IT_ROOT": str(root), "GEMMA_PYTHON": str(interpreter),
                   "PYTHONPATH": os.pathsep.join((str(repo), os.environ.get("PYTHONPATH", ""))),
                   "SRGC_GEMMA_LEGACY_PROTOCOL": "0", "SRGC_GEMMA_IT_PROTOCOL": "1"}
    assert not root.exists()
    command = ["sh", str(repo / "scripts/run_srgc_gemma4.sh"), action, "--json"]
    result = subprocess.run(command, cwd="/tmp", env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "automatically preparing missing IT plans" in result.stderr
    assert "real cohort preparation completed" in result.stderr
    if action == "status":
        assert all(row["prepared"] for row in json.loads(result.stdout)["datasets"])
    else:
        exported = json.loads((tmp_path / "work/results/gemma4_it-results.json").read_text())
        assert all(row["prepared"] for row in exported["datasets"].values())
    for name in ("math", "mbpp"):
        path = root / "experiments" / f"gemma4-12b-it-{name}.json"
        plan = load_plan(path)
        for seed in plan["seeds"]:
            assert json.loads(input_path(path, plan, seed).read_text())["cached_rewards"] == {}
        assert not (root / "runs" / name).exists()
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    environment["GEMMA_PYTHON"] = "/missing"
    repeated = subprocess.run(command, cwd="/tmp", env=environment,
                              capture_output=True, text=True, timeout=30)
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert "automatically preparing" not in repeated.stderr
    assert before == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("explicit_it", [False, True])
def test_automatic_preparation_only_creates_the_missing_dataset(tmp_path, capsys, explicit_it):
    from srgc_research.tests.test_model_results import GemmaITTokenizer
    from srgc_research.dispatch.gemma4_it import adapter
    cohort = make_cohort("gemma4_it", tmp_path)
    cohort.plans["mbpp"].unlink()
    previous = tmp_path / "previous-pt-model"
    atomic_json(previous / "generation_config.json", {"eos_token_id": 1})
    previous_bytes = (previous / "generation_config.json").read_bytes()
    cohort.env.update(SRGC_GEMMA_MODEL_PATH=str(previous), SRGC_GEMMA_MODEL_SHA256="old-PT-parent")
    if explicit_it:
        cohort.env["SRGC_GEMMA_IT_MODEL_PATH"] = str(tmp_path / "explicit-it")
    before = saved(cohort)
    calls = []

    def prepare(command, log, environment, **options):
        assert command[2:4] == ["mbpp", "prepare"]
        assert "SRGC_GEMMA_MODEL_SHA256" not in environment
        if explicit_it:
            assert environment["SRGC_GEMMA_IT_MODEL_PATH"] == cohort.env["SRGC_GEMMA_IT_MODEL_PATH"]
        else:
            assert "SRGC_GEMMA_MODEL_PATH" not in environment
        calls.append(command)
        adapter.prepare("mbpp", adapter.REPO / "srgc_rebuttal/experiments/mbpp_seeds.json",
                        cohort.root, GemmaITTokenizer(), snapshot_sha256="a" * 64)
        print("preparing MBPP inputs")
        return 0

    with patch.object(bootstrap.cluster, "run_child", side_effect=prepare):
        code, output = run_status(cohort, capsys, "--json")
    assert code == 0 and len(calls) == 1
    assert all(row["prepared"] for row in json.loads(output.out)["datasets"])
    assert "preparing MBPP inputs" in output.err
    assert all(path.read_bytes() == data for path, data in before.items())
    assert (previous / "generation_config.json").read_bytes() == previous_bytes


@pytest.mark.parametrize("case", ["peer", "stopped"])
def test_automatic_preparation_respects_another_inspector_and_stop_markers(tmp_path, capsys, case):
    from contextlib import nullcontext
    cohort = make_cohort("gemma4_it", tmp_path)
    for path in cohort.plans.values():
        path.unlink()
    if case == "stopped":
        atomic_json(cohort.root / ".startup-control/math/stop.json", {"immediate": True})
    context = lease(bootstrap.preparation_lock(cohort.root)) if case == "peer" else nullcontext()
    with context, patch.object(bootstrap.cluster, "run_child", side_effect=AssertionError("duplicate preparation")):
        code, output = run_status(cohort, capsys, "--json")
    assert code == 0
    rows = json.loads(output.out)["datasets"]
    assert not any(row["prepared"] for row in rows)
    if case == "peer":
        assert all(row["preparing"] for row in rows)
    else:
        assert rows[0]["stop_requested"]


@pytest.mark.parametrize("child_code", [0, 2])
@pytest.mark.parametrize("action", ["status", "results"])
def test_failed_automatic_preparation_does_not_claim_initialized(tmp_path, capsys, child_code, action):
    from srgc_research.dispatch import model_results
    cohort = make_cohort("gemma4_it", tmp_path)
    for path in cohort.plans.values():
        path.unlink()
    with patch.object(bootstrap.cluster, "run_child", return_value=child_code):
        if action == "status":
            code, output = run_status(cohort, capsys, "--json")
            assert code == 2 and output.out == ""
        else:
            with patch.dict(os.environ, cohort.env, clear=True):
                code = model_results.main(["gemma4_it", "--root", str(cohort.root)])
            output = capsys.readouterr()
            assert code == 1 and "ERROR:" in output.out
            assert not (cohort.work / "results/gemma4_it-results.json").exists()
    error = output.out + output.err
    if child_code == 0:
        assert "did not create required IT plans" in error
    else:
        assert "preparation failed" in error
    assert not any(path.exists() for path in cohort.plans.values())


def test_automatic_preparation_cannot_repurpose_a_pt_cohort(tmp_path, capsys):
    cohort = make_cohort("gemma4", tmp_path)
    before = saved(cohort)
    with patch.object(bootstrap.cluster, "run_child", side_effect=AssertionError("replaced a PT experiment")):
        code, output = run_status(cohort, capsys)
    assert code == 2 and "another model's experiment plan" in output.err
    assert saved(cohort) == before


@pytest.mark.parametrize("action", ["run", "prepare"])
@pytest.mark.parametrize("explicit_it", [False, True])
def test_regular_it_commands_also_handle_an_old_pt_path_automatically(tmp_path, monkeypatch, action, explicit_it):
    from srgc_research.dispatch import gemma_it_run
    previous = tmp_path / "pt-model"
    atomic_json(previous / "generation_config.json", {"eos_token_id": 1})
    before = (previous / "generation_config.json").read_bytes()
    monkeypatch.setenv("GROUP_VOLUME", str(tmp_path))
    monkeypatch.setenv("OM_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("SRGC_GEMMA_MODEL_PATH", str(previous))
    monkeypatch.delenv("SRGC_GEMMA_IT_MODEL_PATH", raising=False)
    if explicit_it:
        monkeypatch.setenv("SRGC_GEMMA_IT_MODEL_PATH", str(tmp_path / "it-model"))
    with patch.object(gemma_it_run, "run", return_value=0):
        assert gemma_it_run.main(["all", action, "--root", str(tmp_path / "work/it-root")]) == 0
    if explicit_it:
        assert os.environ["SRGC_GEMMA_IT_MODEL_PATH"] == str(tmp_path / "it-model")
    else:
        assert "SRGC_GEMMA_MODEL_PATH" not in os.environ
    assert (previous / "generation_config.json").read_bytes() == before


def test_missing_it_plan_with_saved_queue_work_is_still_an_error(tmp_path, capsys):
    cohort = make_cohort("gemma4_it", tmp_path)
    cohort.plans["math"].unlink()
    receipt = cohort.root / "runs/math/.queue/tasks/seed-5.cache.json"
    atomic_json(receipt, {"status": "interrupted", "attempt": 1})
    before = saved(cohort)
    code, output = run_status(cohort, capsys)
    assert code == 1 and "restore the original plan" in output.out
    assert "MBPP PREFLIGHT NOT STARTED" in output.out
    assert saved(cohort) == before


@pytest.mark.parametrize("correct_per_prompt,state", [(0, "PAUSED"), (2, "PASS")])
def test_status_displays_partial_then_validated_preflight_without_a_model(tmp_path, capsys,
                                                                       correct_per_prompt, state):
    cohort = make_cohort("gemma4_it", tmp_path)
    path = cohort.plans["math"]
    plan = load_plan(path)
    data = json.loads(input_path(path, plan, 5).read_text())
    protocol = preflight.identity(path, plan, data)
    directory = preflight.folder(path, plan)
    rows = []
    for candidate in protocol["candidate_ids"]:
        row = {"protocol": protocol, "id": candidate,
               "rewards": [1] * correct_per_prompt + [0] * (8 - correct_per_prompt),
               "responses": ["synthetic status fixture"] * 8, "token_ids": [[9, 106]] * 8}
        rows.append(row)
        atomic_json(directory / f"{hashlib.sha256(candidate.encode()).hexdigest()}.json", row)
        if len(rows) == 1:
            code, output = run_status(cohort, capsys)
            assert code == 0 and "MATH PREFLIGHT PARTIAL | 1/8 prompts recorded" in output.out
            assert "correct (" not in output.out
    summary = preflight.summarize(protocol, rows)
    atomic_json(directory / "summary.json", summary)
    before = saved(cohort)
    code, output = run_status(cohort, capsys)
    assert code == 0 and f"MATH PREFLIGHT {state} | 8/8 prompts recorded" in output.out
    assert f"{8 * correct_per_prompt}/64 correct" in output.out
    assert "full initial-cache accuracy is separate" in output.out
    assert saved(cohort) == before
    code, output = run_status(cohort, capsys, "--json")
    assert code == 0 and output.err == ""
    report = json.loads(output.out)["datasets"][0]
    assert report["preflight"]["state"] == state
    assert report["initial_candidate_accuracy"] == []
    atomic_json(directory / "summary.json", {**summary, "correct": 64})
    code, output = run_status(cohort, capsys)
    assert code == 1 and "PREFLIGHT INVALID" in output.out
    assert "summary differs from raw receipts" in output.out


def test_status_rejects_escaped_plan_metadata(tmp_path, capsys):
    cohort = make_cohort("gemma4_it", tmp_path)
    experiments = cohort.root / "experiments"
    moved = tmp_path / "escaped-plans"
    experiments.rename(moved)
    experiments.symlink_to(moved, target_is_directory=True)
    code, output = run_status(cohort, capsys)
    assert code == 2 and "path must be below group storage" in output.err
