import copy
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from scripts import srgc_experiments as cli
from scripts import srgc_result_collection as collection


@pytest.mark.parametrize("dataset", ["math", "mbpp", "all"])
@pytest.mark.parametrize("experiment", list(cli.EXPERIMENTS))
@pytest.mark.parametrize("action", ["run", "status", "results"])
def test_every_experiment_has_an_explicit_existing_runner(dataset, experiment, action):
    commands = cli.commands(dataset, experiment, action)
    assert all(command[0] == "sh" and Path(command[1]).is_file() for command in commands)
    if action == "run":
        assert len(commands) == 1
    if experiment in {"timing", "rules"} and action == "run":
        assert commands[0][-1] == experiment
    if experiment in {"timing", "rules"} and action == "status":
        assert commands[0][-2:] == ["status", experiment]
    if experiment == "qwen" and action == "run":
        assert commands[0][-1] == dataset


def test_old_all_results_runs_both_datasets_even_when_first_fails():
    with patch.object(cli.subprocess, "call", side_effect=[1, 0]) as run:
        assert cli.main(["all", "replicate", "results"]) == 1
    assert [call.args[0][-2:] for call in run.call_args_list] == [["math", "results"], ["mbpp", "results"]]


def test_single_command_exec_preserves_environment_and_signals():
    class Executed(Exception):
        pass
    with patch.dict(os.environ, {"PAIR_PYTHON": "/recorded/runtime/python"}), \
            patch.object(cli.os, "execvpe", side_effect=Executed) as execute:
        with pytest.raises(Executed):
            cli.main(["all", "mechanism"])
    assert execute.call_args.args[1][-2:] == ["all", "run"]
    assert execute.call_args.args[2]["PAIR_PYTHON"] == "/recorded/runtime/python"


@pytest.mark.parametrize("argv", [["math", "all"], ["math", "reference_sweep"],
                                  ["all", "replicate", "json"], ["math", "qwen", "json"]])
def test_invalid_or_unimplemented_requests_never_launch(argv):
    with patch.object(cli.os, "execvpe") as execute, patch.object(cli.subprocess, "call") as run:
        with pytest.raises(SystemExit) as error:
            cli.main(argv)
    assert error.value.code == 2
    execute.assert_not_called()
    run.assert_not_called()


def test_shell_list_needs_no_training_environment_or_group_volume():
    result = subprocess.run(["sh", str(cli.REPO / "scripts/run_srgc_experiments.sh"), "list"],
                            cwd="/tmp", capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "mechanism" in result.stdout and "tasks/dataset" in result.stdout
    check = subprocess.run(["python3", "-c", "from scripts import srgc_experiments; import sys; "
                            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
                           cwd=cli.REPO, capture_output=True, text=True, timeout=10)
    assert check.returncode == 0, check.stderr


def report_at(tmp_path, dataset="math"):
    root = tmp_path / dataset
    folder = root / "seed-5"
    folder.mkdir(parents=True)
    endpoint = folder / "stage_mechanism-endpoint.json"
    raw = {"seed": 5, "synthetic": True, "rows": [{"reward": .5}]}
    endpoint.write_text(json.dumps(raw))
    return root, endpoint, dict(rows=[dict(dataset=dataset, seed=5, output=str(folder))],
                               errors=[], source_results={str(endpoint): raw})


def test_export_is_self_contained_and_does_not_reread_live_sources(tmp_path):
    root, source, report = report_at(tmp_path)
    original = copy.deepcopy(report["source_results"])
    source.write_text("changed after validation")
    collection.publish(report, "mechanism")
    latest = json.loads((root / "results/mechanism/results.json").read_text())
    assert latest["source_results"] == original
    bundle = Path(latest["collections"][0]["directory"])
    assert json.loads((bundle / "raw/seed-5/stage_mechanism-endpoint.json").read_text()) == original[str(source)]
    assert source.read_text() == "changed after validation"
    assert not (root / "results/results.json").exists()
    collection.publish(report, "mechanism")
    assert len(list((root / "results/mechanism/exports").iterdir())) == 2


def test_export_copy_failure_keeps_previous_latest(tmp_path):
    root, _, report = report_at(tmp_path)
    collection.publish(report, "support")
    latest = root / "results/support/results.json"
    before = latest.read_bytes()
    atomic = collection.atomic_json
    def fail_copy(path, value):
        if "raw" in path.parts:
            raise OSError("full volume")
        return atomic(path, value)
    with patch.object(collection, "atomic_json", side_effect=fail_copy), pytest.raises(OSError):
        collection.publish(report, "support")
    assert latest.read_bytes() == before


def test_multiple_roots_each_receive_full_report_and_local_raw_files(tmp_path):
    root, source, report = report_at(tmp_path)
    other, other_source, extra = report_at(tmp_path, "mbpp")
    report["rows"] += extra["rows"]
    report["source_results"].update(extra["source_results"])
    collection.publish(report, "mechanism")
    assert len(report["collections"]) == 2
    for location in (root, other):
        saved = json.loads((location / "results/mechanism/results.json").read_text())
        assert set(saved["source_results"]) == {str(source), str(other_source)}
    assert all(c["file_count"] == 1 for c in report["collections"])


def test_export_rejects_outside_paths_before_writing(tmp_path):
    root, _, report = report_at(tmp_path)
    report["source_results"][str(tmp_path / "outside.json")] = {}
    with pytest.raises(ValueError, match="outside"):
        collection.publish(report, "support")
    assert not (root / "results").exists()


def test_json_paths_are_stderr_not_json_stdout(tmp_path, capsys):
    _, _, report = report_at(tmp_path)
    collection.publish(report, "mechanism")
    collection.print_paths(report, json_output=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "COLLECTED JSON:" in captured.err and "COLLECTED FILES:" in captured.err
