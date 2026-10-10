"""Gemma transition recovery, live-owner protection and real failure evidence."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import srgc_process_guard as guard
from srgc_rebuttal import cluster
from srgc_rebuttal.runtime import Busy, atomic_json
from srgc_research.dispatch import gemma_run as run
from srgc_research.dispatch import model_launch, model_results
from srgc_research.dispatch.gemma4 import adapter, diagnostics, entry
from srgc_research.tests.test_gemma4 import prepared

UUIDS = ("GPU-a", "GPU-b", "GPU-c", "GPU-d")


def test_delayed_gpu_release_continues_the_gemma_queue_and_restores_hooks(capsys):
    before = adapter.adapter_digest()
    footer, owners = diagnostics.failure_footer, guard.OWNER_MARKERS

    def start(args):
        assert cluster.gpu_identity() == ("0,1,2,3", UUIDS)
        assert cluster.gpu_identity() == ("0,1,2,3", UUIDS)
        return 0

    with patch.object(cluster, "gpu_identity", side_effect=[
        ("0,1,2,3", UUIDS), Busy("CUDA teardown pending"), Busy("CUDA teardown pending"),
        ("0,1,2,3", UUIDS),
    ]) as identity, patch.object(entry, "main", side_effect=start), patch.object(model_launch.time, "sleep") as sleep:
        assert run.main(["all"]) == 0
        assert cluster.gpu_identity is identity
    assert sleep.call_count == 2
    assert diagnostics.failure_footer is footer and guard.OWNER_MARKERS == owners
    assert adapter.adapter_digest() == before
    text = capsys.readouterr().out
    assert "GEMMA waiting for GPU memory release" in text and "GEMMA GPU memory release complete" in text
    assert "LLAMA" not in text


@pytest.mark.parametrize("args", [["all", "status"], ["status"], ["math", "prepare"],
                                  ["all", "results"], ["all", "stop"], ["all", "doctor"]])
def test_nontraining_actions_do_not_install_gpu_recovery(args):
    with patch.object(run, "released_gpu_identity") as wait, patch.object(entry, "main", return_value=0) as original, \
            patch.object(model_results, "export", return_value=0) as export:
        assert run.main(args) == 0
        wait.assert_not_called()
        expected = ["all", "status"] if args == ["status"] else args
        if expected[1] == "results":
            original.assert_not_called()
            assert export.call_args.args[:2] == ("gemma4", "all")
        else:
            original.assert_called_once_with(expected)
            export.assert_not_called()


@pytest.mark.parametrize("owner", ["srgc_research.dispatch.gemma_run", "srgc_research.dispatch.gemma4.entry"])
def test_active_new_and_original_gemma_launchers_are_never_reaped(owner):
    original = guard.OWNER_MARKERS
    table = {
        900001: (1, os.getuid(), f"python -m {owner} all"),
        900002: (900001, os.getuid(), "python -m torch.distributed.run --module srgc_research.dispatch.gemma4.rank"),
        900003: (900002, os.getuid(), "python -m srgc_research.dispatch.gemma4.rank --plan live"),
        900004: (1, os.getuid(), "python -m srgc_research.dispatch.gemma4.rank --plan orphan"),
    }

    def start(args):
        assert owner in guard.OWNER_MARKERS
        assert guard.orphan_pids(table=table) == [900004]
        return 0

    with patch.object(guard, "TARGET_MARKERS", ("srgc_research.dispatch.gemma4.rank",)), \
            patch.object(entry, "main", side_effect=start):
        assert run.main(["all"]) == 0
    assert guard.OWNER_MARKERS == original


@pytest.mark.parametrize("kind", ["smoke", "preflight"])
def test_admission_exception_is_printed_once_below_backup_shutdown(tmp_path, capsys, kind):
    path = tmp_path / "runs/math/.queue/admission/session" / f"{kind}.log"
    path.parent.mkdir(parents=True)
    path.write_text("[rank1]: RuntimeError: actual Gemma admission failure\n")
    message = (f"Gemma generation/backward admission failed: {path}" if kind == "smoke"
               else f"four-GPU admission failed (exit=1); inspect {path}")

    def fail(args):
        diagnostics.failure_footer(tmp_path, "all")
        print("BACKUP final=true")
        raise RuntimeError(message)

    with patch.object(entry, "main", side_effect=fail), pytest.raises(RuntimeError, match="admission failed"):
        run.main(["all", "run", "--root", str(tmp_path)])
    text = capsys.readouterr().out
    assert text.count("GEMMA ADMISSION FAILURE DETAILS") == 1
    assert text.index("GEMMA ADMISSION FAILURE DETAILS") > text.index("BACKUP final=true")
    assert text.endswith("[rank1]: RuntimeError: actual Gemma admission failure\n")


def test_math_only_failure_shows_the_actual_mbpp_resume_error(tmp_path, capsys):
    prepared(tmp_path, "math")
    prepared(tmp_path, "mbpp")
    directory = tmp_path / "runs/mbpp/.queue"
    atomic_json(directory / "tasks/seed-9.prefix.json", {"task": "seed-9.prefix", "status": "failed"})
    log = directory / "logs/seed-9.prefix.log"
    log.parent.mkdir(parents=True)
    log.write_text("RuntimeError: real MBPP failure blocking MATH\n")
    with patch.object(entry, "main", side_effect=RuntimeError("unfinished Gemma tasks")), \
            pytest.raises(RuntimeError, match="unfinished Gemma"):
        run.main(["math", "run", "--root", str(tmp_path)])
    text = capsys.readouterr().out
    assert "mbpp:seed-9.prefix" in text
    assert text.endswith("RuntimeError: real MBPP failure blocking MATH\n")


def test_original_scientific_identity_remains_resumable():
    from srgc_research.storage import runtime_files

    assert adapter.adapter_digest() == "ba0a57d8ffa8802bb52c04b307ccad38a50f232132f0a58345176e3a33a552b1"
    frozen = runtime_files()
    assert "srgc_research/dispatch/gemma_run.py" not in frozen
    assert "srgc_research/dispatch/model_launch.py" not in frozen


def test_actual_new_launcher_status_is_gpu_free_and_read_only(tmp_path):
    root = tmp_path / "work/gemma"
    prepared(root, "math")
    prepared(root, "mbpp")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    wrapper = (
        "import runpy,sys; sys.modules.update(torch=None,transformers=None,peft=None); "
        "sys.argv=['gemma','status','--root',sys.argv[1]]; "
        "runpy.run_module('srgc_research.dispatch.gemma_run',run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-c", wrapper, str(root)],
                            cwd=Path(__file__).resolve().parents[2],
                            env={**os.environ, "GROUP_VOLUME": str(tmp_path), "OM_WORK": str(tmp_path / "work")},
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert "Gemma-4-12B-PT | SRGC" in result.stdout
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_admission_diagnostics_do_not_read_a_log_outside_the_experiment(tmp_path, capsys):
    root = tmp_path / "experiment"
    outside = tmp_path / "outside.log"
    outside.write_text("unrelated private data")
    with patch.object(Path, "open", side_effect=AssertionError("outside file must not be read")):
        model_launch.admission_failure_footer(RuntimeError(f"Gemma generation/backward admission failed: {outside}"),
                                              root, label="GEMMA", model_name="Gemma")
    assert capsys.readouterr().out == ""
