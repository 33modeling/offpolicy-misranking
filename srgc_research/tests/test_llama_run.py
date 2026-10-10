"""Allocated-GPU cleanup, actual process teardown and read-only actions."""

import os
import signal
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import srgc_process_guard as guard
from srgc_rebuttal import cluster
from srgc_rebuttal.runtime import Busy
from srgc_research.dispatch import llama_run as run
from srgc_research.dispatch import model_results
from srgc_research.dispatch.llama31 import adapter, cli

UUIDS = ("GPU-a", "GPU-b", "GPU-c", "GPU-d")


@pytest.mark.parametrize("fault", [None, "duplicates", "small", "non-h100", "missing"])
def test_allocation_accepts_busy_gpus_but_still_validates_hardware(fault):
    rows = [f"{uuid}, {80000 if index < 2 else 4}, NVIDIA H100, 81559"
            for index, uuid in enumerate(UUIDS)]
    if fault == "duplicates":
        rows[1] = rows[0]
    elif fault == "small":
        rows[0] = "GPU-a, 4, NVIDIA H100, 40000"
    elif fault == "non-h100":
        rows[0] = "GPU-a, 4, NVIDIA A100, 81559"
    elif fault == "missing":
        rows.pop()
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}), patch.object(
        run.subprocess, "run", return_value=SimpleNamespace(stdout="\n".join(rows))
    ) as query:
        if fault:
            with pytest.raises(ValueError):
                run.allocated_gpus()
        else:
            assert run.allocated_gpus() == ("0,1,2,3", UUIDS, (80000, 80000, 4, 4))
        assert "--id=0,1,2,3" in query.call_args.args[0]


def test_real_cleanup_stops_retrying_launcher_and_its_gpu_rank_only():
    script = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(child.pid,flush=True); time.sleep(60)"
    )
    launcher = subprocess.Popen(
        [sys.executable, "-u", "-c", script, "srgc_research.dispatch.information_queue"],
        stdout=subprocess.PIPE, text=True,
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    child = int(launcher.stdout.readline())
    try:
        with patch.object(run, "gpu_processes", return_value={
            child: {UUIDS[0]}, unrelated.pid: {"GPU-unallocated"},
        }):
            stopped = run.cleanup_processes(UUIDS, grace_seconds=.2)
        assert child in stopped and launcher.pid in stopped
        assert unrelated.pid not in stopped and unrelated.poll() is None
        assert launcher.wait(timeout=5) in {-signal.SIGTERM, -signal.SIGKILL}
        # A reparented, exiting child may briefly remain a zombie.
        for _ in range(50):
            if run.process_token(child) is None:
                break
            run.time.sleep(.02)
        assert run.process_token(child) is None
    finally:
        for pid in (child, launcher.pid, unrelated.pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        launcher.wait(timeout=5)
        unrelated.wait(timeout=5)
        launcher.stdout.close()


def test_cleanup_escalates_for_an_owned_gpu_process_ignoring_term():
    process = subprocess.Popen(
        [sys.executable, "-u", "-c",
         "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"],
        stdout=subprocess.PIPE, text=True,
    )
    assert process.stdout.readline().strip() == "ready"
    try:
        with patch.object(run, "gpu_processes", return_value={process.pid: {UUIDS[0]}}):
            assert process.pid in run.cleanup_processes(UUIDS, grace_seconds=.05)
        assert process.wait(timeout=5) == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdout.close()


@pytest.mark.parametrize("foreign", [False, True])
def test_other_users_and_processes_spanning_other_allocations_are_not_signaled(foreign):
    pid = 900001
    table = {pid: (1, os.getuid() + int(foreign), "python rank")}
    applications = {pid: {UUIDS[0]} if foreign else {UUIDS[0], "GPU-unallocated"}}
    with patch.object(guard, "process_table", return_value=table), \
            patch.object(run, "gpu_processes", return_value=applications), \
            patch.object(run.os, "kill") as kill:
        with pytest.raises((ValueError, RuntimeError)):
            run.cleanup_processes(UUIDS, grace_seconds=0)
        kill.assert_not_called()


def test_pid_reuse_is_checked_before_each_signal():
    pid, uid = 900001, os.getuid()
    with patch.object(guard, "process_table", return_value={pid: (1, uid, "python rank")}), \
            patch.object(run, "gpu_processes", return_value={pid: {UUIDS[0]}}), \
            patch.object(run, "process_token", side_effect=[(uid, "old"), (uid, "new"), (uid, "new")]), \
            patch.object(run.os, "kill") as kill:
        run.cleanup_processes(UUIDS, grace_seconds=0)
        kill.assert_not_called()


def test_busy_cleanup_waits_for_memory_release_and_preserves_adapter_digest(tmp_path):
    before = adapter.adapter_digest()
    with patch.object(run, "allocated_gpus", return_value=("0,1,2,3", UUIDS, (80000, 64000, 4, 4))), \
            patch.object(guard, "canonical_lock_root", return_value=tmp_path), \
            patch.object(run, "cleanup_processes") as cleanup, \
            patch.object(cluster, "gpu_identity", side_effect=[Busy("releasing"), ("0,1,2,3", UUIDS)]), \
            patch.object(run.time, "sleep"):
        with run.clean_start():
            assert "srgc_research.dispatch.llama_run" in guard.OWNER_MARKERS
            assert list(tmp_path.glob("llama-start-*.lock"))
        cleanup.assert_called_once_with(UUIDS)
    assert adapter.adapter_digest() == before


def test_another_local_llama_launcher_cannot_cleanup_this_launchers_jobs(tmp_path):
    with patch.object(run, "allocated_gpus", return_value=("0,1,2,3", UUIDS, (4, 4, 4, 4))), \
            patch.object(guard, "canonical_lock_root", return_value=tmp_path), \
            patch.object(run, "cleanup_processes") as cleanup:
        with run.clean_start(), pytest.raises(Busy), run.clean_start():
            pytest.fail("second local launcher should not start")
        cleanup.assert_not_called()


@pytest.mark.parametrize("args,clean", [
    ([], True), (["all"], True), (["math", "run"], True), (["mbpp", "resume"], True),
    (["all", "status"], False), (["math", "prepare"], False),
    (["mbpp", "results"], False), (["all", "stop"], False), (["all", "doctor"], False),
])
def test_only_training_start_routes_through_cleanup(args, clean):
    def cli_start(arguments):
        if clean:
            cluster.gpu_identity()
            cluster.gpu_identity()  # Cleanup is once per invocation, not per task.
        return 0
    with patch.object(run, "clean_start", side_effect=lambda **kwargs: nullcontext()) as cleanup, \
            patch.object(cluster, "gpu_identity", return_value=("0,1,2,3", UUIDS)), \
            patch.object(cli, "main", side_effect=cli_start) as original, \
            patch.object(model_results, "main", return_value=0) as export:
        assert run.main(args) == 0
        if len(args) > 1 and args[1] == "results":
            original.assert_not_called()
            export.assert_called_once_with(["llama31", *args])
        else:
            original.assert_called_once_with(args or ["all"])
            export.assert_not_called()
        assert cleanup.call_count == int(clean)


def test_preparation_failure_never_cleans_up_gpu_processes():
    with patch.object(run, "clean_start") as cleanup, \
            patch.object(cli, "main", side_effect=ValueError("bad downloaded model")), \
            pytest.raises(ValueError, match="bad downloaded model"):
        run.main(["all"])
    cleanup.assert_not_called()


def test_later_task_waits_for_delayed_gpu_release_without_stopping_another_job(capsys):
    before = adapter.adapter_digest()

    def cli_start(arguments):
        assert cluster.gpu_identity() == ("0,1,2,3", UUIDS)
        assert cluster.gpu_identity() == ("0,1,2,3", UUIDS)
        return 0

    with patch.object(run, "clean_start", side_effect=lambda **kwargs: nullcontext()), \
            patch.object(cluster, "gpu_identity", side_effect=[
                ("0,1,2,3", UUIDS), Busy("memory is still being released"),
                Busy("memory is still being released"), ("0,1,2,3", UUIDS),
            ]), patch.object(cli, "main", side_effect=cli_start), \
            patch.object(run.time, "sleep") as sleep, patch.object(run, "cleanup_processes") as cleanup:
        assert run.main(["all"]) == 0
    assert sleep.call_count == 2
    cleanup.assert_not_called()
    assert "waiting for GPU memory release" in capsys.readouterr().out
    assert adapter.adapter_digest() == before


def test_permanent_gpu_occupancy_has_a_bounded_timeout(tmp_path):
    checks = 0

    def identity():
        nonlocal checks
        checks += 1
        if checks == 1:
            return "0,1,2,3", UUIDS
        raise Busy("still occupied")

    def cli_start(arguments):
        cluster.gpu_identity()
        return cluster.gpu_identity()

    with patch.object(run, "clean_start", side_effect=lambda **kwargs: nullcontext()), \
            patch.object(cluster, "gpu_identity", side_effect=identity), \
            patch.object(cli, "main", side_effect=cli_start), \
            patch.object(run.time, "monotonic", side_effect=[0, 31]), \
            patch.object(run, "cleanup_processes") as cleanup, \
            pytest.raises(TimeoutError, match="GPU memory release.*30s"):
        run.main(["all", "run", "--root", str(tmp_path)])
    assert checks == 2
    cleanup.assert_not_called()


@pytest.mark.parametrize("kind", ["smoke", "preflight"])
def test_admission_traceback_is_printed_after_backup_messages(tmp_path, capsys, kind):
    path = tmp_path / "runs/math/.queue/admission/session" / f"{kind}.log"
    path.parent.mkdir(parents=True)
    path.write_text("[rank2]: RuntimeError: actual admission failure\n")
    message = (f"Llama generation/backward admission failed: {path}" if kind == "smoke"
               else f"four-GPU admission failed (exit=1); inspect {path}")

    def failure(arguments):
        print("BACKUP final=true")
        raise RuntimeError(message)

    with patch.object(cli, "main", side_effect=failure), pytest.raises(RuntimeError, match="admission failed"):
        run.main(["all", "run", "--root", str(tmp_path)])
    text = capsys.readouterr().out
    assert text.index("LLAMA ADMISSION FAILURE DETAILS") > text.index("BACKUP final=true")
    assert text.endswith("[rank2]: RuntimeError: actual admission failure\n")


def test_gpu_hardware_or_query_error_is_not_retried():
    def invalid_identity():
        raise ValueError("invalid GPU")

    with patch.object(run.time, "sleep") as sleep, pytest.raises(ValueError, match="invalid GPU"):
        run.released_gpu_identity(invalid_identity)
    sleep.assert_not_called()
