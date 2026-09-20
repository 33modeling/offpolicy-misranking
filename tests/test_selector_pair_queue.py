"""Independent Pair branches share states across real processes."""

import json
import fcntl
import multiprocessing
import os
import time

import pytest

import selection_gate as core
import selector_pair as pair
import selector_pair_gpu as gpu
from test_selector_pair_gpu import fake_study


def state_calls(calls, seed, step):
    return [(branch, arm) for branch, s, t, arm in calls if (s, t) == (seed, step)]


def assert_development_order(calls):
    for seed in pair.DEV_SEEDS:
        expected = list(pair.SELECTORS)
        if seed % 2:
            expected.reverse()
        for step in pair.STEPS:
            assert state_calls(calls, seed, step) == [(name, "selection_reduced") for name in expected]


def test_two_real_controllers_share_peer_state_and_execute_each_branch_once(tmp_path, fake_study, monkeypatch):
    p, _, _ = fake_study
    context = multiprocessing.get_context("fork")
    ready, release = context.Event(), context.Event()
    receiver, sender = context.Pipe(duplex=False)
    journal = tmp_path / "executed.jsonl"
    original_execute = gpu.execute
    parent_pid = os.getpid()

    def execute(entry, arm, devices):
        branch, out, config, _, _ = entry
        seed, step = config["config"]["seed"], config["config"]["drift"]
        if os.getpid() != parent_pid and (seed, step) == (0, 25) and branch.name == "on_policy":
            ready.set()
            if not release.wait(15):
                raise AssertionError("second controller did not release the peer test barrier")
        if not (out / arm / "result.json").exists():
            fd = os.open(journal, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            try:
                os.write(fd, (json.dumps([branch.name, seed, step, arm, os.getpid()]) + "\n").encode())
            finally:
                os.close(fd)
        original_execute(entry, arm, devices)

    monkeypatch.setattr(gpu, "execute", execute)

    def worker():
        try:
            gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=.01)
        except BaseException as exc:
            sender.send((type(exc).__name__, str(exc)))
        else:
            sender.send(None)
        finally:
            sender.close()

    process = context.Process(target=worker)
    process.start()
    sender.close()
    original_sleep = time.sleep
    waits = []

    def poll(seconds):
        if not release.is_set():
            rows = [json.loads(line) for line in journal.read_text().splitlines()]
            assert len(rows) == 17
            assert ["cached", 0, 25, "selection_reduced", parent_pid] in rows
            assert not any(row[:3] == ["on_policy", 0, 25] for row in rows)
            assert len(list((tmp_path / "development").glob("*/result.json"))) == 8
            assert not (tmp_path / "model.json").exists()
            assert not (tmp_path / "test-decisions.json").exists()
            waits.append(seconds)
            release.set()
        original_sleep(min(seconds, .01))

    try:
        assert ready.wait(10), "peer did not acquire its first state"
        monkeypatch.setattr(gpu.time, "sleep", poll)
        gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=.01)
        assert receiver.poll(10), "peer controller did not complete"
        assert receiver.recv() is None
        process.join(5)
        assert process.exitcode == 0 and waits
    finally:
        release.set()
        if process.is_alive():
            process.terminate()
        process.join(5)
        receiver.close()

    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(rows) == 18 and len({tuple(row[:4]) for row in rows}) == 18
    assert {row[4] for row in rows} == {parent_pid, process.pid}
    assert len(list((tmp_path / "development").glob("*/result.json"))) == 9
    assert set(state_calls([row[:4] for row in rows], 0, 25)) == {
        ("on_policy", "selection_reduced"), ("cached", "selection_reduced")}


def test_completed_development_is_revalidated_without_another_gpu_attempt(tmp_path, fake_study, monkeypatch):
    p, calls, _ = fake_study
    gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=.01)
    assert len(calls) == 18
    assert_development_order(calls)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in (tmp_path / "development").glob("*/result.json")}
    original_row = gpu.development_row
    validated = set()

    def row(root, protocol, seed, step):
        validated.add((seed, step))
        return original_row(root, protocol, seed, step)

    monkeypatch.setattr(gpu, "development_row", row)
    monkeypatch.setattr(gpu, "execute", lambda *a: pytest.fail("completed state started GPU work"))
    gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=.01)
    assert validated == {(seed, step) for seed in pair.DEV_SEEDS for step in pair.STEPS}
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}


def test_dead_peer_releases_real_flock_and_waiter_resumes_without_deleting_lock(tmp_path, fake_study, monkeypatch):
    p, calls, _ = fake_study
    context = multiprocessing.get_context("fork")
    ready, finish = context.Event(), context.Event()
    lock = tmp_path / "development/s0-t25/.state.lock"
    lock.parent.mkdir(parents=True)

    def peer():
        with lock.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            ready.set()
            finish.wait(20)

    process = context.Process(target=peer)
    process.start()
    polls = []

    def poll(seconds):
        assert not polls, "released peer state was not reclaimed"
        assert len(calls) == 16 and not state_calls(calls, 0, 25)
        polls.append(seconds)
        # Only terminate the explicitly created test peer. OS lock release
        # must make its state resumable without unlinking the shared lock.
        process.terminate()
        process.join(5)
        assert not process.is_alive() and lock.exists()

    try:
        assert ready.wait(10)
        inode = lock.stat().st_ino
        monkeypatch.setattr(gpu.time, "sleep", poll)
        gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=.01)
        assert polls and len(calls) == 18
        assert lock.stat().st_ino == inode
        assert_development_order(calls)
    finally:
        if process.is_alive():
            finish.set()
            process.terminate()
        process.join(5)


def test_tampered_completed_summary_cannot_count_as_finished_or_restart_training(tmp_path, fake_study, monkeypatch):
    p, _, _ = fake_study
    gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=.01)
    path = tmp_path / "development/s0-t25/result.json"
    core.atomic_json(path, {**core.read(path), "state_id": "tampered"})
    before = path.read_bytes()
    monkeypatch.setattr(gpu, "execute", lambda *a: pytest.fail("invalid summary must not restart saved training"))
    with pytest.raises(ValueError):
        gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=.01)
    assert path.read_bytes() == before
    assert not (tmp_path / "model.json").exists()


def test_failed_state_is_not_reclaimed_this_pass_and_independent_work_finishes(tmp_path, fake_study, monkeypatch):
    p, calls, _ = fake_study
    execute = gpu.execute
    failures = []

    def fail_one(entry, arm, devices):
        if (entry[0].name, entry[2]["config"]["seed"], entry[2]["config"]["drift"]) == ("cached", 0, 25):
            failures.append(1)
            raise ValueError("saved contract needs inspection")
        execute(entry, arm, devices)

    monkeypatch.setattr(gpu, "execute", fail_one)
    monkeypatch.setattr(gpu.time, "sleep", lambda *a: pytest.fail("local failure must not cause an endless retry"))
    with pytest.raises(gpu.IncompletePairRun):
        gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=.01)
    assert failures == [1] and len(calls) == 17
    assert len(list((tmp_path / "development").glob("*/result.json"))) == 8
    assert not (tmp_path / "model.json").exists() and not (tmp_path / "test-decisions.json").exists()
    saved = {path: path.read_bytes() for path in tmp_path.glob("branches/*/states/*/point/*/result.json")}
    monkeypatch.setattr(gpu, "execute", execute)
    gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=.01)
    assert len(calls) == 18 and all(path.read_bytes() == value for path, value in saved.items())
    assert len(list((tmp_path / "development").glob("*/result.json"))) == 9


def test_heldout_queue_requires_global_decision_barrier_before_work(tmp_path, fake_study, monkeypatch):
    p, calls, _ = fake_study
    monkeypatch.setattr(gpu, "execute", lambda *a: pytest.fail("test work began without frozen decisions"))
    with pytest.raises((FileNotFoundError, ValueError)):
        gpu.distributed_stage(tmp_path, p, [], "test", wait_seconds=.01)
    assert not calls


def test_heldout_queue_preserves_frozen_barrier_order_and_skips_complete_work(tmp_path, fake_study, monkeypatch):
    p, calls, _ = fake_study
    gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=.01)
    gpu.fit(tmp_path, p)
    choices = gpu.freeze(tmp_path, p)
    barrier = tmp_path / "test-decisions.json"
    preserved = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                 for path in [barrier, tmp_path / "model.json", *tmp_path.glob("decisions/*/decision.json")]}
    gpu.distributed_stage(tmp_path, p, [], "test", wait_seconds=.01)
    assert len(calls) == 42 and len(list((tmp_path / "test").glob("*/result.json"))) == 6
    for seed in pair.TEST_SEEDS:
        for step in pair.STEPS:
            selector = choices[f"s{seed}-t{step}"]["selector"]
            expected = [("on_policy", "selection_full"), ("cached", "selection_full"),
                        ("adaptive-" + selector, "selection_full"), ("on_policy", "random_full")]
            assert state_calls(calls, seed, step) == (expected[::-1] if seed % 2 else expected)
    monkeypatch.setattr(gpu, "execute", lambda *a: pytest.fail("completed held-out state started GPU work"))
    gpu.distributed_stage(tmp_path, p, [], "test", wait_seconds=.01)
    assert preserved == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in preserved}


def test_interruption_releases_state_for_later_resume(tmp_path, fake_study, monkeypatch):
    p, calls, _ = fake_study
    execute = gpu.execute
    monkeypatch.setattr(gpu, "execute", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=.01)
    assert not calls
    monkeypatch.setattr(gpu, "execute", execute)
    gpu.distributed_stage(tmp_path, p, [], "development", wait_seconds=.01)
    assert len(calls) == 18


def test_run_distributed_crosses_fit_and_test_barriers_only_after_all_states(tmp_path, fake_study, monkeypatch):
    p, calls, _ = fake_study
    stage, fit, freeze, report = gpu.distributed_stage, gpu.fit, gpu.freeze, gpu.report
    events = []

    def checked_stage(root, protocol, devices, name):
        if name == "test":
            assert len(gpu.decisions(root, protocol)) == 6
            assert len(list((root / "development").glob("*/result.json"))) == 9
        events.append(name + " start")
        stage(root, protocol, devices, name, wait_seconds=.01)
        events.append(name + " complete")

    def checked_fit(root, protocol):
        assert len(list((root / "development").glob("*/result.json"))) == 9
        assert not (root / "test-decisions.json").exists()
        assert all(seed in pair.DEV_SEEDS for _, seed, _, _ in calls)
        events.append("fit")
        return fit(root, protocol)

    def checked_freeze(root, protocol):
        assert (root / "model.json").exists()
        assert all(seed in pair.DEV_SEEDS for _, seed, _, _ in calls)
        events.append("freeze")
        return freeze(root, protocol)

    def checked_report(root, protocol):
        assert len(list((root / "test").glob("*/result.json"))) == 6
        events.append("report")
        return report(root, protocol)

    monkeypatch.setattr(gpu, "distributed_stage", checked_stage)
    monkeypatch.setattr(gpu, "fit", checked_fit)
    monkeypatch.setattr(gpu, "freeze", checked_freeze)
    monkeypatch.setattr(gpu, "report", checked_report)
    gpu.run_distributed(tmp_path, p, [], "run")
    assert len(calls) == 42
    assert events == ["development start", "development complete", "fit", "freeze",
                      "test start", "test complete", "report"]
    result = core.read(tmp_path / "report.json")
    assert not result["missing_states"] and not result["missing_development_states"]


def test_run_distributed_never_advances_after_incomplete_development(tmp_path, fake_study, monkeypatch):
    p, calls, _ = fake_study

    def incomplete(*args, **kwargs):
        raise gpu.IncompletePairRun("one saved development state needs repair")

    def forbidden(*args, **kwargs):
        pytest.fail("incomplete development advanced the fit/freeze/test barrier")

    monkeypatch.setattr(gpu, "distributed_stage", incomplete)
    for name in ("fit", "freeze", "test", "report"):
        monkeypatch.setattr(gpu, name, forbidden)
    with pytest.raises(gpu.IncompletePairRun):
        gpu.run_distributed(tmp_path, p, [], "run")
    assert not calls
    assert not (tmp_path / "model.json").exists() and not (tmp_path / "test-decisions.json").exists()
