"""CPU-only execution with the actual Pair queue, leases, fit and freeze."""
import contextlib
import fcntl
import multiprocessing
import os
from pathlib import Path
import time

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selector_pair as pair
import selector_pair_gpu as gpu
import selector_pair_parallel as parallel
from test_selector_pair import points
from test_selector_pair_gpu import fake_study


@pytest.fixture
def study(tmp_path, fake_study, monkeypatch):
    protocol, calls, states = fake_study
    protocol.update(schema=pair.SCHEMA, code_hashes=gpu.code_hashes(), branch_manifests={})
    protocol.pop("protocol_id")
    protocol["protocol_id"] = core.fingerprint(protocol)
    core.atomic_json(tmp_path / "pair.json", protocol)

    def canonical(root, seed, step):
        identity, entries = states(root, seed, step)
        return identity, {name: (entry[0], entry[0] / "states" / f"s{seed}-t{step}" / "points" / f"view-{step}",
                                *entry[2:]) for name, entry in entries.items()}

    def execute(entry, arm, devices):
        branch, out, contract, _, _ = entry
        with gpu.pair_lease(out / arm / ".task.lock"):
            if (out / arm / "result.json").exists():
                return
            seed, step = contract["config"]["seed"], contract["config"]["drift"]
            if seed in pair.TEST_SEEDS and (branch.name, arm) not in parallel.FIXED_CONTROLS:
                assert len(gpu.decisions(tmp_path, protocol)) == 6
            calls.append((branch.name, seed, step, arm))
            cost = 100 if "on_policy" in branch.name else 200
            cost += 17 if branch.name.startswith("adaptive-") else 0
            cost = 300 if arm == "random_full" else cost
            value = {"points": points(cost), "artifact_hashes": {}, "path": str(out / arm)}
            core.atomic_json(out / arm / "result.json", value)
            core.atomic_json(out / arm / "curve.json", {"result_sha256": base.digest(out / arm / "result.json")})

    monkeypatch.setattr(gpu, "verify_pair", canonical)
    monkeypatch.setattr(gpu, "execute", execute)
    parallel.validate_activation(tmp_path, protocol)
    base.bind(tmp_path / parallel.RECEIPT, parallel.receipt_value(tmp_path, protocol))
    return protocol, calls, canonical


def test_full_real_scheduler_prioritizes_dev_then_fixed_controls_then_adaptive(tmp_path, study, monkeypatch):
    protocol, calls, states = study
    original = gpu.execute
    claim = tmp_path / "development/s0-t25/queue-branches/cached--selection_reduced.lock"
    _, entries = states(tmp_path, 0, 25)
    peer = tmp_path / "queue-workers/other-container.json"
    core.atomic_json(peer, {"worker": "other-container", "host": base.node_id(), "pid": os.getpid(),
                           "stage": "development", "state": "RUN", "protocol_id": protocol["protocol_id"]})
    before = peer.read_bytes(), peer.stat().st_mtime_ns
    held = contextlib.ExitStack()
    held.enter_context(gpu.pair_lease(claim))
    held.enter_context(gpu.pair_lease(entries["cached"][1] / "selection_reduced/.task.lock"))
    snapshots = {}

    def execute(entry, arm, devices):
        seed = entry[2]["config"]["seed"]
        if seed in pair.TEST_SEEDS and (entry[0].name, arm) in parallel.FIXED_CONTROLS:
            assert not (tmp_path / "model.json").exists()
            assert not (tmp_path / "test-decisions.json").exists()
            assert sum(row[1] in pair.DEV_SEEDS for row in calls) == 17
            records = [core.read(path) for path in tmp_path.glob("queue-workers/*.json") if path != peer]
            assert not any(row["stage"] == "development" and row["state"] == "RUN" for row in records)
            with pytest.raises(gpu.PairLockBusy):
                with gpu.pair_lease(claim):
                    pass
        original(entry, arm, devices)
        if sum(row[1] in pair.TEST_SEEDS for row in calls) == 18:
            snapshots.update({path: (path.read_bytes(), path.stat().st_mtime_ns)
                              for path in tmp_path.glob("branches/*/states/s[34]-*/points/*/*/result.json")})
            held.close()

    monkeypatch.setattr(gpu, "execute", execute)
    try:
        parallel.run_distributed(tmp_path, protocol, [], "run", gpu.run_distributed)
    finally:
        held.close()
    assert len(calls) == 42
    assert all(row[1] in pair.DEV_SEEDS for row in calls[:17])
    assert all((row[0], row[3]) in parallel.FIXED_CONTROLS for row in calls[17:35])
    assert all(row[0].startswith("adaptive-") for row in calls[-6:])
    assert len(gpu.decisions(tmp_path, protocol)) == 6
    assert len(list(tmp_path.glob("test/*/queue-branches/*.json"))) == 24
    assert len(list(tmp_path.glob("test/*/result.json"))) == 6
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == saved for path, saved in snapshots.items())
    assert before == (peer.read_bytes(), peer.stat().st_mtime_ns)
    assert not core.read(tmp_path / "report.json")["missing_states"]
    parallel.run_distributed(tmp_path, protocol, [], "run", gpu.run_distributed)
    assert len(calls) == 42


def test_eighteen_processes_claim_fixed_controls_while_all_development_is_owned(tmp_path, study, monkeypatch):
    protocol, _, states = study
    ctx = multiprocessing.get_context("fork")
    started, finished = ctx.Queue(), ctx.Queue()
    release = ctx.Event()
    execute, stage = gpu.execute, gpu.distributed_stage
    held = contextlib.ExitStack()
    for seed in pair.DEV_SEEDS:
        for step in pair.STEPS:
            folder = tmp_path / "development" / f"s{seed}-t{step}"
            held.enter_context(gpu.pair_lease(folder / ".state.lock", shared=True))
            _, entries = states(tmp_path, seed, step)
            for name in pair.SELECTORS:
                held.enter_context(gpu.pair_lease(folder / "queue-branches" / f"{name}--selection_reduced.lock"))
                held.enter_context(gpu.pair_lease(entries[name][1] / "selection_reduced/.task.lock"))
    held.enter_context(gpu.pair_lease(tmp_path / ".pair.lock", shared=True))

    def paused(entry, arm, devices):
        seed, step = entry[2]["config"]["seed"], entry[2]["config"]["drift"]
        assert seed in pair.TEST_SEEDS and (entry[0].name, arm) in parallel.FIXED_CONTROLS
        started.put((seed, step, entry[0].name, arm, os.getpid()))
        assert release.wait(30), "all eighteen fixed controls were not admitted"
        execute(entry, arm, devices)

    def quick_stage(root, p, devices, role, **kwargs):
        return stage(root, p, devices, role, wait_seconds=.01, idle_timeout=.05)

    monkeypatch.setattr(gpu, "execute", paused)
    monkeypatch.setattr(gpu, "distributed_stage", quick_stage)

    def run():
        try:
            parallel.run_distributed(tmp_path, protocol, [], "run", gpu.run_distributed)
            finished.put("unexpected completion")
        except gpu.PairWaitTimeout:
            finished.put(None)
        except BaseException as exc:
            finished.put((type(exc).__name__, str(exc)))

    processes = [ctx.Process(target=run) for _ in range(18)]
    try:
        for process in processes:
            process.start()
        claims = [started.get(timeout=25) for _ in processes]
        assert len({claim[:4] for claim in claims}) == len({claim[-1] for claim in claims}) == 18
        assert not (tmp_path / "model.json").exists()
        assert not list(tmp_path.glob("test/*/result.json"))
        with pytest.raises(gpu.PairLockBusy):
            with gpu.pair_lease(tmp_path / ".pair.lock"):
                pass
        release.set()
        outcomes = [finished.get(timeout=30) for _ in processes]
        assert outcomes == [None] * 18, outcomes
        for process in processes:
            process.join(5)
            assert process.exitcode == 0
        assert len(list(tmp_path.glob("test/*/queue-branches/*.json"))) == 18
        assert not (tmp_path / "test-decisions.json").exists()
        assert not list(tmp_path.glob("branches/adaptive-*/states/*/points/*/*/result.json"))
        with pytest.raises(gpu.PairLockBusy):
            with gpu.pair_lease(tmp_path / "development/s0-t25/queue-branches/on_policy--selection_reduced.lock"):
                pass
    finally:
        release.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)
        held.close()
        started.close()
        finished.close()


def test_fixed_outcomes_cannot_change_real_fit_or_frozen_choices(tmp_path, study):
    protocol, _, states = study
    gpu.develop(tmp_path, protocol, [])
    gpu.fit(tmp_path, protocol)
    model = (tmp_path / "model.json").read_bytes()
    queue = parallel.FixedQueue(tmp_path, protocol, [])
    for _ in range(18):
        assert queue.step()
    assert not queue.step()
    for seed in pair.TEST_SEEDS:
        for step in pair.STEPS:
            _, entries = states(tmp_path, seed, step)
            for name, arm in parallel.FIXED_CONTROLS:
                core.atomic_json(entries[name][1] / arm / "result.json", {"heldout_reward": 1e300})
    gpu.fit(tmp_path, protocol)
    assert (tmp_path / "model.json").read_bytes() == model
    with parallel.allow_fixed_before_decisions(tmp_path, protocol):
        choices = gpu.freeze(tmp_path, protocol)
    for value in choices.values():
        expected = pair.choose(core.read(tmp_path / "model.json"), value["features"],
                               seed=value["seed"], state_id=value["state_id"], protocol_id=protocol["protocol_id"])
        assert all(value[key] == item for key, item in expected.items())


@pytest.mark.parametrize("branch,arm", [("adaptive-on_policy", "selection_full"),
                                        ("adaptive-cached", "selection_full"),
                                        ("on_policy", "gated"), ("cached", "random_full")])
def test_registered_fixed_exception_never_allows_other_heldout_artifacts(tmp_path, study, branch, arm):
    protocol, _, states = study
    gpu.develop(tmp_path, protocol, [])
    gpu.fit(tmp_path, protocol)
    _, entries = states(tmp_path, 3, 25)
    core.atomic_json(entries[branch][1] / arm / "result.json", {})
    with parallel.allow_fixed_before_decisions(tmp_path, protocol), pytest.raises(ValueError, match="precedes"):
        gpu.freeze(tmp_path, protocol)
    assert not (tmp_path / "test-decisions.json").exists()


@pytest.mark.parametrize("kind", ["symlink", "fifo", "tampered", "missing"])
def test_receipt_validation_is_read_only_and_rejects_nonregular_metadata(tmp_path, study, kind):
    protocol, _, _ = study
    path = tmp_path / parallel.RECEIPT
    if kind == "tampered":
        value = core.read(path)
        value["fixed_controls"].append({"branch": "adaptive-cached"})
        core.atomic_json(path, value)
    else:
        path.unlink()
        if kind == "symlink":
            target = tmp_path / "alias.json"
            core.atomic_json(target, parallel.receipt_value(tmp_path, protocol))
            path.symlink_to(target)
        elif kind == "fifo":
            os.mkfifo(path)
    with pytest.raises(ValueError):
        parallel.validate_receipt(tmp_path, protocol)


@pytest.mark.parametrize("kind", ["fixed_result", "dangling_arm", "symlinked_cost"])
def test_activation_cannot_retroactively_authorize_artifacts_or_symlinks(tmp_path, study, kind):
    protocol, _, states = study
    (tmp_path / parallel.RECEIPT).unlink()
    _, entries = states(tmp_path, 3, 25)
    directory = entries["on_policy"][1] / "selection_full"
    if kind == "fixed_result":
        core.atomic_json(directory / "result.json", {})
    elif kind == "dangling_arm":
        directory.parent.mkdir(parents=True)
        directory.symlink_to(tmp_path / "missing")
    else:
        directory.mkdir(parents=True)
        (directory / "cost.jsonl").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError):
        parallel.validate_activation(tmp_path, protocol)


def test_legacy_strict_freeze_remains_strict_and_hook_restores_on_signal(tmp_path, study):
    protocol, _, _ = study
    gpu.develop(tmp_path, protocol, [])
    gpu.fit(tmp_path, protocol)
    assert parallel.FixedQueue(tmp_path, protocol, []).step()
    original = gpu.training_artifacts
    with pytest.raises(ValueError, match="precedes"):
        gpu.freeze(tmp_path, protocol)
    with pytest.raises(KeyboardInterrupt):
        with parallel.allow_fixed_before_decisions(tmp_path, protocol):
            raise KeyboardInterrupt()
    assert gpu.training_artifacts is original


def test_dev_failures_are_not_retried_after_each_fixed_control(tmp_path, study, monkeypatch):
    protocol, calls, _ = study
    execute = gpu.execute
    attempts = []

    def fail_one(entry, arm, devices):
        if entry[0].name == "cached" and entry[2]["config"] == {"seed": 0, "drift": 25}:
            attempts.append(True)
            raise ValueError("preserved invalid cost evidence")
        return execute(entry, arm, devices)

    monkeypatch.setattr(gpu, "execute", fail_one)
    with pytest.raises(gpu.IncompletePairRun):
        parallel.run_distributed(tmp_path, protocol, [], "run", gpu.run_distributed)
    assert len(attempts) == 1
    assert sum(row[1] in pair.TEST_SEEDS for row in calls) == 18
    assert not (tmp_path / "model.json").exists()


def test_freeze_preparation_uses_same_real_exclusive_lock(tmp_path, study, monkeypatch):
    protocol, _, _ = study
    gpu.develop(tmp_path, protocol, [])
    gpu.fit(tmp_path, protocol)
    original = gpu.verify_pair
    checked = []

    def verify(root, seed, step):
        if seed in pair.TEST_SEEDS:
            with (root / "test" / f"s{seed}-t{step}" / ".prepare-state.lock").open("rb") as handle:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            checked.append((seed, step))
        return original(root, seed, step)

    monkeypatch.setattr(gpu, "verify_pair", verify)
    parallel.run_distributed(tmp_path, protocol, [], "freeze", gpu.run_distributed)
    assert len(checked) == 6
    assert gpu.verify_pair is verify
    assert len(gpu.decisions(tmp_path, protocol)) == 6


def test_pending_controls_do_not_spin_or_trigger_admission(tmp_path, study, monkeypatch):
    protocol, _, _ = study
    attempts = []

    def pending(root, p, entry, arm, devices):
        attempts.append((str(entry[1]), arm))
        raise gpu.PairWorkPending("live shared curve owner")

    monkeypatch.setattr(gpu, "attempt_branch", pending)
    monkeypatch.setattr(gpu, "admit_node", lambda *args: pytest.fail("pending work is not an admission failure"))
    assert not parallel.FixedQueue(tmp_path, protocol, []).step()
    assert len(attempts) == len(set(attempts)) == 18
    assert not list(tmp_path.glob("test/*/queue-branches/*.json"))
    record = core.read(next(tmp_path.glob("queue-workers/*.json")))
    assert record["state"] == "WAIT" and record["total_branches"] == 24


def test_allowed_fixed_artifact_symlinks_are_not_hidden_from_freeze(tmp_path, study):
    protocol, _, states = study
    _, entries = states(tmp_path, 3, 25)
    directory = entries["on_policy"][1] / "selection_full"
    directory.mkdir(parents=True)
    (directory / "result.json").symlink_to(tmp_path / "pair.json")
    with parallel.allow_fixed_before_decisions(tmp_path, protocol), pytest.raises(ValueError, match="symlink"):
        gpu.training_artifacts(entries["on_policy"][1])


def test_existing_frozen_barrier_can_activate_without_reset(tmp_path, study):
    protocol, _, states = study
    gpu.develop(tmp_path, protocol, [])
    gpu.fit(tmp_path, protocol)
    gpu.freeze(tmp_path, protocol)
    _, entries = states(tmp_path, 3, 25)
    gpu.execute(entries["on_policy"], "selection_full", [])
    (tmp_path / parallel.RECEIPT).unlink()
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.rglob("*.json")}
    parallel.validate_activation(tmp_path, protocol)
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in before.items())
    core.atomic_json(tmp_path / "decisions/s3-t25/decision.json", {})
    with pytest.raises(ValueError):
        parallel.validate_activation(tmp_path, protocol)


def test_real_fixed_control_decision_needs_no_gate_or_heldout_measurement(tmp_path, study, monkeypatch):
    _, _, states = study
    _, entries = states(tmp_path, 3, 25)
    _, out, contract, protocol, suite = entries["on_policy"]
    core.atomic_json(out / "contract.json", contract)
    monkeypatch.setattr(gpu.switch.runtime, "measure_once", lambda *args: pytest.fail("fixed control read heldout diagnostics"))
    for arm in ("selection_full", "random_full"):
        assert gpu.switch.decision(out, suite, protocol, arm, {})["reason"] == "control_arm"
    monkeypatch.setattr(gpu.switch.runtime, "measure_once", lambda *args: {
        "status": "complete", "gpu_seconds": 0, "report_sha256": "fixture"})
    with pytest.raises(ValueError, match="gate is not bound"):
        gpu.switch.decision(out, suite, protocol, "gated", {})
