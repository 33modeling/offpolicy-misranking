"""CPU-only regressions for Pair wait bounds and surviving curve shard leases."""

import fcntl
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selector_pair_gpu as gpu
from test_selection_switch_gpu import convergence_manifest
from test_selector_pair_gpu import fake_study
from test_selector_pair_wait import clock, held, prepare_other_states


@pytest.fixture
def queue_runtime():
    path = Path(__file__).resolve().parents[1] / "scripts/queue_selector_pair_gpu.py"
    spec = importlib.util.spec_from_file_location("pair_queue_wait_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name,shared", [
    (".pair.lock", True),
    (".pair-runtime.lock", False),
    (".pair-barrier.lock", False),
])
def test_lease_total_deadline_survives_changing_peer_heartbeats(
        tmp_path, clock, name, shared):
    lock = tmp_path / name
    progress = tmp_path / "branches/on_policy/states/s0-t25/points/view-25/selection_reduced/progress.json"

    def heartbeat():
        core.atomic_json(progress, {"event_id": "live-peer", "state": "running",
                                   "updated": clock.time(), "seconds": clock.elapsed})

    heartbeat()
    clock.after_sleep = heartbeat
    with held(lock):
        inode, contents = lock.stat().st_ino, lock.read_bytes()
        with pytest.raises(gpu.PairWaitTimeout):
            with gpu.queue_lease(lock, shared=shared, wait_seconds=47, max_wait_seconds=180):
                pytest.fail("a peer heartbeat must not bypass the held lease")
        assert clock.elapsed == 180
        assert clock.sleeps == [47, 47, 47, 39]
        assert lock.stat().st_ino == inode and lock.read_bytes() == contents
        with lock.open("a+") as probe, pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_completed_branches_cannot_extend_state_publication_wait_with_heartbeats(
        tmp_path, fake_study, monkeypatch, clock):
    protocol = prepare_other_states(tmp_path, fake_study)
    _, _, states = fake_study
    _, entries = states(tmp_path, 0, 25)
    progress = entries["on_policy"][1] / "selection_reduced/curve/progress.json"

    def heartbeat():
        core.atomic_json(progress, {"event_id": "completed-branch-peer", "state": "running",
                                   "updated": clock.time(), "seconds": clock.elapsed})

    heartbeat()
    clock.after_sleep = heartbeat
    folder = tmp_path / "development/s0-t25"
    lock = folder / ".state.lock"
    with gpu.pair_lease(lock, shared=True):
        inode = lock.stat().st_ino
        with pytest.raises(gpu.PairWaitTimeout):
            gpu.distributed_stage(tmp_path, protocol, [], "development",
                                  wait_seconds=30, idle_timeout=180)
        assert clock.elapsed == 180 and len(clock.sleeps) == 6
        assert len(list((folder / "queue-branches").glob("*.json"))) == 2
        assert not (folder / "result.json").exists()
        worker = core.read(next(tmp_path.glob("queue-workers/*.json")))
        assert worker["verified_branches"] == worker["total_branches"] == 18
        assert worker["verified_states"] == 8 and worker["state"] == "WAIT"
        assert not worker["failures"]
        with lock.open("a+") as probe, pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

    saved = {path: path.read_bytes() for path in folder.glob("queue-branches/*.json")}
    monkeypatch.setattr(gpu, "execute", lambda *a: pytest.fail("completed branches must not execute again"))
    gpu.distributed_stage(tmp_path, protocol, [], "development", wait_seconds=30, idle_timeout=180)
    assert (folder / "result.json").is_file() and lock.stat().st_ino == inode
    assert all(path.read_bytes() == data for path, data in saved.items())


def test_live_curve_peer_can_finish_four_hours_without_ending_its_worker(
        tmp_path, fake_study, clock):
    protocol = prepare_other_states(tmp_path, fake_study)
    _, _, states = fake_study
    _, entries = states(tmp_path, 0, 25)
    progress = entries["on_policy"][1] / "selection_reduced/curve/progress.json"
    lock = tmp_path / "development/s0-t25/queue-branches/on_policy--selection_reduced.lock"
    with held(lock) as owner:
        def heartbeat():
            core.atomic_json(progress, {"event_id": "healthy-long-curve", "state": "running",
                                       "updated": clock.time(), "seconds": clock.elapsed,
                                       "timeout": 14400})
            if clock.elapsed >= 14400:
                fcntl.flock(owner, fcntl.LOCK_UN)

        heartbeat()
        clock.after_sleep = heartbeat
        gpu.distributed_stage(tmp_path, protocol, [], "development",
                              wait_seconds=180, idle_timeout=180)
    assert clock.elapsed == 14400
    assert len(list(tmp_path.glob("development/*/result.json"))) == 9


@pytest.fixture
def curve_case(tmp_path, monkeypatch):
    switch = gpu.switch
    branch = tmp_path / "branches/on_policy"
    out = branch / "states/s3-t25/points/view-25"
    arm = "random_full"
    directory = out / arm
    config = {"config": {"drift": 25}, "scope": {"gpu_type": "H100"}, "eval_k": 8}
    core.atomic_json(directory / "policy/budget_stop.json", {"completed_steps": 125})
    core.atomic_json(directory / "result.json", {"rewards": {"1": .3, "2": .5}})
    for step in (50, 75, 100):
        checkpoint = directory / "policy/curve-checkpoints" / f"step-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_bytes(b"saved-policy")
    launches = []

    def meter(where, phase, gpu_type, **kwargs):
        assert phase == "curve" and kwargs["ledger"] == "reporting"
        assert kwargs["timeout"] == 14400
        for command, _ in kwargs["commands"]:
            step = int(command[command.index("--step") + 1])
            shard = int(command[command.index("--shard") + 1])
            point = switch.curve_point_dir(out, arm, step, 25)
            launches.append((step, shard))
            try:
                with base.lease(point / f"shard-{shard}.lock"):
                    core.atomic_json(point / f"shard-{shard}.done.json", {})
            except BlockingIOError:
                pytest.fail(f"duplicate GPU worker requested for live shard {step}/{shard}")

    monkeypatch.setattr(base, "meter", meter)
    monkeypatch.setattr(switch, "curve_reward", lambda *a: .3)

    def run():
        switch.curve_once(branch, convergence_manifest(), out, config, arm,
                          {"eval_timeout": 14400}, list("0123"), {})

    return SimpleNamespace(root=tmp_path, branch=branch, out=out, arm=arm,
                           directory=directory, config=config, launches=launches, run=run)


@pytest.mark.parametrize("orphan_step", [25, 50])
@pytest.mark.parametrize("orphan_shard", [0, 3])
def test_surviving_curve_shard_defers_point_without_duplicate_worker_or_lost_work(
        curve_case, queue_runtime, orphan_step, orphan_shard):
    case = curve_case
    target = gpu.switch.curve_point_dir(case.out, case.arm, orphan_step, 25)
    target.mkdir(parents=True, exist_ok=True)
    partial = target / f"shard-{orphan_shard}.jsonl"
    partial.write_bytes(b'{"saved": "partial evaluation"}\n')
    saved = {path: path.read_bytes() for path in case.out.rglob("*") if path.is_file()}
    launches = case.launches
    lock = target / f"shard-{orphan_shard}.lock"
    with held(lock):
        inode = lock.stat().st_ino
        # Its supervisor is gone: only the child's shard lease remains held.
        with (target / ".point.lock").open("a+") as point:
            fcntl.flock(point, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with queue_runtime.activated(case.root):
            case.run()
        assert not any(step == orphan_step for step, _ in launches)
        assert {step for step, _ in launches} == {25, 50, 75, 100} - {orphan_step}
        assert not (case.directory / "curve.json").exists()
        assert all(path.read_bytes() == data for path, data in saved.items())
        with lock.open("a+") as probe, pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        core.atomic_json(target / f"shard-{orphan_shard}.done.json", {})

    with queue_runtime.activated(case.root):
        case.run()
    assert len(launches) == len(set(launches)) == 15
    assert (orphan_step, orphan_shard) not in launches
    assert lock.stat().st_ino == inode
    assert all(path.read_bytes() == data for path, data in saved.items())
    assert set(core.read(case.directory / "curve.json")["points"]) == {"25", "50", "75", "100", "125"}


def test_shard_wait_is_pending_without_retry_or_node_readmission(
        curve_case, queue_runtime, monkeypatch):
    case = curve_case
    arm_calls, admissions = [], []
    monkeypatch.setattr(gpu, "manifest", lambda root: {})
    monkeypatch.setattr(gpu, "environment", lambda config: {})
    monkeypatch.setattr(gpu.switch, "manifest", lambda root: convergence_manifest())
    monkeypatch.setattr(gpu.switch.runtime, "run_arm", lambda *args: arm_calls.append(args))
    monkeypatch.setattr(gpu, "admit_node", lambda *args: admissions.append(args))
    entry = (case.branch, case.out, case.config, {}, {"eval_timeout": 14400})
    saved = {path: path.read_bytes() for path in case.out.rglob("*") if path.is_file()}
    with held(case.out / "curve-parent/shard-0.lock"):
        with queue_runtime.activated(case.root), pytest.raises(gpu.PairWorkPending):
            gpu.attempt_branch(case.root, {}, entry, case.arm, list("0123"))
        receipt = core.read(case.directory / "pair-attempt.json")
        assert receipt["state"] == "WAIT" and receipt["attempt"] == 1
        assert "error" not in receipt
        assert len(arm_calls) == 1 and not admissions
        assert set(case.launches) == {(step, shard) for step in (50, 75, 100) for shard in range(4)}
        assert len(case.launches) == 12
        assert not (case.directory / "curve.json").exists()
        assert all(path.read_bytes() == data for path, data in saved.items())
        with base.lease(case.directory / ".task.lock"):
            pass
