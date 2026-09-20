"""Real-process branch concurrency with a CPU-only execution backend."""
import contextlib
import fcntl
import json
import multiprocessing
import os
import time

import pytest

import selection_gate as core
import selection_gate_gpu as base
import selector_pair as pair
import selector_pair_gpu as gpu
from test_selector_pair_gpu import fake_study, bootstrap_predecessor
from test_selector_pair_wait import clock, held, prepare_other_states


@pytest.mark.parametrize("stage,count", [("development", 18), ("test", 24)])
def test_every_branch_can_run_concurrently_without_duplicate_work(tmp_path, fake_study, monkeypatch, stage, count):
    protocol, _, _ = fake_study
    if stage == "test":
        gpu.develop(tmp_path, protocol, [])
        gpu.fit(tmp_path, protocol)
        gpu.freeze(tmp_path, protocol)
    preserved = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                 for path in [*tmp_path.glob("*.json"), *tmp_path.glob("decisions/*/*.json")]
                 if path.name != "development-pass.json"}
    ctx = multiprocessing.get_context("fork")
    started, finished = ctx.Queue(), ctx.Queue()
    release = ctx.Event()
    execute = gpu.execute
    verify_pair = gpu.verify_pair
    journal = tmp_path / "calls.jsonl"

    def prepare(root, seed, step):
        # Detect overlapping cold-state publication, even across real workers.
        with base.lease(tmp_path / "prepare-probes" / f"s{seed}-t{step}.lock"):
            time.sleep(.005)
            return verify_pair(root, seed, step)

    def paused(entry, arm, devices):
        key = [entry[0].name, entry[2]["config"]["seed"], entry[2]["config"]["drift"], arm]
        with journal.open("a") as handle:
            handle.write(json.dumps([*key, os.getpid()]) + "\n")
        started.put(key)
        if not release.wait(25):
            raise AssertionError("not all independent branches were admitted concurrently")
        execute(entry, arm, devices)

    monkeypatch.setattr(gpu, "execute", paused)
    monkeypatch.setattr(gpu, "verify_pair", prepare)

    def worker():
        try:
            gpu.distributed_stage(tmp_path, protocol, [], stage, wait_seconds=.02, idle_timeout=30)
            finished.put(None)
        except BaseException as exc:
            finished.put((type(exc).__name__, str(exc)))

    workers = [ctx.Process(target=worker) for _ in range(count)]
    try:
        for process in workers:
            process.start()
        keys = [started.get(timeout=20) for _ in range(count)]
        assert len({tuple(key) for key in keys}) == count
        assert not list((tmp_path / stage).glob("*/result.json"))
        release.set()
        results = [finished.get(timeout=25) for _ in workers]
        assert results == [None] * count, results
        for process in workers:
            process.join(5)
            assert process.exitcode == 0
    finally:
        release.set()
        for process in workers:
            if process.is_alive():
                process.terminate()
            process.join(5)
        started.close()
        finished.close()
    calls = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(calls) == count
    assert len({row[-1] for row in calls}) == count
    assert len(list((tmp_path / stage).glob("*/queue-branches/*.json"))) == count
    assert len(list((tmp_path / stage).glob("*/result.json"))) == (9 if stage == "development" else 6)
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in preserved.items())
    monkeypatch.setattr(gpu, "execute", lambda *a: pytest.fail("complete branches must not run again"))
    gpu.distributed_stage(tmp_path, protocol, [], stage, wait_seconds=.01)
    workers = [core.read(path) for path in tmp_path.glob("queue-workers/*.json")]
    assert all(row["verified_branches"] == count and row["state"] == "DONE" for row in workers)


def test_death_after_curve_before_queue_receipt_reuses_saved_work(tmp_path, fake_study, monkeypatch):
    protocol, calls, _ = fake_study
    ctx = multiprocessing.get_context("fork")
    execute = gpu.execute

    def crash(entry, arm, devices):
        execute(entry, arm, devices)
        os._exit(23)

    with monkeypatch.context() as patch:
        patch.setattr(gpu, "execute", crash)
        process = ctx.Process(target=gpu.distributed_stage, args=(tmp_path, protocol, [], "development"))
        process.start()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
        assert process.exitcode == 23
    saved = {path: (path.read_bytes(), path.stat().st_mtime_ns)
             for path in tmp_path.glob("branches/*/states/*/point/*/*.json")}
    assert saved and not list(tmp_path.glob("development/*/queue-branches/*.json"))
    lock = tmp_path / "development/s0-t25/queue-branches/on_policy--selection_reduced.lock"
    inode = lock.stat().st_ino
    gpu.distributed_stage(tmp_path, protocol, [], "development", wait_seconds=.01)
    assert len(calls) == 17 and lock.stat().st_ino == inode
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in saved.items())
    assert len(list(tmp_path.glob("development/*/result.json"))) == 9


def test_interrupt_before_result_preserves_checkpoint_and_cost_for_resume(tmp_path, fake_study, monkeypatch):
    protocol, calls, states = fake_study
    _, entries = states(tmp_path, 0, 25)
    directory = entries["on_policy"][1] / "selection_reduced"
    core.atomic_json(directory / "policy/checkpoint_state.json", {"completed_steps": 30, "budget_spent": 77})
    core.atomic_json(directory / "saved-cost.json", {"allocated_gpu_seconds": 77})
    before = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    with monkeypatch.context() as patch:
        patch.setattr(gpu, "execute", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            gpu.distributed_stage(tmp_path, protocol, [], "development", wait_seconds=.01)
    gpu.distributed_stage(tmp_path, protocol, [], "development", wait_seconds=.01)
    assert len(calls) == 18
    assert all(path.read_bytes() == value for path, value in before.items())


def test_peer_curve_wait_resumes_only_publication_without_readmission(tmp_path, fake_study, monkeypatch):
    protocol, calls, states = fake_study
    _, entries = states(tmp_path, 0, 25)
    pending_dir = entries["on_policy"][1] / "selection_reduced"
    execute = gpu.execute
    attempts = []

    def pending(entry, arm, devices):
        execute(entry, arm, devices)
        if entry[1] / arm == pending_dir:
            attempts.append(1)
            if len(attempts) == 1:
                (pending_dir / "curve.json").unlink()
                raise gpu.PairWorkPending("peer owns parent evaluation")
            core.atomic_json(pending_dir / "curve.json", {"result_sha256": base.digest(pending_dir / "result.json")})

    monkeypatch.setattr(gpu, "execute", pending)
    monkeypatch.setattr(gpu, "admit_node", lambda *a: pytest.fail("peer evaluation wait must not re-admit GPUs"))
    polls = []
    monkeypatch.setattr(gpu.time, "sleep", lambda seconds: polls.append(seconds))
    gpu.distributed_stage(tmp_path, protocol, [], "development", wait_seconds=.01)
    assert len(attempts) == 2 and len(calls) == 18 and polls
    assert core.read(pending_dir / "pair-attempt.json")["state"] == "DONE"


def test_real_execute_defers_peer_parent_curve_without_repeating_paid_evaluation(tmp_path, monkeypatch):
    from test_selection_switch_gpu import convergence_manifest

    branch = tmp_path / "branches/on_policy"
    out = branch / "states/s3-t25/points/view-25"
    arm = "random_full"
    directory = out / arm
    config = {"config": {"drift": 25}, "scope": {"gpu_type": "H100"}, "eval_k": 8}
    core.atomic_json(directory / "policy/budget_stop.json", {"completed_steps": 125})
    core.atomic_json(directory / "result.json", {"rewards": {"1": .3, "2": .5}})
    core.atomic_json(directory / "result.sha256.json", {"sha256": base.digest(directory / "result.json")})
    for step in (50, 75, 100):
        checkpoint = directory / "policy/curve-checkpoints" / f"step-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_bytes(b"saved-policy")
    saved = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    charged = []

    def meter(where, phase, gpu_type, **kwargs):
        assert phase == "curve" and kwargs["ledger"] == "reporting"
        charged.append(where)
        for command, _ in kwargs["commands"]:
            step, shard = int(command[command.index("--step") + 1]), command[command.index("--shard") + 1]
            target = gpu.switch.curve_point_dir(out, arm, step, 25)
            core.atomic_json(target / f"shard-{shard}.done.json", {})

    monkeypatch.setattr(gpu, "manifest", lambda root: {})
    monkeypatch.setattr(gpu, "environment", lambda c: {})
    monkeypatch.setattr(gpu, "admit_node", lambda *a: pytest.fail("peer wait must not trigger admission"))
    monkeypatch.setattr(gpu.switch, "manifest", lambda root: convergence_manifest())
    monkeypatch.setattr(gpu.switch.runtime, "run_arm", lambda *a: None)
    monkeypatch.setattr(gpu.switch, "remaining_allocation", lambda *a: pytest.fail("saved result restarted allocation"))
    monkeypatch.setattr(gpu.switch, "curve_reward", lambda *a: .3)
    monkeypatch.setattr(base, "meter", meter)
    entry = (branch, out, config, {}, {"eval_timeout": 10.})
    with held(out / "curve-parent/.point.lock"):
        with pytest.raises(gpu.PairWorkPending):
            gpu.attempt_branch(tmp_path, {}, entry, arm, list("0123"))
    assert charged == [directory / "curve"] * 3
    assert not (directory / "curve.json").exists()
    assert gpu.attempt_branch(tmp_path, {}, entry, arm, list("0123")) is None
    assert charged == [directory / "curve"] * 3 + [out / "curve-parent"]
    assert gpu.switch.branch_finished(convergence_manifest(), directory)
    assert all(path.read_bytes() == value for path, value in saved.items())
    assert gpu.attempt_branch(tmp_path, {}, entry, arm, list("0123")) is None
    assert len(charged) == 4


def test_tampered_branch_receipt_cannot_publish_state_or_restart_work(tmp_path, fake_study, monkeypatch):
    protocol, _, _ = fake_study
    gpu.distributed_stage(tmp_path, protocol, [], "development", wait_seconds=.01)
    folder = tmp_path / "development/s0-t25"
    (folder / "result.json").unlink()
    receipt = folder / "queue-branches/on_policy--selection_reduced.json"
    core.atomic_json(receipt, {**core.read(receipt), "state_id": "tampered"})
    before = receipt.read_bytes()
    monkeypatch.setattr(gpu, "execute", lambda *a: pytest.fail("receipt mismatch restarted training"))
    with pytest.raises(gpu.IncompletePairRun):
        gpu.distributed_stage(tmp_path, protocol, [], "development", wait_seconds=.01)
    assert receipt.read_bytes() == before and not (folder / "result.json").exists()


def test_unrelated_sibling_progress_cannot_extend_busy_branch_deadline(tmp_path, fake_study, clock):
    protocol = prepare_other_states(tmp_path, fake_study)
    _, _, states = fake_study
    _, entries = states(tmp_path, 0, 25)
    sibling_progress = entries["cached"][1] / "selection_reduced/progress.json"
    def heartbeat():
        core.atomic_json(sibling_progress, {"state": "running", "host": "unrelated-sibling", "updated": clock.time()})
    heartbeat()
    clock.after_sleep = heartbeat
    folder = tmp_path / "development/s0-t25"
    with contextlib.ExitStack() as stack:
        stack.enter_context(gpu.pair_lease(folder / ".state.lock", shared=True))
        stack.enter_context(held(folder / "queue-branches/on_policy--selection_reduced.lock"))
        with pytest.raises(gpu.PairWaitTimeout):
            gpu.distributed_stage(tmp_path, protocol, [], "development", wait_seconds=30, idle_timeout=180)
    assert clock.elapsed == 180
    assert not (folder / "result.json").exists()


@pytest.mark.parametrize("offset", [-3600, 3600])
def test_busy_branch_progress_uses_local_monotonic_clock(tmp_path, fake_study, clock, offset):
    protocol = prepare_other_states(tmp_path, fake_study)
    _, _, states = fake_study
    _, entries = states(tmp_path, 0, 25)
    progress = entries["on_policy"][1] / "selection_reduced/curve/progress.json"
    lock = tmp_path / "development/s0-t25/queue-branches/on_policy--selection_reduced.lock"
    with held(lock) as owner:
        def heartbeat():
            core.atomic_json(progress, {"state": "running", "host": "skewed-peer",
                                       "updated": clock.time() + offset, "seconds": clock.elapsed})
            if clock.elapsed >= 240:
                fcntl.flock(owner, fcntl.LOCK_UN)
        heartbeat()
        clock.after_sleep = heartbeat
        gpu.distributed_stage(tmp_path, protocol, [], "development", wait_seconds=30, idle_timeout=180)
    assert clock.elapsed == 240
    assert len(list(tmp_path.glob("development/*/result.json"))) == 9


def released_status_runtime():
    previous = gpu.code_hashes()
    previous.update({"src/selector_pair_gpu.py": "042446a0513d8eaeba2dc93ad0b4401a85f8ae9013c3042f80691afa81901f0f",
                     "scripts/run_selector_pair.sh": "562eea6fbb53a7e024572839860bad02c3af304016ec63e7269a6b054054f12d"})
    assert core.fingerprint(previous) == gpu.PRE_PAIR_BRANCH_QUEUE_CODE
    return previous


@pytest.mark.parametrize("migrated", [False, True])
def test_upgrade_preserves_budget_results_and_entire_receipt_chain(tmp_path, monkeypatch, migrated):
    previous = released_status_runtime()
    p = {"schema": pair.SCHEMA, "code_hashes": bootstrap_predecessor() if migrated else previous,
         "branch_manifests": {}, "target_reward": .35, "training_cap_gpu_seconds": 87120}
    p["protocol_id"] = core.fingerprint(p)
    core.atomic_json(tmp_path / "pair.json", p)
    core.atomic_json(tmp_path / "saved/decision.json", {"selector": "cached"})
    core.atomic_json(tmp_path / "saved/cost.json", {"spent": 87120})
    if migrated:
        with monkeypatch.context() as patch:
            patch.setattr(gpu, "code_hashes", lambda: previous)
            patch.setattr(gpu, "PRE_SHARED_RUNTIME_CODES", gpu.PRE_SHARED_RUNTIME_CODES - {gpu.PRE_PAIR_BRANCH_QUEUE_CODE})
            gpu.bind_startup_runtime(tmp_path, p["code_hashes"])
        (tmp_path / "pair-branch-queue-runtime.json").unlink()
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.rglob("*") if path.is_file()}
    for _ in range(2):
        assert gpu.manifest(tmp_path) == p
        assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in before.items())
    assert core.read(tmp_path / "pair-branch-queue-runtime.json")["runtime_code_hashes"] == gpu.code_hashes()
    assert not gpu.compatible_code({**previous, "src/selector_pair_train.py": "unreviewed"})
