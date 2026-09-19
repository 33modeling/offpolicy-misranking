"""CPU-only branch quarantine: missing saved work must not stop intact siblings."""

import fcntl
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import selection_switch_gpu as switch


def branch_queue(tmp_path, monkeypatch, *, dataset="mbpp"):
    p = {"dataset": dataset, "sources": {str(seed): {"config": {}} for seed in (3, 4)}}
    monkeypatch.setitem(sys.modules, "additive_experiment", SimpleNamespace(model_environment=lambda _: {}))
    monkeypatch.setattr(switch, "manifest", lambda _: p)
    monkeypatch.setattr(switch, "admitted_devices", lambda _: list("0123"))
    monkeypatch.setattr(switch.rule, "DEV_SEEDS", ())
    monkeypatch.setattr(switch.rule, "TEST_SEEDS", (3, 4))
    monkeypatch.setattr(switch.rule, "STEPS", (25, 100))
    monkeypatch.setattr(switch, "fit_once", lambda _: False)
    monkeypatch.setattr(switch, "bind_gate", lambda *args: None)
    monkeypatch.setattr(switch, "status", lambda _: None)
    monkeypatch.setattr(switch, "protocol", lambda child: switch.core.read(child / "net_protocol.json"))
    monkeypatch.setattr(switch.base, "entries", lambda child: iter((child / "points").iterdir()))
    arms = ("selection_full", "random_reduced")
    branches = {}
    for seed in (3, 4):
        for step in (25, 100):
            switch.core.atomic_json(switch.prefix_dir(tmp_path, seed) / f"prefix-{step}.json", {})
            child = switch.child_root(tmp_path, seed, step)
            switch.core.atomic_json(child / "net_protocol.json", {"arms": list(arms)})
            switch.core.atomic_json(child / "suite.json", {})
            out = child / "points" / f"view-{step}"
            switch.core.atomic_json(out / "contract.json", {"seed": seed, "step": step})
            for arm in arms:
                directory = out / arm
                switch.core.atomic_json(directory / "decision.json", {"budget_gpu_seconds": 1000.})
                branches[seed, step, arm] = directory

    calls, current_lease = [], []
    original_lease = switch.base.lease

    @contextmanager
    def lease(path, *args, **kwargs):
        with original_lease(path, *args, **kwargs):
            current_lease.append(path)
            try:
                yield
            finally:
                current_lease.pop()

    monkeypatch.setattr(switch.base, "lease", lease)

    def assert_branch_lease():
        assert current_lease and current_lease[-1].name == ".task.lock"
        path = current_lease[-1]
        with path.open("rb") as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return path.parent

    original_review = switch.mbpp_resume_blocked

    def review(protocol, directory):
        assert assert_branch_lease() == directory
        calls.append(("review", directory))
        return original_review(protocol, directory)

    monkeypatch.setattr(switch, "mbpp_resume_blocked", review)
    monkeypatch.setattr(switch, "freeze_decisions", lambda *args: calls.append(("freeze", assert_branch_lease())))
    monkeypatch.setattr(switch, "remaining_allocation", lambda directory, choice: calls.append(("allocation", directory)))

    def run(out, suite, protocol, arm, devices, env):
        directory = out / arm
        assert assert_branch_lease() == directory
        calls.append(("run", directory))
        switch.core.atomic_json(directory / "result.json", {"reward": .5})
        switch.core.atomic_json(directory / "result.sha256.json", {
            "sha256": switch.base.digest(directory / "result.json")})

    monkeypatch.setattr(switch.runtime, "run_arm", run)
    return branches, calls


def missing_checkpoint(directory):
    switch.core.atomic_json(directory / "policy/checkpoint-8/checkpoint_state.json", {"completed_steps": 8})
    (directory / "policy/grpo_stats.jsonl").write_text('{"step": 8}\n')
    (directory / "cost.jsonl").write_text('{"saved": "existing cost, not a new allocation"}\n')


def snapshot(directory):
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*")
            if path.is_file() and path.name != ".task.lock"}


def complete_checkpoint(directory):
    checkpoint = directory / "policy/checkpoint-8"
    checkpoint.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "grpo_stats.jsonl"):
        (checkpoint / name).write_bytes(b"candidate metadata; full trainer validation remains required")
    switch.core.atomic_json(checkpoint / "checkpoint_state.json", {
        "completed_steps": 8, "adapter_sha256": "a" * 64,
        "optimizer_sha256": "b" * 64, "grpo_stats_sha256": "c" * 64})


def test_missing_checkpoint_branches_are_quarantined_under_lease_and_siblings_finish(tmp_path, monkeypatch, capsys):
    branches, calls = branch_queue(tmp_path, monkeypatch)
    blocked = {branches[3, 25, "selection_full"], branches[4, 100, "random_reduced"]}
    for directory in blocked:
        missing_checkpoint(directory)
    before = {directory: snapshot(directory) for directory in blocked}

    for _ in range(2):
        assert switch.work(tmp_path, idle_timeout=0) == 80
        assert all(snapshot(directory) == before[directory] for directory in blocked)
    assert {directory for phase, directory in calls if phase == "run"} == set(branches.values()) - blocked
    assert sum(phase == "run" for phase, _ in calls) == 6, "published siblings must not run again"
    assert all(phase == "review" for phase, directory in calls if directory in blocked)
    assert all(sum(phase == "review" and seen == directory for phase, seen in calls) == 2
               for directory in blocked), "one quarantine check per branch per invocation, no retry loop"
    output = capsys.readouterr().out
    assert "[WAIT]" in output and "checkpoint review required" in output
    assert all(str(directory) in output for directory in blocked)


def test_quarantine_clears_only_when_checkpoint_candidate_is_available(tmp_path, monkeypatch):
    branches, calls = branch_queue(tmp_path, monkeypatch)
    directory = branches[3, 25, "selection_full"]
    missing_checkpoint(directory)
    cost = (directory / "cost.jsonl").read_bytes()
    assert switch.work(tmp_path, idle_timeout=0) == 80
    assert ("run", directory) not in calls
    complete_checkpoint(directory)
    assert switch.work(tmp_path, idle_timeout=0) == 0
    assert sum(phase == "run" and seen == directory for phase, seen in calls) == 1
    assert (directory / "cost.jsonl").read_bytes() == cost
    assert switch.work(tmp_path, idle_timeout=0) == 0
    assert sum(phase == "run" for phase, _ in calls) == 8


def test_non_mbpp_queue_does_not_apply_mbpp_quarantine(tmp_path, monkeypatch):
    branches, calls = branch_queue(tmp_path, monkeypatch, dataset="math500")
    missing_checkpoint(branches[3, 25, "selection_full"])
    assert switch.work(tmp_path, idle_timeout=0) == 0
    assert sum(phase == "run" for phase, _ in calls) == 8


@pytest.mark.parametrize("other_blocker", ["failure", "busy", "prefix", "gate"])
def test_quarantine_terminal_code_requires_every_other_scoped_task_complete(tmp_path, monkeypatch, other_blocker):
    branches, calls = branch_queue(tmp_path, monkeypatch)
    blocked = branches[3, 25, "selection_full"]
    other = branches[4, 100, "random_reduced"]
    missing_checkpoint(blocked)
    if other_blocker == "failure":
        original_run = switch.runtime.run_arm

        def run(out, suite, protocol, arm, devices, env):
            if out / arm == other:
                raise ValueError("independent input problem")
            original_run(out, suite, protocol, arm, devices, env)

        monkeypatch.setattr(switch.runtime, "run_arm", run)
    elif other_blocker == "busy":
        original_lease = switch.base.lease

        @contextmanager
        def lease(path, *args, **kwargs):
            if path == other / ".task.lock":
                raise BlockingIOError("peer owns another task")
            with original_lease(path, *args, **kwargs):
                yield

        monkeypatch.setattr(switch.base, "lease", lease)
    elif other_blocker == "prefix":
        (switch.prefix_dir(tmp_path, 4) / "prefix-100.json").unlink()
        monkeypatch.setattr(switch, "build_prefix", lambda *args: None)
    else:
        child = switch.child_root(tmp_path, 4, 100)
        switch.core.atomic_json(child / "net_protocol.json", {"arms": ["selection_full", "random_reduced", "gated"]})
    assert switch.work(tmp_path, idle_timeout=0) == 1
    assert ("run", blocked) not in calls


def test_pilot_quarantine_terminal_checks_only_requested_scope(tmp_path, monkeypatch):
    branches, calls = branch_queue(tmp_path, monkeypatch)
    blocked = branches[3, 25, "selection_full"]
    missing_checkpoint(blocked)
    only = {"seeds": {3}, "arms": {"selection_full"}}
    assert switch.work(tmp_path, idle_timeout=0, only=only) == 80
    assert [(phase, directory) for phase, directory in calls if phase == "run"] == [
        ("run", branches[3, 100, "selection_full"])]


def test_mbpp_smoke_checks_saved_work_before_freezing_or_running(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "additive_experiment", SimpleNamespace(model_environment=lambda _: {}))
    monkeypatch.setattr(switch, "manifest", lambda _: {"dataset": "mbpp", "sources": {"0": {"config": {}}}})
    monkeypatch.setattr(switch, "admitted_devices", lambda _: list("0123"))
    switch.core.atomic_json(switch.prefix_dir(tmp_path, 0) / "prefix-25.json", {})
    child = switch.child_root(tmp_path, 0, 25)
    switch.core.atomic_json(child / "net_protocol.json", {})
    out = child / "points/view-25"
    directory = out / "selection_reduced"
    missing_checkpoint(directory)
    before = snapshot(directory)
    monkeypatch.setattr(switch.base, "entries", lambda _: iter([out]))

    def forbidden(*args, **kwargs):
        pytest.fail("quarantined smoke branch must not enter decisions, budget, or GPU work")

    monkeypatch.setattr(switch, "freeze_decisions", forbidden)
    monkeypatch.setattr(switch, "remaining_allocation", forbidden)
    monkeypatch.setattr(switch.runtime, "run_arm", forbidden)
    with pytest.raises(ValueError, match="MBPP smoke checkpoint review required"):
        switch.smoke(tmp_path)
    assert snapshot(directory) == before
    assert "smoke complete" not in capsys.readouterr().out
