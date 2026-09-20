"""Six-suite regression fixture matching the uploaded random-storage inventory."""

import hashlib
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

import selection_gate as core
import selection_switch as rule
import net_gain_gate as net

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


switch_status = module("selection_switch_status")
mopps_status = module("mopps_comparison_status")
combined_status = module("experiments_status")


def branch(root, seed, step, arm="random_reduced"):
    return root / f"states/s{seed}-t{step}/points/view-{step}" / arm


def result(directory, step, schema=rule.SCHEMA):
    core.atomic_json(directory / "result.json", {"schema": schema, "complete": True, "completed_steps": step})
    core.atomic_json(directory / "result.sha256.json", {
        "sha256": hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()})


def policy(directory, step, start, *, checkpoint=False):
    path = directory / "policy"
    if checkpoint:
        path /= f"checkpoint-{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    core.atomic_json(path / "adapter_config.json", {"peft_type": "LORA"})
    (path / "adapter_model.safetensors").write_bytes(b"fixture saved model")
    (path / "optimizer.pt").write_bytes(b"fixture saved optimizer")
    core.atomic_json(path / "grpo_stats.jsonl", {"step": step})
    hashes = {key: hashlib.sha256((path / filename).read_bytes()).hexdigest()
              for key, filename in (("adapter_sha256", "adapter_model.safetensors"),
                                    ("optimizer_sha256", "optimizer.pt"),
                                    ("grpo_stats_sha256", "grpo_stats.jsonl"))}
    if checkpoint:
        core.atomic_json(path / "checkpoint_state.json", {"completed_steps": step, **hashes})
    else:
        budget = {"completed_steps": step, "requested_target_steps": start + 100000,
                  "stop_reason": "budget_exhausted"}
        core.atomic_json(path / "policy_train.json", {"schema": "offpolicy-rlvr-policy/v1",
                                                       "start_step": start, "completed_steps": step,
                                                       "training_budget": budget, **hashes})
        core.atomic_json(path / "budget_stop.json", {**budget, "use_parent_policy": False})


def archive(directory, tag):
    core.atomic_json(directory / f"discarded/{tag}/policy/grpo_stats.jsonl", {"step": 105})
    core.atomic_json(directory / "waivers/old-event.json", {
        "schema": "selection-switch-cost-waiver/v1", "discarded_to": f"discarded/{tag}",
        "discarded_outputs": ["policy"], "round": 1,
        "attempt": {"event_id": "old-event", "phase": "train", "exit_code": 1,
                    "fault": {"kind": "gpu-fault"}},
    })


@pytest.fixture
def six_suites(tmp_path, monkeypatch):
    now = 1900000000.
    for status in (switch_status, mopps_status, combined_status.switch_status, combined_status.mopps_status):
        monkeypatch.setattr(status.node_view, "local_gpus", lambda: {
            "host": "test-node", "available": False, "gpus": [], "processes": []})
    roots = {name: tmp_path / "runs" / name for name in (
        "mopps-comparison-v1", "selection-switch-difficulty-v1", "selection-switch-long-v1",
        "selection-switch-mbpp-quality-v1", "selection-switch-mbpp-v1", "selection-switch-v1")}
    for name, root in roots.items():
        if name.startswith("mopps"):
            continue
        core.atomic_json(root / "switch.json", {"schema": rule.SCHEMA,
                         "gate": "convergence" if "quality" in name else "final"})
        for seed in range(5):
            for step in (25, 50, 100):
                core.atomic_json(root / f"prefixes/seed-{seed}/prefix-{step}.json", {"schema": rule.SCHEMA})
    for name in ("selection-switch-difficulty-v1", "selection-switch-mbpp-v1", "selection-switch-v1"):
        root = roots[name]
        for seed in range(5):
            for step in (25, 50, 100):
                result(branch(root, seed, step), step + 99)
                if seed >= 3:
                    result(branch(root, seed, step, "random_full"), step + 100)
    long = roots["selection-switch-long-v1"]
    for seed, step in ((0, 100), (0, 25), (0, 50), (1, 100), (1, 25),
                       (2, 100), (2, 25), (2, 50), (3, 50)):
        result(branch(long, seed, step), step + 250)
    for seed, step, saved in ((3, 25, 328), (3, 50, 360)):
        policy(branch(long, seed, step, "random_full"), saved, step)
    for seed, step in ((1, 50), (3, 25)):
        policy(branch(long, seed, step), step + 100, step, checkpoint=True)
    quality = roots["selection-switch-mbpp-quality-v1"]
    for index, (seed, step) in enumerate(((0, 100), (0, 25), (0, 50), (1, 25))):
        directory = branch(quality, seed, step)
        policy(directory, step + 100, step, checkpoint=True)
        core.atomic_json(directory / "progress.json", {"state": "running", "phase": "train",
                         "host": f"live-node-{index}", "pid": 5000 + index, "updated": now - 2,
                         "seconds": 123, "timeout": 7000, "event_id": f"live-{index}"})
    mopps = roots["mopps-comparison-v1"]
    core.atomic_json(mopps / "mopps.json", {"seeds": [3, 4], "steps": [25, 50, 100],
                     "arms": ["mopps", "random_online"], "parent": str(roots["selection-switch-v1"])})
    for seed in (3, 4):
        for step in (25, 50, 100):
            result(mopps / f"states/s{seed}-t{step}/random_online", step + 100, "mopps-comparison/v1")
    archived = []
    for name, seed, step, tag in (
        ("selection-switch-difficulty-v1", 0, 25, "20260916T235759Z"),
        ("selection-switch-difficulty-v1", 1, 100, "20260917T023352Z"),
        ("selection-switch-difficulty-v1", 3, 25, "20260917T023352Z"),
        ("selection-switch-long-v1", 1, 50, "20260917T023355Z"),
        ("selection-switch-v1", 3, 100, "20260916T141306Z"),
        ("selection-switch-v1", 4, 25, "20260916T141306Z"),
    ):
        directory = branch(roots[name], seed, step)
        archive(directory, tag)
        archived.append(directory)
    return roots, archived, now


def all_tasks(roots, now):
    return {name: (mopps_status if name.startswith("mopps") else switch_status).snapshot(root, now=now)["tasks"]
            for name, root in roots.items()}


def test_six_suite_snapshot_keeps_all_78_published_random_results(six_suites):
    roots, _, now = six_suites
    tasks = all_tasks(roots, now)
    counts = {name: sum(t["status"] == "DONE" and t.get("arm") in {
        "random_full", "random_reduced", "random_online"} for t in rows) for name, rows in tasks.items()}
    assert counts == {"mopps-comparison-v1": 6, "selection-switch-difficulty-v1": 21,
                      "selection-switch-long-v1": 9, "selection-switch-mbpp-quality-v1": 0,
                      "selection-switch-mbpp-v1": 21, "selection-switch-v1": 21}
    assert sum(counts.values()) == 78


def test_long_completed_training_is_eval_and_checkpoints_are_resume(six_suites):
    roots, _, now = six_suites
    rows = switch_status.snapshot(roots["selection-switch-long-v1"], now=now)["tasks"]
    lookup = {(t["seed"], t["step"], t["arm"]): t for t in rows}
    for seed, step in ((3, 25), (3, 50)):
        task = lookup[seed, step, "random_full"]
        assert task["status"] == "EVAL", task
        assert task["retryable"]
    for seed, step in ((1, 50), (3, 25)):
        task = lookup[seed, step, "random_reduced"]
        assert task["status"] == "RESUME", task
        assert task["retryable"]


def test_quality_four_current_workers_are_not_downgraded_to_ready_or_resume(six_suites):
    roots, _, now = six_suites
    data = switch_status.snapshot(roots["selection-switch-mbpp-quality-v1"], now=now)
    random = [t for t in data["tasks"] if t.get("arm") == "random_reduced" and t.get("host")]
    assert len(random) == 4
    assert {t["status"] for t in random} == {"RUNNING"}
    assert all(not t["retryable"] for t in random)


def test_archive_history_does_not_erase_active_done_or_resume_state(six_suites):
    roots, archived, now = six_suites
    tasks = all_tasks(roots, now)
    observed = {}
    for name, rows in tasks.items():
        for task in rows:
            path = roots[name] / task["directory"]
            if path in archived:
                observed[path] = task["status"]
    assert Counter(observed.values()) == {"DONE": 5, "RESUME": 1}, observed


def test_status_and_combined_sibling_inventory_never_change_saved_files(six_suites):
    roots, _, now = six_suites
    runs = next(iter(roots.values())).parent
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in runs.rglob("*") if p.is_file()}
    all_tasks(roots, now)
    combined = combined_status.snapshot(roots["selection-switch-v1"], roots["mopps-comparison-v1"], now=now)
    siblings = {Path(item["root"]).name: item for item in combined["other_experiments"]}
    assert set(siblings) == {"selection-switch-difficulty-v1", "selection-switch-long-v1",
                             "selection-switch-mbpp-quality-v1", "selection-switch-mbpp-v1"}
    rendered = combined_status.render(combined)
    assert "RANDOM CONTROLS" in rendered and "on-policy" in rendered
    assert "RF DONE 6/6" in rendered and "RR DONE 15/15" in rendered
    assert "HISTORY" in rendered
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in runs.rglob("*") if p.is_file()}
