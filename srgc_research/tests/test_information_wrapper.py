"""Exercise the public shell command without launching GPU work or reports."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/run_srgc_information.sh"


@pytest.fixture
def wrapper(tmp_path):
    work = tmp_path / "work with spaces"
    source = work / "srgc-rebuttal"
    (source / "experiments").mkdir(parents=True)
    (source / "inputs").mkdir()
    for filename, dataset in (("pair_seeds.json", "math"), ("mbpp_pair_seeds.json", "mbpp")):
        plan = json.loads((REPO / "srgc_rebuttal/experiments" / filename).read_text())
        plan["input_pattern"] = f"../inputs/{dataset}-custom-{{seed}}.json"
        (source / "experiments" / filename).write_text(json.dumps(plan))
        for seed in (5, 6, 7, 8, 9):
            (source / "inputs" / f"{dataset}-custom-{seed}.json").write_text("{}")
    binary = tmp_path / "bin"
    binary.mkdir()
    logger = binary / "runner"
    logger.write_text('''#!/usr/bin/python3
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
if sys.argv[1:3] == ["-m", "srgc_research.dispatch.information_queue"]:
    from functools import partial
    from srgc_research.dispatch import information_queue as queue
    queue.node_available = lambda: True
    queue.read_measurement = lambda folder: (queue.read_object(folder / "endpoint.json"), {})
    queue.sweep = partial(queue.sweep, retry_delay=0)
    sys.exit(queue.main(sys.argv[3:]))
if sys.argv[1] != "-m" or sys.argv[2] not in {"srgc_research.information_cli", "srgc_research.dispatch.information_run"}:
    os.execv("/usr/bin/python3", ["/usr/bin/python3", *sys.argv[1:]])
with Path(os.environ["WRAPPER_TEST_LOG"]).open("a") as handle:
    handle.write(json.dumps({"runner": Path(sys.argv[0]).name, "args": sys.argv[1:],
                            "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")
code = int(os.environ.get("WRAPPER_TEST_COLLECT_EXIT", "0")) if sys.argv[4] == "collect" else 0
if code:
    print("ERROR: " + os.environ.get("WRAPPER_TEST_ERROR", "test collector failure"), file=sys.stderr, flush=True)
if sys.argv[4] == "collect" and "--plan" in sys.argv and code == 0:
    from srgc_research.dispatch.information_queue import Task, PHASES, digest, atomic_json
    arg = lambda flag: sys.argv[sys.argv.index(flag) + 1]
    output, plan, inputs = Path(arg("--output")), Path(arg("--plan")), Path(arg("--inputs"))
    task = Task(sys.argv[3], int(arg("--seed")), plan, inputs, output.parents[2], digest(plan), digest(inputs))
    atomic_json(output / "manifest.json", {"identity": task.identity})
    (output / "plan.json").write_bytes(plan.read_bytes())
    (output / "inputs.json").write_bytes(inputs.read_bytes())
    for phase in PHASES:
        atomic_json(output / f"{phase}.json", {"artifacts": []})
    atomic_json(output / "endpoint.json", {"identity": task.identity,
        "phases": {phase: f"{phase}.json" for phase in PHASES}})
sys.exit(code)
''')
    logger.chmod(0o755)
    for name in ("python3", "pair-python", "switch-python"):
        (binary / name).symlink_to(logger)
    env = {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}", "OM_WORK": str(work),
           "GROUP_VOLUME": str(tmp_path),
           "SRGC_STORAGE_ROOT": str(source), "PAIR_PYTHON": str(binary / "pair-python"),
           "SWITCH_PYTHON": str(binary / "switch-python"), "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "WRAPPER_TEST_LOG": str(tmp_path / "calls.jsonl")}

    def run(*args, overrides=None):
        result = subprocess.run(["sh", str(SCRIPT), *args], cwd=tmp_path,
                                env={**env, **(overrides or {})}, capture_output=True, text=True,
                                timeout=20, check=False)
        log = Path(env["WRAPPER_TEST_LOG"])
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run, work, source


@pytest.mark.parametrize("args,dataset,runner", [((), "math", "pair-python"), (("math",), "math", "pair-python"),
                                                 (("mbpp",), "mbpp", "switch-python")])
def test_one_command_defaults_collect_only(wrapper, args, dataset, runner):
    run, work, source = wrapper
    result, calls = run(*args)
    assert result.returncode == 0, result.stderr
    assert "action must" not in result.stderr
    assert len(calls) == 5
    call = calls[0]
    assert call["runner"] == runner and call["args"][2:4] == [dataset, "collect"]
    args = call["args"]
    assert args[args.index("--inputs") + 1] == str(source / "inputs" / f"{dataset}-custom-7.json")
    assert args[args.index("--seed") + 1] == "7"
    assert args[args.index("--stage") + 1] == "0"
    assert args[args.index("--attention") + 1] == "sdpa"
    assert args[args.index("--output") + 1] == str(work / "selection-information" / dataset / "seed-7/t0")
    assert call["cuda"] == "0,1,2,3"
    assert not list(work.rglob("*.html"))


def test_all_dispatches_ten_independent_tasks_without_reports(wrapper):
    run, work, _ = wrapper
    result, calls = run("all")
    assert result.returncode == 0, result.stderr
    assert [c["args"][2:4] for c in calls] == [["math", "collect"], ["mbpp", "collect"]] * 5
    assert [c["runner"] for c in calls] == ["pair-python", "switch-python"] * 5
    assert [int(c["args"][c["args"].index("--seed") + 1]) for c in calls] == [7, 7, 5, 5, 6, 6, 8, 8, 9, 9]
    assert not list(work.rglob("*.html"))


def test_all_missing_second_input_never_starts_first_measurement(wrapper):
    run, _, source = wrapper
    (source / "inputs/mbpp-custom-7.json").unlink()
    result, calls = run("all")
    assert result.returncode != 0 and "input not found" in result.stderr
    assert calls == []


def test_missing_later_seed_is_found_before_first_gpu_job(wrapper):
    run, _, source = wrapper
    (source / "inputs/math-custom-9.json").unlink()
    result, calls = run("all")
    assert result.returncode == 2 and "math-custom-9" in result.stderr and calls == []


def test_collect_without_options_uses_shared_queue(wrapper):
    run, _, _ = wrapper
    result, calls = run("math", "collect")
    assert result.returncode == 0 and len(calls) == 5


def test_restart_skips_completed_tasks_without_another_collector_launch(wrapper):
    run, work, _ = wrapper
    assert run("all")[0].returncode == 0
    before = {p: p.read_bytes() for p in work.rglob("*.pt")}
    result, calls = run("all")
    assert result.returncode == 0 and len(calls) == 10
    assert {p: p.read_bytes() for p in work.rglob("*.pt")} == before


def test_missing_second_interpreter_is_found_before_first_gpu_job(wrapper):
    run, _, _ = wrapper
    result, calls = run("all", overrides={"SWITCH_PYTHON": "/missing/switch-python"})
    assert result.returncode == 2 and "SWITCH_PYTHON" in result.stderr and calls == []


def test_failed_collect_returns_failure_after_independent_tasks_are_attempted(wrapper):
    run, _, _ = wrapper
    result, calls = run("all", overrides={"WRAPPER_TEST_COLLECT_EXIT": "9"})
    assert result.returncode == 1
    assert len(calls) == 30 and all(c["args"][3] == "collect" for c in calls)
    assert "ERROR: test collector failure" in result.stderr and "log:" in result.stderr


def test_old_exhausted_receipts_resume_after_update_and_keep_prior_failures(wrapper):
    run, work, _ = wrapper
    result, calls = run("all", overrides={"WRAPPER_TEST_COLLECT_EXIT": "9"})
    assert result.returncode == 1 and len(calls) == 30
    receipts = list((work / "selection-information/.queue").glob("*.json"))
    previous = []
    for path in receipts:
        row = json.loads(path.read_text())
        row.pop("dispatch_revision")  # Receipts written by the old dispatcher.
        path.write_text(json.dumps(row))
        previous.append(row)
    result, calls = run("all")
    assert result.returncode == 0 and len(calls) == 40, result.stderr
    assert "RETRY" in result.stdout
    archived = [json.loads(p.read_text()) for p in (work / "selection-information/.queue/history").rglob("*.json")]
    assert sorted(archived, key=lambda r: r["task"]) == sorted(previous, key=lambda r: r["task"])
    assert len(list((work / "selection-information/.queue/logs").rglob("*.log"))) == 40
    assert not list(work.rglob("*.html"))


def test_current_exhausted_receipts_show_cause_without_infinite_retries(wrapper):
    run, work, _ = wrapper
    result, calls = run("all", overrides={"WRAPPER_TEST_COLLECT_EXIT": "9",
                                          "WRAPPER_TEST_ERROR": "candidate cache is incomplete"})
    assert result.returncode == 1 and len(calls) == 30
    before = {p: p.read_bytes() for p in (work / "selection-information/.queue/logs").rglob("*.log")}
    result, calls = run("all")
    assert result.returncode == 1 and len(calls) == 30
    assert "candidate cache is incomplete" in result.stderr and "FAILED math.seed-7.t0:" in result.stderr
    footer = result.stdout.split("INFORMATION FAILURE DETAILS\n")[-1]
    assert len(footer.strip().splitlines()) == 10
    assert footer.strip().endswith("mbpp.seed-9.t0: ERROR: candidate cache is incomplete")
    assert ".queue" not in footer and "log:" not in footer
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("dataset,action", [("math", "collect"), ("mbpp", "collect"),
                                            ("all", "status"), ("all", "report")])
def test_explicit_arguments_are_preserved(wrapper, dataset, action):
    run, _, _ = wrapper
    original = [dataset, action, "--output", "/tmp/a path with spaces"]
    result, calls = run(*original)
    assert result.returncode == 0, result.stderr
    assert calls[0]["args"][2:] == original
    assert calls[0]["cuda"] == ("0,1,2,3" if action == "collect" else "")


def test_missing_explicit_interpreter_does_not_fall_back(wrapper):
    run, _, _ = wrapper
    result, calls = run("math", overrides={"PAIR_PYTHON": "/not/a/python"})
    assert result.returncode == 2 and "Python not found" in result.stderr
    assert calls == []


def test_default_seed_must_exist_in_plan(wrapper):
    run, _, source = wrapper
    path = source / "experiments/pair_seeds.json"
    plan = json.loads(path.read_text())
    plan["seeds"] = [5]
    path.write_text(json.dumps(plan))
    result, calls = run("math")
    assert result.returncode == 2 and "seed 7" in result.stderr
    assert calls == []


@pytest.mark.parametrize("dataset,name", [("math", "pair_seeds.json"), ("math", "additional_seeds.json"),
                                         ("mbpp", "mbpp_pair_seeds.json"), ("mbpp", "mbpp_seeds.json")])
def test_active_fresh_plan_is_used_without_top_level_plan(wrapper, dataset, name):
    run, work, source = wrapper
    template = REPO / "srgc_rebuttal/experiments" / name
    plan_path = source / "fresh/candidate40-v2/experiments" / name
    plan_path.parent.mkdir(parents=True)
    plan_path.write_bytes(template.read_bytes())
    plan = json.loads(plan_path.read_text())
    input_path = (plan_path.parent / plan["input_pattern"].format(seed=7)).resolve()
    input_path.parent.mkdir(parents=True)
    for seed in (5, 6, 7, 8, 9):
        path = (plan_path.parent / plan["input_pattern"].format(seed=seed)).resolve()
        path.write_text("{}")
    pointer = source / f".{template.stem}-active.json"
    pointer.write_text(json.dumps({"plan": str(plan_path), "source_plan_sha256": hashlib.sha256(template.read_bytes()).hexdigest()}))
    # Reproduce the missing path from the user's node exactly.
    for path in (source / "experiments").glob("*.json"):
        path.unlink()
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    result, calls = run(dataset)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 5
    args = calls[0]["args"]
    assert args[args.index("--plan") + 1] == str(plan_path)
    assert args[args.index("--inputs") + 1] == str(input_path)
    assert {p: p.read_bytes() for p in source.rglob("*") if p.is_file()} == before
    assert not list(work.rglob("*.html"))


def test_broken_active_pointer_is_not_replaced_with_another_cohort(wrapper):
    run, _, source = wrapper
    pointer = source / ".pair_seeds-active.json"
    pointer.write_text(json.dumps({"plan": str(source / "missing/experiments/pair_seeds.json"),
                                   "source_plan_sha256": "wrong-source-hash"}))
    before = pointer.read_bytes()
    result, calls = run("math")
    assert result.returncode != 0 and "does not match" in result.stderr
    assert calls == [] and pointer.read_bytes() == before


def test_missing_active_input_is_not_replaced_with_staged_inputs(wrapper):
    run, _, source = wrapper
    template = REPO / "srgc_rebuttal/experiments/pair_seeds.json"
    target = source / "fresh/previous/experiments/pair_seeds.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(template.read_bytes())
    (source / ".pair_seeds-active.json").write_text(json.dumps({
        "plan": str(target), "source_plan_sha256": hashlib.sha256(template.read_bytes()).hexdigest()}))
    result, calls = run("math")
    assert result.returncode != 0 and "input not found" in result.stderr
    assert calls == []


def test_saved_previous_run_is_selected_without_changing_active_pointer(wrapper):
    run, _, source = wrapper
    template = REPO / "srgc_rebuttal/experiments/pair_seeds.json"
    paths = [source / f"fresh/{name}/experiments/pair_seeds.json" for name in ("old", "replacement")]
    for path in paths:
        path.parent.mkdir(parents=True)
        path.write_bytes(template.read_bytes())
    old, active = paths
    plan = json.loads(template.read_text())
    old_input = (old.parent / plan["input_pattern"].format(seed=7)).resolve()
    old_input.parent.mkdir(parents=True)
    old_input.write_text("{}")
    for seed in (5, 6, 7, 8, 9):
        path = (active.parent / plan["input_pattern"].format(seed=seed)).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    old_prefix = (old.parent / plan["output_root"] / "seed-7/prefix.pt").resolve()
    old_prefix.parent.mkdir(parents=True)
    old_prefix.write_bytes(b"old checkpoint must remain intact")
    (active.parent.parent / "automatic-restart.json").write_text(json.dumps({
        "reason": "implementation changed", "plan": str(active), "previous_plan": str(old)}))
    pointer = source / ".pair_seeds-active.json"
    pointer.write_text(json.dumps({"plan": str(active),
                                   "source_plan_sha256": hashlib.sha256(template.read_bytes()).hexdigest()}))
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    result, calls = run("math")
    assert result.returncode == 0, result.stderr
    args = calls[0]["args"]
    assert args[args.index("--plan") + 1] == str(old)
    assert args[args.index("--inputs") + 1] == str(old_input)
    assert {p: p.read_bytes() for p in source.rglob("*") if p.is_file()} == before


def test_repository_prepared_plan_is_found_when_group_plan_directory_is_absent(wrapper):
    run, _, source = wrapper
    for path in (source / "experiments").glob("*.json"):
        path.unlink()
    result, calls = run("math")
    assert result.returncode == 0, result.stderr
    args = calls[0]["args"]
    assert args[args.index("--plan") + 1] == str(REPO / "srgc_rebuttal/experiments/additional_seeds.json")
    assert args[args.index("--inputs") + 1] == str(REPO / "srgc_rebuttal/inputs/seed-7.json")


def test_active_plan_cannot_escape_group_volume(wrapper, tmp_path):
    run, _, source = wrapper
    template = REPO / "srgc_rebuttal/experiments/pair_seeds.json"
    (source / ".pair_seeds-active.json").write_text(json.dumps({
        "plan": str(tmp_path.parent / "outside-group/pair_seeds.json"),
        "source_plan_sha256": hashlib.sha256(template.read_bytes()).hexdigest()}))
    result, calls = run("math")
    assert result.returncode != 0 and "does not match" in result.stderr
    assert calls == []
