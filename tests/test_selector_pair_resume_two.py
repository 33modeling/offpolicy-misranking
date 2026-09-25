"""The normal launcher resumes only the two approved saved final policies."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import selector_pair_resume_two as resume
import selector_pair_status as status
from test_selector_pair_finish_saved import experiment, fake_gpu, snapshot


@pytest.fixture
def ready(tmp_path):
    root = tmp_path / "selector-pair-v1"
    p = {"schema": "offpolicy-selector-pair/v1", "branch_manifests": {}}
    for branch in ("on_policy", "cached", "adaptive-cached", "adaptive-on_policy"):
        path = root / "branches" / branch / "switch.json"
        resume.core.atomic_json(path, {"branch": branch})
        p["branch_manifests"][branch] = resume.digest(path)
    p["protocol_id"] = resume.core.fingerprint(p)
    resume.core.atomic_json(root / "pair.json", p)
    barrier = {"protocol_id": p["protocol_id"], "schema": "offpolicy-selector-pair/sr-gc-v1", "decisions": {}}
    count = 0
    for seed in range(5):
        for step in (25, 50, 100):
            arms = [(b, "selection_reduced") for b in ("on_policy", "cached")]
            if seed >= 3:
                key = f"s{seed}-t{step}"
                path = root / "sr-gc" / key / "decision.json"
                resume.core.atomic_json(path, {"selector": "cached"})
                barrier["decisions"][key] = resume.digest(path)
                arms = [(b, "selection_full") for b in ("on_policy", "cached", "adaptive-cached")]
                arms.append(("on_policy", "random_full"))
            for branch, arm in arms:
                path = resume.directory(root, seed, step, branch, arm)
                if branch == "on_policy" and resume.TARGETS.get(seed) == (step, arm):
                    resume.core.atomic_json(path / "policy/policy_train.json", {"completed_steps": 457 if seed == 1 else 256})
                    resume.core.atomic_json(path / "budget-recovery/plan.json", {
                        "runner_sha256": "obsolete", "points": [{"adapter": "missing-checkpoint"}]})
                    continue
                resume.core.atomic_json(path / "result.json", {"schema": resume.RESULT_SCHEMA, "complete": True})
                sha = resume.digest(path / "result.json")
                resume.core.atomic_json(path / "result.sha256.json", {"sha256": sha})
                resume.core.atomic_json(path / "curve.json", {
                    "schema": resume.RESULT_SCHEMA, "result_sha256": sha, "points": {"100": {"reward": .5}}})
                count += 1
    assert count == 40
    resume.core.atomic_json(root / "test-decisions.json", barrier)
    return root


def test_exact_two_are_eligible_without_rebinding_old_hashes(ready):
    before = snapshot(ready)
    assert resume.eligible(ready)
    assert snapshot(ready) == before


@pytest.mark.parametrize("change", ["another_pending", "missing_final", "missing_plan", "wrong_curve", "wrong_decision"])
def test_other_unfinished_or_changed_work_never_uses_two_only_route(ready, change):
    other = resume.directory(ready, 0, 25, "cached", "selection_reduced")
    target = resume.directory(ready, 1, 50, "on_policy", "selection_reduced")
    if change == "another_pending":
        (other / "result.json").unlink()
    elif change == "missing_final":
        (target / "policy/policy_train.json").unlink()
    elif change == "missing_plan":
        (target / "budget-recovery/plan.json").unlink()
    elif change == "wrong_curve":
        resume.core.atomic_json(other / "curve.json", {"points": {"x": .5}, "result_sha256": "wrong"})
    else:
        resume.core.atomic_json(ready / "sr-gc/s3-t25/decision.json", {"selector": "on_policy"})
        with pytest.raises(ValueError, match="decision changed"):
            resume.eligible(ready)
        return
    assert not resume.eligible(ready)


def test_missing_and_setup_root_fall_back_readonly(tmp_path):
    root = tmp_path / "new"
    assert not resume.eligible(root) and not root.exists()
    resume.core.atomic_json(root / "pair.json", {"schema": "offpolicy-selector-pair/setup-v1"})
    assert not resume.eligible(root)


def test_run_resumes_final_not_deleted_checkpoint_and_status_sees_completion(experiment, fake_gpu, monkeypatch):
    root, output, source, seed, c = experiment
    monkeypatch.setattr(resume, "TARGETS", {seed: (100, "random_full")})
    monkeypatch.setattr(status.pair, "DEV_SEEDS", ())
    monkeypatch.setattr(resume, "eligible", lambda _: True)
    monkeypatch.setenv("PAIR_FINAL_EVAL_ROOT", str(output))
    before = snapshot(root)
    assert resume.run(root, list("0123")) == 0
    assert resume.completed(root, seed)
    assert len(fake_gpu) == 12 and snapshot(root) == before
    task = status.observe_branch(root, seed, 100, "random", "on_policy", ready=True, observations=[])
    assert task["status"] == "DONE" and task["canonical_complete"] is False
    assert task["saved_final_evaluation"] and task["training_published"]
    assert not (source / "result.json").exists()
    assert resume.run(root, list("0123")) == 0 and len(fake_gpu) == 12


def test_one_busy_target_does_not_prevent_other_target(ready, monkeypatch):
    import selector_pair_finish_saved as finish
    calls = []
    def evaluate(root, output, seed, devices):
        calls.append(seed)
        if seed == 1:
            raise BlockingIOError("source lease held")
        return {"completed_steps": 256}
    monkeypatch.setattr(finish, "finish", evaluate)
    assert resume.run(ready, list("0123")) == 1
    assert calls == [1, 4]


def test_status_shows_new_evaluation_running(ready, monkeypatch):
    output = ready.parent / "new-evaluation"
    monkeypatch.setenv("PAIR_FINAL_EVAL_ROOT", str(output))
    resume.core.atomic_json(output / "seed-1/progress.json", {
        "state": "running", "updated": 100., "host": "gpu-node", "phase": "evaluate-1"})
    task = status.observe_branch(ready, 1, 50, "on_policy", "on_policy", ready=True, observations=[], now=101.)
    assert task["status"] == "RUN" and task["phase"] == "evaluate-1"
    assert task["host"] == "gpu-node"


@pytest.mark.parametrize("probe", [0, 3, 2])
def test_pinned_worker_dispatches_two_finals_or_ordinary_queue(tmp_path, probe):
    repo = Path(resume.__file__).parents[1]
    python = tmp_path / "python-probe"
    python.write_text('#!/bin/bash\n'
                      'if [[ "$2" == probe ]]; then exit "$PROBE"; fi\n'
                      'printf "%s\\n" "$@"\n')
    python.chmod(0o755)
    result = subprocess.run([
        "bash", "-c", 'set -euo pipefail; source "$1"; shift; selection_run_worker "$@"',
        "test", str(repo / "scripts/_selection_worker.sh"), str(python),
        "src/selector_pair_gpu.py", "run", "--root", str(tmp_path)], cwd=repo,
        env={**os.environ, "EXPERIMENTS_NODE_ID": "test-node", "PROBE": str(probe)},
        capture_output=True, text=True, timeout=20)
    assert result.returncode == (2 if probe == 2 else 0)
    if probe == 2:
        assert not result.stdout
    else:
        script = "selector_pair_resume_two.py" if probe == 0 else "queue_selector_pair_gpu.py"
        assert result.stdout.splitlines() == [arg + " [pair]" for arg in
            [str(repo / "scripts" / script), "run", "--root", str(tmp_path)]]


@pytest.mark.parametrize("probe,worker_exit", [(0, 0), (0, 17), (3, 0), (2, 0)])
def test_normal_shell_routes_two_without_old_hash_gate(tmp_path, probe, worker_exit):
    repo = tmp_path / "checkout"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    source = Path(resume.__file__).parent
    shutil.copyfile(source / "run_selector_pair.sh", scripts / "run_selector_pair.sh")
    (scripts / "_e5_node.sh").write_text(
        "e5_cleanup_lock_helpers() { return 91; }\n"
        "e5_recover_pair_gpu() { return 92; }\n"
        "e5_acquire_node() { e5_cleanup_lock_helpers; e5_recover_pair_gpu; }\n")
    (scripts / "_selection_worker.sh").write_text('selection_run_worker() { "$@"; }\n')
    binaries = tmp_path / "bin"
    binaries.mkdir()
    python = binaries / "python"
    python.write_text('#!/bin/bash\n'
                      'if [[ "$2" == probe ]]; then exit "$PROBE"; fi\n'
                      'if [[ "$2" == ensure-prepared ]]; then echo NORMAL_PREFLIGHT; exit 19; fi\n'
                      'if [[ "$1" == src/bootstrap_math_verify.py ]]; then echo /tmp/verifier; exit 0; fi\n'
                      'echo "WORKER $*"; exit "$WORKER_EXIT"\n')
    nvidia = binaries / "nvidia-smi"
    nvidia.write_text('#!/bin/bash\nprintf "0\\n0\\n0\\n0\\n"\n')
    for path in (python, nvidia):
        path.chmod(0o755)
    env = {**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}", "PAIR_PYTHON": str(python),
           "PAIR_ROOT": str(tmp_path / "data"), "OM_WORK": str(tmp_path / "work"),
           "CUDA_VISIBLE_DEVICES": "0,1,2,3", "PROBE": str(probe), "WORKER_EXIT": str(worker_exit)}
    process = subprocess.run(["bash", str(scripts / "run_selector_pair.sh")], env=env,
                             capture_output=True, text=True, timeout=20)
    if probe == 0:
        assert process.returncode == worker_exit, process.stdout + process.stderr
        assert "selector_pair_resume_two.py run --root" in process.stdout
        assert "NORMAL_PREFLIGHT" not in process.stdout
    elif probe == 3:
        assert process.returncode == worker_exit and "frozen-run --root" in process.stdout
    else:
        assert process.returncode == 2 and "WORKER" not in process.stdout
