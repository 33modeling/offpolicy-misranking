"""Read-only storage preflight must never turn lost work into a fresh run."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/mbpp_storage_audit.py"


@pytest.mark.parametrize("automatic", [False, True])
def test_default_storage_check_separates_run_targets_from_retained_history(tmp_path, automatic):
    import os
    import shutil

    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("check_mbpp_storage.sh", "_mbpp_experiments.sh"):
        shutil.copy2(SCRIPT.parent / name, scripts / name)
    (scripts / "mbpp_storage_audit.py").write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    work = tmp_path / "work"
    names = ("selection-switch-mbpp-quality-v1", "selection-switch-mbpp-v1",
             "selection-switch-mbpp-difficulty-v1", "selection-switch-mbpp-long-v1")
    for name in names:
        (work / "runs" / name).mkdir(parents=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith("SWITCH_")}
    env.update(OM_WORK=str(work), SWITCH_PYTHON=sys.executable,
               MBPP_STORAGE_AUDIT_AUTOMATIC="1" if automatic else "0")
    result = subprocess.run(["bash", "scripts/check_mbpp_storage.sh", "all"], cwd=repo,
                            env=env, capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    roots = [Path(args[i + 1]).name for i, arg in enumerate(args) if arg == "--root"]
    assert roots == list(names[:2] if automatic else names)
    assert ("--report-on-error" in args) == automatic
    assert ("--allow-branch-quarantine" in args) == automatic


@pytest.fixture
def auditor():
    spec = importlib.util.spec_from_file_location("mbpp_storage_audit_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def storage(tmp_path):
    work = tmp_path / "group-volume" / "work"
    root = work / "runs" / "selection-switch-mbpp-v1"
    root.mkdir(parents=True)
    (root / "switch.json").write_text(json.dumps({"dataset": "mbpp", "gate": "final"}))
    return work, root


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def branch(root):
    directory = root / "states/s3-t25/points/view-25/random_full"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def complete_checkpoint(directory, step=30):
    checkpoint = directory / "policy" / f"checkpoint-{step:06d}"
    checkpoint.mkdir(parents=True)
    write_json(checkpoint / "adapter_config.json", {"peft_type": "LORA"})
    (checkpoint / "adapter_model.safetensors").write_bytes(b"fake model payload")
    (checkpoint / "optimizer.pt").write_bytes(b"fake optimizer payload")
    (checkpoint / "grpo_stats.jsonl").write_text(json.dumps({"step": step}) + "\n")
    write_json(checkpoint / "checkpoint_state.json", {
        "completed_steps": step,
        "adapter_sha256": hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "optimizer_sha256": hashlib.sha256((checkpoint / "optimizer.pt").read_bytes()).hexdigest(),
        "grpo_stats_sha256": hashlib.sha256((checkpoint / "grpo_stats.jsonl").read_bytes()).hexdigest(),
    })
    return checkpoint


def sealed_result(directory):
    write_json(directory / "result.json", {"complete": True, "completed_steps": 30, "action": "random"})
    write_json(directory / "result.sha256.json", {
        "sha256": hashlib.sha256((directory / "result.json").read_bytes()).hexdigest(),
    })


def assert_blocked(result):
    assert result["status"] in {"blocked", "missing"}, result
    assert any(item["severity"] == "error" for item in result["findings"]), result


def test_missing_work_is_not_initialized_or_assumed_deleted(auditor, tmp_path):
    work = tmp_path / "missing-volume"
    result = auditor.audit(work, [work / "runs" / "selection-switch-mbpp-v1"])
    assert_blocked(result)
    assert not work.exists()
    assert any("missing" in str(item).lower() for item in result["findings"])


def test_existing_work_without_runs_is_not_treated_as_new_suite(auditor, tmp_path):
    work = tmp_path / "wrong-volume"
    work.mkdir()
    assert_blocked(auditor.audit(work, [work / "runs" / "selection-switch-mbpp-v1"]))
    assert not (work / "runs").exists()


def test_new_suite_can_start_only_on_existing_work_runs(auditor, storage):
    work, root = storage
    absent = work / "runs" / "selection-switch-mbpp-quality-v1"
    result = auditor.audit(work, [root, absent])
    assert result["status"] == "ok", result
    assert not absent.exists()
    assert absent.name in json.dumps(result)


def test_all_existing_suite_roots_missing_does_not_silently_initialize(auditor, storage):
    work, _ = storage
    assert_blocked(auditor.audit(work, [work / 'runs/absent-mbpp-root']))


def test_old_done_in_log_but_missing_result_blocks_rnd_retraining(auditor, storage):
    work, root = storage
    log = root / 'logs/launcher.old-node.log'
    log.parent.mkdir()
    log.write_text('s3/t25      random_full          DONE       reward=0.2 updates=100\n')
    result = auditor.audit(work, [root])
    assert_blocked(result)
    assert any(item['code'] == 'PREVIOUS_DONE_MISSING' for item in result['findings'])


def test_successful_training_cost_without_policy_blocks_rnd_retraining(auditor, storage):
    work, root = storage
    directory = branch(root)
    (directory / 'cost.jsonl').write_text(json.dumps({'phase': 'train', 'state': 'finished', 'exit_code': 0}) + '\n')
    result = auditor.audit(work, [root])
    assert_blocked(result)
    assert any(item['code'] == 'TRAIN_COMPLETION_MISSING' for item in result['findings'])


def test_live_peer_without_first_checkpoint_is_not_misdiagnosed_as_loss(auditor, storage):
    import fcntl
    work, root = storage
    directory = branch(root)
    policy = directory / 'policy'
    policy.mkdir()
    (policy / 'grpo_stats.jsonl').write_text('{"step":26}\n')
    lock = directory / '.task.lock'
    with lock.open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = auditor.audit(work, [root])
        assert result['status'] == 'ok', result
        assert any(item['code'] == 'ACTIVE_WORKER' for item in result['findings'])
    assert_blocked(auditor.audit(work, [root]))


def test_missing_root_manifest_with_saved_work_blocks(auditor, storage):
    work, root = storage
    complete_checkpoint(branch(root))
    (root / "switch.json").unlink()
    assert_blocked(auditor.audit(work, [root]))


def test_result_seal_without_result_blocks_accidental_rerun(auditor, storage):
    work, root = storage
    directory = branch(root)
    write_json(directory / "result.sha256.json", {"sha256": "0" * 64})
    assert_blocked(auditor.audit(work, [root]))


def test_discarded_completed_result_without_active_result_blocks(auditor, storage):
    work, root = storage
    directory = branch(root)
    sealed_result(directory / "discarded/20260918T213631Z")
    result = auditor.audit(work, [root])
    assert_blocked(result)
    assert "discarded" in json.dumps(result)


def test_sealed_active_result_is_preserved_as_complete(auditor, storage):
    work, root = storage
    directory = branch(root)
    sealed_result(directory)
    result = auditor.audit(work, [root])
    assert result["status"] == "ok", result


def test_mismatched_result_seal_blocks(auditor, storage):
    work, root = storage
    directory = branch(root)
    sealed_result(directory)
    write_json(directory / "result.json", {"complete": True, "completed_steps": 999})
    assert_blocked(auditor.audit(work, [root]))


def test_malformed_result_receipt_is_not_assumed_complete(auditor, storage):
    work, root = storage
    directory = branch(root)
    sealed_result(directory)
    (directory / "result.sha256.json").write_text("{interrupted")
    assert_blocked(auditor.audit(work, [root]))


def test_training_stats_without_recoverable_policy_blocks(auditor, storage):
    work, root = storage
    directory = branch(root)
    write_json(directory / "policy/grpo_stats.jsonl", {"step": 30})
    assert_blocked(auditor.audit(work, [root]))


def test_checkpoint_missing_optimizer_is_not_recoverable(auditor, storage):
    work, root = storage
    checkpoint = complete_checkpoint(branch(root))
    (checkpoint / "optimizer.pt").unlink()
    assert_blocked(auditor.audit(work, [root]))


def test_complete_checkpoint_presence_allows_training_resume(auditor, storage):
    work, root = storage
    directory = branch(root)
    complete_checkpoint(directory)
    write_json(directory / "policy/grpo_stats.jsonl", {"step": 32})
    result = auditor.audit(work, [root])
    assert result["status"] == "ok", result


def test_final_policy_survives_normal_intermediate_checkpoint_pruning(auditor, storage):
    work, root = storage
    directory = branch(root)
    policy = directory / "policy"
    policy.mkdir()
    write_json(policy / "policy_train.json", {"completed_steps": 30})
    write_json(policy / "adapter_config.json", {"peft_type": "LORA"})
    write_json(policy / "grpo_stats.jsonl", {"step": 30})
    (policy / "adapter_model.safetensors").write_bytes(b"saved final model")
    (policy / "optimizer.pt").write_bytes(b"saved final optimizer")
    sealed_result(directory)
    result = auditor.audit(work, [root])
    assert result["status"] == "ok", result


def test_final_policy_missing_optimizer_blocks_retraining(auditor, storage):
    work, root = storage
    directory = branch(root)
    policy = directory / "policy"
    policy.mkdir()
    write_json(policy / "policy_train.json", {"completed_steps": 30})
    (policy / "adapter_model.safetensors").write_bytes(b"saved final model")
    sealed_result(directory)
    assert_blocked(auditor.audit(work, [root]))


def test_empty_checkpoint_payload_is_not_recoverable(auditor, storage):
    work, root = storage
    checkpoint = complete_checkpoint(branch(root))
    (checkpoint / "adapter_model.safetensors").write_bytes(b"")
    assert_blocked(auditor.audit(work, [root]))


@pytest.mark.parametrize("state", [None, {"completed_steps": "not-a-step"}])
def test_invalid_checkpoint_state_is_not_recoverable(auditor, storage, state):
    work, root = storage
    checkpoint = complete_checkpoint(branch(root))
    write_json(checkpoint / "checkpoint_state.json", state)
    assert_blocked(auditor.audit(work, [root]))


def test_incomplete_newer_checkpoint_keeps_complete_older_resume(auditor, storage):
    work, root = storage
    directory = branch(root)
    complete_checkpoint(directory, step=30)
    newer = complete_checkpoint(directory, step=35)
    (newer / "optimizer.pt").unlink()
    result = auditor.audit(work, [root])
    assert result["status"] == "ok", result
    assert any(item["severity"] == "warning" for item in result["findings"]), result


def test_audit_never_reads_model_optimizer_or_rollout_payloads(auditor, storage, monkeypatch):
    work, root = storage
    directory = branch(root)
    checkpoint = complete_checkpoint(directory)
    rollout = directory / "fresh-r/candidate-0.jsonl"
    rollout.parent.mkdir()
    rollout.write_text("private potentially large rollout content\n")
    forbidden = {checkpoint / "adapter_model.safetensors", checkpoint / "optimizer.pt", rollout}
    original_open = Path.open

    def checked_open(path, *args, **kwargs):
        assert path not in forbidden, f"Audit read a payload rather than its metadata: {path}"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked_open)
    result = auditor.audit(work, [root])
    assert result["status"] == "ok", result


def test_inventory_does_not_change_any_file_or_directory(auditor, storage):
    work, root = storage
    directory = branch(root)
    complete_checkpoint(directory)
    sealed_result(directory)
    sealed_result(directory / "discarded/old-attempt")
    write_json(directory / "progress.json", {"state": "finished"})
    write_json(directory / "failure.json", {"error": "historical"})

    def inventory():
        return {
            str(path.relative_to(work)): (
                "file", path.read_bytes(), path.stat().st_mtime_ns
            ) if path.is_file() else ("directory", path.stat().st_mtime_ns)
            for path in work.rglob("*")
        }

    before = inventory()
    auditor.audit(work, [root])
    assert before == inventory()


@pytest.mark.parametrize("unsafe", [False, True])
def test_cli_exit_code_and_small_output_match_preflight_safety(storage, unsafe, tmp_path):
    work, root = storage
    for index in range(100):
        directory = root / f"states/s{index}-t25/points/view-25/random_full"
        directory.mkdir(parents=True)
        if unsafe:
            write_json(directory / "result.sha256.json", {"sha256": "0" * 64})
        else:
            sealed_result(directory)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--work", str(work), "--root", str(root),
         "--report-dir", str(tmp_path)],
        text=True, capture_output=True, check=False, timeout=10,
    )
    assert result.returncode == (2 if unsafe else 0), result.stdout + result.stderr
    assert len((result.stdout + result.stderr).encode()) <= 8192
    assert result.stdout.strip()
    reports = list(tmp_path.glob('mbpp-storage-*.txt'))
    assert len(reports) == 1
    assert reports[0].stat().st_size <= 4096
    assert reports[0].read_text() in result.stdout
    assert f'[saved] {reports[0]}' in result.stdout


@pytest.mark.parametrize('automatic', [False, True])
def test_missing_volume_still_saves_report_in_home(tmp_path, monkeypatch, automatic):
    monkeypatch.setenv('HOME', str(tmp_path))
    missing = tmp_path / 'unmounted-volume'
    command = [sys.executable, str(SCRIPT), '--work', str(missing), '--root', str(missing / 'runs/root')]
    if automatic:
        command.append('--report-on-error')
    result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    assert result.returncode == 2
    assert not missing.exists()
    report, = tmp_path.glob('mbpp-storage-*.txt')
    assert 'STORAGE_UNAVAILABLE' in report.read_text()
    assert str(report) in result.stdout


def test_successful_automatic_preflight_does_not_accumulate_files(storage, tmp_path):
    work, root = storage
    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--work', str(work), '--root', str(root),
         '--report-dir', str(tmp_path), '--report-on-error'],
        text=True, capture_output=True, check=False, timeout=10)
    assert result.returncode == 0
    assert not list(tmp_path.glob('mbpp-storage-*.txt'))


def test_report_save_failure_is_explicit(storage, tmp_path):
    work, root = storage
    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--work', str(work), '--root', str(root),
         '--report-dir', str(tmp_path / 'missing')],
        text=True, capture_output=True, check=False, timeout=10)
    assert result.returncode == 2
    assert '[report-save-failed]' in result.stderr
    assert '[saved]' not in result.stdout


@pytest.mark.parametrize('active_checkpoint', [False, True])
def test_archived_training_without_result_blocks_retraining(auditor, storage, active_checkpoint):
    work, root = storage
    directory = branch(root)
    archived = directory / 'discarded/old-auto-waiver/policy'
    write_json(archived / 'grpo_stats.jsonl', {'step': 30})
    write_json(directory / 'waivers/failed-train.json', {'discarded_outputs': ['policy']})
    if active_checkpoint:
        complete_checkpoint(directory)
    report = auditor.audit(work, [root])
    assert report['status'] == 'blocked'
    finding, = (item for item in report['findings'] if item['code'] == 'ARCHIVED_TRAINING')
    assert finding['path'] == str(archived)
    assert (archived / 'grpo_stats.jsonl').is_file()


def test_published_replay_and_archive_are_reported_without_choosing_one(auditor, storage):
    work, root = storage
    directory = branch(root)
    sealed_result(directory)
    write_json(directory / 'discarded/old-policy/policy/grpo_stats.jsonl', {'step': 30})
    report = auditor.audit(work, [root])
    assert any(item['code'] == 'ACTIVE_AND_ARCHIVED_WORK' for item in report['findings'])
    assert (directory / 'result.json').is_file()
    assert (directory / 'discarded/old-policy/policy/grpo_stats.jsonl').is_file()


def test_missing_stop_with_saved_final_can_reach_worker_hash_validated_repair(auditor, storage):
    work, root = storage
    directory = branch(root)
    policy = directory / 'policy'
    write_json(policy / 'policy_train.json', {'completed_steps': 30})
    (policy / 'adapter_model.safetensors').write_bytes(b'adapter')
    (policy / 'optimizer.pt').write_bytes(b'optimizer')
    write_json(directory / 'result.json', {'complete': True, 'artifact_hashes': {
        'random_full/policy/budget_stop.json': '0' * 64}})
    write_json(directory / 'result.sha256.json', {
        'sha256': hashlib.sha256((directory / 'result.json').read_bytes()).hexdigest()})
    report = auditor.audit(work, [root])
    assert report['status'] == 'ok'
    assert any(item['code'] == 'FINAL_STOP_REPAIR_CANDIDATE' for item in report['findings'])
    assert not (policy / 'budget_stop.json').exists()  # Only the worker may repair after full validation.


def test_parent_only_stop_conflicting_with_saved_training_blocks_startup(auditor, storage):
    work, root = storage
    directory = branch(root)
    complete_checkpoint(directory)
    write_json(directory / 'policy/budget_stop.json', {'use_parent_policy': True, 'completed_steps': 25})
    report = auditor.audit(work, [root])
    assert_blocked(report)
    assert any(item['code'] == 'PARENT_STOP_WITH_SAVED_POLICY' for item in report['findings'])


@pytest.mark.parametrize('saved', ['fresh', 'checkpoint', 'final', 'stop'])
def test_resume_guard_does_not_quarantine_fresh_or_recoverable_work(auditor, storage, saved):
    _, root = storage
    directory = branch(root)
    if saved == 'checkpoint':
        complete_checkpoint(directory)
        write_json(directory / 'policy/grpo_stats.jsonl', {'step': 32})
    elif saved == 'final':
        write_json(directory / 'policy/policy_train.json', {'completed_steps': 30})
    elif saved == 'stop':
        write_json(directory / 'policy/budget_stop.json', {'use_parent_policy': True})
    # Final/stop validation belongs to the pre-existing audit/worker checks.
    assert auditor.policy_resume_blocked(directory) is False


@pytest.mark.parametrize('saved', ['stats', 'adapter', 'optimizer', 'partial-checkpoint', 'temporary-checkpoint'])
def test_resume_guard_and_audit_use_same_missing_checkpoint_rule(auditor, storage, saved):
    work, root = storage
    directory = branch(root)
    policy = directory / 'policy'
    if saved == 'stats':
        write_json(policy / 'grpo_stats.jsonl', {'step': 28})
    elif saved in {'adapter', 'optimizer'}:
        policy.mkdir()
        (policy / ('adapter_model.safetensors' if saved == 'adapter' else 'optimizer.pt')).write_bytes(b'saved')
    elif saved == 'partial-checkpoint':
        checkpoint = complete_checkpoint(directory)
        (checkpoint / 'optimizer.pt').unlink()
    else:
        (policy / '.checkpoint-000030.tmp').mkdir(parents=True)
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert auditor.policy_resume_blocked(directory) is True
    report = auditor.audit(work, [root])
    assert {item['code'] for item in report['findings'] if item['severity'] == 'error'} == {'CHECKPOINT_MISSING'}
    assert auditor.quarantinable_branches(report) == [str(directory)]
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_complete_older_checkpoint_keeps_partial_newer_checkpoint_resumable(auditor, storage):
    _, root = storage
    directory = branch(root)
    complete_checkpoint(directory, 30)
    (directory / 'policy/checkpoint-000035').mkdir()
    assert auditor.policy_resume_blocked(directory) is False


@pytest.mark.parametrize('relative', [
    'prefixes/seed-3/segment-25/fresh_r',
    'states/s8-t25/points/view-25/random_full',
    'states/s3-t30/points/view-30/random_full',
    'states/s3-t25/points/view-50/random_full',
    'states/s0-t25/points/view-25/random_full',
])
def test_missing_checkpoint_outside_registered_continuation_is_not_quarantinable(auditor, storage, relative):
    work, root = storage
    directory = root / relative
    write_json(directory / 'policy/grpo_stats.jsonl', {'step': 28})
    report = auditor.audit(work, [root])
    assert_blocked(report)
    assert auditor.quarantinable_branches(report) == []


def test_quarantine_rejects_symlinked_branch_even_inside_root(auditor, storage):
    work, root = storage
    directory = branch(root)
    original = root / 'saved-policy'
    write_json(original / 'grpo_stats.jsonl', {'step': 28})
    (directory / 'policy').symlink_to(original, target_is_directory=True)
    report = auditor.audit(work, [root])
    assert_blocked(report)
    assert auditor.quarantinable_branches(report) == []


@pytest.mark.parametrize('extra', ['orphan-result', 'wrong-dataset', 'missing-root-manifest', 'successful-train'])
def test_other_error_never_acquires_branch_quarantine_exception(auditor, storage, extra):
    work, root = storage
    directory = branch(root)
    write_json(directory / 'policy/grpo_stats.jsonl', {'step': 28})
    if extra == 'orphan-result':
        write_json(directory / 'result.sha256.json', {'sha256': '0' * 64})
    elif extra == 'wrong-dataset':
        write_json(root / 'switch.json', {'dataset': 'math500'})
    elif extra == 'missing-root-manifest':
        (root / 'switch.json').unlink()
    else:
        write_json(directory / 'cost.jsonl', {'phase': 'train', 'state': 'finished', 'exit_code': 0})
    report = auditor.audit(work, [root])
    assert_blocked(report)
    assert auditor.quarantinable_branches(report) == []


@pytest.mark.parametrize('allow', [False, True])
def test_cli_only_guarded_partial_start_preserves_errors_and_saved_work(storage, tmp_path, allow):
    work, root = storage
    for relative in ('states/s3-t25/points/view-25/selection_full',
                     'states/s4-t100/points/view-100/random_reduced'):
        write_json(root / relative / 'policy/grpo_stats.jsonl', {'step': 28})
    before = {str(p): p.read_bytes() for p in work.rglob('*') if p.is_file()}
    command = [sys.executable, str(SCRIPT), '--work', str(work), '--root', str(root),
               '--report-dir', str(tmp_path), '--report-on-error']
    if allow:
        command.append('--allow-branch-quarantine')
    result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    assert result.returncode == (0 if allow else 2), result.stdout + result.stderr
    assert 'MBPP STORAGE AUDIT: BLOCKED' in result.stdout
    assert result.stdout.count('ERROR CHECKPOINT_MISSING') == 2
    assert ('[branch-quarantine] 2 continuation(s) remain BLOCKED' in result.stdout) == allow
    assert ('PARTIAL START ONLY' in result.stdout) == allow
    report, = tmp_path.glob('mbpp-storage-*.txt')
    assert report.stat().st_size <= 4096
    assert before == {str(p): p.read_bytes() for p in work.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault', ['prefix', 'orphan-result', 'missing-volume'])
def test_cli_quarantine_option_never_suppresses_global_blockers(storage, tmp_path, fault):
    work, root = storage
    if fault == 'prefix':
        write_json(root / 'prefixes/seed-0/segment-25/fresh_r/policy/grpo_stats.jsonl', {'step': 3})
    elif fault == 'orphan-result':
        directory = branch(root)
        write_json(directory / 'policy/grpo_stats.jsonl', {'step': 28})
        write_json(directory / 'result.sha256.json', {'sha256': '0' * 64})
    else:
        work = tmp_path / 'unmounted-volume'
    result = subprocess.run([sys.executable, str(SCRIPT), '--work', str(work), '--root', str(root),
                             '--report-dir', str(tmp_path), '--allow-branch-quarantine'],
                            text=True, capture_output=True, check=False, timeout=10)
    assert result.returncode == 2, result.stdout + result.stderr
    assert '[branch-quarantine]' not in result.stdout
