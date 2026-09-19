"""Legacy and direct MBPP launches must audit before any mutating launcher code."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_mbpp_storage_audit import sealed_result, write_json


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    # The real preflight prefix ends before every read-only/mutating dispatch.
    # Replace only that later dispatch with a harmless marker; no GPU command,
    # process cleanup, keepalive or worker can execute in this fixture.
    source = (ROOT / "scripts/run_selection_switch.sh").read_text()
    preflight = source.split('if [ "$MODE" = cpu ]; then', 1)[0]
    (scripts / "run_selection_switch.sh").write_text(preflight + '''
printf '%s\\n' "$SWITCH_ROOT" "$SWITCH_DATASET" "$MODE" "$@" > "$INNER_CAPTURE"
'''.replace('"$SWITCH_DATASET"', '"${SWITCH_DATASET:-}"'))
    shutil.copy2(ROOT / "scripts/run_switch_mbpp.sh", scripts / "run_switch_mbpp.sh")
    shutil.copy2(ROOT / "scripts/mbpp_storage_audit.py", scripts / "_real_audit.py")
    (scripts / "mbpp_storage_audit.py").write_text('''
import json, os, sys
from pathlib import Path
from _real_audit import main
Path(os.environ["AUDIT_CAPTURE"]).write_text(json.dumps({"args": sys.argv[1:], "cuda": os.environ.get("CUDA_VISIBLE_DEVICES")}))
sys.argv += ["--report-dir", os.environ["TEST_REPORT_DIR"]]
raise SystemExit(main())
''')
    work = tmp_path / "volume/work"
    root = work / "runs/selection-switch-mbpp-v1"
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SWITCH_", "EXPERIMENTS_", "OM_")) and key != "OUT_ROOT"}
    env.update(OM_WORK=str(work), SWITCH_ROOT=str(root), SWITCH_PYTHON=sys.executable,
               INNER_CAPTURE=str(tmp_path / "inner.log"), AUDIT_CAPTURE=str(tmp_path / "audit.json"),
               TEST_REPORT_DIR=str(report_dir), CUDA_VISIBLE_DEVICES="0,1,2,3")
    return repo, work, root, env


def invoke(fixture, entry, args=()):
    repo, _, _, env = fixture
    return subprocess.run(["bash", f"scripts/{entry}", *args], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=10, check=False)


@pytest.mark.parametrize("entry", ["run_switch_mbpp.sh", "run_selection_switch.sh"])
@pytest.mark.parametrize("mode", [(), ("smoke",), ("prepare",)])
@pytest.mark.parametrize("evidence", ["orphan-seal", "archived-result", "missing-volume"])
def test_blocked_storage_prevents_legacy_and_direct_worker_dispatch(launcher, entry, mode, evidence):
    _, work, root, env = launcher
    if evidence != "missing-volume":
        write_json(root / "switch.json", {"dataset": "mbpp"})
        directory = root / "states/s3-t25/points/view-25/random_full"
        if evidence == "orphan-seal":
            write_json(directory / "result.sha256.json", {"sha256": "old-sealed-completion"})
        else:
            sealed_result(directory / "discarded/old")
    before = {p: p.read_bytes() for p in work.rglob("*") if p.is_file()} if work.exists() else {}
    result = invoke(launcher, entry, mode)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "no cleanup, preparation or GPU work started" in result.stderr
    assert not Path(env["INNER_CAPTURE"]).exists()
    capture = json.loads(Path(env["AUDIT_CAPTURE"]).read_text())
    assert capture["cuda"] == "" and "--report-on-error" in capture["args"]
    assert capture["args"][capture["args"].index("--root") + 1] == str(root)
    assert before == ({p: p.read_bytes() for p in work.rglob("*") if p.is_file()} if work.exists() else {})
    if evidence == "missing-volume":
        assert not work.exists()


@pytest.mark.parametrize("entry", ["run_switch_mbpp.sh", "run_selection_switch.sh"])
@pytest.mark.parametrize("mode", ["status", "progress", "results", "why", "errors", "stop"])
def test_read_only_and_stop_modes_do_not_run_storage_preflight(launcher, entry, mode):
    result = invoke(launcher, entry, [mode])
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(launcher[3]["INNER_CAPTURE"]).exists()
    assert not Path(launcher[3]["AUDIT_CAPTURE"]).exists()


@pytest.mark.parametrize("entry", ["run_switch_mbpp.sh", "run_selection_switch.sh"])
def test_accepted_actual_root_and_arguments_are_not_rerouted(launcher, entry):
    _, work, _, env = launcher
    root = work / "runs/custom-saved-code-suite"
    env.update(SWITCH_ROOT=str(root), SWITCH_MBPP_ROOT=str(root), SWITCH_DATASET="mbpp")
    write_json(root / "switch.json", {"dataset": "mbpp"})
    sealed_result(root / "states/s3-t25/points/view-25/random_full")
    result = invoke(launcher, entry, ["prepare", "--eval-timeout", "9"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(env["INNER_CAPTURE"]).read_text().splitlines() == [str(root), "mbpp", "prepare", "--eval-timeout", "9"]
    assert not list(Path(env["TEST_REPORT_DIR"]).iterdir())


def test_custom_root_is_detected_from_frozen_mbpp_manifest_without_dataset_env(launcher):
    _, work, _, env = launcher
    root = work / "runs/custom-code-study"
    env["SWITCH_ROOT"] = str(root)
    write_json(root / "switch.json", {"dataset": "mbpp"})
    write_json(root / "states/s0-t25/points/view-25/random_reduced/result.sha256.json", {"sha256": "old"})
    result = invoke(launcher, "run_selection_switch.sh")
    assert result.returncode == 2 and not Path(env["INNER_CAPTURE"]).exists()


def test_non_mbpp_launch_behavior_is_unchanged(launcher):
    _, work, _, env = launcher
    root = work / "runs/selection-switch-v1"
    env["SWITCH_ROOT"] = str(root)
    write_json(root / "switch.json", {"dataset": "math500"})
    result = invoke(launcher, "run_selection_switch.sh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(env["INNER_CAPTURE"]).exists() and not Path(env["AUDIT_CAPTURE"]).exists()


@pytest.mark.parametrize("suite", ["quality", "difficulty", "long"])
def test_new_sibling_root_can_prepare_from_audited_existing_prefix_source(launcher, suite):
    _, work, parent, env = launcher
    target = work / f"runs/selection-switch-mbpp-{suite}-v1"
    write_json(parent / "switch.json", {"dataset": "mbpp"})
    sealed_result(parent / "states/s3-t25/points/view-25/random_full")
    env.update(SWITCH_ROOT=str(target), SWITCH_PREFIX_SOURCE=str(parent), SWITCH_DATASET="mbpp")
    result = invoke(launcher, "run_selection_switch.sh", ["prepare"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(env["INNER_CAPTURE"]).read_text().splitlines() == [str(target), "mbpp", "prepare"]
    args = json.loads(Path(env["AUDIT_CAPTURE"]).read_text())["args"]
    assert [args[i + 1] for i, value in enumerate(args) if value == "--root"] == [str(target), str(parent)]
    assert not target.exists()


@pytest.mark.parametrize("fault", ["saved-target-without-manifest", "wrong-parent-dataset"])
def test_existing_prefix_source_never_bypasses_target_or_source_safety(launcher, fault):
    _, work, parent, env = launcher
    target = work / "runs/selection-switch-mbpp-quality-v1"
    write_json(parent / "switch.json", {"dataset": "math500" if fault == "wrong-parent-dataset" else "mbpp"})
    if fault == "saved-target-without-manifest":
        sealed_result(target / "states/s3-t25/points/view-25/random_full")
    env.update(SWITCH_ROOT=str(target), SWITCH_PREFIX_SOURCE=str(parent), SWITCH_DATASET="mbpp")
    before = {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}
    result = invoke(launcher, "run_selection_switch.sh", ["prepare"])
    assert result.returncode == 2, result.stdout + result.stderr
    assert not Path(env["INNER_CAPTURE"]).exists()
    assert before == {p: p.read_bytes() for p in work.rglob("*") if p.is_file()}


@pytest.mark.parametrize('entry', ['run_switch_mbpp.sh', 'run_selection_switch.sh'])
@pytest.mark.parametrize('mode', ['run', 'prepare', 'smoke'])
def test_missing_continuation_checkpoint_only_passes_guarded_queue_preflight(launcher, entry, mode):
    _, work, root, env = launcher
    write_json(root / 'switch.json', {'dataset': 'mbpp'})
    write_json(root / 'states/s3-t25/points/view-25/selection_full/policy/grpo_stats.jsonl', {'step': 28})
    before = {p: p.read_bytes() for p in work.rglob('*') if p.is_file()}
    result = invoke(launcher, entry, [mode])
    assert result.returncode == (2 if mode == 'smoke' else 0), result.stdout + result.stderr
    assert Path(env['INNER_CAPTURE']).exists() == (mode != 'smoke')
    assert 'MBPP STORAGE AUDIT: BLOCKED' in result.stdout
    args = json.loads(Path(env['AUDIT_CAPTURE']).read_text())['args']
    assert ('--allow-branch-quarantine' in args) == (mode != 'smoke')
    assert before == {p: p.read_bytes() for p in work.rglob('*') if p.is_file()}


@pytest.mark.parametrize('hold', [0, 600])
def test_direct_queue_does_not_hold_or_retry_checkpoint_review_exit(tmp_path, hold):
    source = (ROOT / 'scripts/run_selection_switch.sh').read_text()
    loop = source[source.index('pass=0\nwait_seconds=$HOLD'):]
    script = '''set -euo pipefail
selection_run_worker() { echo worker-called; return 80; }
selection_hold_node() { echo unexpected-hold; return 99; }
switch_complete() { echo unexpected-completion-check; return 99; }
''' + loop
    env = {**os.environ, 'HOLD': str(hold), 'MODE': 'run', 'SWITCH_AUTO_RECOVER': '0',
           'OUT_ROOT': str(tmp_path), 'PY': sys.executable}
    result = subprocess.run(['bash', '-c', script], env=env, cwd=tmp_path,
                            capture_output=True, text=True, check=False, timeout=5)
    assert result.returncode == 80, result.stdout + result.stderr
    assert result.stdout.count('worker-called') == 1
    assert '[WAIT] only checkpoint-review branches remain' in result.stdout
    assert 'unexpected-' not in result.stdout
    assert '[hold]' not in result.stdout
