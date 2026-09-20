"""Pair paper exports regenerate validated reports before packaging partial data."""

import json
import hashlib
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import selector_pair_results as results


@pytest.mark.parametrize("missing,development,complete", [
    (["s4-t25"], [], False),
    ([], ["s1-t25"], False),
    ([], [], True),
])
def test_exports_current_partial_report_and_curves(tmp_path, monkeypatch, missing, development, complete):
    root = tmp_path / "run"
    root.mkdir()
    paired = root / 'test/s3-t25/result.json'
    paired.parent.mkdir(parents=True)
    paired.write_text('{}')
    target = tmp_path / "selector-pair-results.txt"
    report = {"missing_states": missing, "missing_development_states": development,
              "rows": [{"state": "s3-t25", "score": 0.5}]}
    curves = "state,step,score\ns3-t25,10,0.5\n"
    calls = []

    def regenerate(actual_root, repo, timeout):
        calls.append((actual_root, repo, timeout))
        (root / "report.json").write_text(json.dumps(report))
        (root / "curves.csv").write_text(curves)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(results, "run_report", regenerate)
    monkeypatch.setattr(sys, "argv", ["selector_pair_results", "--root", str(root), "--out", str(target)])
    results.main()

    assert len(calls) == 1
    assert calls[0][0] == root.resolve()
    assert calls[0][2] == 60
    content = target.read_text()
    assert curves in content
    data = json.loads(content.split("DATA_JSON\n", 1)[1])
    assert data['rows'] == report['rows']
    assert data['complete'] == complete
    assert data['branch_measurements'] == data['branch_measurement_errors'] == []
    assert data['paired_validation']['status'] == 'validated'
    assert data['exporter']['version'] == 'selector-pair-results/v3'
    assert len(data['exporter']['script_sha256']) == 64
    assert data['exporter']['created_at'] and data['exporter']['export_id']
    assert list(tmp_path.glob("*.txt")) == [target]


def branch_fixture(root, *, selector="on_policy", seed=0, step=25, arm="selection_reduced"):
    directory = root / f"branches/{selector}/states/s{seed}-t{step}/points/view-{step}/{arm}"
    directory.mkdir(parents=True)
    result = {"schema": "offpolicy-selected-prefix-switch/v1", "complete": True,
              "completed_steps": step+10, "rewards": {"q0": .25, "q1": .75},
              "used_gpu_seconds": 80, "cost": {"train": 80}}
    (directory / "result.json").write_text(json.dumps(result))
    digest = hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()
    (directory / "result.sha256.json").write_text(json.dumps({"sha256": digest}))
    curve = {"schema": "offpolicy-selected-prefix-switch/v1", "result_sha256": digest, "points": {
        str(step): {"updates": 0, "reward": .25},
        str(step+5): {"updates": 5, "reward": .375},
        str(step+10): {"updates": 10, "reward": .5, "final": True}}}
    (directory / "curve.json").write_text(json.dumps(curve))
    return directory, result, curve


def test_incomplete_pair_exports_independent_measured_arm_without_h(tmp_path, monkeypatch):
    root = tmp_path / "run"
    directory, endpoint, curve = branch_fixture(root)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
    target = tmp_path / "results.txt"
    def regenerate(*args, **kwargs):
        pytest.fail('no paired result exists; strict report must not run or wait for a lock')
    monkeypatch.setattr(results, "run_report", regenerate)
    monkeypatch.setattr(sys, "argv", ["selector_pair_results", "--root", str(root), "--out", str(target)])
    results.main()
    text = target.read_text()
    data = json.loads(text.split("DATA_JSON\n")[1])
    assert data["rows"] == data["development_rows"] == []
    assert data["summary"] is None
    assert data['paired_validation']['status'] == 'not_run_no_published_paired_results'
    assert data["complete"] is False
    row, = data["branch_measurements"]
    assert row["source_result"] == endpoint and row["source_curve"] == curve
    assert row["mean_reward"] == .5 and row["updates"] == 10
    assert row["question_count"] == 2 and row["issues"] == []
    assert row["eligible_for_paired_comparison"] is False
    assert row["independently_certified"] is False
    assert [p["reward"] for p in row["curve_points"]] == [.25, .375, .5]
    assert all("gpu_seconds" not in p for p in row["curve_points"])
    assert "crossing" not in row and "H" not in row and "cost_to_target" not in row
    assert "INDEPENDENT BRANCH MEASUREMENTS" in text
    assert "selection_reduced,saved_branch_measurement,10,0.5,2" in text
    assert "selection_reduced,5,0.375," in text
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()} == before
    assert list(tmp_path.glob("*.txt")) == [target]


def test_endpoint_export_does_not_require_curve_or_other_selector(tmp_path):
    directory, _, _ = branch_fixture(tmp_path)
    (directory / "curve.json").unlink()
    rows, errors = results.saved_branch_measurements(tmp_path)
    assert not errors
    assert rows[0]["mean_reward"] == .5
    assert rows[0]["curve_points"] == []


def test_real_pair_results_bash_exports_one_partial_txt(tmp_path):
    root = tmp_path / 'run'
    branch_fixture(root)
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    env = {**os.environ, 'HOME': str(tmp_path), 'PAIR_ROOT': str(root),
           'PAIR_PYTHON': sys.executable, 'OM_WORK': str(tmp_path / 'work')}
    result = subprocess.run(['bash', 'scripts/run_paper_results.sh', 'results', 'pair'],
                            cwd=Path(__file__).resolve().parents[1], env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    output, = tmp_path.glob('*.txt')
    data = json.loads(output.read_text().split('DATA_JSON\n', 1)[1])
    assert output.name == 'selector-pair-results.txt' and output.stat().st_size < 1_900_000
    assert data['branch_measurements'][0]['mean_reward'] == .5 and not data['complete']
    assert before == {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}


def test_oversized_raw_metadata_keeps_validated_partial_measurements_in_one_txt(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    directory, endpoint, curve = branch_fixture(root)
    endpoint['unvalidated_debug'] = 'x' * 2_000_000
    raw = json.dumps(endpoint).encode()
    (directory / 'result.json').write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (directory / 'result.sha256.json').write_text(json.dumps({'sha256': digest}))
    curve['result_sha256'] = digest
    curve['points']['25']['unvalidated_debug'] = 'y' * 2_000_000
    (directory / 'curve.json').write_text(json.dumps(curve))
    target = tmp_path / 'results.txt'
    monkeypatch.setattr(sys, 'argv', ['results', '--root', str(root), '--out', str(target)])
    results.main()
    data = json.loads(target.read_text().split('DATA_JSON\n', 1)[1])
    row, = data['branch_measurements']
    assert target.stat().st_size < 1_900_000
    assert list(tmp_path.glob('*.txt')) == [target]
    assert row['mean_reward'] == .5 and row['question_rewards'] == endpoint['rewards']
    assert row['source_result_reference']['sha256'] == digest
    assert row['source_result_reference']['source_bytes'] == len(raw)
    assert row['source_curve_reference']['raw_omitted']
    assert [point['reward'] for point in row['curve_points']] == [.25, .375, .5]
    assert row['issues'] == [] and data['complete'] is False


@pytest.mark.parametrize('schema', [None, 'offpolicy-net-gain-gate/v3-1'])
def test_wrong_schema_does_not_become_pair_measurement(tmp_path, schema):
    directory, endpoint, _ = branch_fixture(tmp_path)
    endpoint['schema'] = schema
    (directory / 'result.json').write_text(json.dumps(endpoint))
    digest = hashlib.sha256((directory / 'result.json').read_bytes()).hexdigest()
    (directory / 'result.sha256.json').write_text(json.dumps({'sha256': digest}))
    rows, errors = results.saved_branch_measurements(tmp_path)
    assert not errors and rows[0]['mean_reward'] is None and rows[0]['issues']


def test_symlink_loop_cannot_prevent_other_partial_results_txt(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    directory, _, _ = branch_fixture(root)
    other, _, _ = branch_fixture(root, selector='cached')
    path = directory / 'result.json'
    path.unlink()
    path.symlink_to(path.name)
    target = tmp_path / 'results.txt'
    monkeypatch.setattr(sys, 'argv', ['results', '--root', str(root), '--out', str(target)])
    with pytest.raises(SystemExit):
        results.main()
    data = json.loads(target.read_text().split('DATA_JSON\n', 1)[1])
    assert len(data['branch_measurements']) == 1
    assert data['branch_measurements'][0]['mean_reward'] == .5
    assert data['branch_measurement_errors']


@pytest.mark.parametrize("damage", ["seal_missing", "seal_changed", "incomplete", "negative", "nan", "stop_before_prefix"])
def test_invalid_endpoint_never_becomes_measured_row(tmp_path, damage):
    directory, endpoint, _ = branch_fixture(tmp_path)
    if damage == "seal_missing":
        (directory / "result.sha256.json").unlink()
    elif damage == "seal_changed":
        (directory / "result.sha256.json").write_text('{"sha256":"wrong"}')
    else:
        if damage == "incomplete":
            endpoint["complete"] = False
        elif damage == "negative":
            endpoint["rewards"]["q0"] = -.2
        elif damage == "nan":
            endpoint["rewards"]["q0"] = float("nan")
        else:
            endpoint["completed_steps"] = 24
        (directory / "result.json").write_text(json.dumps(endpoint))
        digest = hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()
        (directory / "result.sha256.json").write_text(json.dumps({"sha256": digest}))
    rows, errors = results.saved_branch_measurements(tmp_path)
    if damage == "nan":
        assert not rows and errors
    else:
        assert not errors and rows[0]["issues"]
        assert rows[0]["mean_reward"] is None and rows[0]["updates"] is None
        assert rows[0]["curve_points"] == []
        assert rows[0]["status"] == "unverified_source_only"


@pytest.mark.parametrize("damage", ["binding", "updates", "reward", "final", "json"])
def test_bad_curve_keeps_separate_endpoint_but_no_curve_numbers(tmp_path, damage):
    directory, _, curve = branch_fixture(tmp_path)
    if damage == "binding":
        curve["result_sha256"] = "wrong"
    elif damage == "updates":
        curve["points"]["30"]["updates"] = 100
    elif damage == "reward":
        curve["points"]["30"]["reward"] = 2
    elif damage == "final":
        curve["points"]["35"]["reward"] = .25
    (directory / "curve.json").write_text("{" if damage == "json" else json.dumps(curve))
    rows, errors = results.saved_branch_measurements(tmp_path)
    assert not errors and rows[0]["mean_reward"] == .5
    assert rows[0]["curve_points"] == [] and rows[0]["issues"]


def test_only_known_current_layout_arms_are_included(tmp_path):
    branch_fixture(tmp_path, selector="on_policy", seed=3, arm="random_full")
    branch_fixture(tmp_path, selector="cached", seed=3, arm="selection_full")
    branch_fixture(tmp_path, selector="adaptive-cached", seed=3, arm="selection_full")
    branch_fixture(tmp_path, selector="cached", seed=3, arm="random_full")
    branch_fixture(tmp_path, selector="adaptive-cached", seed=0)
    branch_fixture(tmp_path, selector="discarded")
    branch_fixture(tmp_path / "archive")
    rows, errors = results.saved_branch_measurements(tmp_path)
    assert not errors and len(rows) == 3
    assert {row["selector_branch"] for row in rows} == {"on_policy", "cached", "adaptive-cached"}
    assert all(row["role"] == "test" for row in rows)
    assert all(row["eligible_for_paired_comparison"] is False for row in rows)


def test_symlink_outside_root_is_not_exported(tmp_path):
    root = tmp_path / "run"
    outside, _, _ = branch_fixture(tmp_path / "outside")
    link = root / "branches/on_policy/states/s0-t25/points/view-25/selection_reduced"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)
    rows, errors = results.saved_branch_measurements(root)
    assert not rows and errors
    assert "escapes experiment root" in errors[0]["error"]


@pytest.mark.parametrize('returncode', [80, 124])
def test_failed_report_replaces_stale_txt_with_current_error_and_branches(tmp_path, monkeypatch, returncode):
    root = tmp_path / "run"
    root.mkdir()
    paired = root / 'test/s3-t25/result.json'
    paired.parent.mkdir(parents=True)
    paired.write_text('{}')
    branch_fixture(root)
    (root / "report.json").write_text(json.dumps({"missing_states": [], "missing_development_states": []}))
    (root / "curves.csv").write_text("stale curves")
    target = tmp_path / "selector-pair-results.txt"
    target.write_text("previous valid export")
    monkeypatch.setattr(results, "run_report", lambda *a, **kw:
                        SimpleNamespace(returncode=returncode, stdout='', stderr='current validation error'))
    monkeypatch.setattr(sys, "argv", ["selector_pair_results", "--root", str(root), "--out", str(target)])

    with pytest.raises(SystemExit) as failure:
        results.main()

    assert failure.value.code == returncode
    text = target.read_text()
    assert 'previous valid export' not in text and 'stale curves' not in text
    data = json.loads(text.split('DATA_JSON\n')[1])
    assert data['rows'] == data['development_rows'] == []
    assert data['complete'] is False and data['summary'] is None
    assert data['paired_validation']['status'] == 'failed'
    assert data['paired_validation']['stderr_tail'] == 'current validation error'
    assert data['export_exit_code'] == returncode
    assert data['branch_measurements'][0]['mean_reward'] == .5
    assert data['exporter']['version'] == 'selector-pair-results/v3'
    assert list(tmp_path.glob("*.txt")) == [target]


def test_fresh_exports_differ_even_with_unchanged_measurements(tmp_path, monkeypatch):
    branch_fixture(tmp_path / 'run')
    target = tmp_path / 'results.txt'
    monkeypatch.setattr(sys, 'argv', ['results', '--root', str(tmp_path / 'run'), '--out', str(target)])
    results.main()
    first = json.loads(target.read_text().split('DATA_JSON\n')[1])
    results.main()
    second = json.loads(target.read_text().split('DATA_JSON\n')[1])
    assert first['exporter']['export_id'] != second['exporter']['export_id']
    assert first['branch_measurements'] == second['branch_measurements']


def test_report_timeout_terminates_only_export_child_group(tmp_path, monkeypatch):
    calls = []
    class Process:
        pid = 123456
        def communicate(self, timeout=None):
            calls.append(timeout)
            if timeout is not None:
                raise results.subprocess.TimeoutExpired('report', timeout)
            return 'partial CPU output', 'waiting for lock'
    def popen(command, **kwargs):
        assert kwargs['start_new_session'] is True
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert kwargs['env']['PAIR_ROOT'] == str(tmp_path)
        return Process()
    killed = []
    monkeypatch.setattr(results.subprocess, 'Popen', popen)
    monkeypatch.setattr(results.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    result = results.run_report(tmp_path, tmp_path, 0.25)
    assert result.returncode == 124 and 'timed out' in result.stderr
    assert calls == [0.25, None]
    assert killed == [(123456, results.signal.SIGKILL)]


def test_missing_root_writes_fresh_error_txt(tmp_path, monkeypatch):
    target = tmp_path / 'results.txt'
    target.write_text('previous export')
    monkeypatch.setattr(sys, 'argv', ['results', '--root', str(tmp_path / 'missing'), '--out', str(target)])
    with pytest.raises(SystemExit) as exc:
        results.main()
    assert exc.value.code == 2
    data = json.loads(target.read_text().split('DATA_JSON\n')[1])
    assert data['paired_validation']['error'] == 'experiment root does not exist'
    assert data['branch_measurements'] == []


def test_real_cli_exports_zero_paired_states_without_launcher_or_gpu(tmp_path):
    root = tmp_path / 'run'
    branch_fixture(root)
    target = tmp_path / 'results.txt'
    # No pair manifest exists, so invoking the frozen report would fail.
    completed = results.subprocess.run(
        [sys.executable, str(Path(results.__file__).resolve()), '--root', str(root), '--out', str(target)],
        capture_output=True, text=True, timeout=5)
    assert completed.returncode == 0, completed.stderr
    data = json.loads(target.read_text().split('DATA_JSON\n')[1])
    assert data['paired_validation']['status'] == 'not_run_no_published_paired_results'
    assert data['rows'] == [] and data['complete'] is False
    assert data['branch_measurements'][0]['mean_reward'] == .5
    assert data['execution_observations']['planned_branches'] == {'total': 42, 'development': 18, 'test': 24}


def put_metadata(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_execution_observations_distinguish_endpoint_curve_and_worker_records(tmp_path):
    directory, _, _ = branch_fixture(tmp_path)
    (directory / 'curve.json').unlink()
    for shard in (0, 2):
        put_metadata(directory / f'curve/step-30/shard-{shard}.done.json', {'unverified': True})
    put_metadata(directory.parent / 'curve-parent/shard-1.done.json', {})
    put_metadata(directory / 'policy/policy_train.json', {'completed_steps': 35})
    put_metadata(directory / 'pair-attempt.json', {'error': 'evaluation timeout', 'time': 1})
    put_metadata(directory / 'curve/progress.json', {'state': 'running', 'phase': 'curve',
                 'updated': 999999999999, 'host': 'same-host', 'pid': 3, 'event_id': 'curve-30'})
    for worker, timestamp in [('a', 1), ('b', 999999999999)]:
        put_metadata(tmp_path / f'queue-workers/{worker}.json',
                     {'worker': worker, 'host': 'same-host', 'state': 'RUN',
                      'task': 'development/s0-t25/on_policy/selection_reduced', 'updated': timestamp})
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    rows, _ = results.saved_branch_measurements(tmp_path)
    data = results.execution_observations(tmp_path, rows)
    assert data['saved_endpoint_files'] == data['accepted_endpoint_measurements'] == 1
    assert data['saved_curve_files'] == 0
    assert {row['worker'] for row in data['workers']} == {'a', 'b'}
    assert {row['updated'] for row in data['workers']} == {1, 999999999999}
    branch, = data['branches']
    assert branch['curve_checkpoints'] == [{'step': 30, 'shards_done': [0, 2]}]
    assert branch['curve_shards_done_count'] == 2
    assert branch['parent_shards_done'] == [1]
    assert branch['policy']['completed_steps'] == 35
    assert branch['pair-attempt.json']['error'] == 'evaluation timeout'
    assert data['progress'][0]['phase'] == 'curve'
    assert data['independently_certified'] is False
    assert 'no heartbeat age' in data['scope']
    assert not any(key in json.dumps(data) for key in ('"active"', '"live"', '"heartbeat_fresh"'))
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()} == before


def test_execution_observation_limits_and_malformed_metadata(tmp_path, monkeypatch):
    import selector_pair_gpu
    directory, _, _ = branch_fixture(tmp_path)
    for index in range(40):
        put_metadata(tmp_path / f'queue-workers/w{index}.json',
                     {'worker': f'w{index}', 'task': 'x' * 5000, 'failures': [{'error': 'x' * 5000}] * 5})
    put_metadata(directory / 'failure.json', {'error': 'x' * 70000})
    for step in range(80):
        put_metadata(directory / f'curve/step-{step}/shard-0.done.json', {})
    monkeypatch.setattr(selector_pair_gpu, 'pair_progress', lambda root:
        [(index, directory / f'curve/step-{index}/progress.json',
          {'state': 'running', 'phase': 'x' * 10000, 'seconds': float('nan')}) for index in range(100)])
    rows, _ = results.saved_branch_measurements(tmp_path)
    data = results.execution_observations(tmp_path, rows)
    assert len(data['workers']) == 32 and data['omitted']['workers'] == 8
    assert len(data['progress']) == 64 and data['omitted']['progress'] == 36
    assert len(data['branches'][0]['curve_checkpoints']) == 64
    assert data['omitted']['checkpoint_directories'] == 1
    assert any('exceeds 65536' in row['error'] for row in data['errors'])
    assert len(json.dumps(data, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode()) <= results.OBSERVATION_LIMIT
    assert all(len(row['task']) <= 512 for row in data['workers'])


def test_observations_do_not_follow_external_metadata(tmp_path):
    root = tmp_path / 'run'
    directory, _, _ = branch_fixture(root)
    outside = tmp_path / 'secret.json'
    outside.write_text('{"error":"private outside data"}')
    (directory / 'failure.json').symlink_to(outside)
    (directory / 'curve').symlink_to(tmp_path, target_is_directory=True)
    rows, _ = results.saved_branch_measurements(root)
    data = results.execution_observations(root, rows)
    assert 'private outside data' not in json.dumps(data)
    assert any('escapes experiment root' in row['error'] for row in data['errors'])


def test_observation_byte_cap_omits_metadata_without_changing_scientific_rows(tmp_path, monkeypatch):
    directory, _, _ = branch_fixture(tmp_path)
    for index in range(32):
        put_metadata(tmp_path / f'queue-workers/w{index}.json',
                     {key: 'x' * 512 for key in ('worker', 'host', 'task', 'stage', 'protocol_id')})
    rows, _ = results.saved_branch_measurements(tmp_path)
    original = json.dumps(rows, sort_keys=True)
    monkeypatch.setattr(results, 'OBSERVATION_LIMIT', 4000)
    data = results.execution_observations(tmp_path, rows)
    assert len(json.dumps(data, ensure_ascii=True, separators=(',', ':')).encode()) <= 4000
    assert data['omitted']['workers'] > 0
    assert data['saved_endpoint_files'] == data['accepted_endpoint_measurements'] == 1
    assert json.dumps(rows, sort_keys=True) == original
