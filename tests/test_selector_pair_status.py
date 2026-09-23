import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import selection_gate as core
import selector_pair as pair
import selector_pair_gpu as gpu

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pair_status", ROOT / "scripts/selector_pair_status.py")
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


def prepared(root):
    branches = {}
    for name in gpu.BRANCHES:
        path = root / "branches" / name / "switch.json"
        core.atomic_json(path, {"branch": name})
        branches[name] = status.digest(path)
    p = {"schema": pair.SCHEMA, "target_reward": .35, "training_cap_gpu_seconds": 87120,
         "branch_manifests": branches, "code_hashes": gpu.code_hashes()}
    p["protocol_id"] = core.fingerprint(p)
    core.atomic_json(root / "pair.json", p)
    return p


def branch(root):
    return root / "branches/on_policy/states/s0-t25/points/view-25/selection_reduced"


def published(root):
    path = branch(root)
    core.atomic_json(path / "result.json", {"complete": True, "schema": "offpolicy-selected-prefix-switch/v1"})
    value = status.digest(path / "result.json")
    core.atomic_json(path / "result.sha256.json", {"sha256": value})
    core.atomic_json(path / "curve.json", {"schema": status.display.switch_status.rule.SCHEMA,
                                          "result_sha256": value, "points": {"25": {"reward": .1}}})
    return path


def test_missing_root_is_readonly_and_keeps_all_planned_slots(tmp_path):
    root = tmp_path / "missing"
    data = status.snapshot(root)
    assert len(data["tasks"]) == 42
    assert {task["status"] for task in data["tasks"]} == {"WAIT"}
    assert "계획 42개 | 완료 확인 0/42 | 남음 42개" in status.render(data)
    assert "기록 미확인 42개" in status.render(data)
    assert not root.exists()


def test_nonattainment_explains_terminal_adaptive_dependency(tmp_path):
    p = prepared(tmp_path)
    core.atomic_json(tmp_path / "development/s0-t25/result.json", {
        "protocol_id": p["protocol_id"], "seed": 0, "step": 25, "role": "development",
        "contrast": {"status": "censored", "h_gpu_seconds": None}})
    data = status.snapshot(tmp_path)
    adaptive = [task for task in data["tasks"] if task["name"] == "adaptive"]
    assert len(adaptive) == 6
    assert all(task["status"] == "BLOCKED" for task in adaptive)
    assert all("s0/t25" in task["reason"] for task in adaptive)
    assert "테스트 결정 고정: BLOCKED" in status.render(data)


def test_srgc_does_not_wait_for_development_target_and_shows_measurement(tmp_path):
    import selector_pair_srgc as srgc
    p = prepared(tmp_path)
    srgc.activate(tmp_path, p)
    core.atomic_json(tmp_path / "development/s0-t25/result.json", {
        "protocol_id": p["protocol_id"], "seed": 0, "step": 25, "role": "development",
        "contrast": {"status": "censored", "h_gpu_seconds": None}})
    core.atomic_json(tmp_path / "sr-gc/s3-t25/progress.json", {
        "state": "running", "phase": "sr-gc-candidate-a", "host": "node-srgc", "updated": 100.})
    data = status.snapshot(tmp_path, now=100.)
    task = next(row for row in data["tasks"] if row["name"] == "adaptive" and row["seed"] == 3 and row["step"] == 25)
    assert task["status"] == "RUN" and "SR-GC" in task["reason"]
    assert data["adaptive_reason"] == "SR-GC 현재 gradient 측정·결정 0/6"
    assert any(row.get("phase") == "sr-gc-candidate-a" for row in data["activity"])
    rendered = status.render(data)
    assert "SR-GC" in rendered
    assert "회귀" not in rendered
    assert "목표 도달 대기" not in rendered


@pytest.mark.parametrize("damage", [None, "receipt", "curve", "missing_curve", "budget"])
def test_only_published_result_and_bound_curve_count_as_done(tmp_path, damage):
    prepared(tmp_path)
    path = published(tmp_path)
    if damage == "receipt":
        core.atomic_json(path / "result.sha256.json", {"sha256": "wrong"})
    elif damage == "curve":
        core.atomic_json(path / "curve.json", {})
    elif damage == "missing_curve":
        (path / "curve.json").unlink()
    elif damage == "budget":
        for name in ("result.json", "result.sha256.json", "curve.json"):
            (path / name).unlink()
        core.atomic_json(path / "pair-attempt.json", {"error": "budget exhausted"})
    data = status.snapshot(tmp_path)
    assert sum(task["status"] == "DONE" for task in data["tasks"]) == (damage is None)
    assert sum(task["role"] == "development" for task in data["tasks"]) == 18
    assert sum(task["role"] == "test" for task in data["tasks"]) == 24
    assert all(task["status"] == "WAIT" for task in data["tasks"] if task["role"] == "test")


def test_nested_live_progress_full_host_and_stale_queue(tmp_path):
    p = prepared(tmp_path)
    host = "run284441-wts-59-full-node-name"
    core.atomic_json(branch(tmp_path) / "curve/50/progress.json", {
        "host": host, "state": "running", "updated": 995, "phase": "evaluate", "seconds": 5, "timeout": 100})
    core.atomic_json(tmp_path / "queue-workers/worker.json", {
        "host": host, "state": "RUN", "updated": 10, "protocol_id": p["protocol_id"], "task": "development/s0-t25"})
    data = status.snapshot(tmp_path, now=1000)
    assert data["tasks"][0]["status"] == "RUN"
    assert len(data["nodes"]) == 1 and data["nodes"][0]["current"]
    for width in (80, 120):
        rendered = status.render(data, width=width)
        assert host in rendered
        assert all(status.display.columns(line) <= width for line in rendered.splitlines())
    stale = status.snapshot(tmp_path, now=2000)
    assert stale["tasks"][0]["status"] == "WAIT"
    assert not stale["nodes"][0]["current"]


def test_frozen_adaptive_choice_counts_only_one_branch(tmp_path, monkeypatch):
    prepared(tmp_path)
    core.atomic_json(tmp_path / "test-decisions.json", {"saved": True})
    monkeypatch.setattr(gpu, "decisions", lambda *_: {f"s{s}-t{t}": {"selector": "cached"}
                                                    for s in pair.TEST_SEEDS for t in pair.STEPS})
    data = status.snapshot(tmp_path)
    adaptive = [task for task in data["tasks"] if task["name"] == "adaptive"]
    assert len(adaptive) == 6
    assert all("adaptive-cached" in task["directory"] for task in adaptive)
    assert all(task["status"] == "READY" for task in data["tasks"])


@pytest.mark.parametrize('branch_name,arm', [
    ('adaptive-cached', 'selection_full'), ('adaptive-on_policy', 'selection_full'),
    ('on_policy', 'selection_full'), ('cached', 'selection_full'), ('on_policy', 'random_full'),
])
def test_visible_training_row_keeps_completed_updates_after_deduplication(tmp_path, monkeypatch, branch_name, arm):
    prepared(tmp_path)
    choice = branch_name.removeprefix('adaptive-') if branch_name.startswith('adaptive-') else 'cached'
    core.atomic_json(tmp_path / 'test-decisions.json', {'saved': True})
    monkeypatch.setattr(gpu, 'decisions', lambda *_: {f's{s}-t{t}': {'selector': choice}
                                                    for s in pair.TEST_SEEDS for t in pair.STEPS})
    directory = tmp_path / f'branches/{branch_name}/states/s4-t100/points/view-100/{arm}'
    (directory / 'policy').mkdir(parents=True)
    (directory / 'policy/grpo_stats.jsonl').write_text(json.dumps({'step': 107}) + '\n')
    core.atomic_json(directory / 'progress.json', {
        'host': 'training-node', 'pid': 123, 'state': 'running', 'updated': 995,
        'phase': 'train', 'event_id': 'current-train', 'seconds': 95, 'timeout': 1000})
    for phase in ('first', 'second'):
        for state in ('started', 'finished'):
            gpu.base.journal(directory / 'cost.jsonl', {
                'event_id': phase, 'phase': phase, 'state': state, 'exit_code': 0})
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    data = status.snapshot(tmp_path, now=1000)
    assert next(row for row in data['activity'] if row.get('phase') == 'train')['training_step'] == 107
    suite = status.dashboard_data(data)['suites'][0]
    visible = status.display.current_work(suite)
    assert len(visible) == 1
    task, shared = visible[0]
    assert not shared and task.get('training_step') == 107
    progress, note = status.display.task_progress(tmp_path, task)
    assert progress == '100.0% / 100.0% / 7회 완료'
    assert '학습 업데이트 7회 완료' in note
    assert '7회 완료' in status.render(data, width=200)
    assert all(p.read_bytes() == content for p, content in before.items())


def test_training_count_refreshes_and_missing_rows_remain_unknown(tmp_path):
    prepared(tmp_path)
    directory = branch(tmp_path)
    core.atomic_json(directory / 'progress.json', {
        'host': 'training-node', 'state': 'running', 'updated': 995,
        'phase': 'train', 'event_id': 'current-train'})
    stats = directory / 'policy/grpo_stats.jsonl'
    stats.parent.mkdir(parents=True)
    for raw, expected in [('', '?'), ('{"step":26}\n', '1회 완료'),
                          ('{"step":26}\n{"step":27}\n{"step":', '2회 완료')]:
        stats.write_text(raw)
        data = status.snapshot(tmp_path, now=1000)
        task = status.display.current_work(status.dashboard_data(data)['suites'][0])[0][0]
        assert status.display.task_progress(tmp_path, task)[0] == expected
        assert task['status'] == 'RUNNING'
        assert not any(row['status'] == 'DONE' for row in data['tasks'])


def parallel_receipt(root, protocol):
    import selector_pair_parallel as parallel

    core.atomic_json(root / parallel.RECEIPT, parallel.receipt_value(root, protocol))


def test_parallel_receipt_opens_only_eighteen_fixed_controls_before_gate(tmp_path):
    protocol = prepared(tmp_path)
    parallel_receipt(tmp_path, protocol)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in tmp_path.rglob('*') if path.is_file()}
    data = status.snapshot(tmp_path)
    fixed = [task for task in data['tasks'] if task['role'] == 'test' and task['name'] != 'adaptive']
    adaptive = [task for task in data['tasks'] if task['name'] == 'adaptive']
    assert len(fixed) == 18 and {task['status'] for task in fixed} == {'READY'}
    assert len(adaptive) == 6 and {task['status'] for task in adaptive} == {'WAIT'}
    assert all(task['directory'] == '' for task in adaptive)
    assert len(data['tasks']) == 42 and data['parallel_controls_ready']
    assert not data['test_decisions_frozen'] and not data['parallel_controls_error']
    output = status.render(data, width=160)
    assert '고정 대조군 병렬 실행 승인: READY' in output and '테스트 결정 고정: WAIT' in output
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in tmp_path.rglob('*') if path.is_file()}


@pytest.mark.parametrize('name,selector,arm', [
    ('on_policy', 'on_policy', 'selection_full'), ('cached', 'cached', 'selection_full'),
    ('random', 'on_policy', 'random_full'),
])
@pytest.mark.parametrize('record', ['RUN', 'DONE'])
@pytest.mark.parametrize('authorized', [False, True])
def test_pre_gate_fixed_controls_keep_canonical_assignment_and_results(tmp_path, name, selector, arm, record, authorized):
    protocol = prepared(tmp_path)
    if authorized:
        parallel_receipt(tmp_path, protocol)
    directory = tmp_path / f'branches/{selector}/states/s3-t25/points/view-25/{arm}'
    if record == 'RUN':
        core.atomic_json(tmp_path / 'queue-workers/early-fixed.json', {
            'worker': 'early-fixed', 'host': 'new-fixed-node', 'state': 'RUN', 'updated': 995,
            'protocol_id': protocol['protocol_id'], 'stage': 'test',
            'task': f'test/s3-t25/{selector}/{arm}'})
    else:
        core.atomic_json(directory / 'result.json', {'complete': True,
                         'schema': status.display.switch_status.rule.SCHEMA})
        digest = status.digest(directory / 'result.json')
        core.atomic_json(directory / 'result.sha256.json', {'sha256': digest})
        core.atomic_json(directory / 'curve.json', {'schema': status.display.switch_status.rule.SCHEMA,
                         'result_sha256': digest, 'points': {'25': {'reward': .1}}})
    data = status.snapshot(tmp_path, now=1000)
    task = next(task for task in data['tasks'] if task['seed'] == 3 and task['step'] == 25
                and task['name'] == name)
    assert task['status'] == record and task['directory'] == str(directory.relative_to(tmp_path))
    assert not data['test_decisions_frozen']
    assert {task['status'] for task in data['tasks'] if task['name'] == 'adaptive'} == {'WAIT'}


@pytest.mark.parametrize('damage', ['missing', 'changed', 'symlink', 'directory', 'fifo'])
def test_unapproved_parallel_schedule_keeps_fixed_controls_waiting(tmp_path, damage):
    protocol = prepared(tmp_path)
    parallel_receipt(tmp_path, protocol)
    path = tmp_path / 'pair-parallel-controls-runtime.json'
    if damage == 'changed':
        value = core.read(path)
        value['protocol_id'] = 'changed'
        core.atomic_json(path, value)
    else:
        path.unlink()
        if damage == 'symlink':
            path.symlink_to(tmp_path / 'pair.json')
        elif damage == 'directory':
            path.mkdir()
        elif damage == 'fifo':
            os.mkfifo(path)
    data = status.snapshot(tmp_path)
    assert not data['parallel_controls_ready']
    assert bool(data['parallel_controls_error']) == (damage != 'missing')
    assert {task['status'] for task in data['tasks'] if task['role'] == 'test'} == {'WAIT'}


def test_copied_export_layout_keeps_eight_endpoints_and_one_curve_out_of_42(tmp_path):
    prepared(tmp_path)
    # Matches the eight independently saved branches in export_2.txt. None is
    # a completed two-selector state; that must not hide the eight endpoints.
    saved = [(0, 100, 'cached'), (0, 25, 'cached'), (2, 25, 'cached'), (2, 50, 'cached'),
             (0, 50, 'on_policy'), (1, 100, 'on_policy'), (1, 25, 'on_policy'), (2, 100, 'on_policy')]
    for seed, step, selector in saved:
        directory = tmp_path / f'branches/{selector}/states/s{seed}-t{step}/points/view-{step}/selection_reduced'
        core.atomic_json(directory / 'result.json', {'complete': True,
            'schema': 'offpolicy-selected-prefix-switch/v1', 'completed_steps': step+100,
            'rewards': {'0': .5, '1': .75}})
        digest = status.digest(directory / 'result.json')
        core.atomic_json(directory / 'result.sha256.json', {'sha256': digest})
        if (seed, step, selector) == (0, 50, 'on_policy'):
            core.atomic_json(directory / 'curve.json', {'schema': 'offpolicy-selected-prefix-switch/v1',
                'result_sha256': digest, 'points': {'150': {'updates': 100, 'reward': .625, 'final': True}}})
    data = status.snapshot(tmp_path, now=10000)
    assert sum(bool(task.get('training_published')) for task in data['tasks']) == 8
    suite = status.dashboard_data(data)['suites'][0]
    counts = status.display.counts(suite)
    assert (counts['planned'], counts['done'], counts['remaining']) == (42, 1, 41)
    assert sum(task['status'] == 'EVAL' for task in data['tasks']) == 7
    text = status.render(data, width=160)
    assert '최종 평가 저장 8/42' in text
    assert '결과·곡선 저장 1/42' in text
    assert '개발 9상태 x 2분기 + 검증 6상태 x 4분기' in text


def test_old_net_gain_schema_cannot_count_as_pair_result(tmp_path):
    prepared(tmp_path)
    path = published(tmp_path)
    core.atomic_json(path / 'result.json', {'complete': True, 'schema': 'offpolicy-net-gain-gate/v3-1'})
    digest = status.digest(path / 'result.json')
    core.atomic_json(path / 'result.sha256.json', {'sha256': digest})
    curve = core.read(path / 'curve.json')
    core.atomic_json(path / 'curve.json', {**curve, 'result_sha256': digest})
    task = status.snapshot(tmp_path)['tasks'][0]
    assert task['status'] == 'WAIT' and not task.get('training_published')


def test_bad_test_barrier_preserves_published_development(tmp_path):
    prepared(tmp_path)
    published(tmp_path)
    core.atomic_json(tmp_path / "test-decisions.json", {"invalid": True})
    data = status.snapshot(tmp_path)
    assert data["error"]
    assert data["tasks"][0]["status"] == "DONE"
    assert all(task["status"] == "WAIT" for task in data["tasks"] if task["role"] == "test")


def test_launcher_observes_locked_run_without_writes(tmp_path):
    prepared(tmp_path)
    published(tmp_path)
    with (tmp_path / ".pair.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
        process = subprocess.run(["bash", "scripts/run_selector_pair.sh", "status", "--json"],
                                 cwd=ROOT, env={**os.environ, "PAIR_ROOT": str(tmp_path), "PAIR_PYTHON": sys.executable},
                                 text=True, capture_output=True, timeout=15)
        assert process.returncode == 0, process.stderr
        assert len(json.loads(process.stdout)["tasks"]) == 42
        assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf", "1.5"])
def test_watch_rejects_invalid_interval(tmp_path, interval):
    process = subprocess.run(["bash", "scripts/run_selector_pair.sh", "status", "--watch", interval],
                             cwd=ROOT, env={**os.environ, "PAIR_ROOT": str(tmp_path / "missing")},
                             text=True, capture_output=True, timeout=15)
    assert process.returncode == 2
    assert "watch interval must be a positive integer" in process.stdout
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("options,interval", [(["--watch"], "15"), (["--watch", "1", "--all"], "1")])
def test_watch_refreshes_like_mbpp_without_gpu_work(tmp_path, options, interval):
    from test_mbpp_status_watch import two_frame_sleep
    root = tmp_path / "run"
    prepared(root)
    published(root)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.rglob("*") if path.is_file()}
    extra = two_frame_sleep(tmp_path)
    process = subprocess.run(["bash", "scripts/run_selector_pair.sh", "status", *options],
                             cwd=ROOT, env={**os.environ, **extra, "PAIR_ROOT": str(root), "PAIR_PYTHON": sys.executable},
                             text=True, capture_output=True, timeout=15)
    assert process.returncode == 143, process.stderr
    assert process.stdout.count("SELECTOR PAIR EXPERIMENTS") == 2
    assert process.stdout.count("계획 42개 | 완료 확인 1/42 | 남음 41개") == 2
    assert json.loads(Path(extra["SLEEP_LOG"]).read_text()) == [[interval], [interval]]
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.rglob("*") if path.is_file()}


def test_pair_uses_mbpp_renderer_with_same_sections_columns_and_korean_labels(tmp_path, monkeypatch):
    prepared(tmp_path)
    published(tmp_path)
    data = status.snapshot(tmp_path)
    shared_render = status.display.render
    calls = []
    def render(adapted, **kwargs):
        calls.append(adapted)
        return shared_render(adapted, **kwargs)
    monkeypatch.setattr(status.display, "render", render)
    output = status.render(data, width=160)
    assert len(calls) == 1
    assert output == shared_render(status.dashboard_data(data), width=160)
    assert "총 계획 42개 | 완료 확인 1개 | 남음 41개" in output
    assert "진단·학습 시간 한도: 87120 GPU-seconds" in output
    summary = next(line.split() for line in output.splitlines() if line.startswith("Experiment"))
    assert summary == ["Experiment", "계획", "DONE", "남음", "Progress", "READY", "WAIT", "RUN", "Remarks"]
    matrix = next(line.split() for line in output.splitlines() if line.startswith("Seed / Step"))
    assert matrix == ["Seed", "/", "Step", "Role", "Prefix", "On-policy", "Cached", "Adaptive", "Random", "Remarks"]
    sections = ["FULL STATUS", "CURRENT RUN", "NODE ASSIGNMENTS", "작업 없는 노드:"]
    assert [output.index(section) for section in sections] == sorted(output.index(section) for section in sections)
    assert "개발" in output and "검증" in output and "ROOT " not in output
    assert "MBPP" not in output and "Full selection" not in output


def test_same_mbpp_numbered_node_table_and_idle_list(tmp_path):
    p = prepared(tmp_path)
    core.atomic_json(branch(tmp_path) / "progress.json", {
        "host": "run-active-01", "state": "running", "updated": 995, "phase": "train", "seconds": 5, "timeout": 100})
    for host, updated in (("run-idle-02", 998), ("run-old-03", 1)):
        core.atomic_json(tmp_path / "queue-workers" / f"{host}.json", {
            "host": host, "state": "WAIT", "updated": updated, "protocol_id": p["protocol_id"]})
    data = status.snapshot(tmp_path, now=1000)
    output = status.render(data, width=160)
    header = next(line.split() for line in output.splitlines() if line.startswith("# "))
    assert header == ["#", "Node", "Experiment", "Status", "Progress", "Remarks"]
    assert "CURRENT RUN 1" in output and "WORKERS 2 current" in output
    assert "작업 없는 노드: 1개 (배정 대기 확인)" in output
    assert "1. run-idle-02 | WAIT | 작업 배정 대기; 확인 2초 전" in output
    assert "run-old-03" not in output and "run-old-03" in status.render(data, width=160, all_tasks=True)
    assert "branches/" not in output


def test_new_branch_assignment_is_visible_before_phase_heartbeat(tmp_path):
    p = prepared(tmp_path)
    core.atomic_json(tmp_path / "queue-workers/worker.json", {
        "host": "parallel-branch-node", "state": "RUN", "updated": 995,
        "protocol_id": p["protocol_id"], "task": "test/s3-t25/adaptive-cached/selection_full"})
    output = status.render(status.snapshot(tmp_path, now=1000), width=160)
    assert "parallel-branch-node" in output and "CURRENT RUN 1" in output
    assert "Selector pair / seed 3 / step 25 / Adaptive" in output


def test_same_host_independent_branch_heartbeats_keep_worker_identity(tmp_path):
    p = prepared(tmp_path)
    for index, name in enumerate(("on_policy", "cached")):
        worker = f"worker-{index}"
        core.atomic_json(tmp_path / f"queue-workers/{worker}.json", {
            "host": "same-node", "worker": worker, "state": "RUN", "updated": 995,
            "protocol_id": p["protocol_id"], "task": f"development/s0-t25/{name}/selection_reduced"})
        point = tmp_path / f"branches/{name}/states/s0-t25/points/view-25/selection_reduced"
        core.atomic_json(point / "curve/progress.json", {
            "host": "same-node", "state": "running", "updated": 995, "phase": "curve"})
    data = status.snapshot(tmp_path, now=1000)
    assert len(data["nodes"]) == 2
    assert {task["worker_id"] for task in data["activity"]} == {"worker-0", "worker-1"}
    assert {task["worker_id"] for task in data["tasks"] if task["status"] == "RUN"} == {
        "worker-0", "worker-1"}
    assert "CURRENT RUN 2" in status.render(data, width=160)


def test_previous_runtime_and_receipts_are_preserved(tmp_path, monkeypatch):
    from test_selector_pair_gpu import bootstrap_predecessor
    previous = gpu.code_hashes()
    previous['src/selection_switch_gpu.py'] = '7cd13cca9a1299bd3bd571cfb1e4109b65d5f790b82811a85e7604b9ad034606'
    previous.update({"src/selector_pair_gpu.py": "5e2c5ca5446a609dad0f999ad134fef39d17fc6ac490cdea0b66d2479292e84f",
                     "scripts/run_selector_pair.sh": "40a7df854a6b866118164198956225468351bca81f87309fa6700a2c51549092"})
    assert core.fingerprint(previous) == gpu.PRE_PAIR_STATUS_CODE
    assert gpu.compatible_code(previous)
    p = {"schema": pair.SCHEMA, "code_hashes": bootstrap_predecessor(), "branch_manifests": {}}
    p["protocol_id"] = core.fingerprint(p)
    core.atomic_json(tmp_path / "pair.json", p)
    with monkeypatch.context() as patch:
        patch.setattr(gpu, "code_hashes", lambda: previous)
        patch.setattr(gpu, "PRE_SHARED_RUNTIME_CODES", gpu.PRE_SHARED_RUNTIME_CODES - {gpu.PRE_PAIR_STATUS_CODE})
        gpu.bind_startup_runtime(tmp_path, p["code_hashes"])
    (tmp_path / "pair-status-runtime.json").unlink()
    (tmp_path / "pair-curve-progress-runtime.json").unlink(missing_ok=True)
    (tmp_path / "pair-branch-queue-runtime.json").unlink(missing_ok=True)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert gpu.manifest(tmp_path) == p
    assert gpu.manifest(tmp_path) == p
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert core.read(tmp_path / "pair-status-runtime.json")["runtime_code_hashes"] == gpu.code_hashes()
    tampered = {**previous, "scripts/run_selector_pair.sh": "unreviewed"}
    assert not gpu.compatible_code(tampered)


@pytest.mark.parametrize('offset', [-3600, 3600])
def test_curve_meter_is_visible_with_clock_skew_and_live_lease(tmp_path, offset):
    prepared(tmp_path)
    directory = branch(tmp_path) / 'curve'
    core.atomic_json(directory / 'progress.json', {'host': 'curve-peer', 'state': 'running',
        'phase': 'curve', 'updated': 10000 + offset, 'seconds': 200, 'timeout': 14400, 'event_id': 'curve-event'})
    with (directory / '.cost.lock').open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = status.snapshot(tmp_path, now=10000)
        assert data['tasks'][0]['status'] == 'RUN'
        assert data['tasks'][0]['owner_active'] and not data['tasks'][0]['heartbeat_fresh']
        assert data['nodes'][0]['current']
        output = status.render(data, width=160)
        assert 'CURRENT RUN 1' in output and 'curve-peer' in output
        assert '실행 신호 끊김' not in output
    assert status.snapshot(tmp_path, now=10000)['tasks'][0]['status'] == 'WAIT'


def test_curve_direct_read_survives_exhausted_recursive_scan(tmp_path, monkeypatch):
    prepared(tmp_path)
    directory = branch(tmp_path) / 'curve'
    core.atomic_json(directory / 'progress.json', {'host': 'curve-peer', 'state': 'running',
        'phase': 'curve', 'updated': 995, 'seconds': 200, 'timeout': 14400})
    # Force the recursive scan's two-second deadline to expire immediately.
    ticks = iter(range(0, 100000, 3))
    monkeypatch.setattr(gpu.time, 'monotonic', lambda: next(ticks))
    data = status.snapshot(tmp_path, now=1000)
    assert data['tasks'][0]['status'] == 'RUN'
    assert data['nodes'][0]['current']


@pytest.mark.parametrize('offset', [-3600, 3600])
def test_branch_lease_keeps_skewed_assignment_visible_between_meters(tmp_path, offset):
    p = prepared(tmp_path)
    worker = {'host': 'same-name', 'worker': 'branch-owner', 'state': 'RUN',
              'updated': 10000 + offset, 'protocol_id': p['protocol_id'],
              'task': 'development/s0-t25/on_policy/selection_reduced'}
    core.atomic_json(tmp_path / 'queue-workers/branch-owner.json', worker)
    lock = tmp_path / 'development/s0-t25/queue-branches/on_policy--selection_reduced.lock'
    lock.parent.mkdir(parents=True)
    with lock.open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in tmp_path.rglob('*') if path.is_file()}
        data = status.snapshot(tmp_path, now=10000)
        assert data['tasks'][0]['status'] == 'RUN'
        assert data['tasks'][0]['owner_active']
        active = [node for node in data['nodes'] if node['current']]
        assert len(active) == 1 and active[0]['state'] == 'RUN'
        assert active[0]['host'] == 'unknown-owner'
        assert active[0]['worker_id'].startswith('lease-')
        assert data['tasks'][0]['host'] == 'unknown-owner'
        assert 'CURRENT RUN 1' in status.render(data, width=160)
        json.dumps(data)
        assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}
    data = status.snapshot(tmp_path, now=10000)
    assert data['tasks'][0]['status'] != 'RUN'
    assert not data['nodes'][0]['current']


def test_root_shared_lease_does_not_revive_stale_branch_assignment(tmp_path):
    p = prepared(tmp_path)
    core.atomic_json(tmp_path / 'queue-workers/old.json', {
        'host': 'old-node', 'state': 'RUN', 'updated': 1, 'protocol_id': p['protocol_id'],
        'task': 'development/s0-t25/on_policy/selection_reduced'})
    with (tmp_path / '.pair.lock').open('w') as peer:
        fcntl.flock(peer, fcntl.LOCK_SH | fcntl.LOCK_NB)
        data = status.snapshot(tmp_path, now=10000)
    assert not any(node['current'] for node in data['nodes'])
    assert not data['activity']


@pytest.mark.parametrize('legacy', [False, True])
def test_live_lease_before_first_worker_record_is_visible_readonly(tmp_path, legacy):
    prepared(tmp_path)
    lock = tmp_path / 'development/s0-t25' / ('.state.lock' if legacy else
                                             'queue-branches/on_policy--selection_reduced.lock')
    lock.parent.mkdir(parents=True)
    with lock.open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
        data = status.snapshot(tmp_path, now=10000)
        active = [node for node in data['nodes'] if node['current']]
        assert len(active) == 1 and active[0]['host'] == 'unknown-owner'
        assert data['activity'][0]['activity_identity_unconfirmed']
        assert 'CURRENT RUN 1' in status.render(data, width=160)
        if not legacy:
            assert data['tasks'][0]['status'] == 'RUN'
        assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
        fcntl.flock(owner, fcntl.LOCK_SH)
        assert not status.snapshot(tmp_path, now=10000)['activity']
    assert not any(node['current'] for node in status.snapshot(tmp_path, now=10000)['nodes'])


def test_legacy_ex_state_lease_identifies_assignment_without_heartbeat(tmp_path):
    p = prepared(tmp_path)
    core.atomic_json(tmp_path / 'queue-workers/legacy.json', {
        'host': 'legacy-node', 'state': 'RUN', 'updated': 1, 'protocol_id': p['protocol_id'],
        'task': 'development/s0-t25'})
    lock = tmp_path / 'development/s0-t25/.state.lock'
    lock.parent.mkdir(parents=True)
    with lock.open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = status.snapshot(tmp_path, now=10000)
        assert len(data['activity']) == 1 and data['activity'][0]['owner_active']
        assert 'CURRENT RUN 1' in status.render(data, width=160)
        fcntl.flock(owner, fcntl.LOCK_SH)
        assert not status.snapshot(tmp_path, now=10000)['activity']


def test_same_hostname_and_pid_independent_meters_remain_separate(tmp_path):
    prepared(tmp_path)
    for name in ('on_policy', 'cached'):
        point = tmp_path / f'branches/{name}/states/s0-t25/points/view-25/selection_reduced/curve'
        core.atomic_json(point / 'progress.json', {'host': 'same-node', 'pid': 123,
            'state': 'running', 'updated': 995, 'phase': 'curve', 'event_id': name})
    data = status.snapshot(tmp_path, now=1000)
    assert len(data['nodes']) == 2
    assert len({task['worker_id'] for task in data['activity']}) == 2
    assert 'CURRENT RUN 2' in status.render(data, width=160)


def test_parent_and_nested_phase_share_one_fallback_work_identity(tmp_path):
    prepared(tmp_path)
    directory = branch(tmp_path)
    for path in (directory, directory / 'curve'):
        core.atomic_json(path / 'progress.json', {'host': 'legacy-worker', 'pid': 123,
            'state': 'running', 'updated': 995, 'phase': 'curve', 'event_id': 'same-owner'})
    data = status.snapshot(tmp_path, now=1000)
    assert len([node for node in data['nodes'] if node['current']]) == 1
    assert 'CURRENT RUN 1' in status.render(data, width=160)


@pytest.mark.parametrize('identity_key', ['worker_id', 'pid', 'event_id'])
def test_nested_meters_with_explicit_different_owners_remain_separate(tmp_path, identity_key):
    prepared(tmp_path)
    directory = branch(tmp_path)
    for index, path in enumerate((directory, directory / 'curve')):
        core.atomic_json(path / 'progress.json', {'host': 'same-node', 'pid': 123,
            'state': 'running', 'updated': 995, 'phase': 'curve', identity_key: str(index)})
    data = status.snapshot(tmp_path, now=1000)
    assert len([node for node in data['nodes'] if node['current']]) == 2
    assert 'CURRENT RUN 2' in status.render(data, width=160)


def test_skewed_legacy_meter_without_event_id_is_backed_by_real_lease(tmp_path):
    prepared(tmp_path)
    point = branch(tmp_path) / 'curve'
    core.atomic_json(point / 'progress.json', {'host': 'legacy-peer', 'state': 'running',
                                              'updated': 1, 'phase': 'curve'})
    with (point / '.cost.lock').open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert status.snapshot(tmp_path, now=10000)['tasks'][0]['status'] == 'RUN'
    assert status.snapshot(tmp_path, now=10000)['tasks'][0]['status'] == 'WAIT'


@pytest.mark.parametrize('updated', [1, 9995])
def test_restarted_branch_with_old_claims_shows_one_unattributed_live_lease(tmp_path, updated):
    p = prepared(tmp_path)
    for worker in ('old', 'new'):
        core.atomic_json(tmp_path / f'queue-workers/{worker}.json', {
            'host': 'same-host', 'pid': 123, 'state': 'RUN', 'updated': updated,
            'protocol_id': p['protocol_id'], 'task': 'development/s0-t25/on_policy/selection_reduced'})
    lock = tmp_path / 'development/s0-t25/queue-branches/on_policy--selection_reduced.lock'
    lock.parent.mkdir(parents=True)
    with lock.open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = status.snapshot(tmp_path, now=10000)
        active = [node for node in data['nodes'] if node['current']]
        assert len(active) == 1 and active[0]['worker_id'].startswith('lease-')
        assert active[0]['host'] == 'unknown-owner'
        assert data['tasks'][0]['status'] == 'RUN'
        assert active[0]['work_id'] == active[0]['worker_id']
        assert data['activity'][0]['activity_identity_unconfirmed']
        assert data['tasks'][0]['activity_identity_unconfirmed']
        dashboard = status.dashboard_data(data)
        adapted_node = next(node for node in dashboard['suites'][0]['nodes'] if node['host'] == 'unknown-owner')
        assert adapted_node['work_id'] == active[0]['work_id']
        work = status.display.current_work(dashboard['suites'][0])
        assert len(work) == 1 and work[0][0]['activity_identity_unconfirmed']
        assert 'CURRENT RUN 1' in status.render(data, width=160)
    assert not any(node['current'] for node in status.snapshot(tmp_path, now=10000)['nodes'])


def test_heartbeat_advancing_during_lease_verification_is_not_lost(tmp_path, monkeypatch):
    prepared(tmp_path)
    point = branch(tmp_path) / 'curve'
    path = point / 'progress.json'
    core.atomic_json(path, {'host': 'skewed-peer', 'pid': 5, 'event_id': 'current-curve',
                           'state': 'running', 'updated': 1, 'seconds': 1, 'phase': 'curve'})
    original = status.read
    reads = []
    def advancing(candidate):
        value = original(candidate)
        if candidate == path:
            reads.append(1)
            return {**value, 'updated': len(reads), 'seconds': len(reads)}
        return value
    monkeypatch.setattr(status, 'read', advancing)
    with (point / '.cost.lock').open('w') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = status.snapshot(tmp_path, now=10000)
        assert len(reads) >= 2
        assert data['tasks'][0]['status'] == 'RUN' and data['tasks'][0]['owner_active']
