"""Actual startup registration and read-only plan reuse, without model/GPU loads."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import srgc_launch_probe as probe
from srgc_rebuttal.runtime import lease
from srgc_research.dispatch import qwen_status, startup
from srgc_research.dispatch.gemma4_it import cli
from srgc_research.dispatch.gemma_it_control import RunControl
from srgc_research.tests.test_model_results import make_cohort, saved


def test_registration_precedes_entry_and_is_outside_experiment_root(tmp_path):
    root = tmp_path / 'untouched-experiment'

    def entry(args):
        assert len(startup.snapshot(root)) == 1
        assert startup.snapshot(root)[0]['stage'] == 'starting'
        assert not root.exists()
        startup.progress('packages')
        assert startup.snapshot(root)[0]['stage'] == 'packages'
        return 76

    assert startup.wrap_entry(entry, root)(['all']) == 76
    assert startup.snapshot(root) == []
    assert not root.exists()


def test_registration_hands_off_to_real_worker_without_inventing_run(tmp_path):
    root = tmp_path / 'experiment'
    with startup.registration(root):
        rows = qwen_status.node_rows([], model='gemma4_it', root=root)
        assert len(rows) == 1 and rows[0]['status'] == 'preparing'
        assert rows[0]['source'] == 'startup' and rows[0]['task'] is None
        startup.worker_registered()
        assert startup.snapshot(root) == []


def test_registration_ends_after_failed_startup(tmp_path):
    root = tmp_path / 'experiment'
    with pytest.raises(ValueError, match='failed'), startup.registration(root):
        startup.progress('model_verify')
        raise ValueError('failed')
    assert startup.snapshot(root) == []


def test_stale_or_malformed_startup_never_looks_live(tmp_path):
    root = tmp_path / 'experiment'
    with startup.registration(root):
        row = startup.snapshot(root)[0]
        assert startup.snapshot(root, now=row['heartbeat'] + 31) == []
    (startup.directory(root) / 'bad.json').write_text('[]')
    assert startup.snapshot(root) == []


def test_ready_plans_validate_without_waiting_for_global_prepare_lock(tmp_path, monkeypatch):
    cohort = make_cohort('gemma4_it', tmp_path)
    monkeypatch.setenv('GROUP_VOLUME', cohort.env['GROUP_VOLUME'])
    monkeypatch.setenv('OM_WORK', cohort.env['OM_WORK'])
    monkeypatch.delenv('SRGC_DATASET_PHASE', raising=False)
    monkeypatch.setattr(cli, 'resolve_snapshot', lambda env: (tmp_path / 'model', 'a' * 64))
    monkeypatch.setattr(cli.adapter, 'runtime_packages', dict)
    control = RunControl(SimpleNamespace(root=cohort.root, dataset='all'))
    with lease(cohort.root / '.start-prepare.lock'):
        before = saved(cohort)
        result = control.preparing(lambda *a: pytest.fail('ready plans entered serialized preparation'))(
            ['math', 'mbpp'], cohort.root, dict(cohort.env))
    assert result == [cohort.plans['math'], cohort.plans['mbpp']]
    assert saved(cohort) == before


def test_existing_plan_fast_path_still_rejects_wrong_model_identity(tmp_path, monkeypatch):
    cohort = make_cohort('gemma4_it', tmp_path)
    monkeypatch.setattr(cli, 'resolve_snapshot', lambda env: (tmp_path / 'model', 'b' * 64))
    monkeypatch.setattr(cli.adapter, 'runtime_packages', dict)
    control = RunControl(SimpleNamespace(root=cohort.root, dataset='all'))
    with pytest.raises(ValueError, match='different local weights'):
        control.preparing(lambda *a: None)(['math'], cohort.root, dict(cohort.env))


def test_missing_plans_owned_by_peer_yield_without_blocking_or_gpu_work(tmp_path):
    from srgc_rebuttal import runtime

    control = RunControl(SimpleNamespace(root=tmp_path, dataset='all'))

    def original(datasets, root, environment):
        with runtime.lease(root / '.start-prepare.lock', wait=True):
            pytest.fail('peer-owned prepare lock was bypassed')

    with lease(tmp_path / '.start-prepare.lock'), pytest.raises(SystemExit) as stopped:
        control.preparing(original)(['math', 'mbpp'], tmp_path, {'SRGC_MODEL_CHAIN_CHILD': '1'})
    assert stopped.value.code == 75


def test_stop_is_checked_before_model_verification(tmp_path):
    control = RunControl(SimpleNamespace(root=tmp_path, dataset='all'))
    path = tmp_path / '.startup-control/math/stop.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'immediate': True, 'requested': 1}))
    with patch.object(cli, 'resolve_snapshot', side_effect=AssertionError('unexpected model work')), \
            pytest.raises(KeyboardInterrupt):
        control.preparing(lambda *a: None)(['math'], tmp_path, {})


def test_probe_only_recognizes_actual_module_arguments():
    assert probe.role(['python', '-m', 'srgc_research.dispatch.gemma_it_run']) == 'gemma-it-launcher'
    assert probe.role(['python', '-m', 'other', 'srgc_research.dispatch.gemma_it_run']) is None
    assert probe.role(['echo', 'srgc_research.dispatch.model_chain']) is None


def test_probe_distinguishes_held_and_waiting_kernel_locks(tmp_path):
    (tmp_path / 'locks').write_text(
        '1: FLOCK ADVISORY WRITE 10 00:01:123 0 EOF\n'
        '1: -> FLOCK ADVISORY WRITE 11 00:01:123 0 EOF\n')
    assert probe.lock_rows(tmp_path) == {(10, 0, 1, 123): 'held', (11, 0, 1, 123): 'waiting'}
    assert probe.stage('/group-volume/work/.start-prepare.lock') == 'plan 준비'
    assert probe.stage('/home/user/.ssh/private-key') is None


def test_retries_reuse_one_latest_startup_record_per_allocation(tmp_path):
    root = tmp_path / 'experiment'
    identities = []
    for _ in range(3):
        with startup.registration(root):
            rows = startup.snapshot(root)
            assert len(rows) == 1
            identities.append(rows[0]['worker_id'])
    assert len(set(identities)) == 3
    assert len(list(startup.directory(root).glob('*.json'))) == 1
    assert startup.snapshot(root) == []


def test_launcher_row_shows_why_the_last_child_ended(tmp_path):
    # 2026-10-31: every node cycled through LAUNCH and status gave no reason.
    import time

    root = tmp_path / 'experiment'
    with pytest.raises(SystemExit), startup.registration(root):
        startup.progress('plans_validate')
        raise SystemExit(76)
    host = json.loads(next(startup.directory(root).glob('*.json')).read_text())['host']
    assert startup.recent_exits(root)[host][1] == '기존 plan·입력 검증 중 종료 (exit 76)'
    coordinator = {'status': 'running', 'model': 'gemma4', 'status_model': 'gemma4_it', 'host': host,
                   'root': str(root.resolve()), 'heartbeat_age_seconds': 0, 'active_dataset': 'math',
                   'started': time.time()}
    text = qwen_status.render([], root.resolve(), status_model='gemma4_it', coordinators=[coordinator])
    assert '직전 child' in text and '기존 plan·입력 검증 중 종료 (exit 76)' in text


def test_entry_return_code_is_recorded_as_the_exit(tmp_path):
    root = tmp_path / 'experiment'
    assert startup.wrap_entry(lambda args: 75, root)(['all']) == 75
    assert [value[1] for value in startup.recent_exits(root).values()] == ['worker 초기 등록 중 종료 (exit 75)']


def test_yielded_worker_reason_reaches_the_launcher_row():
    snapshots = [{'workers': [
        {'host': 'n1', 'status': 'yielded', 'heartbeat_age_seconds': 40,
         'idle_reason': 'no claimable tasks; waiting to resume this model'},
        {'host': 'n1', 'status': 'yielded', 'heartbeat_age_seconds': 4000, 'idle_reason': 'old'},
        {'host': 'n2', 'status': 'running', 'heartbeat_age_seconds': 1, 'idle_reason': 'live'}]}]
    exits = qwen_status.previous_exits(snapshots, None, 'qwen35')
    assert exits == {'n1': (40, 'worker yielded (no claimable tasks; waiting to resume this model)')}
