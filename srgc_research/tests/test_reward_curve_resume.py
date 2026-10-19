"""Resume recorder upgrades without relabeling old rewards or accepting changed experiments."""

import json
from unittest.mock import patch

import pytest

from srgc_rebuttal.runtime import atomic_json
from srgc_research.dispatch import reward_curves as curves
from srgc_research.dispatch.reward_curve_results import collect
from srgc_research.tests.test_model_results import (
    cohort as cohort,  # noqa: PLC0414 - pytest fixture registration
)
from srgc_research.tests.test_model_results import finish, read_result
from srgc_research.tests.test_reward_curves import make_engine
from srgc_research.tests.test_rollout_records import make, read


def protocol():
    return {"schema": curves.SCHEMA, "seed": 5, "arm": "random",
            "plan_sha256": "a" * 64, "input_sha256": "b" * 64,
            "implementation_sha256": "c" * 64, "extension_sha256": "d" * 64,
            "interval_updates": 25, "responses_per_prompt": 8,
            "evaluation_ids_sha256": "e" * 64, "question_count": 300,
            "stream": "reporting-evaluation", "training_batch_reward_is_held_out": False}


@pytest.mark.parametrize('seed,arm', [(5, 'random'), (8, 'random'), (9, 'on_policy')])
def test_recorder_update_retains_original_protocol_and_adds_separate_receipt(tmp_path, seed, arm):
    path = tmp_path / f'runs/mbpp/seed-{seed}/reward-curves/{arm}/protocol.json'
    original = {**protocol(), 'seed': seed, 'arm': arm}
    atomic_json(path, original)
    before = path.read_bytes(), path.stat().st_mtime_ns
    proposed = {**original, "extension_sha256": "f" * 64}
    assert curves.bind_protocol(path, proposed, "f" * 64) == original
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    document = {"recording_extension_sha256": "f" * 64}
    assert curves.recording_extension(path.parent, original, document) == "f" * 64
    assert curves.recording_extension(path.parent, original, {}) == "d" * 64
    receipt = path.parent / 'recording-extensions' / ('f' * 64 + '.json')
    before_receipt = receipt.read_bytes(), receipt.stat().st_mtime_ns
    curves.bind_protocol(path, proposed, "f" * 64)
    assert (receipt.read_bytes(), receipt.stat().st_mtime_ns) == before_receipt


@pytest.mark.parametrize('field,value', [
    ('schema', 'different'), ('seed', 6), ('arm', 'sr'), ('plan_sha256', '0' * 64),
    ('input_sha256', '0' * 64), ('implementation_sha256', '0' * 64),
    ('interval_updates', 50), ('responses_per_prompt', 4),
    ('evaluation_ids_sha256', '0' * 64), ('question_count', 299),
    ('stream', 'training'), ('training_batch_reward_is_held_out', True),
    ('extra_setting', 1),
])
def test_real_protocol_changes_fail_before_touching_existing_records(tmp_path, field, value):
    path = tmp_path / 'protocol.json'
    original = protocol()
    atomic_json(path, original)
    before = path.read_bytes()
    proposed = {**original, field: value, 'extension_sha256': 'f' * 64}
    with pytest.raises(ValueError, match=f'protocol changed:.*changed fields:.*{field}'):
        curves.bind_protocol(path, proposed, 'f' * 64)
    assert path.read_bytes() == before
    assert not (tmp_path / 'recording-extensions').exists()


@pytest.mark.parametrize('bad', [None, '../invalid', 'x' * 64, 'f' * 63])
def test_invalid_recorder_hash_cannot_create_provenance(tmp_path, bad):
    path = tmp_path / 'protocol.json'
    with pytest.raises(ValueError, match='SHA-256'):
        curves.bind_protocol(path, protocol(), bad)
    assert not path.exists()


def test_modified_or_missing_receipt_is_not_accepted(tmp_path):
    path = tmp_path / 'protocol.json'
    original = protocol()
    writer = 'f' * 64
    curves.bind_protocol(path, original, writer)
    document = {'recording_extension_sha256': writer}
    receipt = tmp_path / 'recording-extensions' / f'{writer}.json'
    receipt.unlink()
    with pytest.raises(ValueError, match='missing or changed'):
        curves.recording_extension(tmp_path, original, document)
    curves.bind_protocol(path, original, writer)
    changed = json.loads(receipt.read_text())
    changed['protocol_sha256'] = '0' * 64
    atomic_json(receipt, changed)
    with pytest.raises(ValueError, match='receipt changed'):
        curves.bind_protocol(path, original, writer)
    with pytest.raises(ValueError, match='missing or changed'):
        curves.recording_extension(tmp_path, original, document)


def test_resumed_engine_preserves_old_checkpoints_and_exports_actual_recorder_versions(tmp_path):
    pytest.importorskip('torch')
    with patch.object(curves, 'extension_digest', return_value='d' * 64):
        engine, _, plan, data, identity = make_engine(tmp_path, True)
    engine.run_until(50)
    state = engine.state_dict()
    immutable = {p: p.read_bytes() for p in (tmp_path / 'reward-curves').rglob('*')
                 if p.is_file() and p.name != 'training.json' and not p.name.endswith('.lock')}
    previous_training = json.loads((tmp_path / 'reward-curves/sr/training.json').read_text())['points']
    with patch.object(curves, 'extension_digest', return_value='f' * 64):
        restored, backend, *_ = make_engine(tmp_path, True)
    restored.load_state_dict(state)
    restored.run_until(75)
    assert restored.step == 75 and len(backend.evaluations) == 1
    assert all(p.read_bytes() == contents for p, contents in immutable.items())
    training = json.loads((tmp_path / 'reward-curves/sr/training.json').read_text())
    assert training['points'][:25] == previous_training
    assert training['extension_sha256'] == 'd' * 64
    assert training['recording_extension_sha256'] == 'f' * 64
    result = collect(tmp_path, 'sr', plan, identity, data['evaluation_ids'], read)
    assert [(p['step'], p['extension_sha256']) for p in result['held_out_rewards']] == [
        (25, 'd' * 64), (50, 'd' * 64), (75, 'f' * 64)]
    assert len(result['training_rewards']) == 50
    assert result['training_windows_25'][-1]['end_step'] == 75


def test_upgrade_retry_after_response_write_keeps_original_rollout_bytes(tmp_path):
    pytest.importorskip('torch')
    with patch.object(curves, 'extension_digest', return_value='d' * 64):
        engine, *_ = make(tmp_path)
    with patch.object(engine, 'save_training_rewards', side_effect=OSError('interrupted')), pytest.raises(OSError):
        engine.update()
    path = tmp_path / 'reward-curves/on_policy/rollouts/step-26.json.gz'
    before = path.read_bytes()
    with patch.object(curves, 'extension_digest', return_value='f' * 64):
        restored, plan, data, identity = make(tmp_path)
    restored.update()
    assert path.read_bytes() == before
    assert (tmp_path / 'on_policy-latest.pt').exists()
    result = collect(tmp_path, 'on_policy', plan, identity, data['evaluation_ids'], read)
    assert len(result['training_rewards']) == 1
    assert result['rollout_sources'][0]['extension_sha256'] == 'd' * 64
    restored.update()
    result = collect(tmp_path, 'on_policy', plan, identity, data['evaluation_ids'], read)
    assert result['rollout_sources'][1]['extension_sha256'] == 'f' * 64


def test_protocol_mismatch_is_checked_before_the_optimizer_update(tmp_path):
    pytest.importorskip('torch')
    engine, backend, *_ = make_engine(tmp_path, True)
    engine.protocol('sr')
    path = tmp_path / 'reward-curves/sr/protocol.json'
    original = json.loads(path.read_text())
    atomic_json(path, {**original, 'responses_per_prompt': 4})
    with pytest.raises(ValueError, match='responses_per_prompt'):
        engine.update()
    assert engine.step == 25 and backend.optimizer_steps == 0


def test_public_export_reads_new_recorder_receipts_for_every_model(cohort):
    from srgc_research.dispatch import model_results
    from srgc_research.tests.test_reward_curves import history

    folder = finish(cohort, 'math')
    identity = json.loads((folder / 'run.json').read_text())
    identity.pop('status')
    plan_path = cohort.plans['math']
    from srgc_rebuttal.plan import input_path, load_plan
    plan = load_plan(plan_path)
    data = json.loads(input_path(plan_path, plan, 5).read_text())
    bound = {**protocol(), **identity, 'arm': 'on_policy',
             'question_count': len(data['evaluation_ids']),
             'evaluation_ids_sha256': curves.fingerprint(data['evaluation_ids'])}
    directory = folder / 'reward-curves/on_policy'
    atomic_json(directory / 'protocol.json', bound)
    curves.bind_protocol(directory / 'protocol.json', {**bound, 'extension_sha256': 'f' * 64}, 'f' * 64)
    atomic_json(directory / 'training.json', {
        **bound, 'recording_extension_sha256': 'f' * 64, 'step': 26,
        'points': curves.training_points(history(26, 26), start=25, stop=26)})
    assert model_results.export(cohort.model, 'math', environment=cohort.env) == 0
    result = read_result(cohort)
    curve = next(row for row in result['datasets']['math']['reward_curves']
                 if row['seed'] == 5 and row['arm'] == 'on_policy')
    assert [point['step'] for point in curve['training_rewards']] == [26]
    receipt = directory / 'recording-extensions' / ('f' * 64 + '.json')
    receipt.write_text('{}\n')
    assert model_results.export(cohort.model, 'math', environment=cohort.env) == 1
    assert read_result(cohort)['errors']
