import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts import srgc_extra_status as status
from scripts import srgc_replicate_worker as worker
from scripts import srgc_support_report as report
from scripts.srgc_sr_matched import MatchedSRRefreshEngine
from scripts.srgc_sr_refresh import _endpoint, make_engine, scope_of
from srgc_rebuttal.runtime import atomic_json
from srgc_rebuttal.srgc import Config
from srgc_rebuttal.tests import test_extra_arm_launch as fixtures
from srgc_rebuttal.tests.test_sr_refresh import EvalToyBackend
from srgc_rebuttal.toy_backend import make_problem

ROOT = fixtures.ROOT


class MatchedSRTests(unittest.TestCase):
    def pair(self, seed=5):
        features, answers, candidates, validation, _, _ = make_problem(seed)
        cache = {prompt: [0., 1.] * 4 for prompt in candidates}
        data = dict(candidate_ids=candidates, ranking_validation_ids=validation, cached_rewards=cache)
        engines = []
        for arm in ('sr_hold', 'sr_refresh_matched'):
            backend = EvalToyBackend(features, answers, projection_dim=64, seed=seed)
            backend.evaluate = lambda ids, **kw: {prompt: sum(cache[prompt]) / 8 for prompt in ids}
            engine, _ = make_engine(arm, backend, data, Config(seed=seed, projection_dim=64))
            engines.append(engine)
        return engines, cache

    def test_equal_rewards_give_identical_selected_prompts_including_ties(self):
        for seed in range(5, 10):
            (held, fresh), cache = self.pair(seed)
            for step in (25, 50, 150):
                held.step = fresh.step = step
                ids, _, ranked = held._refresh_ranking()
                other_ids, _, other_ranked = fresh._refresh_ranking()
                self.assertEqual(ids, other_ids)
                self.assertEqual(ranked, other_ranked)
                self.assertEqual(fresh._rank_refreshed(tuple(reversed(ids)), cache), ranked)
            self.assertEqual(scope_of('sr_refresh_matched'), 'candidates')

    def test_fresh_scores_still_take_precedence_over_tie_order(self):
        (_, fresh), cache = self.pair()
        ids = fresh._draw_candidates()
        ordered = fresh._rank_refreshed(ids, cache)
        rewards = {prompt: [0.] * 8 for prompt in ids}
        rewards[ordered[-1]] = [0., 1.] * 4
        self.assertEqual(fresh._rank_refreshed(ids, rewards)[0], ordered[-1])

    def test_matched_checkpoint_resumes_but_cannot_relabel_legacy_refresh(self):
        (_, fresh), _ = self.pair()
        fresh.run_until(7)
        state = fresh.state_dict()
        (_, resumed), _ = self.pair()
        resumed.load_state_dict(state)
        self.assertEqual(resumed.step, 7)
        expected = fresh.update()['train_ids']
        self.assertEqual(resumed.update()['train_ids'], expected)
        legacy = {key: value for key, value in state.items() if key != 'sr_tie_protocol'}
        with self.assertRaisesRegex(ValueError, 'tie protocol'):
            resumed.load_state_dict(legacy)

    def test_prefix_fork_preserves_step_and_has_no_extra_gradient_scoring(self):
        (prefix, fresh), _ = self.pair()
        prefix.arm = 'on_policy'
        prefix.run_until(25)
        fresh.load_state_dict(prefix.state_dict(), fork_arm='sr_refresh')
        record = fresh.update()
        self.assertEqual(record['selection_step'], 25)
        self.assertEqual(fresh.backend.score_calls, [])
        self.assertEqual(len(record['train_ids']), 4)


class SupportQueueAndStatusTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ExtraArmLaunchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.plan = self.fixture.plan()
        self.folder, self.identity, self.prefix = self.fixture.prefix(self.plan)
        self.task = worker.Task('math', 5, 0, 'sr_hold', self.plan)

    def test_support_contains_only_the_30_matched_controls(self):
        with patch.object(worker, 'default_plan', return_value=self.plan), \
                patch.object(worker, 'route_plan', side_effect=lambda p, **kw: p), \
                patch.object(worker, 'select_plan', side_effect=lambda p, s: p):
            tasks = worker.tasks_for('all', 'support')
            self.assertEqual(len(tasks), 30)
            self.assertEqual(len({t.key for t in tasks}), 30)
            self.assertEqual({(t.dataset, t.seed, t.arm) for t in tasks},
                             {(d, s, a) for d in ('math', 'mbpp') for s in range(5, 10)
                              for a in worker.SUPPORT_ARMS})
            self.assertTrue(all(t.repeat == 0 for t in tasks))
            self.assertFalse(any(t.arm.startswith('switch_fixed') for t in tasks))
            self.assertEqual(len(worker.tasks_for('math', 'support')), 15)
            self.assertEqual(len(worker.tasks_for('all', 'all')), 120)

    def test_status_is_read_only_and_never_claims_or_checks_gpus(self):
        progress = self.folder / 'sr_hold-progress.json'
        atomic_json(progress, {'seed': 5, 'arm': 'sr_hold', 'step': 75})
        atomic_json(self.task.receipt, {'status': 'running', 'started': 80, 'attempt': 1})
        atomic_json(self.task.out / 'launches/sr_hold/test/worker.json',
                    {'status': 'running', 'heartbeat': 99, 'host': 'remote-node'})
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.fixture.storage.rglob('*') if p.is_file()}
        with patch.object(worker, 'node_available', side_effect=AssertionError('GPU access')), \
                patch.object(worker, 'sweep', side_effect=AssertionError('job claimed')):
            row = status.inspect(self.task, now=100)
        self.assertEqual((row['state'], row['step'], row['host']), ('reported_running', 75, 'remote-node'))
        after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.fixture.storage.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_status_distinguishes_stale_workers_and_unverified_endpoints(self):
        worker_path = self.task.out / 'launches/sr_hold/test/worker.json'
        atomic_json(worker_path, {'status': 'running', 'heartbeat': 1})
        self.assertEqual(status.inspect(self.task, now=1000)['state'], 'stale_worker')
        atomic_json(self.task.out / 'sr_hold-endpoint.json', {'fixture': True})
        self.assertEqual(status.inspect(self.task)['state'], 'endpoint_unverified')
        atomic_json(self.task.receipt, {'status': 'complete', 'verified_files': worker.signature(self.task)})
        self.assertEqual(status.inspect(self.task)['state'], 'complete')
        atomic_json(self.task.out / 'sr_hold-endpoint.json', {'changed': True})
        self.assertEqual(status.inspect(self.task)['state'], 'endpoint_unverified')

    def test_status_has_waiting_failed_and_corrupt_states(self):
        self.assertEqual(status.inspect(self.task)['state'], 'waiting')
        (self.folder / 'prefix-ready.json').unlink()
        self.assertEqual(status.inspect(self.task)['state'], 'waiting_prefix')
        atomic_json(self.task.receipt, {'status': 'failed', 'attempt': 3})
        self.assertEqual(status.inspect(self.task)['state'], 'failed')
        atomic_json(self.task.out / 'sr_hold-progress.json', {'seed': 99, 'arm': 'sr_hold', 'step': 50})
        self.assertEqual(status.inspect(self.task)['state'], 'invalid')

    def test_read_only_all_status_keeps_healthy_dataset_when_other_plan_is_missing(self):
        def tasks(dataset, scope):
            if dataset == 'math':
                raise ValueError('missing math plan')
            return [self.task]
        output = io.StringIO()
        with patch.object(status, 'tasks_for', side_effect=tasks), contextlib.redirect_stdout(output):
            rc = status.main(['--dataset', 'all', '--scope', 'support', '--json'])
        report = json.loads(output.getvalue())
        self.assertEqual(rc, 1)
        self.assertEqual(len(report['rows']), 1)
        self.assertIn('missing math plan', report['errors'][0])

    def test_matched_endpoint_requires_its_own_protocol_receipt(self):
        path, value = self.fixture.endpoint(self.plan)
        arm = 'sr_refresh_matched'
        path = path.with_name(arm + '-endpoint.json')
        atomic_json(path, {**value, 'arm': arm})
        from srgc_rebuttal.plan import load_plan
        plan = load_plan(self.plan)
        with self.assertRaisesRegex(ValueError, 'tie protocol'):
            _endpoint(path, self.plan, plan, 5, self.folder, arm)
        atomic_json(path, {**value, 'arm': arm, 'sr_tie_protocol': MatchedSRRefreshEngine.SR_TIE_PROTOCOL})
        self.assertEqual(_endpoint(path, self.plan, plan, 5, self.folder, arm)['reward_percent'], 50.)

    def test_support_shell_runs_or_reports_without_seed_arguments(self):
        fake = self.fixture.base / 'python'
        fake.write_text(f'#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
        fake.chmod(0o755)
        env = {**os.environ, 'PAIR_PYTHON': str(fake), 'SWITCH_PYTHON': str(fake)}
        for dataset in ('math', 'mbpp', 'all'):
            for action in ('run', 'status', 'results'):
                result = subprocess.run(['sh', str(ROOT / 'scripts/run_srgc_support.sh'), dataset, action],
                                        cwd='/tmp', env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                script = {'run': 'srgc_replicate_worker.py', 'status': 'srgc_extra_status.py',
                          'results': 'srgc_support_report.py'}[action]
                expected = ['scripts/' + script, '--dataset', dataset]
                if action != 'results':
                    expected += ['--scope', 'support']
                self.assertEqual(json.loads(result.stdout), expected)

    def test_support_report_excludes_invalid_results_and_keeps_healthy_arms(self):
        _, value = self.fixture.endpoint(self.plan)
        value["checkpoint_policy"] = {"attention": "eager"}
        atomic_json(self.folder / 'on_policy-endpoint.json', {**value, 'arm': 'on_policy'})
        atomic_json(self.folder / 'sr_hold-endpoint.json', {**value, 'arm': 'sr_hold', 'seed': 6})
        atomic_json(self.folder / 'direction_removed-endpoint.json', {**value, 'arm': 'direction_removed'})
        with patch.object(report, 'tasks_for', return_value=[self.task, self.task]):
            result = report.collect('math')
        self.assertEqual(len(result['rows']), 1)
        arms = result['rows'][0]['arms']
        self.assertNotIn('sr_hold', arms)
        self.assertEqual(arms['on_policy']['reward_percent'], 50.)
        self.assertEqual(len(result['errors']), 1)
        ranking = result['comparisons'][0]
        self.assertEqual((ranking['n'], ranking['mean_pp'], ranking['sample_sd_pp']), (1, 0., None))

    def test_matched_sr_forks_and_resumes_the_archived_prefix_engine(self):
        from scripts.srgc_saved_runtime import SAVED_HASH
        for name in ('prefix-ready.json', 'run.json'):
            path = self.folder / name
            atomic_json(path, {**json.loads(path.read_text()), 'implementation_sha256': SAVED_HASH})
        result = subprocess.run([sys.executable, '-c', '''
import sys
from scripts.srgc_saved_runtime import bootstrap
bootstrap(['--plan', sys.argv[1], '--seed', '5'])
from srgc_rebuttal.srgc import Engine
from srgc_rebuttal.tests.test_support_experiments import MatchedSRTests
assert Engine.SAMPLING_PROTOCOL == 'random-candidate40-training4-contrast40-v2'
(held, fresh), cache = MatchedSRTests().pair()
held.arm = 'on_policy'
held.run_until(25)
fresh.load_state_dict(held.state_dict(), fork_arm='sr_refresh')
fresh.update()
(_, resumed), _ = MatchedSRTests().pair()
resumed.load_state_dict(fresh.state_dict())
assert resumed.update()['train_ids'] == fresh.update()['train_ids']
assert resumed.backend.score_calls == []
''', str(self.plan)], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_report_pairs_training_seeds_and_does_not_fill_missing_with_zero(self):
        def row(dataset, seed, **values):
            return dict(dataset=dataset, seed=seed,
                        arms={key: {'reward_percent': value, 'implementation_sha256': 'runtime',
                                    'checkpoint_policy': {'attention': 'eager'}}
                              for key, value in values.items()})
        results = report.summarize([
            row('math', 5, on_policy=40, direction_removed=30),
            row('math', 6, on_policy=31, direction_removed=35),
            row('math', 7, on_policy=99),
            row('mbpp', 5, on_policy=20, direction_removed=20),
        ])
        self.assertEqual(results[0]['seeds'], [5, 6])
        self.assertEqual(results[0]['differences_pp'], [10, -4])
        self.assertEqual(results[0]['mean_pp'], 3.)
        self.assertAlmostEqual(results[0]['sample_sd_pp'], 98 ** .5)
        self.assertEqual(results[1]['n'], 0)
        self.assertIsNone(results[1]['mean_pp'])
        self.assertEqual(results[4]['n'], 1)
        self.assertEqual(results[4]['mean_pp'], 0.)

    def test_report_is_read_only_when_no_endpoints_exist(self):
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.fixture.storage.rglob('*') if p.is_file()}
        with patch.object(report, 'tasks_for', return_value=[self.task]), \
                patch.object(report, 'result_identity', side_effect=AssertionError('prefix loaded')):
            result = report.collect('math')
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['rows'][0]['arms'], {})
        after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.fixture.storage.rglob('*') if p.is_file()}
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
