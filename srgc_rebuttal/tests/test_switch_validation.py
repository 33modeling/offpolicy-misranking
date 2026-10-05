import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from scripts import srgc_replicate_worker as worker
from scripts import srgc_switch_validation_report as report
from scripts.srgc_sr_refresh import _endpoint, extra_arm, make_engine
from scripts.srgc_switch_fixed import SwitchFixedEngine
from scripts.srgc_switch_rules import (ConsecutiveNegativeRule, RULE_ARMS, RULE_PROTOCOL,
                                       SingleNegativeRule)
from srgc_rebuttal.plan import load_plan
from srgc_rebuttal.runtime import atomic_json
from srgc_rebuttal.srgc import Config, Engine, TemporalRule
from srgc_rebuttal.tests import test_extra_arm_launch as fixtures
from srgc_rebuttal.toy_backend import ToyBackend, make_problem

ROOT = fixtures.ROOT


def engine(arm, seed=5):
    features, answers, candidates, validation, _, cache = make_problem(seed)
    backend = ToyBackend(features, answers, projection_dim=64, seed=seed)
    data = dict(candidate_ids=candidates, ranking_validation_ids=validation, cached_rewards=cache)
    config = Config(seed=seed, projection_dim=64)
    if arm in RULE_ARMS:
        return make_engine(arm, backend, data, config)[0]
    return Engine(backend, candidates, validation, cache, arm=arm, config=config)


class RuleTests(unittest.TestCase):
    def test_single_and_consecutive_are_not_the_current_bridge_rule(self):
        single, consecutive, current = SingleNegativeRule(), ConsecutiveNegativeRule(), TemporalRule()
        self.assertTrue(single.observe(25, -2))
        for step, d in ((25, -2), (50, 1)):
            self.assertFalse(consecutive.observe(step, d))
            self.assertFalse(current.observe(step, d))
        self.assertFalse(consecutive.observe(75, -2))
        self.assertTrue(current.observe(75, -2))
        self.assertTrue(consecutive.observe(100, -1))

    def test_zero_gaps_duplicate_nonfinite_and_after_switch(self):
        rule = ConsecutiveNegativeRule()
        self.assertFalse(rule.observe(25, -1))
        self.assertFalse(rule.observe(50, 0))
        self.assertFalse(rule.observe(75, -1))
        self.assertFalse(rule.observe(125, -1))
        with self.assertRaises(ValueError):
            rule.observe(125, -1)
        with self.assertRaises(ValueError):
            rule.observe(150, float('nan'))
        self.assertTrue(rule.observe(150, -1))
        with self.assertRaises(RuntimeError):
            rule.observe(175, -1)

    def test_both_rules_use_own_path_and_stop_all_checks_after_transition(self):
        prefix = engine('on_policy')
        prefix.run_until(25)
        for arm, switch_step in (('switch_single', 25), ('switch_consecutive', 50)):
            with self.subTest(arm=arm):
                fork = engine(arm)
                fork.load_state_dict(prefix.state_dict(), fork_arm='switch_rule')
                with patch('srgc_rebuttal.srgc.gradient_contrast', return_value=-1):
                    fork.run_until(switch_step + 1)
                self.assertEqual(fork.switched_at, switch_step)
                self.assertEqual(fork.history[-1]['selector'], 'sr')
                self.assertGreater(fork.costs['training_wall_seconds'], 0)
                calls, selection_cost = len(fork.backend.score_calls), fork.costs['selection_gpu_seconds']
                with patch('srgc_rebuttal.srgc.gradient_contrast', side_effect=AssertionError('late check')):
                    fork.run_until(80)
                self.assertEqual(len(fork.backend.score_calls), calls)
                self.assertEqual(fork.costs['selection_gpu_seconds'], selection_cost)
                for record in fork.history:
                    if record['d'] is not None:
                        self.assertEqual(len(record['on_ids']), 40)
                        self.assertEqual(len(record['sr_ids']), 40)
                        self.assertEqual(record['scoring_responses_per_prompt'], 8)
                        self.assertEqual(len(record['train_ids']), 4)
                        self.assertIn('on_mean_cos', record)

    def test_resume_retains_negative_window_and_stays_identical(self):
        uninterrupted = engine('switch_consecutive')
        with patch('srgc_rebuttal.srgc.gradient_contrast', return_value=-1):
            uninterrupted.run_until(40)
            resumed = engine('switch_consecutive')
            resumed.load_state_dict(uninterrupted.state_dict())
            uninterrupted.run_until(70)
            resumed.run_until(70)
        self.assertEqual(uninterrupted.switched_at, 50)
        self.assertEqual([r['train_ids'] for r in uninterrupted.history], [r['train_ids'] for r in resumed.history])
        after = engine('switch_consecutive')
        after.load_state_dict(resumed.state_dict())
        self.assertEqual(after.update()['train_ids'], resumed.update()['train_ids'])
        self.assertEqual(after.arm, 'switch_rule')

    def test_rejects_relabelled_or_legacy_rule_checkpoint(self):
        single = engine('switch_single')
        single.run_until(10)
        state = single.state_dict()
        with self.assertRaisesRegex(ValueError, 'protocol'):
            engine('switch_consecutive').load_state_dict(state)
        state.pop('rule_protocol')
        with self.assertRaisesRegex(ValueError, 'protocol'):
            single.load_state_dict(state)

    def test_registry_and_gradients_unchanged_until_first_different_decision(self):
        original, variant = engine('switch'), engine('switch_consecutive')
        with patch('srgc_rebuttal.srgc.gradient_contrast', return_value=1):
            original.run_until(65)
            variant.run_until(65)
        self.assertEqual([r['train_ids'] for r in original.history], [r['train_ids'] for r in variant.history])
        self.assertEqual(original.backend.score_calls, variant.backend.score_calls)
        for arm in RULE_ARMS:
            self.assertEqual(extra_arm(arm), arm)


def row(seed, fixed=None, dataset='math', switch=45, **extra):
    values = {'switch': switch, **dict(zip(worker.TIMING_ARMS, fixed or [30, 31, 32, 33, 34])), **extra}
    return dict(dataset=dataset, seed=seed, arms={arm: {
        'reward_percent': value, 'implementation_sha256': 'verified-runtime',
        'checkpoint_policy': {'attention': 'eager'},
        'cost': {'selection_training_preparation_gpu_seconds': 3600 + value}}
        for arm, value in values.items()})


class AnalysisTests(unittest.TestCase):
    def test_uniform_expectation_requires_all_schedules_and_is_not_a_new_seed(self):
        rows = [row(5), row(6)]
        del rows[1]['arms']['switch_fixed50']
        summary = report.summarize(rows)[0]
        result = next(c for c in summary['comparisons'] if c['right'] == 'uniform_grid_expectation')
        self.assertEqual(result['reward']['n'], 1)
        self.assertEqual(result['reward']['mean'], 13)
        self.assertEqual(summary['loso_folds'], [])
        self.assertNotIn('uniform_grid_expectation', rows[0]['arms'])

    def test_loso_does_not_choose_on_the_held_out_seed(self):
        rows = [row(5, [99, 20, 20, 20, 20]), *[row(seed, [1, 2, 3, 4, 50]) for seed in range(6, 10)]]
        first = report.summarize(rows)[0]
        chosen = next(f for f in first['loso_folds'] if f['seed'] == 5)
        self.assertEqual(chosen['arm'], 'switch_fixed250')
        self.assertNotIn(5, chosen['training_seeds'])
        rows[0]['arms']['switch_fixed50']['reward_percent'] = 0
        second = report.summarize(rows)[0]
        self.assertEqual(chosen, next(f for f in second['loso_folds'] if f['seed'] == 5))

    def test_loso_ties_prefer_earlier_step(self):
        result = report.summarize([row(seed, [30] * 5) for seed in range(5, 10)])[0]
        self.assertEqual({f['arm'] for f in result['loso_folds']}, {'switch_fixed50'})

    def test_missing_costs_and_mismatched_kernels_are_not_zero_or_paired(self):
        rows = [row(5), row(6)]
        rows[0]['arms']['switch'].pop('cost')
        rows[1]['arms']['switch_fixed50']['checkpoint_policy']['attention'] = 'sdpa'
        comparison = report.paired_summary(rows, 'switch', 'switch_fixed50')
        self.assertEqual(comparison['reward']['n'], 1)
        self.assertEqual(comparison['core_cost']['n'], 0)
        self.assertIsNone(comparison['core_cost']['mean'])
        rows[0]['arms']['switch'].pop('checkpoint_policy')
        self.assertEqual(report.paired_summary(rows, 'switch', 'switch_fixed50')['reward']['n'], 0)

    def test_dataset_separation_sample_sd_and_duplicate_rejection(self):
        rows = [row(5), row(6, switch=47), row(5, dataset='mbpp', switch=20)]
        output = report.summarize(rows)
        self.assertEqual([r['dataset'] for r in output], ['math', 'mbpp'])
        result = output[0]['comparisons'][2]['reward']
        self.assertEqual((result['n'], result['mean']), (2, 16))
        self.assertAlmostEqual(result['sample_sd'], 2 ** .5)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            report.summarize([row(5), row(5)])

    def test_different_runtime_generations_are_not_pooled_across_seeds(self):
        rows = [row(5), row(6)]
        for arm in rows[1]['arms'].values():
            arm['implementation_sha256'] = 'another-runtime'
        summaries = report.summarize(rows)
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(s['comparisons'][2]['reward']['n'] == 1 for s in summaries))
        self.assertTrue(all(s['loso_folds'] == [] for s in summaries))

    def test_costs_charge_cache_once_and_do_not_add_phases_to_sessions(self):
        current = row(5, on_policy=40, random=35)
        receipt = lambda seconds: dict(complete=True, recorded_phases=1, total_gpu_seconds=seconds)
        accounting = dict(cache_build={'complete': True, 'invocations': receipt(100)},
                          invocations={'prefix': receipt(200), **{a: receipt(300) for a in current['arms']}},
                          arms={a: dict(complete=True, selection_training_preparation_gpu_seconds=250)
                                for a in current['arms']})
        report.attach_costs(current, accounting)
        self.assertEqual(current['arms']['switch']['cost']['protocol_cold_inclusive_gpu_seconds'], 600)
        self.assertEqual(current['arms']['on_policy']['cost']['protocol_cold_inclusive_gpu_seconds'], 500)
        self.assertEqual(current['arms']['random']['cost']['protocol_cold_inclusive_gpu_seconds'], 500)
        accounting['cache_build'] = None
        report.attach_costs(current, accounting)
        self.assertIsNone(current['arms']['switch']['cost']['protocol_cold_inclusive_gpu_seconds'])
        self.assertEqual(current['arms']['on_policy']['cost']['protocol_cold_inclusive_gpu_seconds'], 500)
        accounting['invocations']['switch']['complete'] = False
        report.attach_costs(current, accounting)
        self.assertIsNone(current['arms']['switch']['cost']['continuation_inclusive_gpu_seconds'])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ExtraArmLaunchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.plan = self.fixture.plan()
        self.folder, self.expected, _ = self.fixture.prefix(self.plan)
        self.task = worker.Task('math', 5, 0, 'switch_single', self.plan)

    def test_new_queue_has_70_unique_jobs_without_changing_previous_queues(self):
        with patch.object(worker, 'default_plan', return_value=self.plan), \
                patch.object(worker, 'route_plan', side_effect=lambda p, **kw: p), \
                patch.object(worker, 'select_plan', side_effect=lambda p, seed: p):
            tasks = worker.tasks_for('all', 'switch_validation')
            self.assertEqual(len(tasks), 70)
            self.assertEqual(len({t.key for t in tasks}), 70)
            self.assertEqual(len(worker.tasks_for('math', 'timing')), 25)
            self.assertEqual(len(worker.tasks_for('all', 'rules')), 20)
            self.assertEqual(len(worker.tasks_for('all', 'all')), 120)
            self.assertEqual(len(worker.tasks_for('all', 'support')), 30)
            self.assertEqual({t.name for t in tasks}, {*worker.TIMING_ARMS, *RULE_ARMS})

    def test_shell_all_actions_and_bad_arguments(self):
        fake = self.fixture.base / 'python'
        fake.write_text(f'#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
        fake.chmod(0o755)
        env = {**os.environ, 'PAIR_PYTHON': str(fake), 'SWITCH_PYTHON': str(fake)}
        script = ROOT / 'scripts/run_srgc_switch_validation.sh'
        for dataset in ('math', 'mbpp', 'all'):
            for action in ('run', 'timing', 'rules', 'status', 'results', 'json'):
                result = subprocess.run(['sh', str(script), dataset, action], cwd='/tmp', env=env,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                output = json.loads(result.stdout)
                self.assertEqual(output[1:3], ['--dataset', dataset])
                if action in ('results', 'json'):
                    self.assertEqual(output[0], 'scripts/srgc_switch_validation_report.py')
                    self.assertEqual(output[3:], ['--json'] if action == 'json' else [])
                else:
                    self.assertEqual(output[-2:], ['--scope', action if action in ('timing', 'rules') else 'switch_validation'])
        for args in ([], ['wrong'], ['math', 'wrong'], ['math', 'run', 'extra']):
            result = subprocess.run(['sh', str(script), *args], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)

    def test_control_endpoint_requires_metadata_and_replayable_checks(self):
        _, raw = self.fixture.endpoint(self.plan)
        arm = 'switch_consecutive'
        raw.update(arm=arm, rule_arm=arm, rule_protocol=RULE_PROTOCOL, switched_at=50,
                   checks=[{'step': 25, 'd': -1}, {'step': 50, 'd': -1}])
        path = self.folder / f'{arm}-endpoint.json'
        atomic_json(path, raw)
        plan = load_plan(self.plan)
        _endpoint(path, self.plan, plan, 5, self.folder, arm)
        report.validate_control(raw, plan, self.folder, arm)
        for change in ({'switched_at': 75}, {'checks': []}, {'checks': raw['checks'] + [{'step': 75, 'd': -1}]}):
            with self.assertRaises((ValueError, RuntimeError)):
                report.validate_control({**raw, **change}, plan, self.folder, arm)
        raw.pop('rule_protocol')
        atomic_json(path, raw)
        with self.assertRaisesRegex(ValueError, 'protocol'):
            _endpoint(path, self.plan, plan, 5, self.folder, arm)

    def test_fixed_control_new_metadata_and_legacy_boundary_checks(self):
        _, raw = self.fixture.endpoint(self.plan)
        arm = 'switch_fixed200'
        raw.update(arm=arm, switched_at=200, fixed_step=200,
                   fixed_transition_protocol=SwitchFixedEngine.TRANSITION_PROTOCOL)
        plan = load_plan(self.plan)
        report.validate_control(raw, plan, self.folder, arm)
        with self.assertRaises(ValueError):
            report.validate_control({**raw, 'fixed_step': 125}, plan, self.folder, arm)
        raw.pop('fixed_transition_protocol')
        progress = dict(seed=5, arm=arm, step=275, history=[dict(checkpoint=step,
                        selector='on_policy' if step < 200 else 'sr') for step in range(25, 275)])
        path = self.folder / f'{arm}-progress.json'
        atomic_json(path, progress)
        report.validate_control(raw, plan, self.folder, arm)
        progress['history'][175]['selector'] = 'on_policy'
        atomic_json(path, progress)
        with self.assertRaisesRegex(ValueError, 'boundary'):
            report.validate_control(raw, plan, self.folder, arm)

    def test_report_is_read_only_and_preserves_valid_results_when_one_is_invalid(self):
        _, raw = self.fixture.endpoint(self.plan)
        raw.update(checkpoint_policy={'attention': 'eager'})
        for arm in ('switch', 'on_policy'):
            atomic_json(self.folder / f'{arm}-endpoint.json', {**raw, 'arm': arm})
        atomic_json(self.folder / 'switch_single-endpoint.json', {**raw, 'arm': 'switch_single'})
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.fixture.storage.rglob('*') if p.is_file()}
        with patch.object(report, 'tasks_for', return_value=[self.task, self.task]):
            result = report.collect('math')
        self.assertEqual(len(result['rows']), 1)
        self.assertEqual(set(result['rows'][0]['arms']), {'switch', 'on_policy'})
        self.assertEqual(len(result['errors']), 1)
        self.assertEqual(result['summaries'][0]['comparisons'][0]['reward']['n'], 1)
        self.assertIsNone(result['rows'][0]['arms']['switch']['cost']['training_gpu_seconds'])
        after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.fixture.storage.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_no_endpoints_does_not_load_prefix_or_claim_job(self):
        with patch.object(report, 'tasks_for', return_value=[self.task]), \
                patch.object(report, 'result_identity', side_effect=AssertionError('prefix accessed')), \
                patch.object(worker, 'sweep', side_effect=AssertionError('job claimed')):
            self.assertEqual(report.collect('math')['errors'], [])

    def test_p0_attention_is_recovered_from_cpu_checkpoint_without_loading_tensors(self):
        try:
            import torch
        except ImportError:
            self.skipTest('CPU PyTorch is needed for reading checkpoint metadata')
        path = self.folder / 'switch-latest.pt'
        state = dict(arm='switch', step=275, config={'seed': 5},
                     checkpoint_policy={'attention': 'sdpa'}, backend={'weight': torch.ones(2)})
        torch.save(state, path)
        policy = report.recorded_policy({}, self.folder, 'switch', 5, 275)
        self.assertEqual(policy['attention'], 'sdpa')
        self.assertEqual(policy['attention_source'], 'final-checkpoint-metadata')
        state.pop('checkpoint_policy')
        torch.save(state, path)
        self.assertTrue(report.recorded_policy({}, self.folder, 'switch', 5, 275)['legacy_eager_default'])
        with self.assertRaisesRegex(ValueError, 'completed arm'):
            report.recorded_policy({}, self.folder, 'switch', 6, 275)

    def test_unknown_attention_is_visible_and_not_assumed_eager(self):
        _, raw = self.fixture.endpoint(self.plan)
        atomic_json(self.folder / 'switch-endpoint.json', {**raw, 'arm': 'switch'})
        with patch.object(report, 'tasks_for', return_value=[self.task]):
            result = report.collect('math')
        self.assertEqual(result['rows'][0]['arms']['switch']['reward_percent'], 50.)
        self.assertTrue(any('pairing excluded' in warning for warning in result['warnings']))

    def test_both_archived_runtimes_can_fork_and_resume_new_rules(self):
        from scripts.srgc_saved_runtime import PREVIOUS_HASH, SAVED_HASH
        for implementation in (PREVIOUS_HASH, SAVED_HASH):
            for name in ('prefix-ready.json', 'run.json'):
                path = self.folder / name
                atomic_json(path, {**json.loads(path.read_text()), 'implementation_sha256': implementation})
            result = subprocess.run([sys.executable, '-c', '''
import sys
from scripts.srgc_saved_runtime import bootstrap
bootstrap(['--plan', sys.argv[1], '--seed', '5'])
from srgc_rebuttal.tests.test_switch_validation import engine
from unittest.mock import patch
prefix = engine('on_policy')
prefix.run_until(25)
for arm in ('switch_single', 'switch_consecutive'):
    variant = engine(arm)
    variant.load_state_dict(prefix.state_dict(), fork_arm='switch_rule')
    with patch('srgc_rebuttal.srgc.gradient_contrast', return_value=-1):
        variant.run_until(40)
        resumed = engine(arm)
        resumed.load_state_dict(variant.state_dict())
        variant.run_until(70)
        resumed.run_until(70)
    assert variant.switched_at == resumed.switched_at
    assert [r['train_ids'] for r in variant.history] == [r['train_ids'] for r in resumed.history]
''', str(self.plan)], cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
