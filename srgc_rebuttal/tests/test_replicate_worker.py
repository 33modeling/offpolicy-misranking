import csv
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from scripts import srgc_replicate_worker as worker
from srgc_rebuttal.runtime import atomic_json, lease
from srgc_rebuttal.tests import test_extra_arm_launch as fixtures

ROOT, SCRIPT = fixtures.ROOT, fixtures.SCRIPT


def hold_task(task, ready, release):
    def run(current, handle):
        ready.put(current.key)
        release.wait(10)
        return 143
    worker.sweep([task], runner=run)


class ReplicateWorkerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ExtraArmLaunchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.plan = self.fixture.plan()
        self.fixture.prefix(self.plan)
        self.tasks = [worker.Task('math', 5, 1, arm, self.plan) for arm in ('sr', 'switch')]

    @staticmethod
    def complete(task, handle):
        atomic_json(task.out / 'replicate.json', {'fixture': True})
        atomic_json(task.out / f'{task.arm}-run.json', {'fixture': True})
        atomic_json(task.out / f'{task.arm}-endpoint.json', {'fixture': True})
        return 0

    def test_planned_matrix_is_exactly_40_replicates_not_p0(self):
        mbpp = self.fixture.plan('mbpp')
        plans = {'math': self.plan, 'mbpp': mbpp}
        with patch.object(worker, 'default_plan', side_effect=lambda repo, name, env, **kw: plans[name]), \
                patch.object(worker, 'route_plan', side_effect=lambda p, **kw: p), \
                patch.object(worker, 'select_plan', side_effect=lambda p, s: p):
            tasks = worker.tasks_for('all')
            self.assertEqual(len(tasks), 40)
            self.assertEqual(len({t.key for t in tasks}), 40)
            self.assertEqual({(t.dataset, t.seed, t.repeat, t.arm) for t in tasks},
                             {(d, s, k, a) for d in ('math', 'mbpp') for s in range(5,10)
                              for k in (1,2) for a in ('sr','switch')})
            self.assertEqual(len(worker.tasks_for('math')), 20)

    def test_two_workers_take_different_tasks(self):
        context = multiprocessing.get_context('fork')
        ready, release = context.Queue(), context.Event()
        process = context.Process(target=hold_task, args=(self.tasks[0], ready, release))
        process.start()
        try:
            self.assertEqual(ready.get(timeout=5), self.tasks[0].key)
            seen = []
            def run(task, handle):
                seen.append(task.key)
                return self.complete(task, handle)
            counts, code = worker.sweep(self.tasks, runner=run)
            self.assertEqual(code, 0)
            self.assertEqual(counts['busy'], 1)
            self.assertEqual(seen, [self.tasks[1].key])
        finally:
            release.set()
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join()

    def test_all_extra_tasks_match_recorded_120_rows_and_priority(self):
        with patch.object(worker, 'default_plan', return_value=self.plan), \
                patch.object(worker, 'route_plan', side_effect=lambda p, **kw: p), \
                patch.object(worker, 'select_plan', side_effect=lambda p, s: p):
            tasks = worker.tasks_for('all', 'all')
            with (ROOT / 'docs/REBUTTAL_EXTRA_TASKS.tsv').open() as handle:
                expected = {(r['dataset'], int(r['seed']), r['arm']) for r in csv.DictReader(handle, delimiter='\t')}
            self.assertEqual({(t.dataset, t.seed, t.name) for t in tasks}, expected)
            self.assertEqual(len(tasks), 120)
            self.assertEqual(len({t.key for t in tasks}), 120)
            self.assertTrue(all(t.name == 'switch_fixed200' for t in tasks[:10]))
            self.assertTrue(all(t.repeat in (1, 2) for t in tasks[10:50]))
            self.assertEqual(len(worker.tasks_for('math', 'all')), 60)
            self.assertEqual(len(worker.tasks_for('mbpp', 'all')), 60)

    def test_all_scopes_use_existing_worker_arm_names(self):
        from scripts.srgc_sr_refresh import extra_arm
        for scope in worker.SCOPES:
            for repeat, arm in worker.conditions(scope):
                task = worker.Task('math', 5, repeat, arm, self.plan)
                self.assertEqual(extra_arm(task.name), task.name)
                if not repeat:
                    self.assertEqual(task.out, task.folder)

    def test_fixed_task_completion_and_manual_lock_are_respected(self):
        task = worker.Task('math', 5, 0, 'switch_fixed200', self.plan)
        with lease(task.folder / '.switch_fixed200.launch.lock'):
            counts, code = worker.sweep([task], runner=lambda *a: self.fail('duplicate fixed started'))
            self.assertEqual(counts['busy'], 1)
        def finish(current, handle):
            atomic_json(current.out / f'{current.arm}-endpoint.json', {'fixture': True})
            return 0
        self.assertEqual(worker.sweep([task], runner=finish)[1], 0)
        counts, code = worker.sweep([task], runner=lambda *a: self.fail('fixed rerun'))
        self.assertEqual(counts['complete'], 1)
        self.assertIsNone(code)
        self.assertFalse((task.out / 'replicate.json').exists())

    def test_dataset_only_shell_and_scoped_queues_need_no_seed(self):
        fake = self.fixture.base / 'python'
        fake.write_text(f'#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
        fake.chmod(0o755)
        env = {**os.environ, 'PAIR_PYTHON': str(fake), 'SWITCH_PYTHON': str(fake)}
        for dataset in ('math', 'mbpp', 'all'):
            for scope in (None, 'switch_fixed200', 'candidates', 'switch_repeat', 'sr_hold', 'pool', 'direction'):
                args = [dataset] + ([scope] if scope else [])
                result = subprocess.run(['sh', str(SCRIPT), *args], cwd='/tmp', env=env,
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), ['scripts/srgc_replicate_worker.py',
                                 '--dataset', dataset, '--scope', scope or 'all'])

    def test_manual_launch_lock_is_respected(self):
        task = self.tasks[0]
        with lease(task.out / '.sr.launch.lock'):
            counts, code = worker.sweep([task], runner=lambda *a: self.fail('duplicate started'))
        self.assertIsNone(code)
        self.assertEqual(counts['busy'], 1)

    def test_verified_completion_is_shared_and_changed_files_rechecked(self):
        original = {p: p.read_bytes() for p in self.fixture.storage.rglob('*') if p.is_file()}
        task = self.tasks[0]
        self.assertEqual(worker.sweep([task], runner=self.complete)[1], 0)
        counts, code = worker.sweep([task], runner=lambda *a: self.fail('completed task rerun'))
        self.assertIsNone(code)
        self.assertEqual(counts['complete'], 1)
        atomic_json(task.out / 'sr-endpoint.json', {'changed': True})
        self.assertEqual(worker.sweep([task], runner=lambda *a: 2)[1], 2)
        self.assertEqual(json.loads(task.receipt.read_text())['status'], 'failed')
        self.assertEqual(original, {p: p.read_bytes() for p in original})

    def test_failures_have_shared_limit_even_with_invalid_endpoint(self):
        task = self.tasks[0]
        self.complete(task, None)
        for _ in range(3):
            self.assertEqual(worker.sweep([task], retry_delay=0, runner=lambda *a: 2)[1], 2)
        counts, code = worker.sweep([task], retry_delay=0, runner=lambda *a: self.fail('retry limit bypassed'))
        self.assertIsNone(code)
        self.assertEqual(counts['failed'], 1)

    def test_busy_and_interruption_do_not_consume_attempts(self):
        task = self.tasks[0]
        for code in (75, 130, 143):
            self.assertEqual(worker.sweep([task], runner=lambda *a: code)[1], code)
            self.assertEqual(json.loads(task.receipt.read_text())['attempt'], 0)

    def test_success_without_endpoint_is_failure_and_missing_prefix_waits(self):
        task = self.tasks[0]
        self.assertEqual(worker.sweep([task], runner=lambda *a: 0)[1], 2)
        (task.folder / 'prefix-ready.json').unlink()
        counts, code = worker.sweep([task], runner=lambda *a: self.fail('missing prefix started'))
        self.assertIsNone(code)
        self.assertEqual(counts['waiting'], 1)

    def test_waiting_fixed_task_does_not_block_ready_replicate(self):
        from dataclasses import replace
        pending = worker.Task('math', 6, 0, 'switch_fixed200', self.plan)
        ready = replace(self.tasks[0], seed=5)
        seen = []
        def finish(task, handle):
            seen.append(task.key)
            return self.complete(task, handle)
        counts, code = worker.sweep([pending, ready], runner=finish)
        self.assertEqual(code, 0)
        self.assertEqual(counts['waiting'], 1)
        self.assertEqual(seen, [ready.key])

    def test_manual_completion_is_validated_after_retry_limit(self):
        task = worker.Task('math', 5, 0, 'switch_fixed200', self.plan)
        for _ in range(3):
            worker.sweep([task], retry_delay=0, runner=lambda *args: 1)
        atomic_json(task.out / 'switch_fixed200-endpoint.json', {'manual completion': True})
        seen = []
        def validate(current, handle):
            seen.append(current.key)
            return 0
        self.assertEqual(worker.sweep([task], runner=validate)[1], 0)
        self.assertEqual(seen, [task.key])
        self.assertEqual(worker.sweep([task], runner=lambda *args: self.fail('reran complete task'))[0]['complete'], 1)

    def test_busy_gpu_idles_without_claiming(self):
        with patch.object(worker, 'tasks_for', return_value=self.tasks), \
                patch.object(worker, 'node_available', return_value=False), \
                patch.object(worker, 'sweep', side_effect=AssertionError('claimed on busy node')), \
                patch.object(worker.time, 'sleep', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                worker.main(['--dataset', 'all'])

    def test_gpu_lease_blocks_same_devices_but_not_another_allocation(self):
        from srgc_rebuttal import cluster
        from scripts.srgc_process_guard import canonical_lock_root
        with patch.dict(os.environ, self.fixture.environment, clear=True), \
                patch('scripts.srgc_process_guard.reap_orphans'):
            root = canonical_lock_root()
            with cluster.device_leases(root, ('GPU-a', 'GPU-b', 'GPU-c', 'GPU-d')):
                with patch.object(cluster, 'gpu_identity', return_value=('0,1,2,3', ('GPU-a', 'GPU-b', 'GPU-c', 'GPU-d'))):
                    self.assertFalse(worker.node_available())
                with patch.object(cluster, 'gpu_identity', return_value=('0,1,2,3', ('GPU-e', 'GPU-f', 'GPU-g', 'GPU-h'))):
                    self.assertTrue(worker.node_available())

    def test_main_exits_when_all_complete_and_reports_exhausted_failures(self):
        with patch.object(worker, 'tasks_for', return_value=self.tasks), \
                patch.object(worker, 'node_available', return_value=True):
            with patch.object(worker, 'sweep', return_value=(dict(complete=2, failed=0, busy=0, waiting=0), None)):
                self.assertEqual(worker.main(['--dataset', 'all']), 0)
            with patch.object(worker, 'sweep', return_value=(dict(complete=1, failed=1, busy=0, waiting=0), None)):
                self.assertEqual(worker.main(['--dataset', 'all']), 1)

    def test_shell_single_command_uses_matching_python(self):
        fake = self.fixture.base / 'python'
        fake.write_text(f'#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
        fake.chmod(0o755)
        for dataset in ('all', 'math', 'mbpp'):
            env = {**os.environ, 'PAIR_PYTHON': str(fake), 'SWITCH_PYTHON': str(fake)}
            result = subprocess.run(['sh', str(SCRIPT), dataset, 'replicate'], env=env,
                                    cwd='/tmp', capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), ['scripts/srgc_replicate_worker.py', '--dataset', dataset])

    def test_run_task_forwards_sigterm_and_waits_for_child(self):
        fake = self.fixture.base / 'python'
        ready, stopped = self.fixture.base / 'ready', self.fixture.base / 'stopped'
        fake.write_text(f'#!{sys.executable}\nimport pathlib, signal, time\n'
                        f'def stop(*a):\n    pathlib.Path({str(stopped)!r}).write_text("stopped")\n    raise SystemExit(143)\n'
                        'signal.signal(signal.SIGTERM, stop)\n'
                        f'pathlib.Path({str(ready)!r}).write_text("ready")\n'
                        'while True: time.sleep(.05)\n')
        fake.chmod(0o755)
        code = ('from pathlib import Path\n'
                'from scripts.srgc_replicate_worker import Task, run_task\n'
                'from srgc_rebuttal.runtime import lease\n'
                f'task=Task("math",5,1,"sr",Path({str(self.plan)!r}))\n'
                'with lease(task.out/".sr.dispatch.lock") as handle:\n'
                '    raise SystemExit(run_task(task,handle))\n')
        process = subprocess.Popen([sys.executable, '-c', code], cwd=ROOT,
                                   env={**os.environ, 'PAIR_PYTHON': str(fake)},
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
                time.sleep(.05)
            self.assertTrue(ready.exists())
            process.send_signal(signal.SIGTERM)
            _, error = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 143, error)
            self.assertTrue(stopped.exists())
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)


if __name__ == '__main__':
    unittest.main()
