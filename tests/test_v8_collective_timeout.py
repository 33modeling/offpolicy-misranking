import os, sys, types, unittest
from datetime import timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class CollectiveTimeoutTest(unittest.TestCase):
    def fake(self):
        calls = []
        dist = types.ModuleType('torch.distributed')
        def init_process_group(*args, **kwargs):
            calls.append(kwargs.get('timeout'))
        dist.init_process_group = init_process_group
        torch = types.ModuleType('torch'); torch.distributed = dist
        return torch, dist, calls

    def run_entry(self, env, timeout):
        torch, dist, calls = self.fake()
        saved = {k: sys.modules.get(k) for k in ('torch', 'torch.distributed')}
        sys.modules['torch'] = torch; sys.modules['torch.distributed'] = dist
        old = os.environ.get('V8_COLLECTIVE_TIMEOUT_SECONDS')
        try:
            if env is None: os.environ.pop('V8_COLLECTIVE_TIMEOUT_SECONDS', None)
            else: os.environ['V8_COLLECTIVE_TIMEOUT_SECONDS'] = env
            import v8_task_entry
            v8_task_entry.extend_collective_timeout()
            v8_task_entry.extend_collective_timeout()   # second call must not wrap twice
            dist.init_process_group('nccl', timeout=timeout)
        finally:
            for k, v in saved.items():
                if v is None: sys.modules.pop(k, None)
                else: sys.modules[k] = v
            if old is None: os.environ.pop('V8_COLLECTIVE_TIMEOUT_SECONDS', None)
            else: os.environ['V8_COLLECTIVE_TIMEOUT_SECONDS'] = old
        return calls

    def test_frozen_1800s_is_raised_to_6h(self):
        self.assertEqual(self.run_entry(None, timedelta(seconds=1800)), [timedelta(hours=6)])

    def test_longer_timeout_is_kept(self):
        self.assertEqual(self.run_entry(None, timedelta(hours=9)), [timedelta(hours=9)])

    def test_zero_keeps_frozen_value(self):
        self.assertEqual(self.run_entry('0', timedelta(seconds=1800)), [timedelta(seconds=1800)])

    def test_custom_value(self):
        self.assertEqual(self.run_entry('7200', timedelta(seconds=1800)), [timedelta(seconds=7200)])

    def test_main_installs_it_before_the_child_loads(self):
        source = (Path(__file__).resolve().parents[1] / 'scripts' / 'v8_task_entry.py').read_text()
        body = source.split('def main():', 1)[1]
        self.assertLess(body.index('extend_collective_timeout()'), body.index('exec_module'))


if __name__ == '__main__':
    unittest.main()
