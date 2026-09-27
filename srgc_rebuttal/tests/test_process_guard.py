from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import srgc_process_guard as guard
from srgc_rebuttal import cluster
from srgc_rebuttal.runtime import Busy, lease
from srgc_rebuttal.tests.test_cluster import simulate_worker, write_inputs

PLAN = "/tmp/srgc-guard-test/experiments/additional_seeds.json"


def spawn(marker, *, owner=False):
    """A detached python process whose cmdline looks like an SRGC child, optionally under a fake launcher."""
    child_code = "import time; time.sleep(120)"
    if owner:
        owner_code = ("import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', "
                      f"{child_code!r}, {marker!r}, '--plan', {PLAN!r}], start_new_session=True); "
                      "print(p.pid, flush=True); time.sleep(120)")
        proc = subprocess.Popen([sys.executable, "-c", owner_code, "run_srgc_rebuttal.py", "worker"],
                                stdout=subprocess.PIPE, text=True, start_new_session=True)
        return proc, int(proc.stdout.readline())
    return subprocess.Popen([sys.executable, "-c", child_code, marker, "--plan", PLAN], start_new_session=True), None


def alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[-1][:1] != "Z"
    except ProcessLookupError:
        return False


def simulate_guarded_node(plan, barrier, node):
    # Both allocations expose local indices 0..3 but have different physical UUIDs.
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}), \
            guard.process_guard(plan), \
            cluster.device_leases(Path(plan).parent / "gpu-node-locks",
                                  tuple(f"node-{node}-GPU-{index}" for index in range(4))):
        return simulate_worker(plan, barrier)


class ProcessGuardTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="srgc-process-guard-")
        self.addCleanup(folder.cleanup)
        plan = Path(folder.name) / "plan.json"
        self.plan_patch = patch(__name__ + ".PLAN", str(plan))
        self.plan_patch.start()
        self.addCleanup(self.plan_patch.stop)
        self.child_pids = []

    def tearDown(self):
        for pid in self.child_pids:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        for proc in getattr(self, "procs", []):
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)

    def test_shell_wrappers_are_never_targets(self):
        self.assertFalse(guard.is_target(f"bash -c python -m srgc_rebuttal.run_experiment --plan {PLAN}", PLAN))
        self.assertTrue(guard.is_target(f"/x/bin/python3 -m torch.distributed.run -m srgc_rebuttal.run_experiment --plan {PLAN}", PLAN))
        self.assertFalse(guard.is_target(f"python3 -m srgc_rebuttal.run_experiment --plan /other/plan.json", PLAN))

    def test_orphan_of_plan_is_reaped_but_owned_and_foreign_processes_survive(self):
        self.procs = []
        orphan, _ = spawn("srgc_rebuttal.run_experiment")
        foreign, _ = spawn("some_other_program")
        owner, owned = spawn("srgc_rebuttal.run_experiment", owner=True)
        self.procs += [orphan, foreign, owner]
        self.child_pids.append(owned)
        time.sleep(0.3)
        found = guard.orphan_pids(PLAN)
        self.assertIn(orphan.pid, found)
        self.assertNotIn(foreign.pid, found)
        self.assertNotIn(owned, found)
        reaped = guard.reap_orphans(PLAN, grace=5)
        self.assertEqual(reaped, [orphan.pid])
        orphan.wait(timeout=10)
        self.assertFalse(alive(orphan.pid))
        self.assertTrue(alive(foreign.pid))
        self.assertTrue(alive(owned))

    def test_guarded_run_child_kills_detached_grandchildren_after_child_exits(self):
        self.procs = []
        marker = Path(tempfile.mkdtemp()) / "grandchild.pid"
        # The child spawns a setsid grandchild (like a torchrun rank) and exits, leaving it behind.
        command = [sys.executable, "-c",
                   "import subprocess, sys, time; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], "
                   f"start_new_session=True); open({str(marker)!r}, 'w').write(str(p.pid)); time.sleep(1)"]
        with tempfile.TemporaryDirectory() as folder:
            run_child = guard.guarded_run_child(cluster.run_child, PLAN)
            code = run_child(command, Path(folder) / "task.log", dict(os.environ), interval=0.2)
        self.assertEqual(code, 0)
        grandchild = int(marker.read_text())
        deadline = time.monotonic() + 10
        while alive(grandchild) and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertFalse(alive(grandchild))

    def test_process_guard_patches_and_restores_cluster_run_child(self):
        original = cluster.run_child
        with guard.process_guard(PLAN):
            self.assertIsNot(cluster.run_child, original)
        self.assertIs(cluster.run_child, original)

    def test_orphan_gpu_lease_is_released_before_worker_admission(self):
        # A leftover torchrun can hold the inherited GPU fd without GPU work.
        # Recovery must happen on guard entry, before worker() acquires this fd.
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "gpu-node-locks/gpu.lock"
            lock.parent.mkdir()
            child = ("import fcntl, time; "
                     f"handle = open({str(lock)!r}, 'a+'); "
                     "fcntl.flock(handle, fcntl.LOCK_EX); print('locked', flush=True); time.sleep(120)")
            proc = subprocess.Popen([sys.executable, "-c", child, "torch.distributed.run", "--plan", PLAN],
                                    stdout=subprocess.PIPE, text=True, start_new_session=True)
            self.procs = [proc]
            self.assertEqual(proc.stdout.readline().strip(), "locked")
            with self.assertRaises(Busy), lease(lock):
                pass
            original = cluster.run_child
            with guard.process_guard(PLAN):
                with lease(lock):
                    self.assertFalse(alive(proc.pid))
                self.assertIsNot(cluster.run_child, original)
            proc.wait(timeout=10)
            proc.stdout.close()
            self.assertIs(cluster.run_child, original)

    def test_initial_recovery_keeps_a_live_workers_gpu_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "gpu-node-locks/gpu.lock"
            lock.parent.mkdir()
            child = ("import fcntl, time; "
                     f"handle = open({str(lock)!r}, 'a+'); "
                     "fcntl.flock(handle, fcntl.LOCK_EX); print('locked', flush=True); time.sleep(120)")
            owner_code = ("import subprocess, sys, time; "
                          f"p = subprocess.Popen([sys.executable, '-c', {child!r}, "
                          f"'torch.distributed.run', '--plan', {PLAN!r}], "
                          "stdout=subprocess.PIPE, text=True, start_new_session=True); "
                          "p.stdout.readline(); print(p.pid, flush=True); time.sleep(120)")
            owner = subprocess.Popen([sys.executable, "-c", owner_code, "run_srgc_rebuttal.py", "worker"],
                                     stdout=subprocess.PIPE, text=True, start_new_session=True)
            self.procs = [owner]
            child_pid = int(owner.stdout.readline())
            try:
                with guard.process_guard(PLAN):
                    self.assertTrue(alive(owner.pid))
                    self.assertTrue(alive(child_pid))
                    with self.assertRaises(Busy), lease(lock):
                        pass
            finally:
                os.kill(child_pid, 9)
                owner.stdout.close()

    def test_two_allocations_share_queue_without_sharing_device_locks_or_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = write_inputs(Path(directory), pending=True)
            context = multiprocessing.get_context("spawn")
            with context.Manager() as manager:
                barrier = manager.Barrier(2)
                with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
                    futures = [pool.submit(simulate_guarded_node, str(plan), barrier, node) for node in range(2)]
                    jobs = [job for future in futures for job in future.result(timeout=30)]
            self.assertEqual(len(jobs), 30)
            self.assertEqual(len({key for key, _, _ in jobs}), 30)
            times = {key: (start, end) for key, start, end in jobs}
            for seed in range(5, 10):
                self.assertGreaterEqual(times[f"seed-{seed}.prefix"][0], times[f"seed-{seed}.cache"][1])
                for arm in ("on_policy", "switch", "sr", "random"):
                    self.assertGreaterEqual(times[f"seed-{seed}.{arm}"][0], times[f"seed-{seed}.prefix"][1])
            events = sorted([(start, 1) for _, start, _ in jobs] + [(end, -1) for _, _, end in jobs])
            active = peak = 0
            for _, change in events:
                active += change
                peak = max(peak, active)
            self.assertEqual(peak, 2)


if __name__ == "__main__":
    unittest.main()
