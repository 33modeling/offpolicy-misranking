import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from scripts import srgc_process_guard as guard
from srgc_rebuttal import cluster

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


class ProcessGuardTest(unittest.TestCase):
    def tearDown(self):
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


if __name__ == "__main__":
    unittest.main()
