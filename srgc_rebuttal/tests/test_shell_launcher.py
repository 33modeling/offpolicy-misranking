import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_srgc.sh"


class ShellLauncherTests(unittest.TestCase):
    def invoke(self, dataset, mode=None, **environment):
        with tempfile.TemporaryDirectory(prefix="srgc launcher ") as directory:
            python = Path(directory) / "python"
            python.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                              "print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), "
                              "'cuda': os.environ.get('CUDA_VISIBLE_DEVICES'), "
                              "'threads': os.environ.get('OPENBLAS_NUM_THREADS')}))\n")
            python.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k not in
                   ("PAIR_PYTHON", "SWITCH_PYTHON", "CUDA_VISIBLE_DEVICES", "SRGC_RUN_NAME", "SRGC_MAX_ATTEMPTS")}
            env.update({"PAIR_PYTHON" if dataset in ("math", "all") else "SWITCH_PYTHON": str(python),
                        "SRGC_SKIP_GPU_CLEANUP": "1", **environment})
            command = ["sh", str(SCRIPT), dataset, *([mode] if mode else [])]
            result = subprocess.run(command, cwd="/tmp", env=env, text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

    def test_run_never_kills_existing_gpu_processes(self):
        script = SCRIPT.read_text()
        self.assertNotIn("clear_gpu_memory", script)
        self.assertNotIn("kill -TERM", script)
        self.assertNotIn("kill -KILL", script)
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "python"
            python.write_text("#!/bin/sh\necho '{}'\n")
            python.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "SRGC_SKIP_GPU_CLEANUP")}
            env.update(PAIR_PYTHON=str(python), SRGC_SKIP_GPU_CLEANUP="1")
            result = subprocess.run(["sh", str(SCRIPT), "math", "run"], cwd="/tmp", env=env, text=True,
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("TERM sent", result.stderr)
            self.assertEqual(result.stdout.strip(), "{}")

    def test_worker_is_restarted_after_a_crash_but_not_after_an_interrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "python"
            counter = Path(directory) / "count"
            # Fails twice (exit 1), then succeeds; each start appends a line.
            python.write_text("#!/bin/sh\necho start >> '%s'\nn=$(wc -l < '%s')\n[ \"$n\" -ge 3 ] && exit 0\nexit 1\n" % (counter, counter))
            python.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "SRGC_SKIP_GPU_CLEANUP")}
            env.update(PAIR_PYTHON=str(python), SRGC_SKIP_GPU_CLEANUP="1", SRGC_WORKER_RESTART_DELAY="0")
            result = subprocess.run(["sh", str(SCRIPT), "math", "run"], cwd="/tmp", env=env, text=True,
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(counter.read_text().count("start"), 3)
            self.assertEqual(result.stderr.count("[worker-restart]"), 2)
            # An interrupt (130) is not restarted.
            python.write_text("#!/bin/sh\necho start >> '%s'\nexit 130\n" % counter)
            counter.write_text("")
            result = subprocess.run(["sh", str(SCRIPT), "math", "run"], cwd="/tmp", env=env, text=True,
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 130)
            self.assertEqual(counter.read_text().count("start"), 1)
            # The restart budget is respected.
            python.write_text("#!/bin/sh\necho start >> '%s'\nexit 1\n" % counter)
            counter.write_text("")
            env["SRGC_WORKER_RESTARTS"] = "3"
            result = subprocess.run(["sh", str(SCRIPT), "math", "run"], cwd="/tmp", env=env, text=True,
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(counter.read_text().count("start"), 3)

    def test_all_runs_one_worker_for_both_queues(self):
        report = self.invoke("all")
        self.assertEqual(report["args"], ["scripts/run_srgc_rebuttal.py", "worker", "--dataset", "math",
                                          "--with-dataset", "mbpp", "--retry-failed", "--max-attempts", "50",
                                          "--retry-delay", "120"])
        self.assertEqual(report["cuda"], "0,1,2,3")

    def test_attempt_limit_override_reaches_the_worker(self):
        report = self.invoke("math", SRGC_MAX_ATTEMPTS="6")
        self.assertEqual(report["args"], ["scripts/run_srgc_rebuttal.py", "worker", "--dataset", "math",
                                          "--retry-failed", "--max-attempts", "6", "--retry-delay", "120"])
        report = self.invoke("math", SRGC_MAX_ATTEMPTS="6", SRGC_RUN_NAME="explicit-study")
        self.assertEqual(report["args"][-2:], ["--fresh", "explicit-study"])
        self.assertIn("--max-attempts", report["args"])

    def test_math_and_mbpp_start_or_continue_with_one_command_and_automatic_retry(self):
        for dataset in ("math", "mbpp"):
            with self.subTest(dataset=dataset):
                report = self.invoke(dataset)
                self.assertEqual(report["args"], ["scripts/run_srgc_rebuttal.py", "worker", "--dataset", dataset,
                                                  "--retry-failed", "--max-attempts", "50", "--retry-delay", "120"])
                self.assertEqual(report["cuda"], "0,1,2,3")
                self.assertEqual(report["threads"], "1")
                self.assertEqual(report["cwd"], str(SCRIPT.parents[1]))

    def test_scheduler_gpu_visibility_is_preserved(self):
        self.assertEqual(self.invoke("math", CUDA_VISIBLE_DEVICES="4,5,6,7")["cuda"], "4,5,6,7")
        self.assertEqual(self.invoke("math", CUDA_VISIBLE_DEVICES="")["cuda"], "")

    def test_reports_use_selected_dataset_without_starting_or_resetting_work(self):
        for dataset in ("math", "mbpp"):
            for mode in ("status", "results", "costs", "backup", "backup-watch"):
                report = self.invoke(dataset, mode)
                self.assertEqual(report["args"], ["scripts/run_srgc_rebuttal.py", mode, "--dataset", dataset])
                self.assertEqual(report["cuda"], "")

    def test_explicit_run_name_is_only_used_when_the_user_sets_it(self):
        for dataset in ("math", "mbpp", "all"):
            report = self.invoke(dataset, SRGC_RUN_NAME="explicit-study")
            self.assertEqual(report["args"], ["scripts/run_srgc_rebuttal.py", "worker", "--dataset",
                                              "math" if dataset == "all" else dataset,
                                              *(["--with-dataset", "mbpp"] if dataset == "all" else []),
                                              "--retry-failed", "--max-attempts", "50", "--retry-delay", "120",
                                              "--fresh", "explicit-study"])
            self.assertEqual(report["cuda"], "0,1,2,3")

    def test_invalid_invocations_fail_before_python(self):
        for args in ([], ["wrong"], ["math", "wrong"], ["math", "resume"], ["math", "run", "extra"]):
            result = subprocess.run(["sh", str(SCRIPT), *args], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
