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
                   ("PAIR_PYTHON", "SWITCH_PYTHON", "CUDA_VISIBLE_DEVICES", "SRGC_RUN_NAME")}
            env.update({"PAIR_PYTHON" if dataset == "math" else "SWITCH_PYTHON": str(python), **environment})
            command = ["sh", str(SCRIPT), dataset, *([mode] if mode else [])]
            result = subprocess.run(command, cwd="/tmp", env=env, text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

    def test_math_and_mbpp_start_shared_fresh_workers_without_user_options(self):
        for dataset in ("math", "mbpp"):
            with self.subTest(dataset=dataset):
                report = self.invoke(dataset)
                self.assertEqual(report["args"], ["scripts/run_srgc_rebuttal.py", "worker", "--dataset", dataset,
                                                  "--fresh", "restart1"])
                self.assertEqual(report["cuda"], "0,1,2,3")
                self.assertEqual(report["threads"], "1")
                self.assertEqual(report["cwd"], str(SCRIPT.parents[1]))

    def test_scheduler_gpu_visibility_is_preserved(self):
        self.assertEqual(self.invoke("math", CUDA_VISIBLE_DEVICES="4,5,6,7")["cuda"], "4,5,6,7")
        self.assertEqual(self.invoke("math", CUDA_VISIBLE_DEVICES="")["cuda"], "")

    def test_reports_use_selected_dataset_without_starting_or_resetting_work(self):
        for dataset in ("math", "mbpp"):
            for mode in ("status", "results", "costs"):
                report = self.invoke(dataset, mode)
                self.assertEqual(report["args"], ["scripts/run_srgc_rebuttal.py", mode, "--dataset", dataset])
                self.assertEqual(report["cuda"], "")

    def test_invalid_invocations_fail_before_python(self):
        for args in ([], ["wrong"], ["math", "wrong"], ["math", "run", "extra"]):
            result = subprocess.run(["sh", str(SCRIPT), *args], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
