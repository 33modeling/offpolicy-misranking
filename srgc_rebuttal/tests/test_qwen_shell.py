import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_srgc_qwen35.sh"


class QwenShellTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="qwen shared env ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "work"
        self.env = {"PATH": os.environ["PATH"], "OM_WORK": str(self.work)}

    def python(self, path, name):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                        f"print(json.dumps({{'python': {name!r}, 'args': sys.argv[1:], "
                        "'threads': os.environ.get('OPENBLAS_DEFAULT_NUM_THREADS')}))\n")
        path.chmod(0o755)
        return str(path)

    def launch(self, dataset="all", **environment):
        return subprocess.run(["sh", str(SCRIPT), dataset, "status"], cwd="/tmp",
                              env={**self.env, **environment}, capture_output=True, text=True, timeout=10)

    def test_default_uses_the_olmo_shared_environment(self):
        self.python(self.work / ".venv-cu126/bin/python", "shared")
        for dataset in ("math", "mbpp", "all"):
            result = self.launch(dataset)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(result.stdout)
            self.assertEqual(value["python"], "shared")
            self.assertEqual(value["threads"], "1")
            self.assertEqual(value["args"], ["scripts/srgc_qwen35_diagnostics.py", dataset, "status"])

    def test_error_command_uses_read_only_diagnostics_entry(self):
        self.python(self.work / ".venv-cu126/bin/python", "shared")
        result = subprocess.run(["sh", str(SCRIPT), "all", "error"], cwd="/tmp",
                                env=self.env, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["args"],
                         ["scripts/srgc_qwen35_diagnostics.py", "all", "error"])

    def test_one_command_starts_both_queues_with_shared_python(self):
        pair = self.python(self.root / "pair/python", "pair")
        result = subprocess.run(["sh", str(SCRIPT)], cwd="/tmp",
                                env={**self.env, "PAIR_PYTHON": pair},
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["python"], "pair")
        self.assertEqual(value["args"], ["scripts/srgc_qwen35_start.py", "all"])

    def test_dataset_specific_interpreters_match_olmo(self):
        pair = self.python(self.root / "pair/python", "pair")
        switch = self.python(self.root / "switch/python", "switch")
        for dataset, expected in (("math", "pair"), ("all", "pair"), ("mbpp", "switch")):
            result = self.launch(dataset, PAIR_PYTHON=pair, SWITCH_PYTHON=switch)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["python"], expected)

    def test_dataset_only_starts_automatic_queue(self):
        pair = self.python(self.root / "pair/python", "pair")
        switch = self.python(self.root / "switch/python", "switch")
        for dataset, expected in (("math", "pair"), ("mbpp", "switch"), ("all", "pair")):
            result = subprocess.run(["sh", str(SCRIPT), dataset], cwd="/tmp",
                                    env={**self.env, "PAIR_PYTHON": pair, "SWITCH_PYTHON": switch},
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(result.stdout)
            self.assertEqual(value["python"], expected)
            self.assertEqual(value["args"], ["scripts/srgc_qwen35_start.py", dataset])

    def test_venv_dir_and_explicit_legacy_override(self):
        venv = self.root / "custom"
        self.python(venv / "bin/python", "custom")
        result = self.launch(VENV_DIR=str(venv))
        self.assertEqual(json.loads(result.stdout)["python"], "custom")
        legacy = self.python(self.root / "legacy/python", "legacy")
        result = self.launch(VENV_DIR=str(venv), QWEN_PYTHON=legacy)
        self.assertEqual(json.loads(result.stdout)["python"], "legacy")

    def test_missing_explicit_interpreter_fails_without_switching_environments(self):
        self.python(self.work / ".venv-cu126/bin/python", "shared")
        for variable in ("PAIR_PYTHON", "QWEN_PYTHON"):
            result = self.launch(**{variable: str(self.root / "missing")})
            self.assertEqual(result.returncode, 2)
            self.assertIn("Python not found", result.stderr)

    def test_default_falls_back_to_python3_like_olmo(self):
        self.python(self.root / "bin/python3", "fallback")
        result = self.launch(PATH=str(self.root / "bin") + os.pathsep + self.env["PATH"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["python"], "fallback")


if __name__ == "__main__":
    unittest.main()
