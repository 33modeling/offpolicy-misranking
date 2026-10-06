"""Qwen must attach the offline verifier before checking package metadata."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class QwenVerifierBootstrapTests(unittest.TestCase):
    def test_missing_verifier_bootstraps_offline_and_reaches_children(self):
        # -S removes site-packages. Only the GPU/model dependencies are mocked;
        # the verifier, metadata, extraction and inherited child path are real.
        code = r'''
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

sys.path[:0] = [sys.argv[1], str(Path(sys.argv[1]) / "scripts")]
try:
    metadata.version("math-verify")
except metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("test did not start without math-verify")

versions = {"torch": "test-cuda-build", "transformers": "5.14.1",
            "peft": "0.20.0", "numpy": "2.1.0", "fla-core": "0.5.2"}
original_version = metadata.version
metadata.version = lambda name: versions[name] if name in versions else original_version(name)
version_module = ModuleType("packaging.version")
version_module.Version = lambda value: value
sys.modules["packaging"] = ModuleType("packaging")
sys.modules["packaging.version"] = version_module
transformers = ModuleType("transformers")
transformers.Qwen3_5ForCausalLM = object
sys.modules["transformers"] = transformers
model_matrix = ModuleType("model_matrix")
model_matrix._require_runtime = lambda spec: None
sys.modules["model_matrix"] = model_matrix
engine_package = ModuleType("srgc_rebuttal")
engine_package.__path__ = [str(Path(sys.argv[1]) / "srgc_rebuttal")]
sys.modules["srgc_rebuttal"] = engine_package

import srgc_qwen35
result = srgc_qwen35.runtime_packages()
assert result["math-verify"] == "0.9.0", result
assert result["torch"] == "test-cuda-build"
from math_verify import parse, verify
assert verify(parse(r"\frac{1}{2}"), parse("0.5"))
import math_verify
assert Path(math_verify.__file__).is_relative_to(Path(os.environ["OM_WORK"]) / "runtime-deps")
child = subprocess.run([sys.executable, "-S", "-c",
    "import importlib.metadata as m; from math_verify import parse, verify; "
    "assert m.version('math-verify') == '0.9.0'; assert verify(parse('0.5'), parse('0.5'))"],
    check=True, capture_output=True, text=True, env=dict(os.environ))
print(json.dumps(result))
'''
        with tempfile.TemporaryDirectory() as directory:
            environment = {**os.environ, "OM_WORK": directory, "PYTHONPATH": "",
                           "PYTHONNOUSERSITE": "1", "PIP_NO_INDEX": "1",
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            result = subprocess.run([sys.executable, "-S", "-c", code, str(ROOT)],
                                    env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"math-verify": "0.9.0"', result.stdout)


if __name__ == "__main__":
    unittest.main()
