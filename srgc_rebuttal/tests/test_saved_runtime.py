import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import srgc_saved_runtime as saved_runtime
from scripts.srgc_saved_runtime import (ARCHIVED_MODULES, PREVIOUS_HASH, SAVED_ENGINE,
                                      SAVED_HASH, ENGINE_HASH, engine_digest)
from srgc_rebuttal import runtime
from srgc_rebuttal.tests import test_extra_arm_launch


ROOT = Path(__file__).resolve().parents[2]


class SavedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_extra_arm_launch.ExtraArmLaunchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.plan = self.fixture.plan()
        self.folder, _, _ = self.fixture.prefix(self.plan)
        for name in ("prefix-ready.json", "run.json"):
            path = self.folder / name
            value = json.loads(path.read_text())
            value["implementation_sha256"] = SAVED_HASH
            path.write_text(json.dumps(value))

    def child(self, code):
        environment = {**os.environ, **self.fixture.environment}
        environment.pop("SRGC_STORAGE_ROOT", None)
        return subprocess.run([sys.executable, "-c", code, str(self.plan)],
                              cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30)

    def test_archived_engine_matches_the_original_full_package_hash(self):
        source = SAVED_ENGINE.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(), ENGINE_HASH)
        self.assertEqual(engine_digest(Path(runtime.__file__).parent, source), SAVED_HASH)

    def test_previous_release_matches_its_original_full_package_hash(self):
        package = Path(runtime.__file__).parent
        self.assertEqual(engine_digest(package, (package / "srgc.py").read_bytes()), PREVIOUS_HASH)

    def test_both_saved_releases_load_verified_old_support_modules_and_verifier(self):
        for release in (SAVED_HASH, PREVIOUS_HASH):
            with self.subTest(release=release):
                path = self.folder / "prefix-ready.json"
                receipt = json.loads(path.read_text())
                receipt["implementation_sha256"] = release
                path.write_text(json.dumps(receipt))
                result = self.child('''
import sys
from pathlib import Path
from scripts.srgc_saved_runtime import bootstrap, ARCHIVED_MODULES
bootstrap(["--plan", sys.argv[1], "--seed", "5"])
from srgc_rebuttal import cluster, cluster_queue, code_check, verifiers, plan, reports
from srgc_rebuttal.runtime import code_digest
assert code_digest() == RELEASE
for module in (cluster, cluster_queue, code_check, verifiers, plan, reports):
    assert Path(module.__file__).parent == ARCHIVED_MODULES, module.__file__
assert verifiers.CODE_REWARD_VERSION == "assertion-completion-v2"
assert verifiers.code_reward({"answer": "assert add(2, 3) == 5"}, "def add(a, b): return a + b") == 1
print("saved reward protocol retained")
'''.replace("RELEASE", repr(release)))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("saved reward protocol retained", result.stdout)

    def test_changed_archive_is_rejected_before_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)
            for path in ARCHIVED_MODULES.iterdir():
                if path.is_file():
                    (archive / path.name).write_bytes(path.read_bytes())
            verifier = archive / "verifiers.py"
            verifier.write_bytes(verifier.read_bytes() + b"\n# modified\n")
            with patch.object(saved_runtime, "ARCHIVED_MODULES", archive):
                with self.assertRaisesRegex(ValueError, "verifiers.py"):
                    engine_digest(Path(runtime.__file__).parent, SAVED_ENGINE.read_bytes())

    def test_late_activation_fails_without_mutating_loaded_modules_or_digest(self):
        result = self.child('''
import sys
from srgc_rebuttal import runtime, verifiers, srgc
import srgc_rebuttal
from scripts.srgc_saved_runtime import activate
from pathlib import Path
before = (list(srgc_rebuttal.__path__), runtime.code_digest, srgc_rebuttal.Engine, srgc)
try:
    activate(Path(sys.argv[1]), 5)
except ValueError as error:
    assert "before importing" in str(error)
else:
    raise AssertionError("must not overwrite a loaded verifier")
assert before == (list(srgc_rebuttal.__path__), runtime.code_digest, srgc_rebuttal.Engine, srgc_rebuttal.srgc)
assert verifiers.CODE_REWARD_VERSION == "parent-checked-values-v3"
''')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_saved_engine_forks_and_resumes_fixed200_with_original_scoring(self):
        result = self.child('''
import sys
from pathlib import Path
from scripts.srgc_saved_runtime import bootstrap, SAVED_HASH
bootstrap(["--plan", sys.argv[1], "--seed", "5"])
from srgc_rebuttal.runtime import code_digest
from srgc_rebuttal.srgc import Engine
from srgc_rebuttal.tests.test_switch_fixed import make_engine
assert code_digest() == SAVED_HASH
assert Engine.SAMPLING_PROTOCOL == "random-candidate40-training4-contrast40-v2"
prefix, _ = make_engine(200, seed=5, arm="on_policy")
prefix.run_until(25)
state = prefix.state_dict()
control, _ = make_engine(200, seed=5)
control.load_state_dict(state, fork_arm="switch_fixed")
control.run_until(200)
assert control.switched_at is None
assert control.update()["selector"] == "sr"
assert control.switched_at == 200
assert all(len(r["sr_ids"]) == 40 for r in control.history if r["selection_refreshed"])
resumed, _ = make_engine(200, seed=5)
resumed.load_state_dict(control.state_dict())
assert resumed.update()["selector"] == "sr"
print("saved runtime fork/resume passed")
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("saved runtime fork/resume passed", result.stdout)

    def test_direction_ablation_scores_what_the_saved_on_policy_scores(self):
        result = self.child('''
import sys
from scripts.srgc_saved_runtime import bootstrap
bootstrap(["--plan", sys.argv[1], "--seed", "5"])
from srgc_rebuttal.srgc import Config, Engine
from srgc_rebuttal.toy_backend import ToyBackend, make_problem
from scripts.srgc_direction_ablation import DirectionAblationEngine
assert Engine.SAMPLING_PROTOCOL == "random-candidate40-training4-contrast40-v2"
features, answers, candidates, validation, _, cache = make_problem(5)
config = Config(seed=5, projection_dim=64)
reference = ToyBackend(features, answers, projection_dim=64, seed=5)
Engine(reference, candidates, validation, cache, arm="on_policy", config=config).update()
for mode in ("removed", "magnitude", "replaced"):
    backend = ToyBackend(features, answers, projection_dim=64, seed=5)
    record = DirectionAblationEngine(backend, candidates, validation, cache, arm="direction_ablation",
                                     config=config, mode=mode).update()
    assert backend.score_calls == reference.score_calls, mode  # same 40+40 union and validation set
    assert len(record["sr_ids"]) == 40 and record["scored_distinct_prompts"] == len(set(reference.score_calls[0]))
print("ablation matches saved on-policy scoring")
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ablation matches saved on-policy scoring", result.stdout)

    def test_verified_extra_can_use_parked_prefix_without_unparking_base_queue(self):
        receipt = self.plan.parent.parent / "automatic-restart.json"
        receipt.write_text(json.dumps({"plan": str(self.plan), "previous_plan": "/old/plan.json"}))
        before = {p: p.read_bytes() for p in self.folder.iterdir() if p.is_file()}
        result = self.child('''
import sys
from pathlib import Path
from types import SimpleNamespace
from scripts.srgc_saved_runtime import bootstrap
bootstrap(["--plan", sys.argv[1], "--seed", "5"])
from scripts.srgc_sr_refresh import prepare_run_storage
from scripts.srgc_shared_storage import route_plan, RunConflict
plan = Path(sys.argv[1])
options = SimpleNamespace(plan=plan, seed=5)
prepare_run_storage(options)
assert options.plan == plan
try:
    route_plan(plan, writing=True)
except RunConflict:
    pass
else:
    raise AssertionError("base queue must remain parked")
print("verified extra admitted; base queue preserved")
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("verified extra admitted", result.stdout)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_unknown_saved_engine_is_not_relabelled_as_current(self):
        path = self.folder / "prefix-ready.json"
        value = json.loads(path.read_text())
        value["implementation_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        result = self.child('''
import sys
from scripts.srgc_saved_runtime import bootstrap
bootstrap(["--plan", sys.argv[1], "--seed", "5"])
''')
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported saved prefix implementation", result.stderr)

    def test_archived_runtime_still_rejects_different_inputs(self):
        path = self.folder / "prefix-ready.json"
        value = json.loads(path.read_text())
        value["input_sha256"] = "wrong-input"
        path.write_text(json.dumps(value))
        result = self.child('''
import sys
from pathlib import Path
from types import SimpleNamespace
from scripts.srgc_saved_runtime import bootstrap
bootstrap(["--plan", sys.argv[1], "--seed", "5"])
from scripts.srgc_sr_refresh import prepare_run_storage
prepare_run_storage(SimpleNamespace(plan=Path(sys.argv[1]), seed=5))
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("input_sha256: saved='wrong-input'", result.stderr)

    def test_both_cli_entrypoints_select_saved_runtime_before_extra_arm_imports(self):
        path = self.folder / "run.json"
        value = json.loads(path.read_text())
        value["seed"] = 9
        path.write_text(json.dumps(value))
        environment = {**os.environ, **self.fixture.environment}
        environment.pop("SRGC_STORAGE_ROOT", None)
        for script, arguments in (("srgc_extra_worker.py", []), ("srgc_sr_refresh.py", ["run"])):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *arguments,
                    "--plan", str(self.plan), "--seed", "5", "--arm", "switch_fixed200"],
                    cwd=ROOT, env=environment, capture_output=True, text=True, timeout=15)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("RUNTIME saved-prefix f581eb043e89409e", result.stdout)
                self.assertIn("run manifest belongs to a different experiment", result.stderr)
                self.assertNotIn("prefix belongs to a different experiment", result.stderr)


if __name__ == "__main__":
    unittest.main()
