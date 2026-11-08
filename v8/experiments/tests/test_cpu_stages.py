"""CPU-stage regressions (Claude debugging pass 2026-11-08): the audit stage with spaces in paths and
harvest while workers keep writing. No model is loaded."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from helpers import HERE, tmpdir

STUB = """#!/bin/sh
if [ "$1" = -c ]; then exec "$REAL_PYTHON" "$@"; fi
"$REAL_PYTHON" -c 'import json,sys; open(sys.argv[1],"a").write(json.dumps(sys.argv[2:])+"\\n")' "$FAKE_LOG" "$@"
"""


class AuditStageQuoting(unittest.TestCase):
    def test_run_root_and_bundles_with_spaces_reach_the_cli_intact(self):
        d = tmpdir() / "dir with spaces"
        root = d / "runs root" / "pair-seeds"
        loc = {"seeds": {str(s): {"folder": f"{root}/seed-{s}", "input_bundle": f"{d}/in puts/seed {s}.json"}
                         for s in (5, 6)}}
        v8 = d / "v8 root"
        (v8 / "olmo_math").mkdir(parents=True)
        (v8 / "olmo_math" / "locator.json").write_text(json.dumps(loc))
        stub = d / "python"
        stub.write_text(STUB)
        stub.chmod(0o755)
        log = d / "argv.jsonl"
        env = dict(os.environ, V8_PYTHON=str(stub), REAL_PYTHON=sys.executable, V8_ROOT=str(v8),
                   V8_CODE_ROOT=str(d / "code"), FAKE_LOG=str(log), CUDA_VISIBLE_DEVICES="")
        r = subprocess.run(["sh", str(HERE / "run_h100.sh"), "olmo_math", "audit"], env=env,
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        argv = [json.loads(line) for line in log.read_text().splitlines()][-1]
        self.assertEqual(argv[:3], ["-m", "switching.cli", "audit"])
        self.assertIn(f"olmo_math={root}", argv)
        bundles = [argv[i + 1] for i, a in enumerate(argv) if a == "--bundle"]
        self.assertEqual(bundles, [f"{d}/in puts/seed 5.json", f"{d}/in puts/seed 6.json"])
        self.assertEqual(argv[argv.index("--out") + 1], f"{v8}/audit/olmo_math")



class HarvestWhileWorkersWrite(unittest.TestCase):
    """GNU tar exits 1 when a log grows during archiving; harvest must still finish (sha256 + path)."""
    def run_harvest(self, tar_rc):
        d = tmpdir() / "with space"
        v8 = d / "v8"
        (v8 / "olmo_math" / "runs" / "r1").mkdir(parents=True)
        (v8 / "olmo_math" / "runs" / "r1" / "manifest.json").write_text("{}")
        bindir = d / "bin"
        bindir.mkdir()
        real = subprocess.run(["sh", "-c", "command -v tar"], capture_output=True, text=True).stdout.strip()
        (bindir / "tar").write_text(f'#!/bin/sh\n"{real}" "$@"\nexit {tar_rc}\n')
        (bindir / "tar").chmod(0o755)
        env = dict(os.environ, V8_ROOT=str(v8), PATH=f"{bindir}:{os.environ['PATH']}")
        return subprocess.run(["sh", str(HERE / "harvest.sh")], env=env, capture_output=True, text=True, timeout=60), v8

    def test_file_changed_warning_is_not_a_failure(self):
        r, v8 = self.run_harvest(1)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout.splitlines()[0]
        self.assertTrue(Path(out + ".sha256").exists() and out.startswith(str(v8 / "harvest")))

    def test_real_tar_failure_still_fails(self):
        r, v8 = self.run_harvest(2)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("harvest failed", r.stderr)


if __name__ == "__main__":
    unittest.main()
