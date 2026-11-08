"""Bind to the existing trainer/selector implementation (never copied here).

Code repository: 33modeling/offpolicy-misranking (``srgc_rebuttal/``).
Integration points used (verified locally at master 0937526):
  srgc_rebuttal.srgc.Engine / Config / TemporalRule / stream_seed / cached_sr_set
  srgc_rebuttal.toy_backend.ToyBackend, make_problem          (CPU smoke)
  srgc_rebuttal.torch_backend.TorchBackend                    (GRPO + scoring gradients)
  srgc_rebuttal.timing.CostMeter / torch_meter                (phase receipts)
  srgc_rebuttal.plan.validate_inputs / load_plan              (split isolation)
  srgc_rebuttal.runtime.code_digest                           (implementation identity)
  scripts.srgc_pair_inputs.default_plan, scripts.srgc_shared_storage.route_plan (cluster run roots)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

DEFAULT_LOCAL = Path("/home/kms/dev/offpolicy-misranking")
FROZEN_FILES = ("srgc.py", "objectives.py", "torch_backend.py", "toy_backend.py", "timing.py",
                "cost_ledger.py", "torch_state.py", "plan.py", "runtime.py")


def code_root(explicit=None) -> Path:
    root = Path(explicit or os.environ.get("V8_CODE_ROOT") or os.environ.get("OM_CODE_ROOT") or DEFAULT_LOCAL)
    if not (root / "srgc_rebuttal" / "srgc.py").is_file():
        raise FileNotFoundError(
            f"code repository with srgc_rebuttal/srgc.py not found at {root}; set V8_CODE_ROOT to the "
            "offpolicy-misranking checkout that produced the seed runs")
    return root.resolve()


def activate(root=None) -> Path:
    root = code_root(root)
    for p in (str(root / "scripts"), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def srgc():
    activate()
    from srgc_rebuttal import srgc as module
    return module


def implementation_digest(root=None) -> str:
    """Same digest as srgc_rebuttal.runtime.code_digest (run.json implementation_sha256)."""
    root = code_root(root)
    value = hashlib.sha256()
    for path in sorted((root / "srgc_rebuttal").glob("*.py")):
        value.update(path.name.encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def frozen_file_hashes(root=None) -> dict:
    root = code_root(root)
    return {name: hashlib.sha256((root / "srgc_rebuttal" / name).read_bytes()).hexdigest()
            for name in FROZEN_FILES if (root / "srgc_rebuttal" / name).is_file()}


def git_commit(root=None):
    """HEAD commit of the code checkout, or None (cluster copies may be rsynced without .git)."""
    import subprocess
    root = code_root(root)
    try:
        out = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def identity(root=None) -> dict:
    root = code_root(root)
    return {"code_root": str(root), "code_commit": git_commit(root),
            "implementation_sha256": implementation_digest(root), "frozen_files": frozen_file_hashes(root)}
