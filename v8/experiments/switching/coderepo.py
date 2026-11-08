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
import importlib.util
import json
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


def source_runtime(root, profile, recorded):
    """Verify actual source bytes, including the existing saved engine/Qwen adapter."""
    root=code_root(root)
    current=implementation_digest(root)
    adapter=None
    if profile=='qwen':
        path=root/'scripts/srgc_qwen35.py'
        spec=importlib.util.spec_from_file_location('_v8_qwen_identity',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        adapter=module.adapter_digest()
    options=[(current,'current')]
    helper_path=root/'scripts/srgc_saved_runtime.py'
    helper=None
    if helper_path.is_file():
        spec=importlib.util.spec_from_file_location('_v8_saved_identity',helper_path)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        options += [(helper.PREVIOUS_HASH,'saved'),(helper.SAVED_HASH,'saved')]
    for base,mode in options:
        identity=hashlib.sha256((base+adapter).encode()).hexdigest() if adapter else base
        if identity!=recorded:continue
        if mode=='saved':
            engine=helper.SAVED_ENGINE if base==helper.SAVED_HASH else root/'srgc_rebuttal/srgc.py'
            data=engine.read_bytes()
            if base==helper.SAVED_HASH and hashlib.sha256(data).hexdigest()!=helper.ENGINE_HASH:
                raise ValueError('Archived engine bytes differ from the original release')
            if helper.engine_digest(root/'srgc_rebuttal',data)!=base:
                raise ValueError('Archived source package does not match its recorded digest')
        return dict(mode=mode,implementation_sha256=identity,base_implementation_sha256=base,
                    adapter_sha256=adapter)
    raise ValueError(f'Recorded training runtime {recorded} is unavailable; current base={current}, adapter={adapter}')


def activate_source_runtime(root,plan,seed,selection):
    """Use the updated selector with the original checkout's verified archive."""
    if selection['mode']!='saved':return
    root=code_root(root)
    paths=[Path(__file__).resolve().parents[3]/'scripts/srgc_saved_runtime.py',
           root/'scripts/srgc_saved_runtime.py']
    spec=importlib.util.spec_from_file_location('_v8_saved_activation',paths[0])
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.ARCHIVED_MODULES=root/'scripts/frozen_srgc/pre_20261004'
    helper.SAVED_ENGINE=root/'scripts/frozen_srgc/f581eb043e89409e.py'
    helper.activate(Path(plan),seed,base_implementation=selection['base_implementation_sha256'],
                    adapter_sha256=selection['adapter_sha256'])


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
