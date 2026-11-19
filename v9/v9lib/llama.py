"""Llama-3.1-8B-Instruct cohorts (X1): locate the original seed 5-9 runs and check their runtime.

The original Llama runs (scripts/run_srgc_llama31.sh -> srgc_research.dispatch.llama31) used the
unchanged srgc_rebuttal Engine and TorchBackend checkpoint format, with a Llama adapter
(srgc_research/dispatch/llama31/adapter.py). Their run.json records
implementation_sha256 = sha256(srgc_rebuttal digest + adapter_digest()). The V8 child cannot run
them as is: switching/cluster.py has no 'llama' plan resolver, switching/coderepo.py::source_runtime
knows no 'llama' adapter digest, and cluster_child.py loads the OLMo registry model. V9 does not
change V8; v9/child.py enters the Llama adapter contexts itself (runtime_adapter + training_adapter,
LlamaBackend, Llama load_model) after this module verified the identity.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import env
from .common import Blocked, allow_fake, read_json, v9c

ADAPTER_MODULE = "srgc_research.dispatch.llama31.adapter"


def plan_path(dataset, environ=None):
    from srgc_research.dispatch.llama31.storage import default_root
    return Path(default_root(environ or os.environ)) / "experiments" / f"llama31-8b-{dataset}.json"


def expected_identity(code_root):
    """sha256(srgc_rebuttal digest + Llama adapter digest), as runtime_adapter() records it."""
    from switching.coderepo import implementation_digest
    import importlib
    adapter = importlib.import_module(ADAPTER_MODULE)
    return hashlib.sha256((implementation_digest(code_root) + adapter.adapter_digest()).encode()).hexdigest(), \
        adapter.adapter_digest()


def locate(cohort):
    """Write v9/<cohort>/locator.json for a Llama cohort, or raise Blocked with the exact reason."""
    from switching.common import atomic_write_json, sha256_file, utc_now
    dst = v9c(cohort) / "locator.json"
    fake = env.work() / "v9-fake-inputs" / f"{cohort}-locator.json"
    if allow_fake() and fake.is_file():
        if not dst.is_file():
            atomic_write_json(dst, read_json(fake))
        return read_json(dst)
    dataset = cohort.split("_", 1)[1]
    code = env.code_root()
    try:
        path = plan_path(dataset)
    except Exception as exc:  # noqa: BLE001
        raise Blocked(f"Llama storage module unavailable: {type(exc).__name__}: {exc}") from None
    if not path.is_file():
        raise Blocked(f"Llama plan not found: {path} (original Llama runs not on this volume)")
    try:
        from srgc_research.dispatch.llama31.adapter import validate_extension
        plan = validate_extension(path, read_only=True)
    except Exception as exc:  # noqa: BLE001
        raise Blocked(f"Llama plan rejected by the Llama adapter: {type(exc).__name__}: {exc}") from None
    from srgc_rebuttal.plan import input_path
    from srgc_rebuttal.runtime import run_root
    identity, adapter_sha = expected_identity(code)
    out = {"cohort": cohort, "located_at": utc_now(), "profile": "llama", "code": {"code_root": str(code)},
           "llama_identity_expected": identity, "seeds": {}}
    problems = []
    for seed in (5, 6, 7, 8, 9):
        folder = run_root(path, plan) / f"seed-{seed}"
        rec = {"plan_path": str(path), "plan_sha256": sha256_file(path), "folder": str(folder),
               "model": plan.get("model"), "model_revision": plan.get("model_revision"),
               "input_bundle": str(input_path(path, plan, seed))}
        ready, run = folder / "prefix-ready.json", folder / "run.json"
        if not ready.is_file():
            problems.append(f"s{seed}: no prefix-ready.json in {folder}")
        else:
            r = read_json(ready)
            rec.update(prefix_path=str(folder / "prefix.pt"), prefix_sha256=r["checkpoint_sha256"],
                       prefix_completed_updates=r["completed_updates"])
        if not run.is_file():
            problems.append(f"s{seed}: no run.json in {folder}")
        else:
            impl = read_json(run).get("implementation_sha256")
            rec["run_implementation_sha256"] = impl
            ok = impl == identity and (not ready.is_file() or read_json(ready).get("implementation_sha256") in (None, impl))
            rec["implementation_matches_code_root"] = ok
            rec["runtime_selection"] = {"mode": "llama_adapter", "implementation_sha256": identity,
                                        "adapter_sha256": adapter_sha}
            if not ok:
                problems.append(f"s{seed}: run.json implementation {str(impl)[:12]} != current srgc_rebuttal+Llama "
                                f"adapter digest {identity[:12]} (code or adapter bytes changed since the run)")
        out["seeds"][str(seed)] = rec
    if problems:
        raise Blocked("Llama runtime not runnable: " + "; ".join(problems[:4]))
    atomic_write_json(dst, out)
    return out


def verify_child(seed_loc):
    """Inside the child: the same identity check, failing before any model is built."""
    identity, _ = expected_identity(env.code_root())
    if seed_loc.get("run_implementation_sha256") != identity:
        raise SystemExit("Llama run identity differs from srgc_rebuttal + adapter digest of this checkout")


_ = json
