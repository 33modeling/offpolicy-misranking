"""Select the exact known training engine recorded by an extra arm's prefix."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


SAVED_HASH = "f581eb043e89409e0e68d8ed77201fa030e34bd93d4d5babbab65304c24a5d6b"
SAVED_ENGINE = Path(__file__).parent / "frozen_srgc/f581eb043e89409e.py"
ENGINE_HASH = "b8b446c291a9bd96c4947a669f7c283220aa89eb4782b2af6775994ea4d42bc0"
ARCHIVED_MODULES = Path(__file__).parent / "frozen_srgc/pre_20261004"
PREVIOUS_HASH = "9435e80003f41e1f65cb9dcc4074f1b06f063880d4823f433d5cd8fb5e2d74a7"


def engine_digest(package, engine):
    """Hash the exact released file set, not later files added to the live package."""
    manifest = json.loads((ARCHIVED_MODULES / "manifest.json").read_text())
    if manifest["package_sha256"] != PREVIOUS_HASH:
        raise ValueError("archived runtime manifest differs from its pinned release")
    value = hashlib.sha256()
    for name, expected in sorted(manifest["files"].items()):
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("invalid archived module name")
        path = ARCHIVED_MODULES / name
        if not path.is_file():
            path = package / name
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"saved runtime package differs: {name}")
        value.update(name.encode())
        value.update(engine if name == "srgc.py" else data)
    return value.hexdigest()


def activate(plan_path, seed):
    from srgc_rebuttal import runtime
    from srgc_rebuttal.plan import load_plan
    plan = load_plan(plan_path)
    folder = runtime.run_root(plan_path, plan) / f"seed-{seed}"
    receipt_path = folder / "prefix-ready.json"
    if not receipt_path.is_file():
        return  # The normal admission path reports the missing prefix.
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict):
        raise ValueError(f"invalid prefix receipt: {receipt_path}")
    saved = receipt.get("implementation_sha256")
    if saved == runtime.code_digest():
        return
    if saved not in {SAVED_HASH, PREVIOUS_HASH}:
        raise ValueError(f"unsupported saved prefix implementation {saved}; current={runtime.code_digest()}")
    package = Path(runtime.__file__).parent
    source_path = SAVED_ENGINE if saved == SAVED_HASH else package / "srgc.py"
    source = source_path.read_bytes()
    if ((saved == SAVED_HASH and hashlib.sha256(source).hexdigest() != ENGINE_HASH)
            or engine_digest(package, source) != saved):
        raise ValueError("saved runtime package differs from the pinned prefix implementation")
    for filename in ("cluster", "cluster_queue", "code_check", "verifiers", "reports"):
        qualified = f"srgc_rebuttal.{filename}"
        if qualified in sys.modules:
            raise ValueError(f"saved runtime must be selected before importing {qualified}")
    # Do not relabel a new engine as an old one. Load the archived engine whose
    # entire package digest was just verified, before importing any extra arms.
    parent = sys.modules["srgc_rebuttal"]
    # Changed support modules are loaded from their verified release copies.
    # The new verifier must never reinterpret old rewards as v3-checked data.
    parent.__path__ = [str(ARCHIVED_MODULES), str(package)]
    # plan was needed to locate the receipt. Replace it with its release copy
    # before training imports its validators or the changed support modules.
    for short_name, path in (("plan", ARCHIVED_MODULES / "plan.py"), ("srgc", source_path)):
        name = f"srgc_rebuttal.{short_name}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        setattr(parent, short_name, module)
    for exported in parent.__all__:
        setattr(parent, exported, getattr(module, exported))

    def selected_digest():
        return engine_digest(package, source_path.read_bytes())

    runtime.code_digest = selected_digest
    print(f"RUNTIME saved-prefix {saved[:16]} protocol={module.Engine.SAMPLING_PROTOCOL}", flush=True)


def bootstrap(arguments):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--seed", type=int)
    args, _ = parser.parse_known_args(arguments)
    if args.plan is not None and args.seed is not None:
        try:
            activate(args.plan, args.seed)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"INVALID RUNTIME: {exc}", file=sys.stderr, flush=True)
            raise SystemExit(2) from None
