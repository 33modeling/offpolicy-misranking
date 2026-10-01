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


def engine_digest(package, engine):
    value = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        value.update(path.name.encode())
        value.update(engine if path.name == "srgc.py" else path.read_bytes())
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
    if saved != SAVED_HASH:
        raise ValueError(f"unsupported saved prefix implementation {saved}; current={runtime.code_digest()}")
    source = SAVED_ENGINE.read_bytes()
    package = Path(runtime.__file__).parent
    if hashlib.sha256(source).hexdigest() != ENGINE_HASH or engine_digest(package, source) != saved:
        raise ValueError("saved runtime package differs from the pinned prefix implementation")
    # Do not relabel a new engine as an old one. Load the archived engine whose
    # entire package digest was just verified, before importing any extra arms.
    name = "srgc_rebuttal.srgc"
    spec = importlib.util.spec_from_file_location(name, SAVED_ENGINE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    parent = sys.modules["srgc_rebuttal"]
    parent.srgc = module
    for exported in parent.__all__:
        setattr(parent, exported, getattr(module, exported))

    def selected_digest():
        return engine_digest(package, SAVED_ENGINE.read_bytes())

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
