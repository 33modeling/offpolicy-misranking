"""Command line: python -m switching.cli <command> (run from v8/experiments).

CPU commands: inspect-repo, seed-registry, validate-config, estimate, audit, make-matrix,
pending, freeze-budget, tune, freeze, collect, analyze, smoke.
GPU command (inside torchrun on the cluster): child  (see cluster_child.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]


def _print(obj):
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def cmd_inspect(args):
    from . import coderepo
    ident = coderepo.identity(args.code_root)
    root = Path(ident["code_root"])
    points = {
        "engine (On-policy grad-alignment selector, cached-SR selector, SR-GC D, TemporalRule)": "srgc_rebuttal/srgc.py",
        "GRPO/RLOO trainer, scoring gradients, LoRA AdamW, evaluate": "srgc_rebuttal/torch_backend.py",
        "advantages/CountSketch": "srgc_rebuttal/objectives.py", "verifiers": "srgc_rebuttal/verifiers.py",
        "phase timers (CostMeter)": "srgc_rebuttal/timing.py", "phase receipts": "srgc_rebuttal/cost_ledger.py",
        "one-seed runner (prefix + four arms)": "srgc_rebuttal/run_experiment.py",
        "extra-arm child (fork from prefix)": "scripts/srgc_sr_refresh.py",
        "fixed-switch control": "scripts/srgc_switch_fixed.py", "rule ablations": "scripts/srgc_switch_rules.py",
        "repeated switching (E8)": "scripts/srgc_switch_repeat.py", "per-update checkpoints": "scripts/srgc_step_checkpoints.py",
        "plan routing to group storage": "scripts/srgc_shared_storage.py", "OLMo plans": "scripts/srgc_pair_inputs.py",
        "Qwen adapter": "scripts/srgc_qwen35.py", "toy CPU backend": "srgc_rebuttal/toy_backend.py"}
    _print({**ident, "integration_points": {k: {"path": v, "present": (root / v).is_file()} for k, v in points.items()}})


def cmd_seed_registry(args):
    from . import coderepo
    from .seeds import choose, registry, write_registry
    rows, scanned = registry(coderepo.code_root(args.code_root), HERE.parents[1])
    write_registry(args.output, rows)
    _print({"written": str(args.output), "scanned": scanned,
            "choice": {c: choose(rows, c) for c in ("olmo_math", "qwen_mbpp")}})


def cmd_validate(args):
    from . import spec
    cfg = spec.resolve(args.cohort, role=args.mode)
    res = spec.validate(cfg, mode=args.mode)
    _print(res)
    sys.exit(0 if res["ok"] or args.mode in ("smoke", "pilot") else 2)


def cmd_estimate(args):
    from .estimate import estimate
    _print([estimate(c) for c in args.cohorts])


def cmd_audit(args):
    from .audit import run_audit
    from .common import atomic_write_json
    from .analyze import write_csv
    roots = dict(kv.split("=", 1) for kv in args.run_root or [])
    res = run_audit(HERE.parents[1], run_roots=roots, bundles=args.bundle or [], code_root=args.code_root)
    out = Path(args.out)
    atomic_write_json(out / "audit.json", res)
    write_csv(out / "reconciliation.csv", [{k: v for k, v in c.items() if k != "checks"} for c in res["cells"]])
    atomic_write_json(out / "verified_runs.json", res["verified_runs"])
    _print({"status_counts": res["status_counts"], "split_overlap": len(res["split_overlap"]), "out": str(out)})
    if args.strict and any(c["status"] == "inconsistent" for c in res["cells"]):
        sys.exit(3)


def cmd_smoke(args):
    from .smoke import main
    res = main(args.out, keep=args.keep, fast=args.fast)
    _print({k: v for k, v in res.items()})


def main(argv=None):
    p = argparse.ArgumentParser(prog="switching.cli", description=__doc__)
    p.add_argument("--code-root", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect-repo").set_defaults(fn=cmd_inspect)
    s = sub.add_parser("seed-registry")
    s.add_argument("--output", type=Path, default=HERE / "registry" / "seed_registry.csv")
    s.set_defaults(fn=cmd_seed_registry)
    s = sub.add_parser("validate-config")
    s.add_argument("--cohort", required=True)
    s.add_argument("--mode", default="rebuttal_bank", choices=["test", "development", "rebuttal_bank", "pilot", "smoke"])
    s.set_defaults(fn=cmd_validate)
    s = sub.add_parser("estimate")
    s.add_argument("cohorts", nargs="*", default=["olmo_math", "qwen_mbpp", "olmo_mbpp"])
    s.set_defaults(fn=cmd_estimate)
    s = sub.add_parser("audit")
    s.add_argument("--run-root", action="append", help="cohort=path of the seed run root (cluster)")
    s.add_argument("--bundle", action="append", help="input bundle JSON for split checks")
    s.add_argument("--out", default="/tmp/v8-audit")
    s.add_argument("--strict", action="store_true")
    s.set_defaults(fn=cmd_audit)
    s = sub.add_parser("smoke")
    s.add_argument("--out", default="/tmp/v8-smoke")
    s.add_argument("--keep", action="store_true")
    s.add_argument("--fast", action="store_true")
    s.set_defaults(fn=cmd_smoke)
    from . import cluster
    cluster.register(sub)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
