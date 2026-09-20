#!/usr/bin/env python3
"""Run an explicitly prepared MBPP repair without modifying its original run."""

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import selection_switch_gpu as switch

HERE = Path(__file__).resolve()
RERUN = {(2, 50, "selection_reduced"), (3, 25, "selection_full"),
         (4, 25, "random_full"), (4, 50, "selection_reduced"),
         (4, 100, "random_reduced")}


def branch_name(seed, step, arm):
    return f"states/s{seed}-t{step}/points/view-{step}/{arm}"


def find_root(path):
    if path is None:
        return None
    path = Path(path).absolute()
    return next((parent for parent in (path, *path.parents)
                 if (parent / "repair.json").is_file()), None)


def _relative(name):
    path = Path(name)
    if path.is_absolute() or not path.parts or ".." in path.parts or str(path) != name:
        raise ValueError(f"invalid repair-relative path: {name!r}")
    return path


def validate(root):
    root = Path(root).resolve()
    p = switch.core.read(root / "repair.json")
    if p.get("schema") != "mbpp-repair/v1":
        raise ValueError("unsupported MBPP repair manifest")
    source = Path(p["source_root"]).resolve()
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("MBPP repair and original must be separate roots")
    if switch.base.digest(source / "switch.json") != p["source_switch_sha256"]:
        raise ValueError("original MBPP switch manifest changed")
    if switch.base.digest(root / "switch.json") != p["source_switch_sha256"]:
        raise ValueError("repair must preserve the original switch manifest bytes")
    protocol = switch.core.read(root / "switch.json")
    if protocol.get("schema") != switch.rule.SCHEMA or protocol.get("dataset") != "mbpp":
        raise ValueError("repair requires the frozen MBPP switch protocol")
    switch.validate_code_hashes(protocol["code_hashes"])
    expected_rerun = {branch_name(*item) for item in RERUN}
    expected_dependent = {branch_name(seed, step, "gated")
                          for seed in switch.rule.TEST_SEEDS for step in switch.rule.STEPS}
    all_branches = {branch_name(seed, step, arm)
                    for seeds, arms in ((switch.rule.DEV_SEEDS, switch.rule.DEV_ARMS),
                                        (switch.rule.TEST_SEEDS, switch.rule.TEST_ARMS))
                    for seed in seeds for step in switch.rule.STEPS for arm in arms}
    for key, expected in (("rerun_branches", expected_rerun),
                          ("dependent_branches", expected_dependent),
                          ("reused_branches", all_branches - expected_rerun - expected_dependent)):
        values = p[key]
        if not isinstance(values, list) or len(values) != len(expected) or set(values) != expected:
            raise ValueError(f"repair {key} differs from the authorized branch set")
    snapshots = p["snapshot_files"]
    if not isinstance(snapshots, dict) or not snapshots:
        raise ValueError("repair requires immutable metadata snapshots")
    for name, digest in snapshots.items():
        if switch.base.digest(root / _relative(name)) != digest:
            raise ValueError(f"repair snapshot changed: {name}")
    for seed in (*switch.rule.DEV_SEEDS, *switch.rule.TEST_SEEDS):
        for step in switch.rule.STEPS:
            if not (switch.prefix_dir(root, seed) / f"prefix-{step}.json").is_file():
                raise ValueError("repair requires all fifteen certified prefixes")
            child = switch.child_root(root, seed, step)
            if not (child / "net_protocol.json").is_file():
                raise ValueError("repair requires all published state protocols")
    return root, source, p


def install(path):
    """Install process-local guards; return an idempotent cleanup callable."""
    root = find_root(path)
    if root is None:
        return lambda: None
    root, source, p = validate(root)
    allowed = set(p["rerun_branches"]) | set(p["dependent_branches"])
    originals = []

    def replace(module, name, value):
        originals.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    def writable(out, arm):
        directory = (Path(out) / arm).resolve()
        try:
            name = str(directory.relative_to(root))
        except ValueError as exc:
            raise ValueError("repair worker cannot write outside its repair root") from exc
        if name not in allowed:
            raise ValueError(f"repair cannot execute a reused or unregistered branch: {name}")

    original_manifest = switch.manifest

    def manifest(candidate):
        candidate = Path(candidate).resolve()
        if candidate == source:
            if switch.base.digest(source / "switch.json") != p["source_switch_sha256"]:
                raise ValueError("original MBPP switch manifest changed")
            value = switch.core.read(source / "switch.json")
            if value.get("schema") != switch.rule.SCHEMA:
                raise ValueError("original MBPP switch schema changed")
            switch.validate_code_hashes(value["code_hashes"])
            return value
        if candidate != root:
            raise ValueError("repair manifest access outside its registered roots")
        return original_manifest(candidate)

    def guarded(original):
        def call(out, suite, protocol, arm, devices, env):
            writable(out, arm)
            return original(out, suite, protocol, arm, devices, env)
        return call

    def guarded_evaluation(original):
        def call(out, arm, *args, **kwargs):
            writable(out, arm)
            return original(out, arm, *args, **kwargs)
        return call

    def forbidden(*args, **kwargs):
        raise ValueError("repair cannot prepare states, rebuild prefixes, or repeat shared diagnostics")

    # switch.main()/preparation install scientific callbacks inside this scope.
    # Restore those too when a caller returns to another experiment in-process.
    for module, names in ((switch.runtime, ("net", "TEST_ARMS", "SELECTORS", "CODE_FILES", "study",
                                           "protocol", "select_once", "decision")),
                          (switch.base, ("verify", "train_command"))):
        for name in names:
            replace(module, name, getattr(module, name))
    replace(switch, "manifest", manifest)
    replace(switch, "HERE", HERE)
    replace(switch.runtime, "HERE", HERE)
    replace(switch.runtime, "run_arm", guarded(switch.runtime.run_arm))
    replace(switch.base, "evaluate", guarded_evaluation(switch.base.evaluate))
    replace(switch, "curve_evaluate", guarded_evaluation(switch.curve_evaluate))
    for name in ("build_prefix", "publish_state", "prepare", "measurement_worker"):
        replace(switch, name, forbidden)
    replace(switch.runtime, "measurement_worker", forbidden)

    def cleanup():
        while originals:
            module, name, value = originals.pop()
            setattr(module, name, value)
    return cleanup


@contextmanager
def activated(root):
    cleanup = install(root)
    try:
        yield
    finally:
        cleanup()


def cli_root(argv=None, *, required=True):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=required)
    return parser.parse_known_args(argv)[0].root


def main():
    with activated(cli_root()):
        return switch.main()


if __name__ == "__main__":
    from light_selection_gate_gpu import install_signal_handlers
    install_signal_handlers()
    raise SystemExit(main())
