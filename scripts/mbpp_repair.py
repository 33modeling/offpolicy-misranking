#!/usr/bin/env python3
"""Prepare the explicitly authorized five-branch MBPP rerun without editing its source."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SCHEMA = "mbpp-repair/v1"
STEPS = (25, 50, 100)
DEV_ARMS = ("random_reduced", "selection_reduced")
TEST_ARMS = (*DEV_ARMS, "random_full", "selection_full", "gated")


def branch(seed, step, arm):
    return f"states/s{seed}-t{step}/points/view-{step}/{arm}"


RERUN_BRANCHES = (
    branch(2, 50, "selection_reduced"),
    branch(3, 25, "selection_full"),
    branch(4, 25, "random_full"),
    branch(4, 50, "selection_reduced"),
    branch(4, 100, "random_reduced"),
)
DEPENDENT_BRANCHES = tuple(branch(s, t, "gated") for s in (3, 4) for t in STEPS)
ALL_BRANCHES = tuple(branch(s, t, a) for s in range(5) for t in STEPS
                     for a in (DEV_ARMS if s < 3 else TEST_ARMS))
REUSED_BRANCHES = tuple(p for p in ALL_BRANCHES if p not in (*RERUN_BRANCHES, *DEPENDENT_BRANCHES))
HEAVY_SUFFIXES = {".pt", ".pth", ".safetensors", ".bin", ".npy", ".npz"}
SKIP_NAMES = {"model.json", "development.json", "development-report.json", "test-report.json",
              "gate.json", "gate-frozen.json", "gate-fit", "fit-cost", "workers", "controllers",
              "logs", "repair.json"}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


@contextlib.contextmanager
def source_locks(source):
    # Existing lock inodes are opened read-only: never manufacture or remove an
    # original lease. The before/after digest also detects non-cooperating writers.
    with contextlib.ExitStack() as stack:
        for path in sorted(source.rglob("*.lock")):
            if path.is_symlink() or not path.is_file():
                continue
            handle = stack.enter_context(path.open("rb"))
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f"original work is still active: {path}; use an idle source") from exc
        yield


def _validate_source(source):
    import selection_switch_gpu as switch

    p = read(source / "switch.json")
    if (p.get("schema") != switch.rule.SCHEMA or p.get("dataset") != "mbpp"
            or p.get("selector") != "fresh_r" or p.get("accounting") != "matched"
            or p.get("gate") != "convergence" or p.get("steps") != list(STEPS)):
        raise ValueError("repair requires the frozen MBPP quality matched/convergence experiment")
    switch.validate_code_hashes(p.get("code_hashes"))
    if (source / "model.json").exists():
        raise ValueError("source gate is already fitted; the approved five-branch repair no longer matches")
    for seed in range(5):
        for step in STEPS:
            if not (source / "prefixes" / f"seed-{seed}" / f"prefix-{step}.json").is_file():
                raise ValueError(f"missing certified prefix s{seed}/t{step}; repair never retrains prefixes")
            point = source / branch(seed, step, "placeholder").rsplit("/", 1)[0]
            c = read(point / "contract.json")
            if Path(c["selected_prefix"]["root"]).resolve() != source:
                raise ValueError(f"unexpected selected-prefix root: {point}")
            if c["budget_gpu_seconds"] != p["budget_gpu_seconds"]:
                raise ValueError(f"branch budget differs from source: {point}")
            freeze = read(point / "decisions-frozen.json")
            arms = DEV_ARMS if seed < 3 else tuple(a for a in TEST_ARMS if a != "gated")
            for arm in arms:
                decision = point / arm / "decision.json"
                if freeze["decisions"][arm] != digest(decision):
                    raise ValueError(f"frozen decision changed: {decision}")
    for relative in REUSED_BRANCHES:
        directory = source / relative
        if not switch.branch_finished(p, directory):
            raise ValueError(f"expected one of 37 complete branches: {relative}")
        if read(directory / "curve.json")["result_sha256"] != digest(directory / "result.json"):
            raise ValueError(f"curve/result mismatch: {relative}")
    for relative in RERUN_BRANCHES:
        if switch.branch_finished(p, source / relative):
            raise ValueError(f"approved rerun is now complete; refusing to replace it: {relative}")
    for relative in DEPENDENT_BRANCHES:
        directory = source / relative
        if any((directory / name).exists() for name in ("execution.json", "result.json", "cost.jsonl", "policy")):
            raise ValueError(f"dependent branch already executed: {relative}")
    return p


def _skip(path):
    return (path.name in SKIP_NAMES or path.name.endswith(".lock")
            or path.name.endswith(".log") or path.name.startswith(".nfs"))


def _copy(source, target, *, prefix=False, seen=()):
    if _skip(source):
        return
    resolved = source.resolve(strict=True)
    if resolved in seen:
        raise ValueError(f"cyclic source link: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        # Certified prefix checkpoint paths must retain their original resolved
        # identity for strict parent-policy lineage validation.
        if prefix and source.name.startswith("policy_step_"):
            target.symlink_to(resolved, target_is_directory=True)
            return
        target.mkdir(exist_ok=True)
        for item in sorted(source.iterdir()):
            _copy(item, target / item.name, prefix=prefix, seen=(*seen, resolved))
    elif source.is_file():
        if source.suffix in HEAVY_SUFFIXES:
            target.symlink_to(resolved)
        else:
            shutil.copy2(source, target)
    else:
        raise ValueError(f"unsupported source artifact: {source}")


def _snapshot(root):
    return {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*"))
            if path.is_file() and not path.is_symlink() and not _skip(path)
            and path.suffix not in HEAVY_SUFFIXES}


def verify_snapshot(root, receipt):
    if receipt.get("schema") != SCHEMA:
        raise ValueError("not a registered MBPP repair")
    for key, expected in (("rerun_branches", RERUN_BRANCHES), ("dependent_branches", DEPENDENT_BRANCHES),
                          ("reused_branches", REUSED_BRANCHES)):
        if receipt.get(key) != list(expected):
            raise ValueError(f"repair branch allowlist changed: {key}")
    for relative, sha in receipt["snapshot_files"].items():
        path = root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or digest(path) != sha:
            raise ValueError(f"preserved repair artifact changed: {relative}")
    if digest(Path(receipt["source_root"]) / "switch.json") != receipt["source_switch_sha256"]:
        raise ValueError("original experiment manifest changed")


def _validate_clone(root):
    import selection_switch_gpu as switch
    from mbpp_repair_runtime import activated

    with activated(root):
        switch.install_runtime()
        for seed in range(5):
            switch.validate_prefix(root, seed, 100)
        for relative in REUSED_BRANCHES:
            directory = root / relative
            out = directory.parent
            c = switch.verify(out)
            protocol = switch.protocol(out.parent.parent)
            result = switch.runtime.validate_result(out, protocol, directory.name)
            switch.base.policy(out, c, directory.name)
            if result["rewards"] != switch.base.rewards(out, c, directory.name):
                raise ValueError(f"saved rewards differ from evaluation evidence: {relative}")


def _clone(source, target):
    _copy(source / "switch.json", target / "switch.json")
    _copy(source / "prefixes", target / "prefixes", prefix=True)
    for state in sorted((source / "states").iterdir()):
        if not state.is_dir():
            continue
        target_state = target / "states" / state.name
        for item in sorted(state.iterdir()):
            if item.name != "points":
                _copy(item, target_state / item.name)
                continue
            for point in sorted(item.iterdir()):
                if not point.is_dir():
                    continue
                target_point = target_state / "points" / point.name
                for artifact in sorted(point.iterdir()):
                    relative = str(artifact.relative_to(source))
                    destination = target_point / artifact.name
                    if relative in DEPENDENT_BRANCHES:
                        continue
                    if relative in RERUN_BRANCHES:
                        _copy(artifact, target / "original-attempts" / relative)
                        _copy(artifact / "decision.json", destination / "decision.json")
                    elif artifact.name in {"subsets", "selector-work"}:
                        for member in sorted(artifact.iterdir()):
                            arm = member.name if artifact.name == "selector-work" else member.name.removeprefix("subset-").split(".")[0]
                            branch_path = str((point / arm).relative_to(source))
                            if branch_path in (*RERUN_BRANCHES, *DEPENDENT_BRANCHES):
                                _copy(member, target / "original-attempts" / str(member.relative_to(source)))
                            else:
                                _copy(member, destination / member.name)
                    else:
                        _copy(artifact, destination)


def prepare(source, root):
    source, root = Path(source).resolve(), Path(root).resolve()
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("source and repair output must be separate, non-overlapping roots")
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f".{root.name}.prepare.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if root.exists():
            receipt = read(root / "repair.json")
            if Path(receipt["source_root"]).resolve() != source:
                raise ValueError("existing repair belongs to a different source")
            verify_snapshot(root, receipt)
            print(f"[repair] ready: preserved=37 rerun=5 dependent=6 root={root}", flush=True)
            return receipt
        with source_locks(source):
            print("[repair] validating original 37 results and five retry targets (CPU only)", flush=True)
            _validate_source(source)
            before = _snapshot(source)
            staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.prepare-", dir=root.parent))
            try:
                _clone(source, staging)
                receipt = {"schema": SCHEMA, "source_root": str(source), "root": str(root),
                           "source_switch_sha256": digest(source / "switch.json"),
                           "rerun_branches": list(RERUN_BRANCHES), "dependent_branches": list(DEPENDENT_BRANCHES),
                           "reused_branches": list(REUSED_BRANCHES), "snapshot_files": _snapshot(staging),
                           "cost_policy": "Original attempts are retained separately; five retries receive their original frozen per-attempt allocation. No historical cost is refunded."}
                (staging / "repair.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
                print("[repair] checking copied results, policies and certified prefixes (CPU only)", flush=True)
                _validate_clone(staging)
                # Validation may recover a completed receipt in the private copy.
                # Such changes must never silently alter a preserved result.
                verify_snapshot(staging, receipt)
                if _snapshot(source) != before:
                    raise ValueError("original evidence changed during preparation; no repair was published")
                os.rename(staging, root)
            except BaseException:
                shutil.rmtree(staging)
                raise
        print(f"[repair] prepared: preserved=37 rerun=5 dependent=6 root={root}", flush=True)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.source, args.root)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[repair blocked] {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
