#!/usr/bin/env python3
"""Keep surviving Pair curve shards pending without changing the frozen worker."""
from __future__ import annotations

import contextlib
import fcntl
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import selector_pair_gpu as worker
import selector_pair_cost_recovery as cost_recovery

MODES = {"run", "develop", "test", "freeze"}
RECEIPT = "pair-curve-shard-guard-runtime.json"
COST_RECEIPT = "pair-cost-recovery-runtime.json"
PRE_COST_GUARD_SHA256 = "3199888c2460768a09da64c098efd7aaaf8898e1707575e67a20f61abe5d4e43"


def known_point(root, branch, out, arm):
    root, branch, out = Path(root).resolve(), Path(branch).resolve(), Path(out).resolve()
    if not out.is_relative_to(root):
        return False
    parts = out.relative_to(root).parts
    if (len(parts) != 6 or parts[0] != "branches" or parts[1] not in worker.BRANCHES
            or parts[2] != "states" or parts[4] != "points"
            or branch != root / "branches" / parts[1]
            or arm not in worker.switch.rule.TEST_ARMS):
        return False
    state = re.fullmatch(r"s([0-9]+)-t([0-9]+)", parts[3])
    return bool(state and int(state[1]) in (*worker.pair.DEV_SEEDS, *worker.pair.TEST_SEEDS)
                and int(state[2]) in worker.pair.STEPS and parts[5] == f"view-{state[2]}")


def check_shards(target):
    """Probe existing leases only; a stale lock file does not own a shard."""
    for shard in range(4):
        if (target / f"shard-{shard}.done.json").exists():
            continue
        lock = target / f"shard-{shard}.lock"
        if lock.is_symlink():
            raise RuntimeError(f"refusing symlinked curve shard lock: {lock}")
        try:
            handle = lock.open("rb")
        except FileNotFoundError:
            continue
        except BlockingIOError as exc:
            raise RuntimeError(f"cannot read curve shard lock: {lock}") from exc
        with handle:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)


@contextlib.contextmanager
def activated(root):
    """Guard only the frozen curve call, restoring both hooks on every exit."""
    root = Path(root).resolve()
    switch = worker.switch
    original_curve = switch.curve_once

    def curve(branch, protocol, out, config, arm, suite, devices, env):
        if not known_point(root, branch, out, arm):
            return original_curve(branch, protocol, out, config, arm, suite, devices, env)
        original_lease = worker.base.lease
        out = Path(out).resolve()

        @contextlib.contextmanager
        def lease(path, *, blocking=False):
            with original_lease(path, blocking=blocking):
                path = Path(path).resolve()
                target = path.parent
                parent = target == out / "curve-parent"
                checkpoint = (target.parent == out / arm / "curve"
                              and re.fullmatch(r"step-[0-9]+", target.name))
                if path.name == ".point.lock" and (parent or checkpoint):
                    check_shards(target)
                yield

        worker.base.lease = lease
        try:
            return original_curve(branch, protocol, out, config, arm, suite, devices, env)
        finally:
            worker.base.lease = original_lease

    switch.curve_once = curve
    try:
        yield
    finally:
        switch.curve_once = original_curve


def guard_receipt(root, protocol):
    return {
            "schema": "offpolicy-selector-pair/curve-shard-guard-v1",
            "root": str(root), "protocol_id": protocol["protocol_id"],
            "pair_manifest_sha256": worker.base.digest(root / "pair.json"),
            "guard_sha256": worker.base.digest(Path(__file__).resolve()),
            "frozen_code_hashes": protocol["code_hashes"],
            "change": "defer curve points while an unfinished shard holds its exclusive lease",
            "cost_policy": "preserve all costs, budgets, checkpoints, evaluations and task leases",
        }


def recovery_receipt(root, protocol):
    scripts = Path(__file__).resolve().parent
    return {
        "schema": "offpolicy-selector-pair/cost-recovery-v1",
        "root": str(root), "protocol_id": protocol["protocol_id"],
        "pair_manifest_sha256": worker.base.digest(root / "pair.json"),
        "curve_guard_receipt_sha256": worker.base.digest(root / RECEIPT),
        "runtime_code_hashes": {name: worker.base.digest(scripts / name) for name in (
            "queue_selector_pair_gpu.py", "selector_pair_cost_recovery.py",
            "recover_selection_switch_cost.py", "_recovery_owners.py", "mbpp_storage_audit.py")},
        "frozen_code_hashes": protocol["code_hashes"],
        "change": "recover interrupted costs before Pair admission using the existing stale-owner policy",
        "cost_policy": "append explicit evidence-based estimates; preserve prior costs, budgets, live leases and published costs",
    }


def validate_receipts(root, protocol):
    """Read-only preflight, also used by handoff before stopping any worker."""
    root = Path(root).resolve()
    if worker.manifest(root, bind_runtime=False) != protocol:
        raise ValueError("Pair protocol changed before curve guard activation")
    for name in (RECEIPT, COST_RECEIPT):
        if (root / name).is_symlink():
            raise ValueError(f"refusing a symlinked Pair runtime receipt: {name}")
    expected = guard_receipt(root, protocol)
    if (root / RECEIPT).exists():
        previous = worker.core.read(root / RECEIPT)
        if previous not in (expected, {**expected, "guard_sha256": PRE_COST_GUARD_SHA256}):
            raise ValueError(f"frozen contract changed: {root / RECEIPT}")
    if (root / COST_RECEIPT).exists():
        if not (root / RECEIPT).exists() or worker.core.read(root / COST_RECEIPT) != recovery_receipt(root, protocol):
            raise ValueError(f"frozen contract changed: {root / COST_RECEIPT}")


def bind_receipt(root, protocol):
    with worker.queue_lease(root / ".pair-runtime.lock"):
        validate_receipts(root, protocol)
        if not (root / RECEIPT).exists():
            worker.base.bind(root / RECEIPT, guard_receipt(root, protocol))


def bind_recovery_receipt(root, protocol):
    with worker.queue_lease(root / ".pair-runtime.lock"):
        validate_receipts(root, protocol)
        worker.base.bind(root / COST_RECEIPT, recovery_receipt(root, protocol))


def run():
    if len(sys.argv) != 4 or sys.argv[1] not in MODES or sys.argv[2] != "--root":
        raise SystemExit("Pair queue adapter requires run|develop|test|freeze --root PATH")
    root = Path(sys.argv[3]).resolve()
    original_stage = worker.run_distributed
    original_admission = worker.admit_node
    prepared = False

    def prepare(stage_root, protocol):
        nonlocal prepared
        if Path(stage_root).resolve() != root:
            raise ValueError("Pair queue adapter root changed")
        bind_receipt(root, protocol)
        bind_recovery_receipt(root, protocol)
        if not prepared:
            cost_recovery.recover(root, protocol)
            prepared = True

    def admit(stage_root, protocol):
        prepare(stage_root, protocol)
        return original_admission(stage_root, protocol)

    def stage(stage_root, protocol, devices, mode):
        prepare(stage_root, protocol)
        return original_stage(stage_root, protocol, devices, mode)

    worker.run_distributed = stage
    worker.admit_node = admit
    try:
        with activated(root):
            return worker.main()
    finally:
        worker.run_distributed = original_stage
        worker.admit_node = original_admission


def main():
    from light_selection_gate_gpu import install_signal_handlers
    install_signal_handlers()
    try:
        return run()
    except worker.PairWaitTimeout as exc:
        print(f"[pair-wait-timeout] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(76) from None
    except worker.PairLockBusy as exc:
        if exc.path.name == ".pair.lock":
            worker.show_pair_activity(exc.path.parent)
            raise SystemExit(75) from None
        raise SystemExit(f"[pair] {exc}") from None
    except worker.NodeAdmissionError as exc:
        print(f"[blocked] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(78) from None
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"[pair] {exc}") from None
    except (OSError, RuntimeError):
        worker.resource_diagnostics()
        raise


if __name__ == "__main__":
    sys.exit(main())
