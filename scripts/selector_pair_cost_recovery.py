"""Apply the existing interrupted-cost policy inside validated Pair branches."""
from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
import re

import selector_pair_gpu as worker


def recovery_module():
    # Private globals let close_stale retain its reviewed policy while adding
    # Pair publication guards without changing the standalone Switch helper.
    path = Path(__file__).with_name("recover_selection_switch_cost.py")
    spec = importlib.util.spec_from_file_location("_pair_interrupted_cost_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def location(root, branch, directory):
    if checked(directory) != directory or not directory.is_relative_to(branch):
        raise ValueError("Pair recovery refuses an escaped or symlinked cost directory")
    parts = directory.relative_to(branch).parts
    state = re.fullmatch(r"s([0-9]+)-t([0-9]+)", parts[1]) if len(parts) >= 2 else None
    if (len(parts) not in (5, 6) or parts[0] != "states" or not state
            or parts[2] != "points" or parts[3] != f"view-{state[2]}"
            or int(state[2]) not in worker.pair.STEPS):
        raise ValueError("unsupported Pair cost directory; preserved")
    seed = int(state[1])
    if seed in worker.pair.DEV_SEEDS and branch.name in worker.pair.SELECTORS:
        stage, arms = "development", ("selection_reduced",)
    elif seed in worker.pair.TEST_SEEDS:
        stage = "test"
        arms = ("selection_full", "random_full") if branch.name == "on_policy" else ("selection_full",)
    else:
        raise ValueError("cost directory is not a scheduled Pair state")
    parent = len(parts) == 5 and parts[4] == "curve-parent"
    if not parent and (parts[4] not in arms or len(parts) == 6 and parts[5] != "curve"):
        raise ValueError("cost directory is not a scheduled Pair arm or curve")
    return root / stage / parts[1], branch.joinpath(*parts[:4]), arms if parent else (parts[4],), parent


def checked(path):
    try:
        resolved = path.resolve()
    except RuntimeError as exc:
        raise ValueError(f"Pair recovery refuses a cyclic path: {path}") from exc
    if resolved != path:
        raise ValueError(f"Pair recovery refuses a symlinked path: {path}")
    return path


def checked_cost_files(directory):
    for name in ("cost.jsonl", "progress.json", "cost-events", "pending-costs", ".cost.lock"):
        checked(directory / name)
    for name in ("cost-events", "pending-costs"):
        for path in (directory / name).glob("*.json"):
            checked(path)


def serial_main_ledger(branch, directory, events):
    parts = directory.relative_to(branch).parts
    if len(parts) != 5 or parts[-1] not in worker.switch.rule.TEST_ARMS:
        return
    pending = worker.core.cost_summary(events)["incomplete_events"]
    if not pending:
        worker.pair.finished_events(events)
        return
    if len(pending) != 1:
        raise ValueError("Pair main ledger has overlapping open phases; original costs preserved")
    index = next(i for i, row in enumerate(events) if row["event_id"] == pending[0])
    if any(row["event_id"] != pending[0] or row["state"] != "started" for row in events[index:]):
        raise ValueError("Pair main ledger is not a closed serial prefix plus one open phase; original costs preserved")
    worker.pair.finished_events(events[:index])


@contextlib.contextmanager
def publication_guard(root, branch, directory):
    folder, out, arms, parent = location(root, branch, directory)
    with contextlib.ExitStack() as locks:
        owner = (directory.parent / ".task.lock" if directory.name == "curve" else
                 directory / (".point.lock" if parent else ".task.lock"))
        checked(owner)
        locks.enter_context(worker.pair_lease(checked(folder / ".state.lock"), shared=True))
        for arm in arms:
            receipt = checked(folder / "queue-branches" / f"{branch.name}--{arm}.json")
            locks.enter_context(worker.pair_lease(checked(receipt.with_suffix(".lock"))))
        if ((folder / "result.json").exists()
                or any((folder / "queue-branches" / f"{branch.name}--{arm}.json").exists() for arm in arms)):
            raise ValueError("Pair costs are sealed by a published branch or state result; preserved")
        # A parent ledger is shared across arms. Detached evaluators can retain
        # a shard lease after their controller's point/task lease disappears.
        if parent:
            for arm in arms:
                locks.enter_context(worker.pair_lease(checked(out / arm / ".task.lock"), shared=True))
        points = list(directory.glob("step-*/.point.lock")) if directory.name == "curve" else []
        shards = list(directory.rglob("shard-*.lock"))
        for path in sorted(set(points + shards)):
            locks.enter_context(worker.pair_lease(checked(path), shared=True))
        checked_cost_files(directory)
        yield


def recover_branch(root, branch, *, now=None):
    policy = recovery_module()
    original_recover = policy.recover
    original_inspect = policy.inspect
    original_read = policy.read_events

    def read(directory, *, repair=False):
        checked_cost_files(directory)
        raw, events = original_read(directory, repair=repair)
        serial_main_ledger(branch, directory, events)
        return raw, events

    def inspect(branch_root, *, errors=None):
        pending = []
        for item in original_inspect(branch_root, errors=errors):
            try:
                location(root, branch, branch / item["directory"])
            except ValueError as exc:
                if errors is None:
                    raise
                errors.append({"directory": item["directory"], "event_id": item["start"]["event_id"],
                               "status": "blocked", "reason": str(exc)})
            else:
                pending.append(item)
        return pending

    def recover(branch_root, directory, event_id, **kwargs):
        directory = branch / directory
        if kwargs.get("evidence_kind") in {"stale_owner_last_evidence", "confirmed_local_stop_last_evidence"}:
            kwargs["reason"] = ("Estimated interrupted-attempt duration from last observed evidence plus the existing "
                                "margin; not directly measured and not a guaranteed upper bound")
            kwargs["evidence_extra"] = {**kwargs.get("evidence_extra", {}), "duration_is_estimate": True,
                                       "directly_measured": False}
        try:
            with publication_guard(root, branch, directory):
                result = original_recover(branch_root, directory, event_id, **kwargs)
        except worker.PairLockBusy as exc:
            return {"status": "active", "reason": f"Pair lease held: {exc.path}; costs preserved"}
        return result

    policy.inspect, policy.recover, policy.read_events = inspect, recover, read
    return policy.close_stale(branch, now=now)


def recover(root, protocol, *, now=None):
    """Validate each branch before recovery; a bad branch does not block peers."""
    root = Path(root).resolve()
    outcomes = []
    for name, digest in protocol["branch_manifests"].items():
        try:
            if name not in worker.BRANCHES:
                raise ValueError("unknown Pair branch")
            branch = checked(root / "branches" / name)
            if worker.base.digest(checked(branch / "switch.json")) != digest:
                raise ValueError("Pair branch manifest changed; costs preserved")
            worker.switch.manifest(branch)
            rows = recover_branch(root, branch, now=now)
        except (ValueError, OSError, RuntimeError, KeyError, TypeError, AttributeError) as exc:
            rows = [{"status": "blocked", "reason": str(exc)}]
        outcomes.extend({**row, "branch": name} for row in rows)
    for row in outcomes:
        estimate = row.get("evidence", {}).get("kind", "")
        detail = (f"{row['seconds']:g}s charged; evidence={estimate}; prior costs preserved"
                  if row["status"] == "recovered" else row.get("reason", ""))
        print(f"[pair-recover-cost] {row['branch']}/{row.get('directory', '')} "
              f"{row.get('event_id', '')}: {row['status']} {detail}", flush=True)
    return outcomes
