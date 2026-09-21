"""Avoid GPU admission when MBPP work is reviewed or already owned by peers."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import stat

import selection_switch_status as status
from mbpp_storage_audit import policy_resume_blocked


def terminal_review(data, task):
    if task.get('posthoc_evaluation_saved') is True:
        return True
    try:
        root = Path(data['root']).resolve()
        directory = root / task['directory']
        if not directory.resolve().is_relative_to(root):
            return False
        # REVIEW is a display label, not a worker readiness contract. A saved
        # final policy can repair its missing budget_stop on CPU; parent-only
        # stops can still evaluate. Use the worker's narrow quarantine check.
        return policy_resume_blocked(directory)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False


def registered_branches(data):
    tasks = data.get('tasks', [])
    prefixes = [t for t in tasks if t.get('kind') == 'prefix']
    expected_prefixes = {(seed, step) for seed in (*status.rule.DEV_SEEDS, *status.rule.TEST_SEEDS)
                         for step in status.rule.STEPS}
    if (len(prefixes) != len(expected_prefixes)
            or {(t.get('seed'), t.get('step')) for t in prefixes} != expected_prefixes
            or any(t.get('status') != 'DONE' for t in prefixes)):
        return None
    branches = [t for t in tasks if t.get('kind') == 'branch']
    expected = {(seed, step, arm) for seed in (*status.rule.DEV_SEEDS, *status.rule.TEST_SEEDS)
                for step in status.rule.STEPS
                for arm in (status.rule.DEV_ARMS if seed in status.rule.DEV_SEEDS else status.rule.TEST_ARMS)}
    if (len(branches) != len(expected)
            or {(t.get('seed'), t.get('step'), t.get('arm')) for t in branches} != expected):
        return None
    return branches


def gate_fit_claimable(data):
    """Wake the controller for a validated, currently unowned CPU gate fit."""
    if (not data.get('prepared') or data.get('protocol', {}).get('dataset') != 'mbpp'
            or data.get('protocol', {}).get('gate') != 'convergence'
            or data.get('gate_ready') or data.get('gate_fit_failure') or data.get('notices')
            or data.get('development_done') != 18):
        return False
    branches = registered_branches(data)
    if branches is None or any(
            task.get('status') != 'DONE' or task.get('task_lease_held')
            for task in branches if task['seed'] in status.rule.DEV_SEEDS):
        return False
    if not any(task['arm'] == 'gated' and task.get('status') == 'WAIT'
               and not task.get('task_lease_held') for task in branches):
        return False
    try:
        path = Path(data['root']) / '.fit.lock'
        # Read-only, nonblocking and no symlink following: unknown leases are
        # not permission to wake. The worker rechecks the lease before fitting.
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return False
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
    except FileNotFoundError:
        return True
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False
    return True


def peer_blockers(data):
    """Defer only a complete registry whose unfinished work has real task leases."""
    if (not data.get('prepared') or data.get('protocol', {}).get('dataset') != 'mbpp'
            or data.get('notices')):
        return None
    branches = registered_branches(data)
    if branches is None:
        return None
    peers, dependent = [], False
    for task in branches:
        if task.get('status') == 'DONE':
            continue
        if task.get('task_lease_held'):
            peers.append(task)
        elif task['arm'] == 'gated' and task.get('status') == 'WAIT' and not data.get('gate_ready'):
            dependent = True
        else:
            return None
    if not peers:
        return None
    # Once development curves finish, an idle worker can fit the gate even
    # while other controls remain peer-owned.
    if dependent and not any(task['seed'] in status.rule.DEV_SEEDS for task in peers):
        return None
    return peers


def review_blockers(data):
    if (not data.get('prepared') or data.get('protocol', {}).get('dataset') != 'mbpp'
            or data.get('gate_ready')):
        return None
    tasks = data.get('tasks', [])
    if any(t.get('status') == 'RUNNING' or t.get('owner_active') or t.get('heartbeat_fresh')
           or t.get('task_lease_held') for t in tasks):
        return None
    branches = registered_branches(data)
    if branches is None:
        return None
    reviewed = [t for t in branches if t.get('status') == 'REVIEW']
    if not any(t['seed'] in status.rule.DEV_SEEDS for t in reviewed):
        return None
    if not all(terminal_review(data, task) for task in reviewed):
        return None
    for task in branches:
        if task.get('status') in {'DONE', 'REVIEW'}:
            continue
        if task['arm'] == 'gated' and task.get('status') == 'WAIT':
            continue
        # BUDGET without a sealed recovery is not terminal here: saved-policy
        # evaluation may still be runnable. FAILED/EVAL/RESUME also go to worker.
        return None
    return reviewed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repair-source', type=Path,
                        help='print only a verified existing repair root; no preparation or worker checks')
    args = parser.parse_args()
    if args.repair_source is not None:
        from mbpp_status import verified_repair_root
        repair = verified_repair_root(args.repair_source, args.root)
        if repair is not None:
            print(repair)
        return 0
    data = status.snapshot(args.root, local_gpus=False)
    reviewed = review_blockers(data)
    if reviewed is None:
        peers = peer_blockers(data)
        if peers is not None:
            gate_wait = sum(task.get('arm') == 'gated' and task.get('status') == 'WAIT'
                            for task in data['tasks'])
            print(f'[waiting] MBPP unfinished branches are held by {len(peers)} peer task leases; '
                  f'gate-wait={gate_wait}; no GPU admission on this idle node [mbpp]', flush=True)
            for task in peers:
                print(f"[peer] {task['directory']}: task lease held [mbpp]", flush=True)
            return 82
        return 0
    counts = Counter(t['status'] for t in data['tasks'] if t.get('kind') == 'branch')
    print(f"[blocked] MBPP has no runnable branch: saved={counts['DONE']} review={counts['REVIEW']} "
          f"gate-wait={counts['WAIT']}; no GPU admission, no training retry, NOT complete [mbpp]", flush=True)
    for task in reviewed:
        print(f"[review] {task['directory']}: {task.get('reason', 'saved work requires review')} [mbpp]", flush=True)
    print('[review] Missing canonical development results cannot be replaced by posthoc evaluations; '
          'preserve checkpoints and costs [mbpp]', flush=True)
    return 80


if __name__ == '__main__':
    raise SystemExit(main())
