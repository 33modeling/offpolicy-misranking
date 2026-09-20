"""Avoid GPU admission when only reviewed MBPP work and its gate remain."""

import argparse
from collections import Counter
from pathlib import Path

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


def review_blockers(data):
    if (not data.get('prepared') or data.get('protocol', {}).get('dataset') != 'mbpp'
            or data.get('gate_ready')):
        return None
    tasks = data.get('tasks', [])
    if any(t.get('status') == 'RUNNING' or t.get('owner_active') or t.get('heartbeat_fresh')
           or t.get('task_lease_held') for t in tasks):
        return None
    prefixes = [t for t in tasks if t.get('kind') == 'prefix']
    expected_prefixes = {(seed, step) for seed in (*status.rule.DEV_SEEDS, *status.rule.TEST_SEEDS)
                         for step in status.rule.STEPS}
    if (len(prefixes) != len(expected_prefixes)
            or {(t['seed'], t['step']) for t in prefixes} != expected_prefixes
            or any(t.get('status') != 'DONE' for t in prefixes)):
        return None
    branches = [t for t in tasks if t.get('kind') == 'branch']
    expected = {(seed, step, arm) for seed in (*status.rule.DEV_SEEDS, *status.rule.TEST_SEEDS)
                for step in status.rule.STEPS
                for arm in (status.rule.DEV_ARMS if seed in status.rule.DEV_SEEDS else status.rule.TEST_ARMS)}
    if (len(branches) != len(expected)
            or {(t['seed'], t['step'], t['arm']) for t in branches} != expected):
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
    args = parser.parse_args()
    data = status.snapshot(args.root, local_gpus=False)
    reviewed = review_blockers(data)
    if reviewed is None:
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
