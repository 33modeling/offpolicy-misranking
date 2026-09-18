"""Shared display-only labels/counts; never change experiment state."""
from collections import Counter
from pathlib import Path

RANDOM_ARMS = {'random_full': 'RF', 'random_reduced': 'RR', 'random_online': 'RO'}


def suite_label(root):
    name = Path(root).name
    if name == 'selection-switch-v1':
        return 'on-policy'
    if name == 'selection-switch-mbpp-v1':
        return 'MBPP on-policy'
    if name == 'mopps-comparison-v1':
        return 'MoPPS'
    return name.removeprefix('selection-switch-').removesuffix('-v1')


def random_counts(tasks):
    return {label: dict(Counter(task['status'] for task in tasks
                               if task.get('kind', 'branch') == 'branch' and task.get('arm') == arm))
            for arm, label in RANDOM_ARMS.items()
            if any(task.get('kind', 'branch') == 'branch' and task.get('arm') == arm for task in tasks)}


def random_text(counts):
    parts = []
    for arm, states in counts.items():
        value = f'{arm} DONE {states.get("DONE", 0)}/{sum(states.values())}'
        pending = [f'{"RUN" if key == "RUNNING" else key} {count}'
                   for key, count in states.items() if key != 'DONE' and count]
        parts.append(value + (' ' + ' '.join(pending) if pending else ''))
    return ' | '.join(parts)
