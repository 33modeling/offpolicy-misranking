"""Shared display-only labels/counts; never change experiment state."""
from collections import Counter
from pathlib import Path

RANDOM_ARMS = {'random_full': 'RF', 'random_reduced': 'RR', 'random_online': 'RO'}
MBPP_SUITE_LABELS = {
    'fresh': 'On-policy · 선택비용 포함',
    'quality': 'On-policy · 선택비용 별도',
    'difficulty': 'Difficulty · 선택비용 포함',
    'long': 'On-policy · 장시간 예산',
}
_MBPP_ROOT_SUITES = {
    'selection-switch-mbpp-v1': 'fresh',
    'selection-switch-mbpp-quality-v1': 'quality',
    'selection-switch-mbpp-difficulty-v1': 'difficulty',
    'selection-switch-mbpp-long-v1': 'long',
}


def mbpp_suite_label(root, protocol=None):
    """Prefer explicit frozen settings; otherwise identify only known roots."""
    name = Path(root).name
    if isinstance(protocol, dict):
        if protocol.get('dataset') not in (None, 'mbpp'):
            return name
        selector, accounting = protocol.get('selector'), protocol.get('accounting')
        if (selector == 'fresh_r' and accounting == 'budget'
                and (protocol.get('budget_gpu_seconds') == 87120 or name == 'selection-switch-mbpp-long-v1')):
            return MBPP_SUITE_LABELS['long']
        suite = ({('fresh_r', 'budget'): 'fresh', ('fresh_r', 'matched'): 'quality',
                  ('difficulty', 'budget'): 'difficulty'}.get((selector, accounting))
                 if isinstance(selector, str) and isinstance(accounting, str) else None)
        if suite:
            return MBPP_SUITE_LABELS[suite]
        if selector is not None and accounting is not None:
            return name  # Explicit unregistered settings must not inherit a familiar label.
    return MBPP_SUITE_LABELS.get(_MBPP_ROOT_SUITES.get(name), name)


def selector_label(raw):
    return {'fresh_r': 'On-policy', 'difficulty': 'Difficulty'}.get(raw, raw) if isinstance(raw, str) else raw


def accounting_label(raw):
    return {'budget': '선택비용 포함', 'matched': '선택비용 별도'}.get(raw, raw) if isinstance(raw, str) else raw


def gate_label(raw):
    return {'final': '최종 보상 기준', 'convergence': '비용 보정 학습 효율 기준'}.get(raw, raw) if isinstance(raw, str) else raw


def suite_label(root, protocol=None):
    name = Path(root).name
    if isinstance(protocol, dict) and protocol.get('dataset') == 'mbpp':
        return 'MBPP ' + mbpp_suite_label(root, protocol)
    if (name.startswith('selection-switch-mbpp-') and isinstance(protocol, dict)
            and protocol.get('dataset') not in (None, 'mbpp')):
        return name
    if name == 'selection-switch-v1':
        return 'on-policy'
    if name in _MBPP_ROOT_SUITES:
        return 'MBPP ' + mbpp_suite_label(root, protocol)
    if name.startswith('selection-switch-mbpp-'):
        return name
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
