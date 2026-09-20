"""Read-only MBPP metadata export split into upload-sized UTF-8 text files."""

import json
import re
import tempfile
from pathlib import Path

import mbpp_failure_summary as summary

PART_BYTES = 8 * 1024
LOG_BYTES = 8 * 1024
ARMS = {'selection_reduced', 'random_reduced', 'selection_full', 'random_full', 'gated'}
POLICY_FILES = ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt',
                'grpo_stats.jsonl', 'checkpoint_state.json', 'policy_train.json', 'budget_stop.json')


def metadata(root, path, keys=None):
    value = summary.record(root, path)
    if keys is not None and '_read_error' not in value:
        value = {key: value[key] for key in keys if key in value}
    return f'FILE {path.relative_to(root)}\n{json.dumps(value, ensure_ascii=False, sort_keys=True)}\n'


def policy_inventory(root, policy):
    yield f'POLICY {policy.relative_to(root)} (presence only; hashes NOT validated)\n'
    try:
        summary.checked(root, policy)
        candidates = [policy, *sorted(policy.glob('checkpoint-*')), *sorted(policy.glob('.checkpoint-*.tmp')),
                      *sorted((policy / 'curve-checkpoints').glob('step-*'))]
        for candidate in candidates:
            summary.checked(root, candidate)
            files = {}
            for name in POLICY_FILES:
                path = summary.checked(root, candidate / name)
                files[name] = path.stat().st_size if path.is_file() else None
            yield f'INVENTORY {candidate.relative_to(root)} bytes={json.dumps(files, sort_keys=True)}\n'
            for name in ('checkpoint_state.json', 'policy_train.json', 'budget_stop.json'):
                path = candidate / name
                if path.exists():
                    yield metadata(root, path, ('schema', 'completed_steps', 'start_step', 'target_steps',
                        'requested_target_steps', 'stop_reason', 'use_parent_policy', 'training_objective',
                        'adapter_sha256', 'optimizer_sha256', 'grpo_stats_sha256'))
    except (OSError, ValueError) as exc:
        yield f'INVENTORY ERROR {exc}\n'


def log_excerpt(root, path):
    yield f'LOG {path.relative_to(root)} (last {LOG_BYTES} bytes at most)\n'
    try:
        summary.checked(root, path)
        with path.open('rb') as handle:
            size = handle.seek(0, 2)
            handle.seek(max(0, size - LOG_BYTES))
            data = handle.read(LOG_BYTES)
        if size > LOG_BYTES:
            yield '[Earlier log bytes not collected; metadata records below are exported separately.]\n'
        yield data.decode('utf-8', errors='replace') + '\n'
    except (OSError, ValueError) as exc:
        yield f'LOG ERROR {exc}\n'


def sections(work, roots):
    yield ('MBPP DIAGNOSTIC DETAILS\nREAD-ONLY. No training, repair or lock creation.\n'
           'No model/optimizer/rollout payloads. Presence is NOT hash/lineage validation.\n'
           'All discovered branch records included, not only two recent failures.\n'
           'JSON reads limited to 1 MiB each; oversize/unreadable records show errors.\n'
           'Log excerpts are bounded; this is not an atomic snapshot of live workers.\n')
    for root in roots:
        yield f'\nROOT {root}\n'
        if not root.is_dir():
            yield 'ROOT MISSING\n'
            continue
        yield metadata(root, root / 'switch.json', ('dataset', 'selector', 'accounting', 'gate'))
        yield f'gate model present={int((root / "model.json").is_file())}\n'
        for path in sorted((root / 'gate-fit').glob('failure.json')):
            yield metadata(root, path)
        for directory in sorted(root.glob('states/*/points/*/*')):
            if directory.name not in ARMS or not directory.is_dir():
                continue
            yield f'\nBRANCH {directory.relative_to(root)}\n'
            try:
                summary.checked(root, directory)
                yield f'charged deployment GPU-s={summary.charged(directory)}\n'
                for name in ('decision.json', 'failure.json', 'progress.json',
                             'budget-recovery/review.json', 'budget-recovery/failure.json',
                             'budget-recovery/progress.json'):
                    path = directory / name
                    yield metadata(root, path) if path.exists() else f'MISSING {path.relative_to(root)}\n'
                for name, keys in (
                    ('result.json', ('complete', 'completed_steps', 'stop_reason', 'used_gpu_seconds', 'budget_gpu_seconds')),
                    ('curve.json', ('result_sha256',)),
                    ('budget-recovery/plan.json', ('schema', 'canonical_complete', 'purpose', 'completed_steps',
                                                  'start_step', 'used_gpu_seconds', 'budget_gpu_seconds', 'over_budget_gpu_seconds')),
                    ('budget-recovery/result.json', ('schema', 'canonical_complete', 'evaluation_complete',
                                                    'plan_sha256', 'used_gpu_seconds', 'budget_gpu_seconds', 'over_budget_gpu_seconds'))):
                    path = directory / name
                    yield metadata(root, path, keys) if path.exists() else f'MISSING {path.relative_to(root)}\n'
                yield from policy_inventory(root, directory / 'policy')
                progress = summary.record(root, directory / 'progress.json')
                phase = progress.get('phase')
                if isinstance(phase, str) and re.fullmatch(r'[\w-]+', phase):
                    for path in summary.recent(directory, (f'{phase}-*.log',))[:4]:
                        yield f'ERROR EXCERPT {path.relative_to(root)} (bounded log context)\n'
                        yield summary.error_excerpt(root, path) + '\n'
            except (OSError, ValueError) as exc:
                yield f'BRANCH ERROR {exc}\n'
        for path in sorted(root.glob('prefixes/seed-*/segment-*/failure.json')):
            yield metadata(root, path)
        for path in sorted(root.glob('states/*/failure.json')):
            yield metadata(root, path)
        for path in sorted(root.glob('node-preflight/*/admission.json')):
            yield metadata(root, path)
    for pattern in ('runs/experiments/logs/console.mbpp.*.log', 'runs/experiments/logs/cleanup.mbpp.*.log'):
        for path in sorted(work.glob(pattern)):
            yield from log_excerpt(work, path)


def write_parts(chunks, destination):
    """Split bytes without dropping content or splitting a UTF-8 code point."""
    destination.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='mbpp-why-', dir=destination))
    paths, buffer = [], b''
    # Fixed payload headroom keeps headers inside the per-file byte cap.
    capacity = PART_BYTES - 256

    def save(data):
        path = folder / f'mbpp-why-{len(paths) + 1:03d}.txt'
        header = (f'MBPP WHY part {len(paths) + 1}; set={folder.name}\n'
                  'Parts concatenate in numeric order; content may continue from the preceding part.\n\n').encode()
        if len(header) > 256:
            raise ValueError('diagnostic part header exceeds reserved space')
        path.write_bytes(header + data)
        paths.append(path)

    for chunk in chunks:
        buffer += chunk.encode('utf-8')
        while len(buffer) >= capacity:
            end = capacity
            while end < len(buffer) and buffer[end] & 0xC0 == 0x80:
                end -= 1
            save(buffer[:end])
            buffer = buffer[end:]
    if buffer:
        save(buffer)
    return paths
