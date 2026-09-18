"""Read-only MBPP storage preflight. Missing evidence is never silently reset.

Metadata/presence checks only: tensor and rollout payloads are not read. This
cannot identify who deleted a file, nor certify a checkpoint's tensor hashes.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
from pathlib import Path

MAX_OUTPUT_BYTES = 4096
JSON_LIMIT = 1024 * 1024
ARMS = {'random_full', 'random_reduced', 'selection_full', 'selection_reduced', 'gated'}
CHECKPOINT_FILES = ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt',
                    'grpo_stats.jsonl', 'checkpoint_state.json')


def json_bytes(path):
    with path.open('rb') as handle:
        raw = handle.read(JSON_LIMIT + 1)
    if len(raw) > JSON_LIMIT:
        raise ValueError('metadata exceeds 1 MiB; not fully checked')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError('metadata is not a JSON object')
    return value, raw


def present(path):
    return path.exists() or path.is_symlink()


def worker_active(path):
    """Probe an EXISTING lease only; never create/change/delete lock files."""
    if not path.is_file():
        return False
    with path.open('rb') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def complete_checkpoint(path):
    if not all((path / name).is_file() and (path / name).stat().st_size > 0 for name in CHECKPOINT_FILES):
        return False
    state, _ = json_bytes(path / 'checkpoint_state.json')
    return (type(state.get('completed_steps')) is int and state['completed_steps'] > 0
            and all(isinstance(state.get(key), str) and len(state[key]) == 64
                    for key in ('adapter_sha256', 'optimizer_sha256', 'grpo_stats_sha256')))


def audit(work, roots):
    work = Path(work)
    findings, summaries = [], []

    def note(severity, code, path, detail):
        findings.append({'severity': severity, 'code': code, 'path': str(path), 'detail': detail})

    if not work.is_dir() or not (work / 'runs').is_dir():
        note('error', 'STORAGE_UNAVAILABLE', work,
             'work/runs unavailable: check volume mount and path; cannot infer deletion')
        return {'status': 'missing', 'work': str(work), 'roots': [], 'findings': findings}
    if not any((Path(root) / 'switch.json').is_file() for root in roots):
        note('error', 'NO_EXISTING_RUN', work / 'runs',
             'no requested existing MBPP manifest found; cannot distinguish fresh setup from lost/wrong storage; no automatic initialization')

    def inspect_policy(directory, summary):
        policy = directory / 'policy'
        owner = directory / '.task.lock'
        if directory.name == 'fresh_r' and directory.parent.name.startswith('segment-'):
            owner = directory.parent.parent / '.prefix.lock'
        if worker_active(owner):
            note('warning', 'ACTIVE_WORKER', directory,
                 'owner holds lease; mutable checkpoint inspection deferred, do not interrupt or reset it')
            return
        checkpoints = sorted(policy.glob('checkpoint-*'))
        good = []
        for checkpoint in checkpoints:
            try:
                if complete_checkpoint(checkpoint):
                    good.append(checkpoint)
            except (OSError, ValueError, TypeError):
                pass
        summary['checkpoints'] += len(good)
        final = policy / 'policy_train.json'
        stop = policy / 'budget_stop.json'
        adapter = policy / 'adapter_model.safetensors'
        stats = policy / 'grpo_stats.jsonl'
        parent_only = False
        if stop.is_file():
            value, _ = json_bytes(stop)
            parent_only = value.get('use_parent_policy') is True
        if final.is_file():
            manifest, _ = json_bytes(final)
            if (not adapter.is_file() or adapter.stat().st_size == 0
                    or not (policy / 'optimizer.pt').is_file() or (policy / 'optimizer.pt').stat().st_size == 0):
                note('error', 'FINAL_POLICY_MISSING', policy,
                     'final policy manifest exists but adapter/optimizer is missing; do not retrain')
            else:
                summary['policies'] += 1
            if manifest.get('completed_steps') is None:
                note('warning', 'POLICY_METADATA', final, 'completed step not recorded; full lineage validation required')
        elif stop.is_file() and not parent_only:
            note('error', 'FINAL_MANIFEST_MISSING', policy,
                 'training stop exists but final policy manifest is missing; do not restart from parent')
        elif not parent_only and not good and (checkpoints or any(policy.glob('.checkpoint-*.tmp'))
                or (stats.is_file() and stats.stat().st_size > 0) or present(adapter)
                or present(policy / 'optimizer.pt')):
            note('error', 'CHECKPOINT_MISSING', policy,
                 'prior training state exists but no complete checkpoint metadata; refusing parent restart')
        if good and len(good) < len(checkpoints):
            note('warning', 'PARTIAL_CHECKPOINT', policy,
                 'incomplete checkpoint also exists; older complete checkpoint present, trainer must validate hashes')
        if good:
            steps = [json_bytes(p / 'checkpoint_state.json')[0]['completed_steps'] for p in good]
            summary['latest_checkpoint_step'] = max(summary['latest_checkpoint_step'], max(steps))
        ledger = directory / 'cost.jsonl'
        if ledger.is_file() and not final.is_file() and not stop.is_file():
            with ledger.open('rb') as handle:
                raw = handle.read(2 * JSON_LIMIT + 1)
            if len(raw) > 2 * JSON_LIMIT:
                note('error', 'COST_AUDIT_LIMIT', ledger, 'cannot check full completion history within metadata read limit')
            else:
                for line in raw.splitlines():
                    event = json.loads(line)
                    if event.get('state') == 'finished' and event.get('phase') == 'train' and event.get('exit_code') == 0:
                        note('error', 'TRAIN_COMPLETION_MISSING', directory,
                             'ledger records successful training but final policy/stop is absent; do not start over')
                        break

    for root in map(Path, roots):
        summary = {'root': str(root), 'results': 0, 'policies': 0, 'checkpoints': 0,
                   'latest_checkpoint_step': 0, 'archived_results': 0, 'waivers': 0, 'discards': 0}
        summaries.append(summary)
        try:
            if not root.is_dir():
                note('warning', 'ROOT_ABSENT', root,
                     'root not present here; may be unprepared or wrong path, not proof of deletion')
                continue
            if not (root / 'switch.json').is_file():
                # Logs alone can be created before prepare; saved state cannot.
                if any((root / 'states').glob('*')) or any((root / 'prefixes').glob('*')):
                    note('error', 'ROOT_MANIFEST_MISSING', root, 'saved run directories exist but switch.json is absent')
                continue
            manifest, _ = json_bytes(root / 'switch.json')
            if manifest.get('dataset') not in (None, 'mbpp'):
                note('error', 'WRONG_DATASET', root, 'requested MBPP root contains another dataset')
            # Worker status lines are surviving evidence of previously published
            # results, even if a cleanup removed both the result and its seal.
            logs = sorted((root / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
            witnessed = set()
            for log in logs[:4]:
                with log.open('rb') as handle:
                    handle.seek(0, 2)
                    handle.seek(max(0, handle.tell() - 65536))
                    tail = handle.read().decode(errors='replace')
                for seed, step, arm in re.findall(r'^s(\d+)/t(\d+)\s+(random_full|random_reduced|selection_full|selection_reduced|gated)\s+DONE\b', tail, re.MULTILINE):
                    key = (seed, step, arm)
                    if key in witnessed:
                        continue
                    witnessed.add(key)
                    points = root / f'states/s{seed}-t{step}/points'
                    if not any(points.glob(f'*/{arm}/result.json')):
                        note('error', 'PREVIOUS_DONE_MISSING', points / f'view-{step}/{arm}',
                             f'prior DONE in {log.name}, but active result absent; path/storage recovery required')
            for state in sorted((root / 'states').glob('s*-t*')):
                points = sorted(p for p in (state / 'points').glob('*') if p.is_dir())
                if len(points) > 1:
                    note('error', 'AMBIGUOUS_POINTS', state, 'multiple state points; status and worker may read different results')
                for point in points:
                    for directory in sorted((p for p in point.iterdir() if p.is_dir() and p.name in ARMS),
                                            key=lambda p: (not p.name.startswith('random'), p.name)):
                        result, seal = directory / 'result.json', directory / 'result.sha256.json'
                        archives = sorted((directory / 'discarded').glob('*/result.json'))
                        summary['archived_results'] += len(archives)
                        summary['waivers'] += len(list((directory / 'waivers').glob('*.json')))
                        discards = list((directory / 'discards').glob('*.json'))
                        summary['discards'] += len(discards)
                        if present(result):
                            value, raw = json_bytes(result)
                            if value.get('complete') is not True:
                                note('error', 'RESULT_INCOMPLETE', result, 'completion record is not complete')
                            elif seal.is_file():
                                receipt, _ = json_bytes(seal)
                                if receipt.get('sha256') != hashlib.sha256(raw).hexdigest():
                                    note('error', 'RESULT_HASH_MISMATCH', result, 'result differs from its saved seal; do not replace/retrain')
                                else:
                                    summary['results'] += 1
                            else:
                                note('warning', 'RESULT_UNSEALED', result, 'saved result awaits seal repair; do not retrain')
                            # Paths explicitly named by a result are durable completion evidence.
                            for relative in value.get('artifact_hashes', {}):
                                artifact = point / relative
                                if not artifact.resolve().is_relative_to(point.resolve()):
                                    note('error', 'ARTIFACT_PATH_ESCAPE', result, 'recorded result artifact escapes its state point')
                                elif not artifact.is_file():
                                    note('error', 'RESULT_ARTIFACT_MISSING', artifact, 'completed result references a now-missing artifact')
                            if manifest.get('gate') == 'convergence' and not (directory / 'curve.json').is_file():
                                note('warning', 'CURVE_PENDING', directory,
                                     'result exists: training is complete; remaining curve is evaluation, not new training')
                        elif present(seal):
                            note('error', 'ORPHAN_RESULT_SEAL', directory, 'result seal remains but result.json is missing')
                        if archives and not result.is_file():
                            note('error', 'ARCHIVED_RESULT', archives[-1],
                                 'completed result moved under discarded; active result absent; review before any retraining')
                        elif discards and not result.is_file():
                            note('error', 'RESET_RECEIPT', directory / 'discards',
                                 'explicit reset receipt exists; active completed result absent; inspect archived files')
                        inspect_policy(directory, summary)
            for policy in sorted((root / 'prefixes').glob('seed-*/segment-*/fresh_r/policy')):
                inspect_policy(policy.parent, summary)
            # Imported prefixes are pointers to the original saved policy. Broken
            # pointers are not new tasks and must not trigger recomputation.
            for seed in sorted((root / 'prefixes').glob('seed-*')):
                for path in seed.glob('*'):
                    if path.is_symlink() and not path.exists():
                        note('error', 'BROKEN_PREFIX_LINK', path, 'saved prefix points to missing shared storage')
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            note('error', 'AUDIT_UNREADABLE', root, str(exc))
    return {'status': 'blocked' if any(f['severity'] == 'error' for f in findings) else 'ok',
            'work': str(work), 'roots': summaries, 'findings': findings}


def render(report):
    lines = [f"MBPP STORAGE AUDIT: {report['status'].upper()}", f"WORK {report['work']}",
             'READ-ONLY. No model/rollout payloads read; no files moved/deleted; no GPU work.',
             'Cannot prove who/when deleted files. Missing path may mean unmounted storage.',
             'Checkpoint presence is NOT tensor-hash/lineage validation; trainer must verify.']
    for item in report['roots']:
        lines += [f"ROOT {item['root']}",
                  (f"  results={item['results']} policies={item['policies']} checkpoints={item['checkpoints']} "
                  f"latest_step={item['latest_checkpoint_step']} archived_results={item['archived_results']} "
                  f"waivers={item['waivers']} resets={item['discards']}")]
    for finding in sorted(report['findings'], key=lambda f: f['severity'] != 'error'):
        lines.append(f"{finding['severity'].upper()} {finding['code']} {finding['path']}: {finding['detail']}")
    lines.append('START BLOCKED; preserve files and inspect the findings.' if report['status'] != 'ok'
                 else 'Metadata preflight passed. Historical deletion without surviving evidence cannot be ruled out.')
    raw = ('\n'.join(lines) + '\n').encode()
    if len(raw) <= MAX_OUTPUT_BYTES:
        return raw.decode()
    footer = b'\n[More findings omitted; output capped at 4 KiB. See first errors above.]\n'
    return raw[:MAX_OUTPUT_BYTES-len(footer)].decode(errors='ignore') + footer.decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--root', type=Path, action='append', required=True)
    args = parser.parse_args()
    report = audit(args.work, args.root)
    print(render(report), end='')
    return 0 if report['status'] == 'ok' else 2


if __name__ == '__main__':
    raise SystemExit(main())
