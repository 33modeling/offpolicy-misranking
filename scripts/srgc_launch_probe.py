"""Inspect this user's local SRGC launch processes without signals or GPU calls."""

import argparse
import json
import os
import socket
from pathlib import Path

MODULES = {
    'srgc_research.dispatch.model_chain': 'coordinator',
    'srgc_research.dispatch.gemma_it_run': 'gemma-it-launcher',
    'srgc_research.dispatch.gemma4_it.entry': 'gemma-it-startup',
    'srgc_research.dispatch.gemma4_it.cli': 'gemma-it-prepare',
    'srgc_research.dispatch.gemma_run': 'gemma-launcher',
    'srgc_research.dispatch.gemma_text_run': 'gemma-text-launcher',
    'srgc_research.dispatch.llama_run': 'llama-launcher',
    'srgc_research.dispatch.qwen_run': 'qwen-launcher',
}


def role(argv):
    if '-m' not in argv:
        return None
    index = argv.index('-m') + 1
    return MODULES.get(argv[index]) if index < len(argv) else None


def process(path, uid):
    if path.stat().st_uid != uid:
        return None
    raw = (path / 'stat').read_text()
    tail = raw.rsplit(') ', 1)[1].split()
    argv = (path / 'cmdline').read_bytes().decode(errors='replace').split('\0')
    return {'pid': int(path.name), 'ppid': int(tail[1]), 'state': tail[0],
            'start_ticks': tail[19], 'role': role(argv),
            'pip': '-m' in argv and argv[argv.index('-m') + 1:argv.index('-m') + 2] == ['pip']}


def stage(path):
    name = Path(path.removesuffix(' (deleted)')).name
    if name == '.start-prepare.lock' or name.endswith('.auto-prepare.lock'):
        return 'plan 준비'
    if name.endswith('.session.lock'):
        return '실험 session'
    if name.endswith('.download.lock'):
        return '모델 다운로드'
    if name.endswith('.lock') and '/gemma-runtime-cache/packages/' in path:
        return '패키지 준비'
    if name.endswith('.lock') and '/gemma-runtime-cache/verified-models/' in path:
        return '모델 검증'
    return None


def lock_rows(proc):
    rows = {}
    for line in (proc / 'locks').read_text().splitlines():
        parts = line.split()
        try:
            index = next(i for i, value in enumerate(parts) if value in {'READ', 'WRITE'})
            pid = int(parts[index + 1])
            major, minor, inode = parts[index + 2].split(':')
            key = (pid, int(major, 16), int(minor, 16), int(inode))
            rows[key] = 'waiting' if '->' in parts[:index] else 'held'
        except (StopIteration, ValueError, IndexError):
            continue
    return rows


def inspect(proc=Path('/proc'), *, uid=None):
    uid = os.getuid() if uid is None else uid
    processes, errors = {}, []
    for path in proc.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            row = process(path, uid)
            if row:
                processes[row['pid']] = row
        except (OSError, ValueError, IndexError):
            continue
    selected = {pid for pid, row in processes.items() if row['role']}
    # Include only pip descendants of recognized SRGC startup processes.
    while True:
        extra = {pid for pid, row in processes.items()
                 if row['pip'] and row['ppid'] in selected} - selected
        if not extra:
            break
        selected.update(extra)
    try:
        locks = lock_rows(proc)
    except OSError as error:
        locks = {}
        errors.append('kernel lock records unavailable: ' + type(error).__name__)
    result = []
    for pid in sorted(selected):
        row = processes[pid]
        path = proc / str(pid)
        try:
            waiting = (path / 'wchan').read_text().strip()
            files = []
            for fd in (path / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    label = stage(target)
                    if not label:
                        continue
                    info = fd.stat()
                    key = (pid, os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
                    files.append({'path': target, 'stage': label,
                                  'state': locks.get(key, 'open_unconfirmed')})
                except OSError:
                    continue
            after = process(path, uid)
            if after is None or any(row[key] != after[key] for key in ('start_ticks', 'ppid', 'role')):
                continue
        except (OSError, ValueError, IndexError):
            errors.append(f'pid {pid}: process changed or inspection unavailable')
            continue
        observed = [item['stage'] + ' 잠금 대기' for item in files if item['state'] == 'waiting']
        result.append({'pid': pid, 'ppid': row['ppid'], 'role': row['role'] or 'pip-child',
                       'process_state': row['state'], 'wait_channel': waiting,
                       'observed_wait': observed, 'startup_locks': files})
    return {'host': socket.gethostname(), 'scope': 'current user, local host only',
            'read_only': True, 'processes': result, 'errors': errors,
            'note': 'held/open locks do not prove waiting; no process, lock, queue or GPU is changed'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    options = parser.parse_args(argv)
    value = inspect()
    if options.json:
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return
    print(f"SRGC launch probe | {value['host']} | read-only | local processes only")
    for row in value['processes']:
        print(f"pid={row['pid']} ppid={row['ppid']} {row['role']} "
              f"state={row['process_state']} wait={row['wait_channel'] or '-'}")
        for item in row['startup_locks']:
            print(f"  {item['state']}: {item['stage']} | {item['path']}")
    if not value['processes']:
        print('No recognized local SRGC startup process observed; this does not describe other nodes.')
    for error in value['errors']:
        print('Inspection incomplete: ' + error)


if __name__ == '__main__':
    main()
