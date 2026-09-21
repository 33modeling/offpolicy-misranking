"""Recover a live local MBPP controller identity without trusting host names."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def environment(path):
    return dict(item.split('=', 1) for item in path.read_bytes().decode().split('\0')
                if '=' in item)


def start_time(directory):
    fields = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
    return None if fields[0] == 'Z' else int(fields[19])


def gpu_inventory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True, timeout=3)
    rows = [tuple(field.strip() for field in line.split(','))
            for line in result.stdout.splitlines() if line.strip()]
    if not rows or any(len(row) != 2 or not row[0].isdigit()
                       or not row[1].startswith('GPU-') for row in rows):
        raise ValueError('GPU inventory is unavailable')
    return dict(rows)


def same_devices(current, previous, inventory=gpu_inventory):
    # Allocation masks are independent of CUDA's device ordering.
    if current.get('NVIDIA_VISIBLE_DEVICES', '') != previous.get('NVIDIA_VISIBLE_DEVICES', ''):
        return None
    left = tuple(part.strip() for part in current.get('CUDA_VISIBLE_DEVICES', '').split(',') if part.strip())
    right = tuple(part.strip() for part in previous.get('CUDA_VISIBLE_DEVICES', '').split(',') if part.strip())
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        return None
    if set(left) == set(right):
        return True
    try:
        devices = inventory()
        def selected(mask):
            if not mask:
                return set(devices.values())
            return {devices[value] if value in devices else value
                    if value in devices.values() else None for value in mask}
        a, b = selected(left), selected(right)
        if None in a or None in b:
            return None
        return True if a == b else False if a.isdisjoint(b) else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def live_identity(receipt, work, suite, current, *, proc=Path('/proc'), inventory=gpu_inventory,
                  with_pid=False):
    try:
        owner = json.loads(receipt.read_text())
        lock = receipt.with_suffix('').with_suffix('.lock').resolve()
        if (owner.get('schema') != 'mbpp-node-owner-v1' or owner.get('state') != 'active'
                or owner.get('lock') != str(lock) or type(owner.get('pid')) is not int
                or type(owner.get('start_time')) is not int
                or not re.fullmatch('[a-f0-9]{32}', owner.get('token', ''))):
            return None
        directory = proc / str(owner['pid'])
        if directory.stat().st_uid != os.getuid() or start_time(directory) != owner['start_time']:
            return None
        args = (directory / 'cmdline').read_bytes().decode().split('\0')
        if (not any(Path(arg).name == '_mbpp_node_guard.py' for arg in args)
                or '--lock' not in args or args[args.index('--lock') + 1] != str(lock)):
            return None
        if ((directory / 'ns/pid').readlink() != (proc / 'self/ns/pid').readlink()
                or (directory / 'cgroup').read_bytes() != (proc / 'self/cgroup').read_bytes()):
            return None
        previous = environment(directory / 'environ')
        node = previous.get('EXPERIMENTS_NODE_ID', '')
        if (previous.get('OM_WORK') != str(work)
                or previous.get('EXPERIMENTS_MBPP_SUITE') not in {'all', 'fresh', 'quality', 'difficulty', 'long'}
                or not re.fullmatch('[a-zA-Z0-9_.-]+', node)):
            return None
        # A remote receipt can happen to name a local PID/start time. Its random
        # token must also bind the guard's actual local controller child.
        children = (directory / 'task' / str(owner['pid']) / 'children').read_text().split()
        for pid in children:
            child = proc / pid
            try:
                env = environment(child / 'environ')
                command = (child / 'cmdline').read_bytes().decode().split('\0')
                if (child.stat().st_uid == os.getuid() and start_time(child) is not None
                        and any(Path(arg).name == 'run_experiments.sh' for arg in command)
                        and env.get('OM_MBPP_CONTROLLER_TOKEN') == owner['token']
                        and env.get('MBPP_GUARD_PID') == str(owner['pid'])
                        and env.get('OM_WORK') == str(work)
                        and env.get('EXPERIMENTS_NODE_ID') == node
                        and start_time(directory) == owner['start_time']):
                    devices = same_devices(current, previous, inventory)
                    if devices is None:
                        raise RuntimeError('live local MBPP controller found but GPU allocation equivalence '
                                           'is ambiguous; no new controller started')
                    if devices is False:
                        continue
                    previous_suite = previous.get('EXPERIMENTS_MBPP_SUITE')
                    # The default "all" queue contains only quality. The repair
                    # wrapper spells it "quality"; both own the same allocation.
                    if previous_suite != suite and {previous_suite, suite} != {'all', 'quality'}:
                        raise RuntimeError('another MBPP suite already owns these GPUs; '
                                           'existing work preserved')
                    return (node, owner['pid']) if with_pid else node
            except (OSError, ValueError, IndexError, UnicodeError):
                continue
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError, UnicodeError):
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--field', choices=('node', 'pid'), default='node')
    parser.add_argument('--node')
    args = parser.parse_args()
    try:
        found = [value for receipt in args.logs.glob('mbpp-controller.*.owner.json')
                 if (value := live_identity(receipt, args.work, args.suite, os.environ, with_pid=True))]
    except RuntimeError as exc:
        print(f'[blocked] {exc}', file=sys.stderr)
        return 75
    if len(found) > 1:
        print('[blocked] multiple proven local MBPP controllers use these GPUs; '
              'no new controller started and no existing process stopped', file=sys.stderr)
        return 75
    if found and (args.node is None or found[0][0] == args.node):
        print(found[0][0 if args.field == 'node' else 1])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
