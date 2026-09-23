"""Stage the reviewed Pair queue runtime without updating a live checkout."""
from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import selectors
import shutil
import signal
import stat
import subprocess
import tarfile
import tempfile
import time


PINNED_COMMIT = 'c0c38d62e893fdcf25920d5a70d3149a06b5450f'
PRE_SRGC_OPERATIONS_COMMIT = '0baf97d5b32e2453a14a7213802d9c3cd570a70a'
OPERATIONS_COMMIT = '6c5e02f70c87b6a5b0253339063881e24012f89b'
PRE_SRGC_OPERATIONS_FILES = ('scripts/queue_selector_pair_gpu.py',
                            'scripts/selector_pair_parallel.py', 'scripts/selector_pair_status.py')
PRE_BUDGET_RECOVERY_COMMITS = ('29903457ccbab9bbed96221794019004ceeb34ad',
                               '428780570cdba64e779408eb4f87af644b9ced3d')
PRE_BUDGET_RECOVERY_FILES = ('scripts/queue_selector_pair_gpu.py',
                    'scripts/selector_pair_parallel.py', 'scripts/selector_pair_status.py',
                    'scripts/selector_pair_srgc.py', 'scripts/selector_pair_srgc_score.py',
                    'scripts/report_selector_pair_srgc.py', 'scripts/selector_pair_results.py',
                    'scripts/selector_pair_diagnostic.py')
OPERATIONS_FILES = (*PRE_BUDGET_RECOVERY_FILES, 'scripts/selector_pair_budget_recovery.py')
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_FILES = 20000
MANIFEST = '.pair-runtime.json'


def operations_files(commit):
    if commit == PRE_SRGC_OPERATIONS_COMMIT:
        return PRE_SRGC_OPERATIONS_FILES
    if commit in PRE_BUDGET_RECOVERY_COMMITS:
        return PRE_BUDGET_RECOVERY_FILES
    return OPERATIONS_FILES


def git(repo, *args, limit=MAX_ARCHIVE_BYTES, timeout=90):
    """Bound stdout/stderr and elapsed time, including a stalled fetch."""
    command = ['git', '-C', str(repo), '-c', 'tar.umask=0022', *args]
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_NO_REPLACE_OBJECTS='1')
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env=env, start_new_session=True) as process:
        output, errors = bytearray(), bytearray()
        deadline = time.monotonic() + timeout
        try:
            with selectors.DefaultSelector() as poll:
                poll.register(process.stdout, selectors.EVENT_READ, (output, limit))
                poll.register(process.stderr, selectors.EVENT_READ, (errors, 65536))
                while poll.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RuntimeError(f'Pair runtime git {args[0]} timed out; no live checkout changed')
                    for key, _ in poll.select(min(remaining, .2)):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            poll.unregister(key.fileobj)
                            continue
                        buffer, cap = key.data
                        if len(buffer) + len(chunk) > cap:
                            raise RuntimeError(f'Pair runtime git {args[0]} exceeded its output limit')
                        buffer.extend(chunk)
            remaining = deadline - time.monotonic()
            code = process.wait(timeout=max(.001, remaining))
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        if code:
            # Do not echo remote URLs, credentials, environment, or arbitrary stderr.
            raise RuntimeError(f'Pair runtime git {args[0]} failed (exit {code}); existing workers unchanged')
        return bytes(output)


def safe_name(name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or '..' in relative.parts
            or str(relative) != name or '\\' in name):
        raise RuntimeError('unsafe path in pinned Pair runtime')
    return relative


def pinned_files(repo, *, commit=None):
    overlay = commit is None and OPERATIONS_COMMIT is not None
    commit = PINNED_COMMIT if commit is None else commit
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise RuntimeError('invalid pinned Pair commit')
    try:
        kind = git(repo, 'cat-file', '-t', commit, limit=1024).strip()
    except RuntimeError:
        print(f'[pair-runtime] fetching reviewed commit {commit}; no checkout update', flush=True)
        git(repo, 'fetch', '--no-tags', '--no-write-fetch-head', 'origin', commit,
            limit=1024 * 1024)
        kind = git(repo, 'cat-file', '-t', commit, limit=1024).strip()
    if kind != b'commit' or git(repo, 'rev-parse', '--verify', commit + '^{commit}',
                              limit=1024).decode().strip() != commit:
        raise RuntimeError('reviewed Pair commit identity does not match')
    roots = git(repo, 'ls-tree', '--name-only', '-z', commit,
                limit=1024 * 1024).decode().rstrip('\0').split('\0')
    if not {'src', 'scripts'}.issubset(roots):
        raise RuntimeError('reviewed Pair runtime is missing source or launcher')
    paths = [name for name in roots if name in {'src', 'scripts', 'vendor'}
             or name.startswith('requirements') and name.endswith('.txt')]
    tree = git(repo, 'ls-tree', '-r', '-l', '-z', commit, '--', *paths,
               limit=4 * 1024 * 1024)
    expected, total = {}, 0
    for row in tree.split(b'\0'):
        if not row:
            continue
        header, raw_name = row.split(b'\t', 1)
        mode, kind, oid, raw_size = header.split()
        name = raw_name.decode('utf-8')
        safe_name(name)
        if kind != b'blob' or mode not in {b'100644', b'100755'}:
            raise RuntimeError('pinned Pair runtime contains a symlink or non-regular object')
        size = int(raw_size)
        if size < 0:
            raise RuntimeError('invalid pinned Pair file size')
        total += size
        expected[name] = (int(mode, 8) & 0o777, size, oid.decode('ascii'))
        if len(expected) > MAX_FILES or total > MAX_ARCHIVE_BYTES:
            raise RuntimeError('pinned Pair runtime exceeds its size limit')
    if not {'src/selector_pair_gpu.py', 'scripts/run_selector_pair.sh'}.issubset(expected):
        raise RuntimeError('reviewed Pair source or launcher is missing')
    archive = git(repo, 'archive', '--format=tar', commit, '--', *paths)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:') as bundle:
        for member in bundle:
            name = member.name.rstrip('/') if member.isdir() else member.name
            safe_name(name)
            if member.isdir():
                continue
            if not member.isfile() or name not in expected or name in files:
                raise RuntimeError('unsafe or unexpected object in Pair archive')
            mode, size, oid = expected[name]
            if member.size != size or member.mode & 0o777 != mode:
                raise RuntimeError('Pair archive differs from pinned Git tree')
            stream = bundle.extractfile(member)
            if stream is None:
                raise RuntimeError('missing Pair archive data')
            data = stream.read(size + 1)
            if len(data) != size:
                raise RuntimeError('truncated Pair archive data')
            blob = b'blob ' + str(size).encode('ascii') + b'\0' + data
            if hashlib.sha1(blob).hexdigest() != oid:
                raise RuntimeError('Pair archive content differs from its pinned Git blob')
            files[name] = (data, mode)
    if files.keys() != expected.keys():
        raise RuntimeError('Pair archive omits pinned runtime files')
    if overlay:
        # Keep the existing learner and fixed-control source pin. The SR-GC
        # decision/scoring extension and its queue helpers are pinned separately.
        operations = pinned_files(repo, commit=OPERATIONS_COMMIT)
        for name in operations_files(OPERATIONS_COMMIT):
            if name not in operations:
                raise RuntimeError('Pair operations overlay is missing a required helper')
            files[name] = operations[name]
    return files


def manifest_for(files, *, commit=None):
    return {'schema': 'selector-pair-isolated-runtime-v1', 'commit': PINNED_COMMIT if commit is None else commit,
            **({'operations_commit': OPERATIONS_COMMIT} if commit is None and OPERATIONS_COMMIT else {}),
            'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'mode': mode, 'size': len(data)}
                      for name, (data, mode) in sorted(files.items())}}


def verify(target, expected):
    if target.is_symlink() or not target.is_dir():
        raise RuntimeError('Pair runtime cache is not a regular directory; refusing replacement')
    seen = set()
    allowed_dirs = {str(parent) for name in expected['files']
                    for parent in PurePosixPath(name).parents if str(parent) != '.'}
    for directory, folders, names in os.walk(target, followlinks=False):
        for name in folders:
            path = Path(directory) / name
            if path.is_symlink() or path.relative_to(target).as_posix() not in allowed_dirs:
                raise RuntimeError('unexpected or symlinked Pair runtime directory; cache preserved')
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(target).as_posix()
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError('non-regular Pair runtime file; cache preserved')
            if relative == MANIFEST:
                if info.st_size > 8 * 1024 * 1024:
                    raise RuntimeError('invalid Pair runtime manifest; cache preserved')
                if json.loads(path.read_text()) != expected:
                    raise RuntimeError('Pair runtime manifest changed; cache preserved')
                seen.add(relative)
                continue
            record = expected['files'].get(relative)
            if (record is None or stat.S_IMODE(info.st_mode) != record['mode']
                    or info.st_size != record['size']):
                raise RuntimeError('Pair runtime contents changed; cache preserved')
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != record['sha256']:
                raise RuntimeError('Pair runtime file hash changed; cache preserved')
            seen.add(relative)
    if seen != set(expected['files']) | {MANIFEST}:
        raise RuntimeError('Pair runtime files missing; cache preserved')


def directory(path):
    if path.is_symlink():
        raise RuntimeError('refusing symlinked Pair runtime cache directory')
    path.mkdir(exist_ok=True)
    if not path.is_dir():
        raise RuntimeError('Pair runtime cache path is not a directory')


def publish(temporary, target):
    """Linux atomic rename with NOREPLACE: never replace even an empty cache."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise RuntimeError('atomic no-replacement publication unavailable; no runtime replaced')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(temporary), -100, os.fsencode(target), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise RuntimeError('Pair runtime appeared during staging; preserved; run command again')
        raise OSError(code, 'could not atomically publish Pair runtime')


def stage_runtime(repo):
    repo = Path(repo).resolve(strict=True)
    files = pinned_files(repo)
    expected = manifest_for(files)
    directory(repo / '.work')
    cache = repo / '.work/pair-runtimes'
    directory(cache)
    runtime_id = PINNED_COMMIT + ('-' + OPERATIONS_COMMIT if OPERATIONS_COMMIT else '')
    target = cache / runtime_id
    lock_path = cache / '.stage.lock'
    if lock_path.is_symlink():
        raise RuntimeError('refusing symlinked Pair runtime staging lock')
    with lock_path.open('a+b') as lock:
        deadline = time.monotonic() + 30
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('another Pair runtime staging operation is still active')
                time.sleep(.1)
        if target.exists() or target.is_symlink():
            verify(target, expected)
            print(f'[pair-runtime] verified existing queue runtime {runtime_id}', flush=True)
            return target
        temporary = Path(tempfile.mkdtemp(prefix='.stage-', dir=cache))
        try:
            for name, (data, mode) in files.items():
                path = temporary / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as handle:
                    handle.write(data)
                path.chmod(mode)
            (temporary / MANIFEST).write_text(json.dumps(expected, sort_keys=True) + '\n')
            verify(temporary, expected)
            publish(temporary, target)
        finally:
            if temporary.exists():
                # Only our unique unpublished staging directory, never a runtime.
                shutil.rmtree(temporary)
    print(f'[pair-runtime] staged reviewed queue runtime {runtime_id}; live checkout unchanged', flush=True)
    return target
