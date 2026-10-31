"""Real startup registration outside immutable experiment/queue directories."""

import hashlib
import json
import math
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from srgc_rebuttal.runtime import atomic_json

CURRENT = ContextVar('srgc_startup', default=None)
LABELS = {'starting': 'worker 초기 등록', 'model_files': '모델 파일 확인·다운로드',
          'packages': '고정 패키지 활성화', 'model_verify': '모델 snapshot 검증',
          'plans_validate': '기존 plan·입력 검증', 'plans_prepare': '누락 plan 준비',
          'prepare_busy': '공용 plan 생성 중 · 다음 재시도 대기',
          'queue': '작업 큐 등록 준비'}


def directory(root):
    root = Path(root).resolve()
    return root.parent / ('.' + root.name + '.startup-workers')


class Registration:
    def __init__(self, root):
        identity = uuid.uuid4().hex
        allocation = socket.gethostname() + ':' + os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3')
        key = hashlib.sha256(allocation.encode()).hexdigest()
        self.path = directory(root) / (key + '.json')
        self.row = {'worker_id': identity, 'host': socket.gethostname(), 'pid': os.getpid(),
                    'root': str(Path(root).resolve()), 'model': 'gemma4_it',
                    'started': time.time(), 'status': 'preparing', 'stage': 'starting'}
        self.stop = threading.Event()
        self.lock = threading.Lock()

    def update(self, **values):
        with self.lock:
            self.row.update(values)
            atomic_json(self.path, {**self.row, 'heartbeat': time.time()})

    def watch(self):
        while not self.stop.wait(5):
            self.update()


def progress(stage):
    value = CURRENT.get()
    if value is not None:
        value.update(stage=stage)


def worker_registered():
    value = CURRENT.get()
    if value is not None:
        value.update(status='registered')
        value.stop.set()


@contextmanager
def registration(root):
    value = Registration(root)
    value.update()
    token = CURRENT.set(value)
    thread = threading.Thread(target=value.watch, name='srgc-startup', daemon=True)
    thread.start()
    try:
        yield
    except BaseException:
        value.update(status='ended')
        raise
    finally:
        value.stop.set()
        thread.join(timeout=6)
        value.update(status='ended')
        CURRENT.reset(token)


def wrap_entry(original, root):
    def launch(args):
        from unittest.mock import patch

        from .gemma4_it import model, runtime

        def step(function, name):
            def call(*args, **kwargs):
                progress(name)
                result = function(*args, **kwargs)
                progress('queue')
                return result
            return call

        with registration(root), \
                patch.object(model, 'ensure_snapshot', step(model.ensure_snapshot, 'model_files')), \
                patch.object(runtime, 'activate', step(runtime.activate, 'packages')):
            return original(args)
    return launch


def snapshot(root, *, now=None):
    now = time.time() if now is None else now
    rows = []
    for path in directory(root).glob('*.json'):
        try:
            row = json.loads(path.read_text())
            if (row.get('status') != 'preparing' or row.get('model') != 'gemma4_it'
                    or not isinstance(row.get('worker_id'), str) or not row['worker_id']
                    or row.get('root') != str(Path(root).resolve())
                    or row.get('stage') not in LABELS
                    or not isinstance(row.get('host'), str)
                    or type(row.get('pid')) is not int or row['pid'] <= 0):
                continue
            if any(type(row.get(name)) not in (int, float) or not math.isfinite(row[name])
                   for name in ('heartbeat', 'started')):
                continue
            age = now - row['heartbeat']
            if not 0 <= age <= 30:
                continue
            rows.append({**row, 'heartbeat_age_seconds': age,
                         'status': 'preparing', 'idle_reason': LABELS[row['stage']]})
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return rows
