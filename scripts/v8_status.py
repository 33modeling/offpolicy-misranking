#!/usr/bin/env python3
"""V8 run status in plain Korean (read-only: no locks, signals, Torch or queue edits).

    sh scripts/v8_status.sh            # one snapshot
    sh scripts/v8_status.sh --watch    # refresh every 60 s until Ctrl-C (--watch 30 for 30 s)

Shows, per cohort: phase, task counts, every running task (node, elapsed, update n/N,
seconds since its log last changed, last log line), failures, and this node's GPU use.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_probe import active_roots, last_row, read, root, tail  # noqa: E402

PHASES = {'not_started': '시작 전', 'prepare': '준비', 'train': '학습', 'freeze': '예산 고정',
          'extension': '예산 연장', 'evaluation': '평가', 'finalize': '집계', 'complete': '완료'}
STATES = (('done', '완료'), ('running', '진행'), ('pending', '대기'), ('failed', '실패'), ('interrupted', '중단'))
COHORTS = ('olmo_math', 'qwen_mbpp')


def ago(seconds):
    if seconds is None:
        return '?'
    seconds = int(max(0, seconds))
    if seconds < 90:
        return f'{seconds}초'
    if seconds < 5400:
        return f'{seconds // 60}분'
    return f'{seconds // 3600}시간 {seconds % 3600 // 60}분'


def gpu_rows(run=subprocess.run):
    try:
        out = run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                   '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) == 3 and parts[1].isdigit():
            rows.append((parts[0], int(parts[1]), int(parts[2]) if parts[2].isdigit() else 0))
    return rows


def task_label(task):
    stage = task.get('stage') or task.get('kind') or 'task'
    bits = [str(stage)]
    if task.get('method_id'):
        bits.append(str(task['method_id']))
    if task.get('training_seed') is not None:
        bits.append(f"seed {task['training_seed']}")
    return ' '.join(bits)


def running_detail(directory, queue, job, now):
    task, attempt = job['task'], job['attempts'][-1]
    worker = str(attempt.get('worker') or '?')
    item = dict(label=task_label(task), node=worker.split('.')[0], started=attempt.get('started_at'))
    item['elapsed'] = now - item['started'] if item['started'] else None
    rid = task.get('run_id')
    total = task.get('horizon') or task.get('endpoint') or task.get('total_updates')
    done = None
    if rid:
        manifest = read(directory / 'runs' / rid / 'manifest.json')
        if manifest and manifest.get('attempts'):
            row = last_row(directory / 'runs' / rid / 'attempts' / manifest['attempts'][-1]['attempt_id'] / 'updates.jsonl')
            done = row.get('completed_updates') if row else None
    item['update'] = f"{done if done is not None else '-'}/{total if total else '?'}"
    log = queue / 'logs' / (str(attempt.get('id')) + '.log')
    try:
        item['silent'] = now - log.stat().st_mtime
    except OSError:
        item['silent'] = None
    lines = [x for x in tail(log, 16384) if x.strip()]
    item['last'] = lines[-1].strip()[:160] if lines else '(아직 출력 없음)'
    return item


def cohort_report(base, cohort, now, gpus_busy):
    directory = Path(base) / cohort
    queue = directory / '.multi-node'
    state = read(queue / 'state.json') or {}
    phase = state.get('phase', 'not_started')
    counts = {key: 0 for key, _ in STATES}
    running, failed = [], []
    for path in sorted((queue / 'jobs').glob('*.json')):
        job = read(path)
        if not job:
            continue
        counts[job.get('state')] = counts.get(job.get('state'), 0) + 1
        if job.get('state') == 'running' and job.get('attempts'):
            running.append(running_detail(directory, queue, job, now))
        elif job.get('state') == 'failed':
            last = (job.get('attempts') or [{}])[-1]
            failed.append((task_label(job['task']), str(last.get('error') or last.get('exit') or job.get('error') or '')[:160]))
    lines = [f"━ {cohort}  단계: {PHASES.get(phase, phase)}  "
             + ' · '.join(f'{name} {counts.get(key, 0)}' for key, name in STATES)]
    if state.get('failure'):
        lines.append(f"  ⚠ 큐 실패: {str(state['failure'])[:200]}")
    if running:
        lines.append('  진행 중')
        for item in sorted(running, key=lambda x: x['label']):
            warn = ''
            if item['silent'] is not None and item['silent'] > 1800 and gpus_busy is False:
                warn = '  ⚠ 30분 넘게 출력 없음 + 이 노드 GPU 쉬는 중 → 멈춤 의심 (.multi-node/stacks 확인)'
            lines.append(f"   {item['label']:<34} {item['node']:<16} {ago(item['elapsed'])}째  update {item['update']:<9}"
                         f" 출력 {ago(item['silent'])} 전{warn}")
            lines.append(f"     └ {item['last']}")
    for label, reason in failed:
        lines.append(f"  실패  {label}: {reason}")
    if phase == 'not_started' and not (queue / 'jobs').exists():
        lines.append('  (이 경로에 큐가 아직 없어요)')
    return lines


def render(bases, now=None, gpus=None):
    now = time.time() if now is None else now
    gpus = gpu_rows() if gpus is None else gpus
    busy = None if gpus is None else any(util >= 10 for _, util, _ in gpus)
    out = [f"V8 현황 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}  host={socket.gethostname()}"]
    for base in bases:
        out.append(f"V8_ROOT={base}")
        for cohort in COHORTS:
            out += cohort_report(base, cohort, now, busy)
    if gpus is None:
        out.append('이 노드 GPU: 확인 불가 (nvidia-smi 없음)')
    elif gpus:
        out.append('이 노드 GPU: ' + ' | '.join(f'{i}: {u}% {m // 1024}G' for i, u, m in gpus))
    return '\n'.join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--watch', nargs='?', const=60.0, type=float, default=None, metavar='초',
                        help='새로 고침 간격(기본 60초). Ctrl-C 로 끝냄')
    args = parser.parse_args(argv)
    bases = active_roots() or [root(os.environ)]
    if args.watch is None:
        print(render(bases), flush=True)
        return 0
    try:
        while True:
            sys.stdout.write('\033[H\033[2J' + render(bases) + f"\n\n{int(args.watch)}초마다 새로 고침 · Ctrl-C 로 끝내기\n")
            sys.stdout.flush()
            time.sleep(max(5.0, args.watch))
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    sys.exit(main())
