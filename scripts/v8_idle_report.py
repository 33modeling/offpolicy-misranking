"""Read-only V8 idle report: why nodes have no task. Writes ~/v8-idle-report.txt (summary on screen).

Copies every node's last scheduler record (.v8-worker-status/*.json under the V8 root), each cohort's
state.json, job counts by state and phase, and the definition and last attempt of every job that is not done.
Touches no queue, lease, checkpoint or run file."""
import collections
import datetime
import json
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_probe import read, root  # noqa: E402

OUT = Path(os.environ.get('V8_IDLE_REPORT', str(Path.home() / 'v8-idle-report.txt')))
DONE = ('done', 'completed', 'complete')


def when(value):
    try:
        return datetime.datetime.fromtimestamp(float(value), datetime.timezone.utc).strftime('%m-%d %H:%M:%SZ')
    except (TypeError, ValueError, OverflowError, OSError):
        return '?'


def dump(value, limit):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= limit else text[:limit] + ' ...'


def main():
    base = root(os.environ)
    now = time.time()
    report = [f"V8 idle report  written {when(now)}  on {socket.gethostname()}  root {base}", '']
    summary = []
    status_dir = base / '.v8-worker-status'
    rows = []
    for path in sorted(status_dir.glob('*.json')) if status_dir.is_dir() else []:
        row = read(path)
        if isinstance(row, dict):
            rows.append(row)
    rows.sort(key=lambda r: -float(r.get('at') or 0))
    fresh = [r for r in rows if now - float(r.get('at') or 0) < 3600]
    summary.append(f"nodes with a scheduler record in the last hour: {len(fresh)} (all records: {len(rows)})")
    report.append('=' * 100)
    report.append(f"NODE RECORDS ({len(rows)}; newest first)")
    for r in rows:
        age = (now - float(r.get('at') or 0)) / 60
        report.append(f"- {r.get('worker')}  at={when(r.get('at'))} ({age:.0f} min ago)  state={r.get('state')}"
                      f"  cohort={r.get('cohort')}  same_state_since={when(r.get('since'))}  controller={str(r.get('controller_sha256'))[:12]}")
        report.append(f"  reason: {r.get('reason')}")
        if r.get('idle_reason'):
            report.append(f"  idle_reason: {r.get('idle_reason')}")
        extra = {k: v for k, v in r.items() if k not in ('worker', 'at', 'state', 'cohort', 'since', 'reason',
                                                        'idle_reason', 'controller_sha256', 'schema')}
        report.append(f"  details: {dump(extra, 3000)}")
        if age < 60 and r.get('state') != 'running':
            summary.append(f"  {r.get('worker')}: {r.get('state')} {str(r.get('idle_reason') or r.get('reason'))[:200]}")
    try:
        cohorts = sorted(p for p in base.iterdir() if (p / '.multi-node' / 'jobs').is_dir())
    except OSError:
        cohorts = []
    for directory in cohorts:
        queue = directory / '.multi-node'
        state = read(queue / 'state.json')
        jobs = [(p, read(p)) for p in sorted((queue / 'jobs').glob('*.json'))]
        jobs = [(p, j) for p, j in jobs if isinstance(j, dict)]
        counts = collections.Counter((str(j.get('phase')), str(j.get('state'))) for _, j in jobs)
        line = ', '.join(f"{ph}/{st}={n}" for (ph, st), n in sorted(counts.items()))
        failure = (state or {}).get('failure') if isinstance(state, dict) else None
        summary.append(f"{directory.name}: phase={(state or {}).get('phase') if isinstance(state, dict) else '?'} "
                       f"{line}" + (f"  QUEUE FAILURE: {str(failure.get('detail') if isinstance(failure, dict) else failure)[:160]}" if failure else ''))
        report += ['', '=' * 100, f"COHORT {directory.name}", f"job counts (phase/state): {line}"]
        if isinstance(state, dict):
            slim = {k: v for k, v in state.items() if k not in ('actions', 'failure_history', 'task_failures')}
            report.append(f"state.json: {dump(slim, 4000)}")
            report.append(f"last actions: {dump((state.get('actions') or [])[-8:], 3000)}")
        leases = sorted(p.stem for p in (queue / 'leases').glob('*.lock')) if (queue / 'leases').is_dir() else []
        report.append(f"lease files: {len(leases)}")
        for path, job in jobs:
            if str(job.get('state')) in DONE:
                continue
            attempts = [a for a in job.get('attempts') or [] if isinstance(a, dict)]
            last = attempts[-1] if attempts else {}
            slim = {k: v for k, v in job.items() if k not in ('attempts', 'task')}
            report.append(f"- {path.stem}  state={job.get('state')} phase={job.get('phase')} attempts={len(attempts)}")
            report.append(f"  task: {dump(job.get('task'), 1200)}")
            report.append(f"  job fields: {dump(slim, 1500)}")
            if last:
                report.append(f"  last attempt: id={last.get('id')} host={last.get('host')} start={when(last.get('started_at'))}"
                              f" end={when(last.get('ended_at'))} exit={last.get('exit_code')} outcome={last.get('outcome')}"
                              f" error={str(last.get('error'))[:300]}")
    text = '\n'.join(report) + '\n'
    try:
        OUT.write_text(text)
        where = f"Full report: {OUT} ({len(text)} bytes)"
    except OSError as exc:
        where = f"Could not write {OUT}: {exc}"
    print('\n'.join(summary))
    print(where)
    return 0


if __name__ == '__main__':
    sys.exit(main())
