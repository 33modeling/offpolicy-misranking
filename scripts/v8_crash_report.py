"""Read-only V8 native-crash report: every stopped task that ended with a native signal (SIGSEGV etc.),
with each attempt's own evidence, written to ~/v8-crash-report.txt (a short summary goes to the screen).

Touches no queue, lease, checkpoint or run file. Per attempt it copies: where/when it ran, exit code and
signal, the recorded stage, crash class/frames/log lines and environment from the job record, every rank's
last progress record, fault dumps, runtime-repair and CUDA-failure receipts, and the head and tail of the
attempt log. It ends with this node's kernel segfault lines when the kernel log is readable."""
import datetime
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_probe import read, root  # noqa: E402

NATIVE = {4: 'SIGILL', 6: 'SIGABRT', 7: 'SIGBUS', 8: 'SIGFPE', 11: 'SIGSEGV'}
FULL_ATTEMPTS = 3        # newest attempts of a task shown in full; older ones get one line each
LOG_HEAD, LOG_TAIL = 15, 80
OUT = Path(os.environ.get('V8_CRASH_REPORT', str(Path.home() / 'v8-crash-report.txt')))


def when(value):
    try:
        return datetime.datetime.fromtimestamp(float(value), datetime.timezone.utc).strftime('%m-%d %H:%M:%SZ')
    except (TypeError, ValueError, OverflowError, OSError):
        return '?'


def native_signal(attempt):
    rc = attempt.get('exit_code')
    if type(rc) is int and rc < 0 and -rc in NATIVE:
        return -rc
    if type(rc) is int and rc - 128 in NATIVE:
        return rc - 128
    for f in (attempt.get('child_failure') or {}).get('failures') or []:
        if isinstance(f, dict) and f.get('signal') in NATIVE:
            return f['signal']
    return None


def lines_of(path, head=None, tail=None, size=262144):
    try:
        with Path(path).open('rb') as handle:
            end = handle.seek(0, 2)
            handle.seek(0)
            first = handle.read(min(end, 65536)).decode(errors='replace').splitlines()
            handle.seek(max(0, end - size))
            last = handle.read().decode(errors='replace').splitlines()
    except OSError as exc:
        return None, f'(unreadable: {type(exc).__name__})'
    out = []
    if head:
        out += first[:head]
    if tail:
        if head and end <= 65536 and len(first) <= head + tail:
            out = first
        else:
            out += ['...'] + last[-tail:]
    return end, out


def short(value, limit=400):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= limit else text[:limit] + ' ...'


def attempt_block(base, attempt, full):
    aid = str(attempt.get('id') or '?')
    worker = str(attempt.get('worker') or '')
    host = attempt.get('host') or (worker.rsplit('.', 1)[0] if worker else '?')
    gpus = ','.join(str(g) for g in attempt.get('gpus') or []) or '?'
    sig = native_signal(attempt)
    start, end = attempt.get('started_at'), attempt.get('ended_at')
    try:
        took = f"{(float(end) - float(start)) / 60:.1f} min"
    except (TypeError, ValueError):
        took = '?'
    head = (f"attempt {aid}  host={host} gpus={gpus}  start={when(start)} ran={took}  exit={attempt.get('exit_code')}"
            f" signal={NATIVE.get(sig, sig)}  stage={attempt.get('stage') or 'unrecorded'}"
            f" class={attempt.get('crash_class') or '-'}")
    if not full:
        return [head]
    out = [head]
    for key in ('crash_frames', 'crash_log_lines', 'error', 'node_error', 'crash_evidence'):
        if attempt.get(key):
            out.append(f"  {key}: {short(attempt[key], 1200)}")
    env = attempt.get('env')
    if env:
        out.append(f"  env: {short(env, 1500)}")
    if attempt.get('preflight'):
        out.append(f"  preflight: {short(attempt['preflight'], 800)}")
    for f in (attempt.get('child_failure') or {}).get('failures') or []:
        if isinstance(f, dict):
            info = {k: f.get(k) for k in ('rank', 'pid', 'signal', 'exitcode', 'message', 'last_progress', 'error_file',
                                          'native_fault_log') if f.get(k) is not None}
            out.append(f"  rank failure: {short(info, 900)}")
            if f.get('native_python_stack'):
                out.append('  rank stack:')
                out += ['    ' + x for x in str(f['native_python_stack']).splitlines()[:40]]
    progress = base / 'progress' / aid
    files = sorted(progress.glob('*')) if progress.is_dir() else []
    out.append(f"  progress dir: {len(files)} files" + ('' if progress.is_dir() else ' (missing)'))
    for path in files:
        if path.suffix == '.json':
            out.append(f"  {path.name}: {short(read(path), 700)}")
        elif path.suffix in ('.log', '.txt'):
            size, text = lines_of(path, head=5, tail=40, size=65536)
            out.append(f"  {path.name} ({size} bytes):")
            out += ['    ' + x for x in (text if isinstance(text, list) else [text])]
    log = base / 'logs' / (aid + '.log')
    size, text = lines_of(log, head=LOG_HEAD, tail=LOG_TAIL)
    out.append(f"  log {log.name} ({size} bytes):")
    out += ['    ' + x[:400] for x in (text if isinstance(text, list) else [text])]
    return out


def kernel_lines():
    for command in (['dmesg', '-T'], ['journalctl', '-k', '--no-pager', '-n', '4000']):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0 and result.stdout:
            hits = [x for x in result.stdout.splitlines() if re.search(r'segfault|general protection|Xid|traps:', x)]
            return f"{' '.join(command)}: {len(hits)} matching lines", hits[-40:]
    return 'kernel log not readable on this node (dmesg/journalctl refused)', []


def cohort_dirs(base):
    try:
        return sorted(p for p in base.iterdir() if (p / '.multi-node' / 'jobs').is_dir())
    except OSError:
        return []


def main():
    base = root(os.environ)
    report, summary = [], []
    report.append(f"V8 native crash report  written {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%SZ}"
                  f"  on {socket.gethostname()}  root {base}")
    cohorts = cohort_dirs(base)
    if not cohorts:
        summary.append(f"No V8 cohort queue found under {base}.")
    for directory in cohorts:
        queue = directory / '.multi-node'
        for path in sorted((queue / 'jobs').glob('*.json')):
            job = read(path)
            if not isinstance(job, dict):
                continue
            attempts = [a for a in job.get('attempts') or [] if isinstance(a, dict)]
            crashed = [a for a in attempts if native_signal(a)]
            if not crashed or job.get('state') in ('completed', 'done', 'complete'):
                continue
            task = job.get('task') if isinstance(job.get('task'), dict) else {}
            name = str(task.get('task_id') or task.get('run_id') or path.stem)
            stages = sorted({str(a.get('stage') or 'unrecorded') for a in crashed})
            hosts = sorted({str(a.get('host') or str(a.get('worker') or '?').rsplit('.', 1)[0]) for a in crashed})
            times = []
            for a in crashed:
                try:
                    times.append((float(a['ended_at']) - float(a['started_at'])) / 60)
                except (KeyError, TypeError, ValueError):
                    pass
            ran = f"{min(times):.1f}-{max(times):.1f} min" if times else '?'
            summary.append(f"{directory.name} {name}: {len(crashed)}/{len(attempts)} attempts native crash on "
                           f"{len(hosts)} hosts, ran {ran} before dying, stage {', '.join(stages)}")
            report += ['', '=' * 100, f"{directory.name}  {name}  state={job.get('state')}  attempts={len(attempts)}",
                       f"task: {short(task, 1500)}"]
            rid = task.get('run_id')
            if rid:
                manifest = read(directory / 'runs' / rid / 'manifest.json')
                if isinstance(manifest, dict):
                    report.append(f"run manifest attempts: {short(manifest.get('attempts'), 1500)}")
            for index, attempt in enumerate(attempts):
                report += attempt_block(queue, attempt, full=index >= len(attempts) - FULL_ATTEMPTS)
    note, hits = kernel_lines()
    report += ['', '=' * 100, f"this node ({socket.gethostname()}) kernel: {note}"] + hits
    text = '\n'.join(report) + '\n'
    try:
        OUT.write_text(text)
        where = f"Full report: {OUT} ({len(text)} bytes)"
    except OSError as exc:
        where = f"Could not write {OUT}: {exc}"
    print('\n'.join(summary or ['No stopped task ended with a native crash.']))
    print(f"Kernel log here: {note}")
    print(where)
    return 0


if __name__ == '__main__':
    sys.exit(main())
