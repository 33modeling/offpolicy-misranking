"""Read-only V8 node snapshot: no Torch import, model load, signals or queue edits."""
import json
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import time


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return None


def worker_status(base,worker):
    key=hashlib.sha256(json.dumps(worker,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    value=read(Path(base)/'.v8-worker-status'/(key+'.json'))
    return value if isinstance(value,dict) and value.get('worker')==worker and isinstance(value.get('at'),(int,float)) else None


def tail(path,limit=32768):
    try:
        with Path(path).open('rb') as f:
            f.seek(0,2);size=f.tell();f.seek(max(0,size-limit));data=f.read(limit)
        lines=data.decode(errors='replace').splitlines()
        return lines[1:] if size>limit else lines
    except OSError:return []


def last_row(path):
    for line in reversed(tail(path)):
        try:
            row=json.loads(line)
            if isinstance(row,dict):return row
        except ValueError:pass
    return None


def root(environment):
    group=Path(environment.get('GROUP_VOLUME','/group-volume'))
    work=Path(environment.get('OM_WORK',str(group/environment.get('OM_USER','minsoo3.kim')/'offpolicy-misranking')))
    return Path(environment.get('V8_ROOT',str(work/'v8')))


def active_roots(proc=Path('/proc')):
    """Read only the V8_ROOT field of this user's existing V8 process environment."""
    found=set()
    try:entries=list(Path(proc).iterdir())
    except OSError:return []
    for entry in entries:
        if not entry.name.isdigit():continue
        try:
            if entry.stat().st_uid!=os.getuid():continue
            args=(entry/'cmdline').read_bytes().split(b'\0')
            if not any(Path(a.decode(errors='replace')).name in ('srgc_v8_nodes.py','cluster_child.py','v8_dispatch.py','v8_runtime_worker.py') for a in args if a):continue
            for field in (entry/'environ').read_bytes().split(b'\0'):
                if field.startswith(b'V8_ROOT='):
                    found.add(field[len(b'V8_ROOT='):].decode(errors='replace'))
        except (OSError,ValueError):continue
    return sorted(Path(p) for p in found if p)


def snapshot(base,host):
    result=[]
    for cohort in ('olmo_math','qwen_mbpp','olmo_mbpp','qwen_math'):
        directory=Path(base)/cohort;queue=directory/'.multi-node';state_path=queue/'state.json'
        raw=read(state_path);errors=[]
        if not isinstance(raw,dict):
            if state_path.exists():errors.append(f'{state_path}: unreadable or malformed queue state')
            state={}
        else:state=raw
        info=dict(root=str(base),cohort=cohort,phase=state.get('phase','unknown' if errors else 'not_started'),
                  failure=state.get('failure'),local_jobs=[],recorded_job_counts={},record_errors=errors)
        for path in sorted((queue/'jobs').glob('*.json')):
            job=read(path)
            if not isinstance(job,dict):errors.append(f'{path}: unreadable or malformed job');continue
            status=job.get('state','unknown');info['recorded_job_counts'][status]=info['recorded_job_counts'].get(status,0)+1
            if job.get('state')!='running' or not job.get('attempts'):continue
            attempt=job['attempts'][-1]
            if attempt.get('worker','').rsplit('.',1)[0]!=host:continue
            task=job.get('task')
            if not isinstance(task,dict):errors.append(f'{path}: missing task definition');continue
            rid=task.get('run_id');run=directory/'runs'/rid if rid else None
            manifest=read(run/'manifest.json') if run else None
            item=dict(run_id=rid,kind=task.get('kind'),method=task.get('method_id'),seed=task.get('training_seed'),
                      log=str(queue/'logs'/(attempt['id']+'.log')),attempt_started=attempt.get('started_at'),
                      worker=attempt.get('worker'),run_manifest=manifest.get('status') if manifest else 'not_created')
            if manifest and manifest.get('attempts'):
                current=manifest['attempts'][-1]['attempt_id']
                row=last_row(run/'attempts'/current/'updates.jsonl')
                item['reported_update_not_checkpoint_verified']=row.get('completed_updates') if row else None
            if run:
                checkpoint=run/'latest.pt'
                if checkpoint.exists():
                    stat=checkpoint.stat();item['latest_file']=dict(bytes=stat.st_size,age_seconds=round(time.time()-stat.st_mtime,1))
            item['last_log_lines']=tail(queue/'logs'/(attempt['id']+'.log'))[-12:]
            info['local_jobs'].append(item)
        result.append(info)
    return result


def main():
    print('V8 READ-ONLY PROBE host='+socket.gethostname(),flush=True)
    try:
        gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=8)
        print(gpu.stdout or gpu.stderr,flush=True)
    except (OSError,subprocess.TimeoutExpired) as exc:print('GPU query unavailable: '+str(exc),flush=True)
    print('Reading current node queue/log files; this does not restart or signal workers.',flush=True)
    rows=[]
    for base in active_roots() or [root(os.environ)]:rows+=snapshot(base,socket.gethostname())
    print(json.dumps(rows,ensure_ascii=False,indent=2),flush=True)
    print('reported_update is a log record, not proof that its checkpoint was saved. No changing record in this one snapshot does not prove a hang.',flush=True)


if __name__=='__main__':main()
