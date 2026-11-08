"""Resume a queue with its exact recorded source, without editing its binding or results."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runner_identity(root):
    root=Path(root)
    files=sorted([*root.glob('switching/*.py'),*root.glob('configs/**/*.json'),
                  root/'cluster_child.py',root/'run_h100.sh'])
    return {str(p.relative_to(root)):sha(p) for p in files}


def existing_python(binding,env,cohort,explicit=None,identities=None):
    """Find the recorded environment among existing interpreters; never install or relabel packages."""
    identities={} if identities is None else identities
    profile='PAIR_PYTHON' if cohort=='olmo_math' else 'SWITCH_PYTHON'
    work=Path(env.get('OM_WORK',Path(env['V8_ROOT']).parent))
    candidates=[explicit] if explicit else [env.get('V8_PYTHON_'+cohort.upper()),env.get(profile),
        env.get('QWEN_PYTHON') if cohort=='qwen_mbpp' else None,env.get('V8_PYTHON'),
        str(Path(env.get('VENV_DIR',work/'.venv-cu126'))/'bin/python'),sys.executable,shutil.which('python3')]
    expected={k:binding[k] for k in ('python_version','packages') if k in binding}
    for candidate in dict.fromkeys(x for x in candidates if x):
        executable=shutil.which(candidate)
        if not executable:continue
        if not expected:return executable
        if executable not in identities:
            script="""import importlib.metadata,json,sys
packages={}
for name in ('torch','transformers','peft','numpy','math-verify'):
 try:packages[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:packages[name]=None
print(json.dumps({'python_version':sys.version,'packages':packages}))
"""
            try:
                result=subprocess.run([executable,'-c',script],capture_output=True,text=True,timeout=15)
                identities[executable]=json.loads(result.stdout) if result.returncode==0 else {}
            except (OSError,ValueError,subprocess.TimeoutExpired):identities[executable]={}
        if all(identities[executable].get(k)==v for k,v in expected.items()):return executable
    raise ValueError(f'{cohort}: original Python/package versions are not available in the existing environments. '
                     'No package was installed and the queue was not changed.')


def unpack(path):
    """Immutable, private local cache. Never put source cache files in an experiment queue."""
    path=Path(path);key=sha(path)
    base=Path(tempfile.gettempdir())/f'v8-queue-source-{os.getuid()}'
    base.mkdir(mode=0o700,exist_ok=True)
    if base.is_symlink() or base.stat().st_uid!=os.getuid() or base.stat().st_mode&0o077:
        raise ValueError('V8 runtime cache must be a private directory owned by this user')
    root=base/key
    with (base/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with zipfile.ZipFile(path) as archive:
            catalog=json.loads(archive.read('catalog.json'))
            if catalog.get('schema')!=1:raise ValueError('Unsupported V8 source archive schema')
            expected=catalog['files']
            for name in expected:
                pure=PurePosixPath(name)
                if pure.is_absolute() or '..' in pure.parts or str(pure)!=name:
                    raise ValueError('Unsafe V8 source archive path')
            if root.exists():
                if root.is_symlink():raise ValueError('V8 source cache cannot be a symlink')
                for name,digest in expected.items():
                    target=root/name
                    if not target.is_file() or target.is_symlink() or sha(target)!=digest:
                        raise ValueError(f'Cached V8 source bytes changed: {name}')
            else:
                stage=base/(key+'.'+uuid.uuid4().hex+'.tmp');stage.mkdir(mode=0o700)
                try:
                    for name,digest in expected.items():
                        content=archive.read(name)
                        if hashlib.sha256(content).hexdigest()!=digest:raise ValueError(f'Archived V8 source bytes changed: {name}')
                        target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
                    os.replace(stage,root)
                finally:
                    if stage.exists():shutil.rmtree(stage)
    return root,catalog


def selections(tools,env,args):
    """Return original source groups; retain normal all-cohort dispatch when identities match."""
    tools=Path(tools)
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('cohort',nargs='?',default='all')
    for flag in ('root','runner-root','python','gpus','poll-seconds'):parser.add_argument('--'+flag)
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--new-queue',action='store_true')
    options,unknown=parser.parse_known_args(args)
    if unknown:raise ValueError('Unsupported V8 launch arguments: '+' '.join(unknown))
    cohorts=['olmo_math','qwen_mbpp'] if options.cohort=='all' else [options.cohort]
    root=Path(options.root or env['V8_ROOT']).resolve()
    bindings={}
    for cohort in cohorts:
        queue=root/cohort/'.multi-node';path=queue/'binding.json'
        if not path.exists():continue
        old=json.loads(path.read_text());state=json.loads((queue/'state.json').read_text()) if (queue/'state.json').exists() else {}
        # Keep the existing, intentionally narrow source-locator retry; no job has started here.
        if state.get('phase')=='prepare' and state.get('failure') and not list(queue.glob('jobs/*.json')) and not (queue/'input-identity.json').exists():continue
        if old.get('cohort')!=cohort or old.get('root')!=str(root/cohort):
            raise ValueError(f'{cohort}: saved queue root/cohort changed; original={old.get("root")}, current={root/cohort}. Queue untouched.')
        bindings[cohort]=old
    if options.new_queue:bindings={}
    current_script=tools/'scripts/srgc_v8_nodes.py'
    current_runner=Path(options.runner_root or env['V8_RUNNER_ROOT']).resolve()
    current=dict(script=current_script,runner=current_runner,coordinator_sha256=sha(current_script),
                 runner_sha256=runner_identity(current_runner),supports_all=True)
    archived=None;groups={};identities={}
    for cohort in cohorts:
        old=bindings.get(cohort);choice=dict(current)
        if old and (old.get('coordinator_sha256')!=current['coordinator_sha256'] or old.get('runner_sha256')!=current['runner_sha256']):
            path=tools/'v8/queue-runtimes.zip'
            if not path.is_file():raise ValueError(f'{cohort}: historical V8 sources missing from bundle; need coordinator {old.get("coordinator_sha256")}. Queue untouched.')
            if archived is None:archived=unpack(path)
            cache,catalog=archived
            coordinator=next((x for x in catalog['coordinators'] if x['sha256']==old.get('coordinator_sha256')),None)
            runner=next((x for x in catalog['runners'] if x['runner_sha256']==old.get('runner_sha256')),None)
            if old.get('coordinator_sha256')==current['coordinator_sha256']:coordinator=dict(file=str(current_script),supports_all=True)
            if old.get('runner_sha256')==current['runner_sha256']:runner=dict(root=str(current_runner))
            if coordinator is None or runner is None:
                raise ValueError(f'{cohort}: queue source is unavailable (coordinator={old.get("coordinator_sha256")}, runner files={len(old.get("runner_sha256",{}))}). Queue untouched; no identity bypass.')
            choice=dict(script=cache/coordinator['file'],runner=cache/runner['root'],
                        coordinator_sha256=old['coordinator_sha256'],runner_sha256=old['runner_sha256'],
                        supports_all=coordinator['supports_all'])
            if sha(choice['script'])!=old['coordinator_sha256'] or runner_identity(choice['runner'])!=old['runner_sha256']:
                raise ValueError(f'{cohort}: archived source does not match the saved queue. Queue untouched.')
        python=existing_python(old or {},env,cohort,options.python,identities)
        key=(str(choice['script']),str(choice['runner']),None if choice['supports_all'] else cohort)
        group=groups.setdefault(key,dict(choice,cohorts=[],pythons={}))
        group['cohorts'].append(cohort);group['pythons'][cohort]=python
    return list(groups.values())
