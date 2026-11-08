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


class SelectionPlan(list):
    def __init__(self,groups,errors):
        super().__init__(groups);self.errors=errors


def private_cache():
    base=Path(tempfile.gettempdir())/f'v8-queue-source-{os.getuid()}'
    base.mkdir(mode=0o700,exist_ok=True)
    if base.is_symlink() or base.stat().st_uid!=os.getuid() or base.stat().st_mode&0o077:
        raise ValueError('V8 runtime cache must be a private directory owned by this user')
    return base


def existing_python(binding,env,cohort,explicit=None,identities=None):
    """Find the recorded environment among existing interpreters; never install or relabel packages."""
    identities={} if identities is None else identities
    profile='PAIR_PYTHON' if cohort=='olmo_math' else 'SWITCH_PYTHON'
    work=Path(env.get('OM_WORK',Path(env['V8_ROOT']).parent))
    requested=explicit or env.get('V8_PYTHON_'+cohort.upper()) or env.get(profile) or (env.get('QWEN_PYTHON') if cohort=='qwen_mbpp' else None)
    if requested and not shutil.which(requested):
        raise ValueError(f'{cohort}: configured Python is unavailable: {requested}; no replacement was installed')
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
    base=private_cache()
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


def freeze(choice,tools):
    """Copy selected production bytes before exec so a later git pull cannot alter child launches."""
    runner=Path(choice['runner']);payload={'coordinator/srgc_v8_nodes.py':Path(choice['script']).read_bytes()}
    for name in choice['runner_sha256']:
        payload['source/v8/experiments/'+name]=(runner/name).read_bytes()
    if hashlib.sha256(payload['coordinator/srgc_v8_nodes.py']).hexdigest()!=choice['coordinator_sha256']:
        raise ValueError('Selected coordinator changed while freezing; retry after checkout update completes')
    for name,digest in choice['runner_sha256'].items():
        if hashlib.sha256(payload['source/v8/experiments/'+name]).hexdigest()!=digest:
            raise ValueError(f'Selected runner changed while freezing: {name}; queue untouched')
    parent=runner.parents[1]
    helper=parent/'scripts/srgc_saved_runtime.py'
    if not helper.is_file():helper=Path(tools)/'scripts/srgc_saved_runtime.py'
    if helper.is_file():payload['source/scripts/srgc_saved_runtime.py']=helper.read_bytes()
    for path in sorted((parent/'v7/evidence').rglob('*.csv')):
        payload['source/'+str(path.relative_to(parent))]=path.read_bytes()
    hashes={p:hashlib.sha256(data).hexdigest() for p,data in sorted(payload.items())}
    metadata=dict(coordinator_sha256=choice['coordinator_sha256'],runner_sha256=choice['runner_sha256'],
                  supports_all=choice['supports_all'],script='coordinator/srgc_v8_nodes.py',runner='source/v8/experiments')
    key=hashlib.sha256(json.dumps(dict(snapshot_schema=2,files=hashes,source=metadata),sort_keys=True).encode()).hexdigest()
    base=private_cache();archive=base/('selected-'+key+'.zip')
    with (base/('selected-'+key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not archive.exists():
            temporary=base/('selected-'+key+'.'+uuid.uuid4().hex+'.tmp')
            try:
                with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as output:
                    output.writestr('catalog.json',json.dumps(dict(schema=1,files=hashes,source=metadata)))
                    for name,data in payload.items():output.writestr(name,data)
                os.replace(temporary,archive)
            finally:
                if temporary.exists():temporary.unlink()
    cache,_=unpack(archive)
    return dict(choice,script=cache/'coordinator/srgc_v8_nodes.py',runner=cache/'source/v8/experiments')


def cached_source(binding):
    """Previously selected bytes remain usable after a pull, even before a new history zip is built."""
    for path in sorted(private_cache().glob('selected-*.zip')):
        try:
            with zipfile.ZipFile(path) as archive:meta=json.loads(archive.read('catalog.json')).get('source',{})
        except (OSError,ValueError,KeyError,zipfile.BadZipFile):continue
        if (meta.get('coordinator_sha256'),meta.get('runner_sha256'))!=(binding.get('coordinator_sha256'),binding.get('runner_sha256')):continue
        cache,_=unpack(path);script=cache/meta['script'];runner=cache/meta['runner']
        if sha(script)!=binding['coordinator_sha256'] or runner_identity(runner)!=binding['runner_sha256']:
            raise ValueError('Previous local source snapshot differs from its recorded identity')
        return dict(script=script,runner=runner,coordinator_sha256=binding['coordinator_sha256'],
                    runner_sha256=binding['runner_sha256'],supports_all=meta['supports_all'])
    return None


def read_binding(root,cohort):
    queue=root/cohort/'.multi-node';path=queue/'binding.json'
    if not path.exists():return None
    old=json.loads(path.read_text())
    state=json.loads((queue/'state.json').read_text()) if (queue/'state.json').exists() else {}
    if not isinstance(old,dict) or not isinstance(state,dict):raise ValueError(f'{cohort}: malformed queue metadata; queue untouched')
    if state.get('phase')=='prepare' and state.get('failure') and not list(queue.glob('jobs/*.json')) and not (queue/'input-identity.json').exists():return None
    if old.get('cohort')!=cohort or old.get('root')!=str(root/cohort):
        raise ValueError(f'{cohort}: saved queue root/cohort changed; original={old.get("root")}, current={root/cohort}. Queue untouched.')
    if not isinstance(old.get('coordinator_sha256'),str) or not isinstance(old.get('runner_sha256'),dict):
        raise ValueError(f'{cohort}: malformed queue source identity; queue untouched')
    return old


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
    bindings={};errors={}
    for cohort in cohorts:
        try:bindings[cohort]=read_binding(root,cohort)
        except (OSError,ValueError) as exc:errors[cohort]=str(exc)
    if options.new_queue:bindings={}
    current_script=tools/'scripts/srgc_v8_nodes.py'
    current_runner=Path(options.runner_root or env['V8_RUNNER_ROOT']).resolve()
    try:current_hash=sha(current_script)
    except OSError:current_hash=None
    try:current_runner_hash=runner_identity(current_runner)
    except OSError:current_runner_hash=None
    current=dict(script=current_script,runner=current_runner,coordinator_sha256=current_hash,
                 runner_sha256=current_runner_hash,supports_all=True)
    archived=None;groups={};identities={}
    for cohort in cohorts:
        if cohort in errors:continue
        try:
            old=bindings.get(cohort);choice=dict(current)
            different=old and (old.get('coordinator_sha256')!=current['coordinator_sha256'] or old.get('runner_sha256')!=current['runner_sha256'])
            local=cached_source(old) if different else None
            if local:choice=local
            elif different:
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
            if not choice['coordinator_sha256'] or not choice['runner_sha256']:raise ValueError(f'{cohort}: current runtime source missing; queue untouched')
            python=existing_python(old or {},env,cohort,options.python,identities)
            choice=freeze(choice,tools)
            key=(str(choice['script']),str(choice['runner']),python,None if choice['supports_all'] else cohort)
            group=groups.setdefault(key,dict(choice,cohorts=[],pythons={}))
            group['cohorts'].append(cohort);group['pythons'][cohort]=python
        except (OSError,ValueError,zipfile.BadZipFile) as exc:errors[cohort]=str(exc)
    if not groups:raise ValueError('; '.join(f'{c}: {e}' for c,e in errors.items()))
    return SelectionPlan(groups.values(),errors)
