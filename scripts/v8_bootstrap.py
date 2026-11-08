"""Reuse the existing SRGC cluster environment; no installs or artifact migration."""
import json
import os
from pathlib import Path
import shutil
import sys


PATH_VARIABLES=('GROUP_VOLUME','OM_USER','OM_WORK','V8_ROOT','VENV_DIR','V8_CODE_ROOT','OM_CODE_ROOT','V8_RUNNER_ROOT',
                'EXPORT_CODE_ROOT','USER_VOLUME','SRGC_STORAGE_ROOT','V8_PYTHON','PAIR_PYTHON','SWITCH_PYTHON','QWEN_PYTHON',
                'V8_PYTHON_OLMO_MATH','V8_PYTHON_QWEN_MBPP')
PYTHON_VARIABLES=('V8_PYTHON','PAIR_PYTHON','SWITCH_PYTHON','QWEN_PYTHON','V8_PYTHON_OLMO_MATH','V8_PYTHON_QWEN_MBPP')


def environment(tools,source,*,cwd=None,warn=None):
    # An exported-but-empty path variable means unset, like the shell launchers' ${VAR:-default}.
    env={k:v for k,v in dict(source).items() if v!='' or k not in PATH_VARIABLES}
    tools=Path(tools).resolve();cwd=Path(cwd or Path.cwd())
    group=Path(env.get('GROUP_VOLUME','/group-volume'))
    user=env.get('OM_USER','minsoo3.kim')
    work=Path(env.get('OM_WORK',str(group/user/'offpolicy-misranking')))
    env.setdefault('OM_WORK',str(work))
    env.setdefault('V8_ROOT',str(work/'v8'))
    env.setdefault('EXPORT_CODE_ROOT',str(tools))
    env.setdefault('PYTHONDONTWRITEBYTECODE','1')
    for name in PYTHON_VARIABLES:
        # An explicitly configured interpreter must exist; it is never silently replaced.
        if env.get(name) and not shutil.which(env[name]):
            raise ValueError(f'Existing Python is unavailable: {name}={env[name]}')
    # Fallback for every cohort = the existing SRGC venv (scripts/run_srgc.sh). Cohort-specific
    # PAIR_/SWITCH_/QWEN_PYTHON are applied per cohort by srgc_v8_nodes.cohort_python.
    python=env.get('V8_PYTHON') or str(Path(env.get('VENV_DIR',str(work/'.venv-cu126')))/'bin/python')
    if not shutil.which(python):
        fallback=shutil.which('python3') or sys.executable
        if warn:warn(f'existing SRGC venv not found at {python}; using {fallback}. Check GROUP_VOLUME/OM_WORK if '
                     'this node normally uses the group-volume .venv-cu126.')
        python=fallback
    env.setdefault('V8_PYTHON',python)
    if not env.get('V8_CODE_ROOT'):
        candidates=[]
        if env.get('OM_CODE_ROOT'):candidates.append(Path(env['OM_CODE_ROOT']))
        for cohort in ('olmo_math','qwen_mbpp'):
            receipt=Path(env['V8_ROOT'])/cohort/'locator.json'
            if receipt.exists():
                saved=json.loads(receipt.read_text()).get('code',{}).get('code_root')
                if saved:candidates.append(Path(saved))
        volume=Path(env.get('USER_VOLUME','/user-volume'))
        candidates += [tools,*list(tools.parents)[:3],cwd,*list(cwd.parents)[:2],
                       tools.parent/'offpolicy-misranking',volume/user/'offpolicy-misranking',
                       volume/'offpolicy-misranking',work]
        for root in candidates:
            if (root/'srgc_rebuttal/srgc.py').is_file():
                env['V8_CODE_ROOT']=str(root.resolve());break
    if not env.get('V8_RUNNER_ROOT'):
        candidates=[tools/'v8/experiments',tools.parent/'v8/experiments',cwd/'v8/experiments',
                    tools.parent/'rlvr-v8/v8/experiments',tools.parent/'offpolicy-misranking-paper-v2/v8/experiments']
        for root in candidates:
            if all((root/name).is_file() for name in ('cluster_child.py','run_h100.sh','switching/cli.py','configs/base.json')):
                env['V8_RUNNER_ROOT']=str(root.resolve());break
    return env


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    tools=Path(__file__).resolve().parents[1]
    env=environment(tools,os.environ,warn=lambda m:print('V8 startup warning: '+m,file=sys.stderr,flush=True))
    inspect=any(arg in ('--status','--help','-h') for arg in argv)
    if not inspect:
        if not env.get('V8_CODE_ROOT'):
            raise ValueError('the original trainer checkout (srgc_rebuttal/srgc.py) was not found next to this launcher '
                             f'({tools}), in its parent folders, under {env.get("USER_VOLUME","/user-volume")} or in a saved '
                             'locator. Unpack the V8 bundle inside the existing offpolicy-misranking checkout.')
        if not env.get('V8_RUNNER_ROOT') and '--runner-root' not in argv:
            raise ValueError(f'V8 runtime files are missing from this code copy ({tools}/v8/experiments); '
                             'unpack the complete V8 bundle into this checkout.')
    os.execvpe(env['V8_PYTHON'],[env['V8_PYTHON'],str(tools/'scripts/srgc_v8_nodes.py'),*argv],env)


if __name__=='__main__':
    try:main()
    except (OSError,ValueError) as exc:
        print('V8 startup: '+str(exc),file=sys.stderr);raise SystemExit(2)
