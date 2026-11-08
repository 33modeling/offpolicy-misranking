"""Reuse the existing SRGC cluster environment; no installs or artifact migration."""
import json
import os
from pathlib import Path
import shutil
import sys


def environment(tools,source,*,cwd=None):
    env=dict(source);tools=Path(tools).resolve();cwd=Path(cwd or Path.cwd())
    group=Path(env.get('GROUP_VOLUME','/group-volume'))
    user=env.get('OM_USER','minsoo3.kim')
    work=Path(env.get('OM_WORK',str(group/user/'offpolicy-misranking')))
    env.setdefault('OM_WORK',str(work))
    env.setdefault('V8_ROOT',str(work/'v8'))
    env.setdefault('EXPORT_CODE_ROOT',str(tools))
    env.setdefault('PYTHONDONTWRITEBYTECODE','1')
    explicit=env.get('V8_PYTHON') or env.get('PAIR_PYTHON') or env.get('SWITCH_PYTHON')
    python=explicit or str(Path(env.get('VENV_DIR',str(work/'.venv-cu126')))/'bin/python')
    if not shutil.which(python):
        if explicit:raise ValueError('Existing Python is unavailable: '+python)
        python=shutil.which('python3') or sys.executable
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
        candidates=[tools.parent/'v8/experiments',tools/'v8/experiments',cwd/'v8/experiments',
                    tools.parent/'rlvr-v8/v8/experiments',tools.parent/'offpolicy-misranking-paper-v2/v8/experiments']
        for root in candidates:
            if (root/'cluster_child.py').is_file() and (root/'configs/base.json').is_file():
                env['V8_RUNNER_ROOT']=str(root.resolve());break
    return env


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    tools=Path(__file__).resolve().parents[1]
    env=environment(tools,os.environ)
    inspect=any(arg in ('--status','--help','-h') for arg in argv)
    if not inspect:
        if not env.get('V8_CODE_ROOT'):raise ValueError('Existing trainer checkout was not found in the SRGC environment or source receipts.')
        if not env.get('V8_RUNNER_ROOT') and '--runner-root' not in argv:
            raise ValueError('V8 runtime not found beside the launcher; use the complete offline bundle.')
    os.execvpe(env['V8_PYTHON'],[env['V8_PYTHON'],str(tools/'scripts/srgc_v8_nodes.py'),*argv],env)


if __name__=='__main__':
    try:main()
    except (OSError,ValueError) as exc:
        print('V8 startup: '+str(exc),file=sys.stderr);raise SystemExit(2)
