"""Frozen MBPP verifier parity and thread-safe launch controls; CPU, no CUDA inference."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import types

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(params=['srgc_rebuttal/verifiers.py','scripts/frozen_srgc/pre_20261004/verifiers.py'])
def verifier(request,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('srgc_rebuttal.mbpp_spawn_fixture',ROOT/request.param)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import v8_task_entry
    fixed,receipt=v8_task_entry.repair_mbpp_subprocess(module.code_reward)
    assert fixed is not module.code_reward and receipt['reward_rule_changed'] is False
    assert receipt['code_verifier_version']==module.CODE_REWARD_VERSION
    return module,fixed


@pytest.mark.parametrize('response,expected',[
    ('def add(a,b): return a+b',1.0),
    ('def add(a,b): return a-b',0.0),
    ('def add(a,b): raise ValueError("bad candidate")',0.0),
    ('import sys; sys.exit(0)',0.0),
    ('not valid python !',0.0),
    ('import resource,ctypes\nresource.setrlimit(resource.RLIMIT_CORE,(0,0))\nctypes.string_at(0)',0.0),
])
def test_original_current_and_frozen_rewards_are_unchanged(verifier,response,expected):
    module,fixed=verifier;record=dict(answer='assert add(2,3)==5')
    assert module.code_reward(record,response)==fixed(record,response)==expected


def test_limits_are_applied_and_python_preexec_is_removed(verifier,monkeypatch):
    module,fixed=verifier;original=module.subprocess.Popen;calls=[]
    def checked(command,*a,**kwargs):
        assert kwargs.get('preexec_fn') is None
        assert command[:4]==[sys.executable,'-I','-B','-c']
        assert kwargs['start_new_session'] is True
        calls.append(command);return original(command,*a,**kwargs)
    monkeypatch.setattr(module.subprocess,'Popen',checked)
    limits=(module.CODE_MEMORY_BYTES,int(module.CODE_TIMEOUT_SECONDS)+1,1<<20)
    record=dict(answer=f'assert limits()=={limits!r}')
    code='def limits():\n import resource\n return tuple(resource.getrlimit(k)[0] for k in (resource.RLIMIT_AS,resource.RLIMIT_CPU,resource.RLIMIT_FSIZE))'
    assert fixed(record,code)==1.0 and len(calls)==1


def test_checker_keeps_original_main_module_loader_and_argv(verifier):
    module,fixed=verifier
    record=dict(answer='assert context()==("code_check.py", "__main__", None, None, "__main__", 1, True)')
    code='''def context():
 import __main__,os,sys
 return (os.path.basename(__main__.__file__),__main__.__name__,__main__.__spec__,
         __main__.__package__,__main__.__loader__.name,sys.flags.isolated,sys.dont_write_bytecode)
'''
    assert module.code_reward(record,code)==fixed(record,code)==1.0


def test_bad_candidate_timeout_remains_zero_reward(verifier,monkeypatch):
    module,_=verifier
    monkeypatch.setattr(module,'CODE_TIMEOUT_SECONDS',0.2)
    import v8_task_entry
    fixed,_=v8_task_entry.repair_mbpp_subprocess(module.code_reward)
    record=dict(answer='assert add(2,3)==5');code='while True: pass'
    assert module.code_reward(record,code)==fixed(record,code)==0.0


def test_threaded_mbpp_grading_does_not_run_python_between_fork_and_exec(verifier,monkeypatch):
    module,fixed=verifier;original=module.subprocess.Popen;calls=[]
    def checked(command,*a,**kwargs):
        assert kwargs.get('preexec_fn') is None
        calls.append(True);return original(command,*a,**kwargs)
    monkeypatch.setattr(module.subprocess,'Popen',checked)
    def grade(i):return fixed(dict(answer='assert add(2,3)==5'),'def add(a,b): return a+b')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(grade,range(24)))==[1.0]*24
    assert len(calls)==24


def test_runtime_context_installs_spawn_repair_without_changing_scoring_identity(verifier,tmp_path,monkeypatch):
    module,_=verifier
    import v8_task_entry
    backend=types.SimpleNamespace(verifier=module.code_reward)
    identity=dict(code_verifier_version=module.CODE_REWARD_VERSION)
    ctx=types.SimpleNamespace(backend=backend,verifier_identity=identity)
    monkeypatch.setenv('SRGC_PROGRESS_DIR',str(tmp_path));monkeypatch.setenv('RANK','0')
    v8_task_entry.repair_context(ctx)
    assert backend.verifier is not module.code_reward and ctx.verifier_identity is identity
    assert backend.verifier(dict(answer='assert add(2,3)==5'),'def add(a,b): return a+b')==1.0
    report=json.loads((tmp_path/'runtime-repair-rank-0.json').read_text())
    assert report['reward_runtime_changed'] is False
    assert report['repairs']['mbpp_subprocess']['code_verifier_version']==module.CODE_REWARD_VERSION


@pytest.mark.parametrize('filename',['srgc_rebuttal/verifiers.py','scripts/frozen_srgc/pre_20261004/verifiers.py'])
def test_limit_setup_failure_stays_an_infrastructure_error_not_zero_reward(filename):
    script='''import importlib.util,resource,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root));sys.path.insert(0,str(root/'scripts'))
s=importlib.util.spec_from_file_location('srgc_rebuttal.limit_fixture',root/sys.argv[2])
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.CODE_TIMEOUT_SECONDS=100
import v8_task_entry
fixed,_=v8_task_entry.repair_mbpp_subprocess(m.code_reward)
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
for function in (m.code_reward,fixed):
 try:function({'answer':'assert add(2,3)==5'},'def add(a,b): return a+b')
 except m.subprocess.SubprocessError:pass
 else:raise AssertionError('resource setup failure became a reward')
print('SETUP_ERRORS_PRESERVED')
'''
    result=subprocess.run([sys.executable,'-c',script,str(ROOT),filename],text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'SETUP_ERRORS_PRESERVED' in result.stdout
