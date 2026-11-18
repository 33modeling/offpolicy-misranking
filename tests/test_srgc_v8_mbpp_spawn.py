"""Frozen MBPP verifier parity and thread-safe launch controls; CPU, no CUDA inference."""
import concurrent.futures
import errno
import importlib.util
import json
import os
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


# Candidate programs for `assert add(2,3)==5` with a 1 s checker wall timeout (CPU rlimit 2 s).
OUTCOME_PROGRAMS=[
    ('passing','def add(a,b): return a+b',1.0),
    ('failing','def add(a,b): return a-b',0.0),
    ('looping','while True: pass',0.0),
    ('sleeps-within-timeout','import time\ntime.sleep(0.2)\ndef add(a,b): return a+b',1.0),
    ('sleeps-past-timeout','import time\ntime.sleep(3)\ndef add(a,b): return a+b',0.0),
]


def _short_timeout_fixed(module,monkeypatch,bootstrap_delay=None,startup=None):
    import v8_task_entry
    monkeypatch.setattr(module,'CODE_TIMEOUT_SECONDS',1.0)
    if bootstrap_delay is not None:
        monkeypatch.setattr(v8_task_entry,'_MBPP_EXEC_BOOTSTRAP',
                            f'import time;time.sleep({bootstrap_delay})\n'+v8_task_entry._MBPP_EXEC_BOOTSTRAP)
    if startup is None:monkeypatch.delenv('V8_MBPP_STARTUP_SECONDS',raising=False)
    else:monkeypatch.setenv('V8_MBPP_STARTUP_SECONDS',str(startup))
    return v8_task_entry.repair_mbpp_subprocess(module.code_reward)


@pytest.mark.parametrize('name,code,expected',OUTCOME_PROGRAMS,ids=[p[0] for p in OUTCOME_PROGRAMS])
def test_reward_for_passing_failing_looping_programs_is_unchanged(verifier,monkeypatch,name,code,expected):
    module,_=verifier
    fixed,_=_short_timeout_fixed(module,monkeypatch)
    record=dict(answer='assert add(2,3)==5')
    assert module.code_reward(record,code)==fixed(record,code)==expected


@pytest.mark.parametrize('name,code,expected',OUTCOME_PROGRAMS,ids=[p[0] for p in OUTCOME_PROGRAMS])
def test_slow_bootstrap_start_computes_the_same_reward_instead_of_failing_the_rank(verifier,monkeypatch,name,code,expected):
    # The bootstrap takes 1.5 s to start, longer than the 1 s checker timeout. Before, the handshake waited
    # only CODE_TIMEOUT_SECONDS and raised SubprocessError (rank failure), and a start just under it left the
    # checker almost no wall time. Now the handshake has its own window and the test run keeps its full 1 s.
    module,_=verifier
    fixed,_=_short_timeout_fixed(module,monkeypatch,bootstrap_delay=1.5)
    record=dict(answer='assert add(2,3)==5')
    assert fixed(record,code)==module.code_reward(record,code)==expected


def test_receipt_records_unchanged_limits_and_the_startup_window(verifier,monkeypatch):
    module,_=verifier
    _,receipt=_short_timeout_fixed(module,monkeypatch)
    assert receipt['policy']=='mbpp-limits-after-exec-v3' and receipt['reward_rule_changed'] is False
    assert (receipt['wall_timeout_seconds'],receipt['cpu_seconds'],receipt['memory_bytes'])==(1.0,2,module.CODE_MEMORY_BYTES)
    assert receipt['startup_handshake_seconds']==60.0 and receipt['wall_timeout_starts']=='checker exec'


def test_startup_past_its_own_window_stays_an_infrastructure_error(verifier,monkeypatch):
    module,_=verifier
    fixed,receipt=_short_timeout_fixed(module,monkeypatch,bootstrap_delay=1.5,startup=0.3)
    assert receipt['startup_handshake_seconds']==0.3
    with pytest.raises(module.subprocess.SubprocessError,match='startup timeout'):
        fixed(dict(answer='assert add(2,3)==5'),'def add(a,b): return a+b')


@pytest.mark.parametrize('code',[errno.EAGAIN,errno.ENOMEM])
def test_transient_spawn_errors_are_retried_and_the_reward_is_computed(verifier,monkeypatch,code):
    module,fixed=verifier
    import v8_task_entry
    original=module.subprocess.Popen;calls=[];sleeps=[]
    def flaky(command,*a,**kwargs):
        calls.append(command)
        if len(calls)<=2:raise OSError(code,os.strerror(code))
        return original(command,*a,**kwargs)
    monkeypatch.setattr(module.subprocess,'Popen',flaky)
    monkeypatch.setattr(v8_task_entry,'_sleep',sleeps.append)
    record=dict(answer='assert add(2,3)==5')
    assert fixed(record,'def add(a,b): return a+b')==1.0
    assert len(calls)==3 and len(sleeps)==2 and sleeps[0]<sleeps[1]


def test_spawn_retries_are_bounded_and_other_errors_are_not_retried(verifier,monkeypatch):
    module,fixed=verifier
    import v8_task_entry
    calls=[];sleeps=[]
    def failing(number):
        def popen(command,*a,**kwargs):
            calls.append(command);raise OSError(number,os.strerror(number))
        return popen
    monkeypatch.setattr(v8_task_entry,'_sleep',sleeps.append)
    record=dict(answer='assert add(2,3)==5')
    monkeypatch.setattr(module.subprocess,'Popen',failing(errno.EAGAIN))
    with pytest.raises(OSError):fixed(record,'def add(a,b): return a+b')
    assert len(calls)==v8_task_entry.MBPP_SPAWN_ATTEMPTS and len(sleeps)==v8_task_entry.MBPP_SPAWN_ATTEMPTS-1
    calls.clear();sleeps.clear()
    monkeypatch.setattr(module.subprocess,'Popen',failing(errno.EACCES))
    with pytest.raises(PermissionError):fixed(record,'def add(a,b): return a+b')
    assert len(calls)==1 and sleeps==[]


@pytest.mark.parametrize('value',['inf','1e300','nan','-1','0','junk'])
def test_unusable_startup_window_setting_falls_back_to_the_default(verifier,monkeypatch,value):
    module,_=verifier
    monkeypatch.setenv('V8_MBPP_STARTUP_SECONDS',value)
    import v8_task_entry
    fixed,receipt=v8_task_entry.repair_mbpp_subprocess(module.code_reward)
    assert receipt['startup_handshake_seconds']==v8_task_entry.MBPP_STARTUP_SECONDS_DEFAULT
    assert fixed(dict(answer='assert add(2,3)==5'),'def add(a,b): return a+b')==1.0
