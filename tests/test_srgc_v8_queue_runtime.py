"""Exact historical queue resumption; CPU/stdlib only, no model or GPU launch."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import pytest

from scripts import v8_queue_runtime as runtime

REPO=Path(__file__).resolve().parents[1]
ARCHIVE=REPO/'v8/queue-runtimes.zip'


@pytest.fixture
def source_cache(tmp_path,monkeypatch):
    monkeypatch.setattr(tempfile,'tempdir',str(tmp_path))
    return runtime.unpack(ARCHIVE)


def queue(base,cohort,coordinator,runner):
    path=base/cohort/'.multi-node';path.mkdir(parents=True)
    binding=dict(schema=1,cohort=cohort,root=str(base/cohort),
                 coordinator_sha256=coordinator['sha256'],runner_sha256=runner['runner_sha256'])
    (path/'binding.json').write_text(json.dumps(binding))
    (path/'state.json').write_text(json.dumps(dict(phase='train',failure=None)))
    (path/'control.lock').touch()
    (path/'jobs').mkdir();(path/'jobs/a.json').write_text('{"state":"pending"}')
    return binding


def snapshot(base):
    return {str(p.relative_to(base)):p.read_bytes() for p in base.rglob('*') if p.is_file()}


def environment(base):
    return dict(V8_ROOT=str(base),V8_RUNNER_ROOT=str(REPO/'v8/experiments'))


def test_all_shipped_coordinator_runner_combinations_match_without_queue_changes(tmp_path,source_cache):
    cache,catalog=source_cache
    base=tmp_path/'experiments'
    for coordinator in catalog['coordinators']:
        for runner in catalog['runners']:
            cohort='olmo_math';root=base/(coordinator['sha256'][:8]+runner['root'].split('/')[1][:8])
            binding=queue(root,cohort,coordinator,runner)
            before=snapshot(root);inode=(root/cohort/'.multi-node/control.lock').stat().st_ino
            groups=runtime.selections(REPO,environment(root),[cohort])
            assert len(groups)==1
            assert runtime.sha(groups[0]['script'])==binding['coordinator_sha256']
            assert runtime.runner_identity(groups[0]['runner'])==binding['runner_sha256']
            assert snapshot(root)==before
            assert (root/cohort/'.multi-node/control.lock').stat().st_ino==inode


def test_unknown_source_never_relabels_or_moves_queue(tmp_path,source_cache):
    _,catalog=source_cache;base=tmp_path/'experiments'
    coordinator=dict(catalog['coordinators'][0],sha256='f'*64)
    queue(base,'olmo_math',coordinator,catalog['runners'][0]);before=snapshot(base)
    with pytest.raises(ValueError,match='source is unavailable'):
        runtime.selections(REPO,environment(base),['olmo_math'])
    assert snapshot(base)==before


def test_corrupted_cached_sources_are_rejected(tmp_path,source_cache):
    root,catalog=source_cache
    name=next(iter(catalog['files']));(root/name).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='Cached V8 source bytes changed'):runtime.unpack(ARCHIVE)


def test_different_cohort_pins_remain_separate_without_rebinding(tmp_path,source_cache):
    _,catalog=source_cache;base=tmp_path/'experiments';runner=catalog['runners'][-1]
    queue(base,'olmo_math',catalog['coordinators'][-1],runner)
    queue(base,'qwen_mbpp',catalog['coordinators'][-2],runner)
    before=snapshot(base);groups=runtime.selections(REPO,environment(base),[])
    assert {c for g in groups for c in g['cohorts']}=={'olmo_math','qwen_mbpp','olmo_mbpp'}
    assert next(g for g in groups if 'olmo_math' in g['cohorts'])['coordinator_sha256']==catalog['coordinators'][-1]['sha256']
    assert next(g for g in groups if 'qwen_mbpp' in g['cohorts'])['coordinator_sha256']==catalog['coordinators'][-2]['sha256']
    assert snapshot(base)==before


def test_unstarted_default_keeps_three_cohorts_and_does_not_create_queue(tmp_path):
    base=tmp_path/'absent';groups=runtime.selections(REPO,environment(base),[])
    assert len(groups)==1 and groups[0]['cohorts']==['olmo_math','qwen_mbpp','olmo_mbpp']
    assert runtime.sha(groups[0]['script'])==runtime.sha(REPO/'scripts/srgc_v8_nodes.py') and not base.exists()


def test_existing_python_is_selected_by_recorded_versions_without_installing(tmp_path,monkeypatch):
    fake=tmp_path/'python';fake.symlink_to(sys.executable)
    seen=[]
    def identity(command,**kwargs):
        seen.append(command)
        return subprocess.CompletedProcess(command,0,json.dumps(dict(python_version='original',packages={'torch':'original'})),'')
    monkeypatch.setattr(runtime.subprocess,'run',identity)
    binding=dict(python_version='original',packages={'torch':'original'})
    assert runtime.existing_python(binding,dict(V8_ROOT=str(tmp_path/'v8'),V8_PYTHON=str(fake)),'olmo_math')==str(fake)
    assert len(seen)==1 and seen[0][1]=='-c'
    with pytest.raises(ValueError,match='original Python/package versions'):
        runtime.existing_python(dict(binding,packages={'torch':'different'}),dict(V8_ROOT=str(tmp_path/'v8')),
                                'olmo_math',explicit=str(fake))


@pytest.mark.parametrize('cohort_after_flags',[False,True])
def test_no_argument_bootstrap_reexecs_queue_pinned_code(tmp_path,source_cache,monkeypatch,cohort_after_flags):
    from scripts import v8_bootstrap as bootstrap
    _,catalog=source_cache;base=tmp_path/'experiments'
    coordinator=next(x for x in catalog['coordinators'] if x['source_commit'].startswith('cbd68a4'))
    runner=next(x for x in catalog['runners'] if x['source_commit'].startswith('cbd68a4'))
    for cohort in ('olmo_math','qwen_mbpp'):queue(base,cohort,coordinator,runner)
    env=dict(environment(base),V8_CODE_ROOT=str(REPO),V8_PYTHON=sys.executable)
    monkeypatch.setattr(bootstrap,'environment',lambda *a,**k:env)
    before=snapshot(base);seen=[]
    class Reexec(Exception):pass
    def intercept(python,args,child_env):
        seen.append((python,args,child_env));raise Reexec()
    monkeypatch.setattr(bootstrap.os,'execvpe',intercept)
    argv=['--root',str(base),'olmo_math'] if cohort_after_flags else []
    with pytest.raises(Reexec):bootstrap.main(argv)
    assert Path(seen[0][1][1]).name=='v8_dispatch.py'
    plan=json.loads(Path(seen[0][1][2]).read_text())
    assert runtime.sha(plan['entries'][0]['script'])==coordinator['sha256']
    assert runtime.runner_identity(plan['entries'][0]['runner'])==runner['runner_sha256']
    assert plan['argv'].count('olmo_math')==(1 if cohort_after_flags else 0)
    assert snapshot(base)==before


@pytest.mark.parametrize('cohort',['olmo_math','qwen_mbpp'])
@pytest.mark.parametrize('revision',['cbd68a4','f1bb211','7d89fb9'])
def test_real_historical_runtime_binding_reopens_with_exact_original_code(tmp_path,source_cache,cohort,revision,monkeypatch):
    cache,catalog=source_cache
    source=subprocess.check_output(['git','show',revision+':scripts/srgc_v8_nodes.py'],cwd=REPO)
    digest=hashlib.sha256(source).hexdigest()
    coordinator=next(x for x in catalog['coordinators'] if x['sha256']==digest)
    runner=next(x for x in reversed(catalog['runners']) if subprocess.run(
        ['git','merge-base','--is-ancestor',x['source_commit'],revision],cwd=REPO).returncode==0)
    base=tmp_path/'experiments';env=dict(os.environ,V8_CODE_ROOT=str(REPO),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    # Construct the REAL historical binding in its own interpreter, without calling execute/train.
    script=r'''
import importlib.util,json,sys
from pathlib import Path
path,runner,root,cohort=sys.argv[1:]
spec=importlib.util.spec_from_file_location('pinned_nodes',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=m.Runtime(runner,root,cohort,sys.executable,['0','1','2','3'])
p=m.Pool(r.root/'.multi-node',r.binding)
with p.control():p.initialize()
print(json.dumps(r.binding))
'''
    args=[str(cache/coordinator['file']),str(cache/runner['root']),str(base),cohort]
    result=subprocess.run([sys.executable,'-c',script,*args],env=env,text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    binding=json.loads(result.stdout);before=snapshot(base)
    # This is the original reported failure: a different current source must remain rejected.
    from scripts import srgc_v8_nodes as nodes
    with pytest.raises(ValueError,match='identity differs'):
        nodes.Pool(base/cohort/'.multi-node',dict(binding,coordinator_sha256='not-the-original')).initialize()
    groups=runtime.selections(REPO,environment(base),[cohort]);assert len(groups)==1
    # The automatic choice must be accepted by the original Pool, including Python/packages/config.
    args=[str(groups[0]['script']),str(groups[0]['runner']),str(base),cohort]
    result=subprocess.run([sys.executable,'-c',script,*args],env=env,text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==binding and snapshot(base)==before
