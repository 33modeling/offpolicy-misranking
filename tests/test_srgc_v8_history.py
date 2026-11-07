"""Live exporter tests use synthetic JSON records, never real RLVR rewards.

Run with --noconftest: the unrelated historical trainer fixture imports a
removed selection_gate_gpu module; this standalone utility does not use it."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import srgc_v8_history as history


def json_file(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def json_lines(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


@pytest.fixture
def run(tmp_path):
    root = tmp_path/'input'
    rdir = root/'math'/'runs'/'run-1'
    manifest = dict(study_id='fixture',cohort_id='math',run_id='run-1',method_id='on_monitored',
                    training_seed=5,role='smoke',status='active',
                    attempts=[dict(attempt_id='a',start_update=25)],
                    task={'parent':{'completed_update':25},'horizon':100,'retain_updates':[50,75,100]})
    json_file(rdir/'manifest.json',manifest)
    update_file = rdir/'attempts'/'a'/'updates.jsonl'
    json_lines(update_file,[dict(completed_updates=u,metrics={'sample_reward':.25},train_ids=['p']) for u in range(26,51)])
    checkpoint = rdir/'snapshots'/'step-000050.pt'
    checkpoint.parent.mkdir();checkpoint.write_bytes(b'synthetic checkpoint bytes, never unpickled')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    snap = dict(completed_update=50,path='snapshots/step-000050.pt',sha256=digest)
    json_lines(rdir/'snapshots.jsonl',[snap])
    return dict(root=root,rdir=rdir,manifest=manifest,update_file=update_file,checkpoint=checkpoint,snap=snap,tmp=tmp_path)


def exported(run, **kwargs):
    return history.export(run['root'],**kwargs)


def evaluation(run, filename='final', **changes):
    record = dict(run_id='run-1',completed_update=50,split='final',evaluation_spec_hash='e'*64,
                  checkpoint_hash=run['snap']['sha256'],measurement_status='synthetic_fixture',
                  reward_fraction=.375,n_questions=1,responses_per_question=8,
                  per_question_correct={'q':3})
    record.update(changes)
    json_file(run['root']/'math'/'evaluations'/(filename+'.json'),record)
    return record


def test_active_run_exports_without_heldout_evaluation_or_model_imports(run):
    report = exported(run)
    assert report['integrity_ok'] and report['coverage_complete']
    assert report['run_status_counts']=={'active':1}
    data = report['runs'][0]
    assert len(data['training'])==25
    assert data['training'][-1]['reward_fraction']==.25
    assert data['training'][-1]['valid_response_count'] is None  # not guessed from prompt count
    assert data['training'][-1]['reward_scope']=='training_batch'
    assert data['coverage']==[dict(completed_update=50,training_status='recorded',checkpoint_status='present_unhashed',
                                   final_evaluation_status='unmeasured',development_evaluation_status='unmeasured',
                                   final_evaluation_records=0,development_evaluation_records=0)]
    assert not data['evaluations']
    assert data['checkpoints'][0]['payload_loaded'] is False


def test_checkpoint_hash_validation_and_missing_file(run):
    report = exported(run,verify_checkpoints=True)
    assert report['runs'][0]['checkpoints'][0]['status']=='verified_sha256'
    run['checkpoint'].write_bytes(b'changed')
    report = exported(run,verify_checkpoints=True)
    assert not report['integrity_ok'] and not report['coverage_complete']
    assert report['runs'][0]['checkpoints'][0]['status']=='hash_mismatch'
    run['checkpoint'].unlink()
    assert exported(run)['runs'][0]['coverage'][0]['checkpoint_status']=='missing_file'


def test_checkpoint_path_cannot_escape_input_or_run(run):
    outside = run['tmp']/'private.pt';outside.write_bytes(b'must not read')
    run['snap']['path']=str(outside)
    json_lines(run['rdir']/'snapshots.jsonl',[run['snap']])
    report = exported(run,verify_checkpoints=True)
    assert not report['integrity_ok']
    assert str(outside) not in report['sources_sha256']
    assert report['runs'][0]['checkpoints'][0]['status']=='unavailable'


def test_resume_boundary_and_rolled_back_tail_are_not_stitched(run):
    # Old update50 is retained by a resume from50; old51+ is abandoned.
    json_lines(run['update_file'],[dict(completed_updates=u,metrics={'sample_reward':.25}) for u in range(26,76)])
    run['manifest']['attempts'].append(dict(attempt_id='b',start_update=50))
    json_file(run['rdir']/'manifest.json',run['manifest'])
    json_lines(run['rdir']/'attempts'/'b'/'updates.jsonl',[
        dict(completed_updates=u,metrics={'sample_reward':.75}) for u in range(51,61)])
    data = exported(run)['runs'][0]
    rows = {row['completed_update']:row for row in data['training']}
    assert rows[50]['reward_fraction']==.25
    assert rows[51]['reward_fraction']==.75
    assert max(rows)==60 and data['last_logged_update']==60
    assert all('.25' not in str(row['reward_fraction']) for u,row in rows.items() if u>50)


def test_manifest_order_not_filesystem_sort_decides_latest_attempt(run):
    run['manifest']['attempts']=[dict(attempt_id='z',start_update=25),dict(attempt_id='a',start_update=25)]
    json_file(run['rdir']/'manifest.json',run['manifest'])
    json_lines(run['rdir']/'attempts'/'z'/'updates.jsonl',[dict(completed_updates=50,metrics={'sample_reward':1})])
    assert exported(run)['runs'][0]['training'][-1]['reward_fraction']==.25


def test_conflicting_duplicate_is_excluded_and_never_averaged(run):
    with run['update_file'].open('a') as file:
        file.write(json.dumps(dict(completed_updates=50,metrics={'sample_reward':.75},train_ids=['p']))+'\n')
    report = exported(run)
    assert not report['integrity_ok'] and not report['coverage_complete']
    assert report['runs'][0]['training'][-1]['status']=='conflicting'
    assert report['runs'][0]['training'][-1]['reward_fraction'] is None


def test_identical_duplicate_is_idempotent(run):
    line=run['update_file'].read_text().splitlines()[-1]
    with run['update_file'].open('a') as file:file.write(line+'\n')
    assert len(exported(run)['runs'][0]['training'])==25


def test_unfinished_line_while_training_is_omitted_without_source_edit(run):
    with run['update_file'].open('ab') as file:file.write(b'{"completed_updates":51')
    before=run['update_file'].read_bytes()
    report=exported(run)
    assert report['runs'][0]['last_logged_update']==50
    assert report['integrity_ok']
    assert any(i['code']=='unfinished_line' for i in report['issues'])
    assert run['update_file'].read_bytes()==before


def test_missing_expected_rewards_and_checkpoint_not_filled_from_future(run):
    run['manifest'].update(status='completed_horizon',completed_updates=100)
    json_file(run['rdir']/'manifest.json',run['manifest'])
    report=exported(run)
    assert report['runs'][0]['missing_recording_updates']==[75,100]
    assert report['runs'][0]['coverage'][-1]['training_status']=='unmeasured'
    assert report['runs'][0]['coverage'][-1]['checkpoint_status']=='missing_index'
    assert report['runs'][0]['last_logged_update']==50


def test_count_summary_reconciles_actual_denominator_and_preserves_prefix(run):
    run['manifest']['status']='completed_horizon';json_file(run['rdir']/'manifest.json',run['manifest'])
    json_lines(run['rdir']/'training_stats.jsonl',[
        dict(completed_update=25,reward_sum=1,valid_response_count=4),
        dict(completed_update=50,reward_sum=8,valid_response_count=32)])
    rows={r['completed_update']:r for r in exported(run)['runs'][0]['training']}
    assert rows[25]['inherited_prefix']
    assert rows[50]['reward_sum']==8 and rows[50]['valid_response_count']==32
    assert rows[50]['reward_fraction']==.25


def test_summary_and_raw_reward_conflict_is_not_silently_selected(run):
    json_lines(run['rdir']/'training_stats.jsonl',[dict(completed_update=50,reward_sum=16,valid_response_count=32)])
    report=exported(run)
    assert not report['integrity_ok']
    assert report['runs'][0]['training'][-1]['reward_fraction'] is None


def test_active_restart_does_not_use_stale_finalized_tail(run):
    json_lines(run['rdir']/'training_stats.jsonl',[dict(completed_update=75,reward_sum=32,valid_response_count=32)])
    assert exported(run)['runs'][0]['last_logged_update']==50


@pytest.mark.parametrize('changes',[
    {'reward_fraction':.5}, {'per_question_correct':{'q':9}},
    {'per_question_correct':{'q':2.5}}, {'per_question_correct':{}},
    {'checkpoint_hash':'0'*64}, {'n_questions':0}, {'responses_per_question':True},
])
def test_inconsistent_evaluation_preserved_but_not_used_as_measured(run,changes):
    evaluation(run,**changes)
    report=exported(run)
    row=report['runs'][0]['evaluations'][0]
    assert row['status']=='invalid_record' and row['reward_fraction'] is None
    assert not report['integrity_ok']
    assert 'reported_reward_fraction' in row


def test_training_dev_and_final_are_separate_and_no_best_spec_is_selected(run):
    evaluation(run)
    evaluation(run,'dev',split='dev',reward_fraction=.5,per_question_correct={'q':4},evaluation_spec_hash='d'*64)
    evaluation(run,'other-final',evaluation_spec_hash='c'*64,reward_fraction=.125,per_question_correct={'q':1})
    report=exported(run)
    assert report['integrity_ok']
    directory=history.write_outputs(report,run['tmp']/'exports')
    def rows(name):return list(csv.DictReader((directory/name).open()))
    assert len(rows('heldout-final.csv'))==2
    assert len(rows('heldout-development.csv'))==1
    assert len(rows('training-curve.csv'))==25
    assert {r['reward_fraction'] for r in rows('heldout-final.csv')}=={'0.375','0.125'}
    assert all(r['measurement_status']=='synthetic_fixture' for r in rows('heldout-final.csv'))


def test_bad_evaluation_file_does_not_drop_training_or_other_runs(run):
    path=run['root']/'math'/'evaluations'/'broken.json';path.parent.mkdir();path.write_text('{bad')
    report=exported(run)
    assert not report['integrity_ok'] and len(report['runs'])==1
    assert len(report['runs'][0]['training'])==25


def test_checkpoint_and_evaluation_conflicts_are_reported(run):
    json_lines(run['rdir']/'snapshots.jsonl',[run['snap'],dict(run['snap'],sha256='0'*64)])
    evaluation(run)
    report=exported(run)
    assert not report['integrity_ok'] and not report['coverage_complete']
    assert not report['runs'][0]['checkpoints']
    assert report['runs'][0]['evaluations'][0]['reward_fraction'] is None


def test_equal_means_do_not_hide_conflicting_question_counts(run):
    evaluation(run,'one',n_questions=2,responses_per_question=4,per_question_correct={'q1':1,'q2':2})
    evaluation(run,'two',n_questions=2,responses_per_question=4,per_question_correct={'q1':2,'q2':1})
    report=exported(run)
    assert not report['integrity_ok']
    assert report['runs'][0]['evaluations'][0]['status']=='conflicting'
    assert report['runs'][0]['evaluations'][0]['reward_fraction'] is None


def test_immutable_generations_and_source_hashes(run):
    report=exported(run)
    output=run['tmp']/'exports'
    first=history.write_outputs(report,output)
    first_bytes={p.name:p.read_bytes() for p in first.iterdir()}
    second=history.write_outputs(report,output)
    assert first!=second
    assert {p.name:p.read_bytes() for p in first.iterdir()}==first_bytes
    pointer=json.loads((output/'latest.json').read_text())
    assert pointer['generation']==second.name
    assert history.sha((second/'manifest.json').read_bytes())==pointer['manifest_sha256']
    manifest=json.loads((second/'manifest.json').read_text())
    assert all(history.sha((second/name).read_bytes())==digest for name,digest in manifest['files_sha256'].items())
    assert report['sources_sha256']['math/runs/run-1/manifest.json']==history.sha((run['rdir']/'manifest.json').read_bytes())


@pytest.mark.parametrize('where',['input','inside','ancestor','symlink'])
def test_output_cannot_modify_input_tree(run,where):
    path={'input':run['root'],'inside':run['root']/'exports','ancestor':run['tmp']}.get(where)
    if where=='symlink':
        path=run['tmp']/'alias';path.symlink_to(run['root'],target_is_directory=True)
    with pytest.raises(ValueError,match='separate'):
        history.write_outputs(exported(run),path)


def test_multiple_cohorts_and_seed_identity_are_not_combined(run):
    import shutil
    target=run['root']/'mbpp'/'runs'/'run-2';target.parent.mkdir(parents=True)
    shutil.copytree(run['rdir'],target)
    manifest=dict(run['manifest'],cohort_id='mbpp',run_id='run-2',training_seed=6,method_id='bank_sr_t050')
    json_file(target/'manifest.json',manifest)
    report=exported(run)
    assert len(report['runs'])==2
    assert {(r['cohort_id'],r['training_seed']) for r in report['runs']}=={('math',5),('mbpp',6)}


def test_cli_is_cpu_only_and_strict_allows_unmeasured_heldout(run):
    script=Path(history.__file__)
    command=[sys.executable,str(script),'--root',str(run['root']),'--output',str(run['tmp']/'cli'),'--strict']
    result=subprocess.run(command,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['coverage_complete']
    run['checkpoint'].unlink()
    result=subprocess.run(command,text=True,capture_output=True)
    assert result.returncode==3


def test_missing_root_or_no_runs_is_an_error(tmp_path):
    with pytest.raises(ValueError,match='not found'):history.export(tmp_path/'missing')
    with pytest.raises(ValueError,match='no V8 run'):history.export(tmp_path)


def test_abandoned_future_checkpoint_and_eval_are_not_current_lineage(run):
    # A former attempt reached75, but the current attempt resumed from50 and is at60.
    json_lines(run['update_file'],[dict(completed_updates=u,metrics={'sample_reward':.25}) for u in range(26,76)])
    run['manifest']['attempts'].append(dict(attempt_id='b',start_update=50))
    json_file(run['rdir']/'manifest.json',run['manifest'])
    json_lines(run['rdir']/'attempts'/'b'/'updates.jsonl',[
        dict(completed_updates=u,metrics={'sample_reward':.75}) for u in range(51,61)])
    old=run['rdir']/'snapshots'/'step-000075.pt';old.write_bytes(b'old abandoned state')
    snap=dict(completed_update=75,path='snapshots/step-000075.pt',sha256=history.sha(old.read_bytes()))
    json_lines(run['rdir']/'snapshots.jsonl',[run['snap'],snap])
    evaluation(run,completed_update=75,checkpoint_hash=snap['sha256'])
    report=exported(run)
    assert report['runs'][0]['last_logged_update']==60
    assert report['runs'][0]['evaluations'][0]['reward_fraction'] is None
    assert 'surviving training lineage' in report['runs'][0]['evaluations'][0]['reason']


def test_snapshot_publish_failure_preserves_previous_generation_pointer(run,monkeypatch):
    report=exported(run);output=run['tmp']/'exports'
    history.write_outputs(report,output)
    before=(output/'latest.json').read_bytes()
    original=history.os.replace
    def fail(source,dest):
        if Path(dest).name=='latest.json':raise OSError('synthetic publication failure')
        return original(source,dest)
    monkeypatch.setattr(history.os,'replace',fail)
    with pytest.raises(OSError,match='synthetic'):history.write_outputs(report,output)
    assert (output/'latest.json').read_bytes()==before


@pytest.mark.parametrize('field,value', [('completed_update',[]),('evaluation_spec_hash',{}),('split',[])])
def test_malformed_evaluation_identity_does_not_hide_training(run,field,value):
    evaluation(run,**{field:value})
    report=exported(run)
    assert not report['integrity_ok']
    assert len(report['runs'])==1 and len(report['runs'][0]['training'])==25


def test_unknown_attempt_lineage_is_not_sorted_or_guessed(run):
    run['manifest']['attempts']=[dict(attempt_id='a',start='resume@25')]
    json_file(run['rdir']/'manifest.json',run['manifest'])
    report=exported(run)
    assert not report['integrity_ok']
    assert not report['runs'][0]['training']


def test_malformed_metrics_and_nonfinite_json_are_not_measurements(run):
    with run['update_file'].open('a') as file:
        file.write('{"completed_updates":51,"metrics":{"sample_reward":NaN}}\n')
        file.write('{"completed_updates":52,"metrics":[]}\n')
    report=exported(run)
    assert not report['integrity_ok']
    assert all(r['reward_fraction'] is None for r in report['runs'][0]['training'] if r['completed_update']>50)


def test_v8_runtime_interrupt_resume_and_offline_evaluation(tmp_path):
    """Optional integration with the actual V8 runner; CPU software check only."""
    import os
    runtime = os.environ.get('V8_HISTORY_TEST_RUNTIME')
    if not runtime:
        pytest.skip('Set V8_HISTORY_TEST_RUNTIME to v8/experiments for the CPU runner integration')
    script = r'''
from pathlib import Path
import csv,hashlib,json,sys
runtime,work,exporter = map(Path,sys.argv[1:])
sys.path.insert(0,str(runtime))
sys.path.insert(0,str(exporter.parent))
import srgc_v8_history as history
from switching import backends,coderepo,matrix as mx,smoke,spec
from switching.engine import make_engine_class
from switching.runner import execute_train,execute_eval,Interrupted

root=work/'input'/'fixture';root.mkdir(parents=True)
H=100;srgc=coderepo.srgc();PE=make_engine_class(srgc)
feat,ans,data=backends.toy_problem(0)
cfg=spec.resolve('olmo_math',role='smoke',extra={'cohort_id':'fixture','seeds':{'training':[5]},
                 'training':{'matched_update_endpoint':H}})
ctx=smoke.toy_ctx(root,5,srgc,PE,feat,ans,data,H)
smoke.make_prefix(ctx,5,25,srgc)
loc=smoke.Locator(root,cfg,25)
task=mx.build_stage('bank-trajectory',cfg,loc,root,role='smoke',seeds=[5])[0][0]
task['debug_stop_after_update']=60
try:
    execute_train(task,ctx)
except Interrupted:
    pass
else:
    raise AssertionError('expected safe interruption at60')

def capture(label):
    digest=lambda:{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in root.rglob('*') if p.is_file()}
    before=digest()
    report=history.export(work/'input',verify_checkpoints=True)
    assert digest()==before,'exporter changed source artifacts'
    out=history.write_outputs(report,work/'exports')
    assert report['integrity_ok'],report['issues']
    assert report['coverage_complete'],report['runs'][0]['coverage']
    return report,out

live,live_out=capture('interrupted')
assert live['runs'][0]['run_status']=='interrupted'
assert live['runs'][0]['last_logged_update']==60
assert len(live['runs'][0]['training'])==35
assert not live['runs'][0]['evaluations']
assert not (root/'evaluations').exists(),'training must not invoke held-out evaluation'
task.pop('debug_stop_after_update')
done=execute_train(task,ctx)
assert done['completed_updates']==H
trained,trained_out=capture('completed_without_evaluation')
rows=trained['runs'][0]['training']
assert len(rows)==H and len({r['completed_update'] for r in rows})==H
assert len(trained['runs'][0]['checkpoints'])==3
assert all(r['status']=='verified_sha256' for r in trained['runs'][0]['checkpoints'])
assert not trained['runs'][0]['evaluations']

for split in ('final','dev'):
    tasks=mx.eval_tasks(cfg,data,root,[task['run_id']],split=split,updates={H})
    assert len(tasks)==1
    execute_eval(tasks[0],ctx)
evaluated,eval_out=capture('offline_evaluated')
ev=evaluated['runs'][0]['evaluations']
assert len(ev)==2 and all(r['status']=='validated_record' for r in ev)
assert all(r['measurement_status']=='simulated_cpu' for r in ev)
assert {r['split'] for r in ev}=={'dev','final'}
coverage=evaluated['runs'][0]['coverage']
assert [r['completed_update'] for r in coverage]==[50,75,100]
assert all(r['final_evaluation_records']==0 and r['final_evaluation_status']=='unmeasured' for r in coverage[:-1])
assert coverage[-1]['final_evaluation_records']==1
assert len(list(csv.DictReader((eval_out/'heldout-final.csv').open())))==1
assert len(list(csv.DictReader((eval_out/'heldout-development.csv').open())))==1
assert len(list(csv.DictReader((eval_out/'training-curve.csv').open())))==100
summary={'scope':'CPU ToyBackend software integration; NOT RLVR scientific measurements',
 'runtime_source':str(runtime),'completed_updates':H,'interruption_update':60,
 'interrupted_training_rows':35,'resumed_training_rows':len(rows),'verified_retained_checkpoints':3,
 'offline_final_evaluations':1,'offline_development_evaluations':1,'new_GPU_jobs':0,
 'generations':[str(live_out),str(trained_out),str(eval_out)]}
(work/'integration-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
'''
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',V8_FSYNC='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([sys.executable,'-c',script,runtime,str(tmp_path),history.__file__],
                            env=env,text=True,capture_output=True,timeout=90)
    (tmp_path/'integration-stdout.txt').write_text(result.stdout)
    (tmp_path/'integration-stderr.txt').write_text(result.stderr)
    assert result.returncode==0,result.stderr+'\n'+result.stdout
