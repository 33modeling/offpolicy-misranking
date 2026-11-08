"""Run the exact original v8 child with a separately receipted decode observer."""
import hashlib
import functools
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from v8_generation_probe import install


def repair_context(ctx):
    """Apply the seed runs' documented verifier/cache protections after saved-source selection."""
    backend=ctx.backend;changes={};verifier=getattr(backend,'verifier',None)
    if (getattr(verifier,'__name__',None)=='math_reward'
            and getattr(verifier,'__module__',None) in ('srgc_rebuttal.run_experiment','srgc_rebuttal.verifiers')):
        records=getattr(backend,'records',{})
        if any(not str(row.get('answer','')).strip() for row in records.values()):
            raise ValueError('empty gold answer in MATH inputs; input repair required before training')
        path=Path(__file__).with_name('srgc_verifier_fallback.py')
        spec=importlib.util.spec_from_file_location('_v8_seed_math_fallback',path)
        fallback=importlib.util.module_from_spec(spec);spec.loader.exec_module(fallback)
        backend.verifier=fallback.tolerant_math_reward
        identity=dict(getattr(ctx,'verifier_identity',None) or {})
        identity.update(tolerant_math_fallback=True,legacy_runtime_repair='seed-math-fallback-v1',
                        fallback_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gold_parse_timeout=fallback.GOLD_TIMEOUT)
        ctx.verifier_identity=identity
        changes['verifier']=dict(original=verifier.__module__+':'+verifier.__name__,current=identity)
    rollout=getattr(backend,'_rollout',None)
    if callable(rollout) and not getattr(backend,'_v8_rollout_guard',False):
        # The current child already guards scoring/evaluation. Extend its same
        # guard to training; a rollout retry never repeats an optimizer update.
        if hasattr(backend,'_v8_protect'):
            train=getattr(backend,'train',None)
            if callable(train):
                @functools.wraps(train)
                def guarded_train(*args,**kwargs):
                    previous=backend._v8_protect;backend._v8_protect=True
                    try:return train(*args,**kwargs)
                    finally:backend._v8_protect=previous
                backend.train=guarded_train;changes['memory']='extend-existing-rollout-guard-to-training'
        else:
            import torch
            delay=float(os.environ.get('V8_OOM_RETRY_SECONDS','5'))
            if not 0<=delay<float('inf'):raise ValueError('V8_OOM_RETRY_SECONDS must be finite and nonnegative')
            oom=getattr(torch,'OutOfMemoryError',torch.cuda.OutOfMemoryError)
            def release():
                if getattr(getattr(backend,'device',None),'type',None)=='cuda':torch.cuda.empty_cache()
            @functools.wraps(rollout)
            def guarded_rollout(*args,**kwargs):
                message=None
                try:result=rollout(*args,**kwargs)
                except oom as exc:message=str(exc)
                if message is not None:
                    # The exception scope must end before freeing memory: its
                    # traceback otherwise keeps the failed generation tensors alive.
                    gc.collect();release()
                    print(f'CUDA OOM: released cache; retrying the same rollout once: {message}',flush=True)
                    time.sleep(delay);result=rollout(*args,**kwargs)
                release();return result
            backend._rollout=guarded_rollout;changes['memory']='seed-cache-release-and-one-identical-rollout-retry'
        backend._v8_rollout_guard=True
    if changes:
        record=dict(schema=1,original_child_sha256=os.environ.get('V8_ORIGINAL_CHILD_SHA256'),
                    numerical_source_changed=False,reward_runtime_changed='verifier' in changes,repairs=changes)
        directory=os.environ.get('SRGC_PROGRESS_DIR')
        if directory:
            path=Path(directory)/f'runtime-repair-rank-{os.environ.get("RANK","0")}.json'
            path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(record)+'\n');os.replace(temporary,path)
        print('RUNTIME REPAIR '+json.dumps(record),flush=True)


def install_runtime_repairs():
    from switching.runner import Context
    if getattr(Context,'_v8_runtime_repairs',False):return
    original=Context.__init__
    @functools.wraps(original)
    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs);repair_context(self)
    Context.__init__=initialize;Context._v8_runtime_repairs=True


def retain_cuda_failure(exc):
    """Retain CUDA failure context without launching another CUDA operation after failure."""
    if not isinstance(exc,RuntimeError) or not any(s in str(exc).lower() for s in ('cuda','out of memory','symintarrayref','registercuda')):return
    directory=os.environ.get('SRGC_PROGRESS_DIR')
    if not directory:return
    rank=int(os.environ.get('RANK','0'));path=Path(directory)/f'cuda-failure-rank-{rank}.json'
    record=dict(schema=1,rank=rank,type=type(exc).__name__,message=str(exc),notes=getattr(exc,'__notes__',[]),
                last_progress=None,memory=None)
    try:
        stage=Path(directory)/f'rank-{rank}.json'
        if stage.exists():record['last_progress']=json.loads(stage.read_text())
        torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            device=int(os.environ.get('LOCAL_RANK','0'))
            stats=torch.cuda.memory_stats(device)
            record['memory']={k:stats.get(k) for k in ('allocated_bytes.all.current','reserved_bytes.all.current',
                'allocated_bytes.all.peak','reserved_bytes.all.peak','num_alloc_retries','num_ooms')}
            record.update(device=device,torch_version=torch.__version__,cuda_version=torch.version.cuda)
    except Exception as error:record['diagnostic_error']=str(error)
    try:
        path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record)+'\n');os.replace(temporary,path)
        print(f'CUDA FAILURE CONTEXT rank={rank}: {path}',file=sys.stderr,flush=True)
    except Exception as error:print(f'Cannot retain CUDA failure context: {error}',file=sys.stderr,flush=True)


def main():
    path=Path(sys.argv[1]);expected=os.environ.get('V8_ORIGINAL_CHILD_SHA256')
    if not expected or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Original v8 child bytes differ from the selected runner identity')
    spec=importlib.util.spec_from_file_location('_v8_original_child',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if hasattr(module,'install_progress'):module.install_progress=lambda *args,**kwargs:None
    install_runtime_repairs()
    install(int(os.environ.get('RANK','0')))
    sys.argv=[str(path),*sys.argv[2:]]
    return module.main()


def recorded_main():
    """Let torchrun retain the original rank traceback, including startup/SystemExit checks."""
    try:from torch.distributed.elastic.multiprocessing.errors import record,get_error_handler
    except ModuleNotFoundError as exc:
        # Standalone CPU hash/argv checks need no torch; a real torchrun child always has it.
        if exc.name=='torch':return main()
        raise
    try:return record(main)()
    except BaseException as exc:
        retain_cuda_failure(exc)
        # torchrun's standard decorator deliberately skips SystemExit; failed
        # provenance/config checks often use it. Keep its exact exit semantics.
        if isinstance(exc,SystemExit) and exc.code not in (None,0,75):
            try:get_error_handler().record_exception(exc)
            except Exception as error:print(f'Cannot record original SystemExit: {error}',file=sys.stderr,flush=True)
        directory=os.environ.get('SRGC_PROGRESS_DIR');source=os.environ.get('TORCHELASTIC_ERROR_FILE')
        if directory and source and not (isinstance(exc,SystemExit) and exc.code in (None,0,75)):
            try:
                rank=int(os.environ.get('RANK','0'));path=Path(directory)/f'rank-{rank}-error.json'
                path.parent.mkdir(parents=True,exist_ok=True)
                payload=dict(schema=1,rank=rank,invocation=os.environ.get('V8_FAILURE_TOKEN'),
                             original_error_file=source,error=json.loads(Path(source).read_text()))
                temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(payload)+'\n');os.replace(temporary,path)
            except Exception as error:print(f'Cannot retain original rank error: {error}',file=sys.stderr,flush=True)
        raise


if __name__=='__main__':raise SystemExit(recorded_main())
