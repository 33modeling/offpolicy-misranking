"""Run the exact original v8 child with a separately receipted decode observer."""
import hashlib
import copy
import functools
import gc
import importlib.util
import inspect
import faulthandler
import json
import os
from pathlib import Path
import sys
import time
import textwrap
import types
from contextlib import contextmanager

from v8_generation_probe import install


LEGACY_DENSE_BACKEND_SHA256='c73ee5e60288791bedbc790147fc88903ba6f3429d48b2ab626774b334f4f3fc'
MBPP_VERIFIER_HASHES={'5062befe4ed5a8e21f6de01a73dca4f1d3544dd033c5d2c846bff391d1148a90',
                      '5d2414c2ef8c6091aeace48ac8eccadce3357b5e0854e658980b33bd854a1522'}
_NATIVE_FAULT_STATE=None


def install_native_fault_handler():
    """Keep a raw fd open through interpreter teardown; no Python work in the signal handler."""
    global _NATIVE_FAULT_STATE
    directory=os.environ.get('SRGC_PROGRESS_DIR')
    if not directory:return
    if _NATIVE_FAULT_STATE is None:
        root=Path(directory);root.mkdir(parents=True,exist_ok=True)
        rank=int(os.environ.get('RANK','0'));token=os.environ.get('V8_FAILURE_TOKEN','pid-'+str(os.getpid()))
        key=hashlib.sha256(token.encode()).hexdigest()[:24]
        path=root/f'native-fault-rank-{rank}-{key}.log'
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_APPEND|os.O_CLOEXEC,0o660)
        meta=dict(schema=1,rank=rank,pid=os.getpid(),invocation=token,log=path.name,python=sys.version)
        os.write(fd,('V8 native fault context '+json.dumps(meta)+'\n').encode())
        target=root/f'native-fault-rank-{rank}.json';temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(meta)+'\n');os.replace(temporary,target)
        _NATIVE_FAULT_STATE=fd
    # torch.elastic's record() installs its own default handler; main() rebinds this fd afterwards.
    try:faulthandler.enable(file=_NATIVE_FAULT_STATE,all_threads=True,c_stack=True)
    except TypeError:faulthandler.enable(file=_NATIVE_FAULT_STATE,all_threads=True)


_MBPP_EXEC_BOOTSTRAP='''import os,resource,sys
ack=int(sys.argv[1])
memory,cpu,fsize=map(int,sys.argv[2:5])
sys.argv=sys.argv[5:]
try:
 resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
 resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu))
 resource.setrlimit(resource.RLIMIT_FSIZE,(fsize,fsize))
except BaseException as error:
 os.write(ack,('ERROR '+type(error).__name__+': '+str(error))[:2000].encode())
 os.close(ack)
 raise
os.write(ack,b'OK');os.close(ack)
path=sys.argv[0]
with open(path,'rb') as handle:source=handle.read()
scope={'__name__':'__main__','__file__':path,'__package__':None,'__spec__':None,'__cached__':None,'__builtins__':__builtins__}
exec(compile(source,path,'exec'),scope)
'''


def repair_mbpp_subprocess(verifier):
    """Move audited limits after exec; retain each frozen verifier's checker, scoring and limits."""
    if getattr(verifier,'_v8_mbpp_spawn',None):return verifier,verifier._v8_mbpp_spawn
    namespace=getattr(verifier,'__globals__',{});filename=namespace.get('__file__')
    if getattr(verifier,'__name__',None)!='code_reward' or not filename:return verifier,None
    if not Path(filename).is_file():return verifier,None
    digest=hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    if digest not in MBPP_VERIFIER_HASHES:return verifier,None
    original=namespace['subprocess'];limits=namespace['_limits']
    memory=namespace['CODE_MEMORY_BYTES'];cpu=int(namespace['CODE_TIMEOUT_SECONDS'])+1
    fsize=namespace.get('CODE_RESULT_BYTES',1<<20)
    class ProcessModule:
        def __getattr__(self,name):return getattr(original,name)
        def Popen(self,args,*pos,**kwargs):
            import select,signal
            if kwargs.pop('preexec_fn',None) is not limits or list(args[:3])!=[sys.executable,'-I','-B']:
                raise RuntimeError('MBPP subprocess differs from the audited frozen launch')
            reader,writer=os.pipe();process=None
            try:
                kwargs['pass_fds']=(*kwargs.get('pass_fds',()),writer)
                command=[*args[:3],'-c',_MBPP_EXEC_BOOTSTRAP,str(writer),str(memory),str(cpu),str(fsize),*args[3:]]
                process=original.Popen(command,*pos,**kwargs);started=time.monotonic()
                os.close(writer);writer=None
                ready,_,_=select.select([reader],[],[],float(namespace['CODE_TIMEOUT_SECONDS']))
                status=os.read(reader,4096) if ready else b'startup timeout'
                if status!=b'OK':
                    raise original.SubprocessError('MBPP checker limits/setup failed after exec: '+status.decode(errors='replace'))
                elapsed=time.monotonic()-started;wait=process.wait
                # Keep the original wall timeout measured from Popen's exec completion.
                def timed_wait(timeout=None):return wait(timeout=None if timeout is None else max(0,timeout-elapsed))
                process.wait=timed_wait
                return process
            except BaseException:
                if process is not None:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait()
                raise
            finally:
                os.close(reader)
                if writer is not None:os.close(writer)
    cloned=dict(namespace,subprocess=ProcessModule())
    fixed=types.FunctionType(verifier.__code__,cloned,verifier.__name__,verifier.__defaults__,verifier.__closure__)
    functools.update_wrapper(fixed,verifier)
    receipt=dict(policy='mbpp-limits-after-exec-v1',original_verifier_sha256=digest,
                 code_verifier_version=namespace['CODE_REWARD_VERSION'],preexec_fn=False,
                 memory_bytes=memory,cpu_seconds=cpu,file_bytes=fsize,reward_rule_changed=False)
    fixed._v8_mbpp_spawn=receipt
    return fixed,receipt


def cpu_snapshot(value):
    """Copy directly to host; deepcopy of a CUDA optimizer duplicates it on GPU."""
    import torch
    if isinstance(value,torch.Tensor):return value.detach().to('cpu',copy=True)
    if isinstance(value,dict):return {k:cpu_snapshot(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_snapshot(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_snapshot(v) for v in value)
    return copy.deepcopy(value)


def repair_dense_memory(backend):
    """Apply only the two audited lifetime edits to the exact saved TorchBackend.

    Keep its own reward functions, timing decorator, RNG, rank sharding and single
    gather. Inherited Protected/Qwen overrides still call this same base method.
    Original files and source bindings remain immutable; receipt the overlay.
    """
    for cls in type(backend).__mro__:
        if '_v8_dense_memory_receipt' in cls.__dict__:return cls._v8_dense_memory_receipt
        method=cls.__dict__.get('score_gradients')
        if not callable(method):continue
        original=inspect.unwrap(method)
        filename=getattr(original,'__globals__',{}).get('__file__')
        if not filename or not Path(filename).is_file():continue
        digest=hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        if digest!=LEGACY_DENSE_BACKEND_SHA256:continue
        source=textwrap.dedent(inspect.getsource(original))
        edits=(('            accumulated = [None] * len(params)\n',
                '            accumulated = [None] * len(params)\n            value = None\n'),
               ('            result[prompt_id] = projected\n',
                '            del accumulated, gradient, value, sequences\n            result[prompt_id] = projected\n'))
        for before,after in edits:
            if source.count(before)!=1:raise RuntimeError('Saved dense scorer differs from audited memory repair')
            source=source.replace(before,after,1)
        namespace={}
        # The function keeps the original live module globals (including progress
        # instrumentation); exec locals avoid replacing unrelated module symbols.
        exec(compile(source,'<v8-dense-memory-overlay>','exec'),original.__globals__,namespace)
        cls.score_gradients=namespace['score_gradients']
        def state_dict(self):
            return dict(trainable={n:cpu_snapshot(p) for n,p in self.train_parameters},
                        optimizer=cpu_snapshot(self.optimizer.state_dict()),
                        projection_dim=self.projection_dim,projection_seed=self.projection_seed,
                        score_names=[n for n,_ in self.score_parameters])
        cls.state_dict=state_dict
        receipt=dict(original_backend_sha256=digest,scorer_overlay_sha256=hashlib.sha256(source.encode()).hexdigest(),
                     changes=['release-per-prompt-dense-tensors','copy-optimizer-directly-to-host'],
                     extra_collectives=0)
        cls._v8_dense_memory_receipt=receipt
        return receipt
    return None


def repair_olmo_memory(backend):
    """Recompute zero-dropout OLMo decoder activations; keep its draw/batch protocol."""
    if getattr(backend,'_v8_olmo_memory',None):return backend._v8_olmo_memory
    model=getattr(backend,'model',None)
    if model is None:return None
    base=model.get_base_model() if hasattr(model,'get_base_model') else model
    if getattr(getattr(base,'config',None),'model_type',None)!='olmo3':return None
    import torch
    if (getattr(base.config,'attention_dropout',0)!=0 or any(
            isinstance(m,torch.nn.Dropout) and m.p!=0 for m in model.modules())):
        raise ValueError('OLMo activation recomputation requires zero dropout to preserve the experiment')
    logps=getattr(backend,'_logps_batch',None)
    if not callable(logps):raise ValueError('OLMo backend is missing the expected log-probability function')
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    stage=backend.cost_meter.stage
    @contextmanager
    def identified_stage(name):
        try:
            with stage(name):yield
        except RuntimeError as exc:
            if hasattr(exc,'add_note'):exc.add_note(f'OLMo memory context: stage={name}; activation checkpointing enabled')
            raise
    backend.cost_meter.stage=identified_stage
    @functools.wraps(logps)
    def checkpointed_logps(*args,**kwargs):
        previous=model.training;model.train(True)
        try:return logps(*args,**kwargs)
        finally:model.train(previous)
    backend._logps_batch=checkpointed_logps
    last_logits='logits_to_keep' in inspect.signature(base.forward).parameters
    if last_logits:
        generate=model.generate
        @functools.wraps(generate)
        def last_logit_generate(*args,**kwargs):
            kwargs.setdefault('logits_to_keep',1)
            try:return generate(*args,**kwargs)
            except RuntimeError as exc:
                if hasattr(exc,'add_note'):
                    exc.add_note(f'OLMo generation: responses={kwargs.get("num_return_sequences")} '
                                 f'max_new_tokens={kwargs.get("max_new_tokens")}; original draw batch retained')
                raise
        model.generate=last_logit_generate
    receipt=dict(protocol='olmo-zero-dropout-recomputation-v1',activation_checkpointing=True,
                 use_reentrant=False,generation_last_logit=last_logits,
                 response_count_changed=False,generation_microbatch_changed=False,
                 logprob_microbatch_changed=False,attention_kernel_changed=False)
    backend._v8_olmo_memory=receipt
    return receipt


def repair_context(ctx):
    """Apply the seed runs' documented verifier/cache protections after saved-source selection."""
    backend=ctx.backend;changes={};verifier=getattr(backend,'verifier',None)
    fixed,spawn=repair_mbpp_subprocess(verifier)
    if spawn:backend.verifier=fixed;changes['mbpp_subprocess']=spawn
    dense_memory=repair_dense_memory(backend)
    if dense_memory:changes['dense_memory']=dense_memory
    olmo_memory=repair_olmo_memory(backend)
    if olmo_memory:changes['olmo_memory']=olmo_memory
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
    install_native_fault_handler()
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
    install_native_fault_handler()
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
