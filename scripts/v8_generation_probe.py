"""Versioned observer for the original model's decode calls; no sampling or optimizer changes."""
import faulthandler
import functools
import math
import os
from pathlib import Path
import sys
import threading
import time


class DecodeObserver:
    def __init__(self,record,rank,call_id,prompt_tokens,*,interval=5,clock=time.monotonic):
        self.record,self.rank,self.call_id=record,rank,call_id
        self.prompt_tokens=prompt_tokens;self.interval=interval;self.clock=clock
        self.last_tokens=prompt_tokens;self.steps=0;self.segment=0;self.last_report=-math.inf
        self.last_advance=clock();self.closed=False;self.lock=threading.Lock()

    def __call__(self,input_ids,scores,**kwargs):
        tokens=int(input_ids.shape[-1]);now=self.clock();event=None
        with self.lock:
            if tokens<self.last_tokens:self.segment+=1;self.last_tokens=self.prompt_tokens
            if tokens>self.last_tokens:
                self.steps+=tokens-self.last_tokens;self.last_tokens=tokens;self.last_advance=now
                if now-self.last_report>=self.interval:
                    event=dict(generation_call=self.call_id,sequence_tokens=tokens,decode_steps=self.steps,
                               segment=self.segment,progress_kind='returned_decode_steps')
                    self.last_report=now
        # A blocked progress-file write must not hold the watchdog's state mutex.
        if event is not None:
            self.record('generation',**event)
            if self.rank==0:
                print(f'V8 DECODE rank={self.rank} call={self.call_id} sequence_tokens={tokens} '
                      f'decode_steps={event["decode_steps"]} segment={event["segment"]}',flush=True)
        return False


def observed_generate(original,record,rank,*,timeout=1800,interval=5):
    """Append an always-false criterion; retain every existing criterion and generation argument."""
    if not math.isfinite(timeout) or timeout<=0:raise ValueError('V8_GENERATION_STALL_SECONDS must be finite and positive')
    if not math.isfinite(interval) or interval<0:raise ValueError('V8_DECODE_PROGRESS_SECONDS must be finite and nonnegative')
    calls=0
    @functools.wraps(original)
    def generate(*args,**kwargs):
        nonlocal calls
        from transformers import StoppingCriteriaList
        calls+=1
        inputs=kwargs.get('input_ids',kwargs.get('inputs',args[0] if args else None))
        prompt_tokens=int(inputs.shape[-1]) if inputs is not None else 0
        observer=DecodeObserver(record,rank,calls,prompt_tokens,interval=interval)
        criteria=list(kwargs.get('stopping_criteria') or [])
        kwargs=dict(kwargs,stopping_criteria=StoppingCriteriaList([*criteria,observer]))
        record('generation_start',generation_call=calls,prompt_tokens=prompt_tokens,progress_kind='entered_only')
        finished=threading.Event()
        def watch():
            while not finished.wait(min(1,timeout/10)):
                with observer.lock:
                    if observer.closed or time.monotonic()-observer.last_advance<timeout:continue
                    message=(f'V8 GENERATION STALL rank={rank} call={calls}: no returned decode step for {timeout:g}s; '
                             f'last sequence_tokens={observer.last_tokens} decode_steps={observer.steps}; '
                             'rank exits70; existing checkpoints retained')
                    def diagnostics():
                        try:
                            print(message,file=sys.stderr,flush=True)
                            directory=os.environ.get('SRGC_PROGRESS_DIR')
                            if directory:
                                path=Path(directory)/f'generation-stall-rank{rank}.txt'
                                path.parent.mkdir(parents=True,exist_ok=True)
                                with path.open('a') as handle:
                                    handle.write(message+'\n');handle.flush();faulthandler.dump_traceback(file=handle,all_threads=True)
                            else:faulthandler.dump_traceback(all_threads=True)
                        except Exception:pass  # a broken log path cannot disable the deadline
                    writer=threading.Thread(target=diagnostics,daemon=True);writer.start();writer.join(.2)
                    # Fail this attempt, never report an unmeasured completion or retry it in-place.
                    os._exit(70)
        thread=threading.Thread(target=watch,daemon=True);thread.start()
        try:
            result=original(*args,**kwargs)
            with observer.lock:observer.closed=True
            record('generation_finished',generation_call=calls,decode_steps=observer.steps,progress_kind='generate_returned')
            return result
        finally:
            with observer.lock:observer.closed=True
            finished.set();thread.join(timeout=1)
    return generate


def install(rank):
    # All historical children construct Context after source selection, adapters and
    # their backend. Older children have no install_progress hook at all.
    from switching.runner import Context
    cls=Context
    if getattr(cls,'_v8_decode_observer',False):return
    original=cls.__init__
    @functools.wraps(original)
    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs)
        from srgc_rebuttal import progress
        self.backend.model.generate=observed_generate(self.backend.model.generate,progress.record,rank,
            timeout=float(os.environ.get('V8_GENERATION_STALL_SECONDS','1800')),
            interval=float(os.environ.get('V8_DECODE_PROGRESS_SECONDS','5')))
    cls.__init__=initialize;cls._v8_decode_observer=True
