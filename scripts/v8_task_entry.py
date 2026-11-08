"""Run the exact original v8 child with a separately receipted decode observer."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from v8_generation_probe import install


def main():
    path=Path(sys.argv[1]);expected=os.environ.get('V8_ORIGINAL_CHILD_SHA256')
    if not expected or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Original v8 child bytes differ from the selected runner identity')
    spec=importlib.util.spec_from_file_location('_v8_original_child',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if hasattr(module,'install_progress'):module.install_progress=lambda *args,**kwargs:None
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
