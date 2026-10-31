"""Yield peer-owned startup locks to the coordinator without changing model code."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from srgc_rebuttal.runtime import Busy


def lock_stage(path):
    path = Path(path)
    if path.name.endswith('.download.lock'):
        return 'model_files'
    if path.suffix == '.lock' and path.parent.parent.name == 'gemma-runtime-cache':
        return {'packages': 'packages', 'verified-models': 'model_verify'}.get(path.parent.name)
    return None


def retry_lease(original):
    @contextmanager
    def acquire(path, *, wait=False):
        from . import startup
        from .model_chain import BUSY, enabled

        row = startup.CURRENT.get()
        stage = lock_stage(path)
        retry = (wait and stage is not None and enabled() and row is not None
                 and row.row['status'] == 'preparing')
        if not retry:
            with original(path, wait=wait) as handle:
                yield handle
            return
        startup.progress(stage)
        with ExitStack() as locks:
            try:
                handle = locks.enter_context(original(path, wait=False))
            except Busy:
                row.update(waiting_lock=str(Path(path)))
                print(f'GEMMA startup lock owned by peer: {path}; yielding before GPU admission; '
                      'coordinator will retry the same model', flush=True)
                raise SystemExit(BUSY) from None
            # Only acquisition failure is a peer wait; exceptions inside the
            # protected model/package verification retain their original cause.
            yield handle
    return acquire


@contextmanager
def retrying_startup_locks():
    from .gemma4 import runtime
    from .gemma4_it import model

    # The IT runtime re-exports these functions; their globals live in gemma4.
    # Patch only this launcher's module bindings, never the shared runtime or
    # the on-disk frozen adapter, and restore them before returning.
    with patch.object(model, 'lease', retry_lease(model.lease)), \
            patch.object(runtime, 'lease', retry_lease(runtime.lease)):
        yield


@contextmanager
def retrying_session(root, *, reset=False):
    """A chain join must not block before it can even register startup."""
    from . import startup
    from .gemma4_it import reset as reset_module
    from .model_chain import BUSY, enabled

    if reset or not enabled():
        with reset_module.session(root, reset=reset):
            yield
        return

    original = reset_module.fcntl

    class SharedLease:
        def __getattr__(self, name):
            return getattr(original, name)

        @staticmethod
        def flock(handle, operation):
            if operation != original.LOCK_SH:
                return original.flock(handle, operation)
            try:
                return original.flock(handle, operation | original.LOCK_NB)
            except BlockingIOError:
                # The exclusive reset owner and its inode remain untouched.
                with startup.registration(root):
                    startup.progress('session')
                    startup.CURRENT.get().update(waiting_lock=handle.name)
                    print('GEMMA session owned by reset; yielding before startup; '
                          'coordinator will retry the same model', flush=True)
                    raise SystemExit(BUSY) from None

    # Use a module-local proxy; do not change the process-wide fcntl module.
    with patch.object(reset_module, 'fcntl', SharedLease()), reset_module.session(root):
        yield
