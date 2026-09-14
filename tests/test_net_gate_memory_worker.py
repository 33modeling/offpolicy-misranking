import os
import sys

import pytest

import net_gate_memory_worker as memory


def test_failed_shard_does_not_kill_healthy_siblings(tmp_path):
    scripts = ["import sys; sys.exit(2)"] + [
        f"import time; from pathlib import Path; time.sleep(.25); Path({str(tmp_path / f'done-{i}')!r}).touch()"
        for i in range(1, 4)]
    commands = [[sys.executable, "-c", script] for script in scripts]
    logs = [tmp_path / f"shard-{i}.log" for i in range(4)]
    codes = memory.drain_workers((commands, list("0123")), logs, os.environ)
    assert codes == [2, 0, 0, 0]
    assert all((tmp_path / f"done-{i}").exists() for i in range(1, 4))


def test_parent_deadline_still_stops_supervisor_and_children(tmp_path):
    import selection_gate_gpu as base
    script = '''
import os, signal, sys
from pathlib import Path
from net_gate_memory_worker import drain_workers
def stop(signum, frame):
    raise SystemExit(128 + signum)
signal.signal(signal.SIGTERM, stop)
root = Path(sys.argv[1])
commands = [[sys.executable, "-c", "import os,time; from pathlib import Path; Path("+repr(str(root/f"pid-{i}"))+").write_text(str(os.getpid())); time.sleep(60)"] for i in range(4)]
drain_workers((commands, list("0123")), [root/f"worker-{i}.log" for i in range(4)], os.environ)
'''
    with pytest.raises(TimeoutError):
        base.meter(tmp_path, "autograd-score", "H100", commands=[([sys.executable, "-c", script, str(tmp_path)], "0,1,2,3")],
                   timeout=1.5, env={**os.environ, "PYTHONPATH": str(memory.HERE.parent)}, ledger="deployment")
    pids = list(tmp_path.glob("pid-*"))
    assert len(pids) == 4
    for path in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(int(path.read_text()), 0)
    assert base.cost(tmp_path)["complete"] and base.spent(tmp_path) > 0
