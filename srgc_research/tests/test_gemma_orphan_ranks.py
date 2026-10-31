"""Real local processes: orphaned wrapped Gemma IT ranks are reaped, owned ones are kept."""

import os
import subprocess
import sys
import time

from scripts import srgc_process_guard as guard
from srgc_research.dispatch import gemma_it_run

RANK = ["-c", "import time; time.sleep(120)", "-m", "srgc_research.dispatch.rank_runtime", "--model", "gemma4_it",
        "--entry-module", "srgc_research.dispatch.reward_curve_rank", "--", "--model", "gemma4_it", "--stage", "train"]


def orphan():
    """A rank whose launcher has exited: reparented away from any owner."""
    launcher = subprocess.Popen([sys.executable, "-c", (
        "import subprocess, sys; p = subprocess.Popen(sys.argv[1:], start_new_session=True, "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print(p.pid)"),
        sys.executable, *RANK], stdout=subprocess.PIPE, text=True)
    pid = int(launcher.communicate()[0])
    return pid


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = open(f"/proc/{pid}/stat").read()
    return stat.rsplit(") ", 1)[1].split()[0] != "Z"


def test_wrapped_training_rank_was_invisible_to_the_pinned_reaper_and_is_now_reaped(monkeypatch):
    pid = orphan()
    owned = subprocess.Popen([sys.executable, "-c", (
        "import subprocess, sys, time; subprocess.Popen(sys.argv[1:]); time.sleep(120)"),
        sys.executable, *RANK, "srgc_research.dispatch.gemma_it_run"])
    try:
        time.sleep(1)
        monkeypatch.setattr(guard, "TARGET_MARKERS", ("srgc_research.dispatch.gemma4_it.rank",))
        monkeypatch.setattr(guard, "OWNER_MARKERS", (*guard.OWNER_MARKERS, "srgc_research.dispatch.gemma_it_run"))
        assert pid not in guard.orphan_pids()  # the bug: the wrapped rank is not a target
        monkeypatch.setattr(guard, "is_target", gemma_it_run.wrapped_rank_target(guard.is_target))
        found = guard.orphan_pids()
        assert pid in found
        # A rank whose launcher (owner) is alive is never reaped.
        owned_children = [child for child in guard.descendants(owned.pid)]
        assert owned_children and not set(owned_children) & set(found)
        guard.terminate([pid], grace=5, label="orphan")
        assert not alive(pid)
    finally:
        for child in guard.descendants(owned.pid):
            try:
                os.kill(child, 9)
            except ProcessLookupError:
                pass
        owned.kill()
        owned.wait()
        if alive(pid):
            os.kill(pid, 9)
