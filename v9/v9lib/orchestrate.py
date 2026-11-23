"""V9 worker: `sh v9/run.sh [stage]` -> python -m v9lib.orchestrate [stage].

No stage: freeze-tau -> x1 -> x2 -> x5 -> x3 -> x4 -> collect. GPU tasks are taken in that priority
order, one at a time; after each finished task the worker starts again from the first stage, so a
newly unlocked high-priority task goes first. Stages with nothing ready are skipped. A failing
stage or task is recorded and the worker continues with the rest. When nothing is ready anywhere
the worker runs collect and exits. The same command on every node: a task is leased (flock) while
it runs; a run resumes from its latest.pt; finished outputs are never redone.

Words: status | freeze-tau | x1 | x2 | x3 | x4 | x5 | collect | retry-failed
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from . import env

env.activate()

from .common import Blocked, item, v9c  # noqa: E402

MAX_FAILURES = 2
STOP = {"flag": False, "child": None}


def log_path():
    d = env.v9_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"worker-{socket.gethostname()}-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}.log"


class Out:
    def __init__(self):
        self.path = None

    def __call__(self, text):
        line = f"[v9] {text}"
        print(line, flush=True)
        if self.path is None:
            self.path = log_path()
        with open(self.path, "a") as fh:
            fh.write(line + "\n")


say = Out()


# ------------------------------------------------------------------ failures
def failure_file(task_id):
    return env.v9_root() / ".state" / "failures" / f"{task_id}.json"


def failures(task_id):
    p = failure_file(task_id)
    if not p.is_file():
        return {"count": 0, "errors": []}
    return json.loads(p.read_text())


def record_failure(task_id, error):
    from switching.common import atomic_write_json, utc_now
    f = failures(task_id)
    if f.get("cleared_at") and not f.get("count"):
        f = {"count": 0, "errors": f.get("errors", [])}
    f["count"] = int(f.get("count", 0)) + 1
    f.setdefault("errors", []).append({"at": utc_now(), "host": socket.gethostname(), "error": str(error)[:1500]})
    atomic_write_json(failure_file(task_id), f)


def record_stage_failure(stage, error):
    from switching.common import JsonlLog, utc_now
    JsonlLog(env.v9_root() / ".state" / "stage-failures.jsonl", ("at", "host", "stage")).append(
        {"at": utc_now(), "host": socket.gethostname(), "stage": stage, "error": str(error)[:2000]})


def retry_failed():
    from switching.common import atomic_write_json, utc_now
    d = env.v9_root() / ".state" / "failures"
    n = 0
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        f = json.loads(p.read_text())
        if f.get("count"):
            f["history_count"] = f.get("history_count", 0) + f["count"]
            f["count"], f["cleared_at"] = 0, utc_now()
            atomic_write_json(p, f)
            n += 1
    say(f"retry-failed: cleared the failure count of {n} task(s); history kept")


# ------------------------------------------------------------------ planning
def plan(stage):
    from . import stages, tau
    if stage == "freeze-tau":
        return tau.status_items()
    if stage == "collect":
        from . import collect
        return collect.status_items()
    items = stages.PLANNERS[stage]()
    for it in items:
        if it["state"] == "ready" and it.get("task"):
            tid = task_id(it["task"])
            f = failures(tid)
            if f.get("count", 0) >= MAX_FAILURES:
                it.update(state="failed", reason=f"failed {f['count']} times, no more automatic retries · last error: "
                          f"{f['errors'][-1]['error'][:160]}")
    return items


def task_id(task):
    return task.get("task_id") or task["run_id"]


# ------------------------------------------------------------------ launching one GPU task
def launch(stage, it):
    from switching.common import Busy, atomic_write_json, file_lock
    cohort, task = it["cohort"], it["task"]
    tid = task_id(task)
    tfile = v9c(cohort) / "matrix" / stage / f"{tid}.json"
    atomic_write_json(tfile, {"stage": stage, "cohort": cohort, "tasks": [task]})
    lease = v9c(cohort) / "leases" / f"{tid}.lock"
    try:
        with file_lock(lease):
            from .status_view import running_dir
            if say.path is None:
                say.path = log_path()   # the record names the log this task's output goes to
            record = running_dir() / f"{tid}.json"
            atomic_write_json(record, {"task_id": tid, "stage": stage, "cohort": cohort, "key": it.get("key"),
                                       "host": socket.gethostname(), "pid": os.getpid(), "started_at": time.time(),
                                       "log": str(say.path)})
            try:
                return _run_child(stage, cohort, tid, tfile)
            finally:
                record.unlink(missing_ok=True)
    except Busy:
        return 75


def _run_child(stage, cohort, tid, tfile):
    py = env.cohort_python(cohort, sys.executable)
    exit_file = env.v9_root() / ".state" / "exit" / f"{socket.gethostname()}-{os.getpid()}.json"
    exit_file.parent.mkdir(parents=True, exist_ok=True)
    exit_file.unlink(missing_ok=True)
    nproc = os.environ.get("V9_NPROC", "4")
    cmd = [py, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", "--max_restarts=0",
           str(env.V9_DIR / "child.py"), "--cohort", cohort, "--task-file", str(tfile), "--root", str(env.v9_root())]
    child_env = dict(os.environ, V8_EXIT_FILE=str(exit_file), PYTHONUNBUFFERED="1",
                     CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    say(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {socket.gethostname()} stage={stage} cohort={cohort} task={tid}")
    with open(say.path or log_path(), "a") as fh:
        proc = subprocess.Popen(cmd, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        STOP["child"] = proc
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        rc = proc.wait()
        STOP["child"] = None
    if rc != 0 and exit_file.is_file():
        try:
            if json.loads(exit_file.read_text()).get("exit_code") == 75:
                rc = 75
        except ValueError:
            pass
    exit_file.unlink(missing_ok=True)
    return rc


def _on_signal(signum, frame):  # noqa: ARG001
    STOP["flag"] = True
    child = STOP.get("child")
    if child is not None and child.poll() is None:
        try:
            child.send_signal(signal.SIGTERM)
        except OSError:
            pass


# ------------------------------------------------------------------ CPU stages
def run_freeze_tau(quiet=False):
    from . import tau
    try:
        rec, how = tau.freeze()
        if quiet and how == "reused":
            return True
        say(f"freeze-tau: {how} tau={rec['tau']} (basis {rec['basis']}, best mean {100 * rec['best_mean_reward_fraction']:.2f}%) "
            f"-> {env.v9_root() / 'frozen' / 'tau.json'}")
        return True
    except Blocked as exc:
        say(f"freeze-tau: not ready: {exc}")
        return False


def run_collect():
    from . import collect
    try:
        res = collect.collect()
        say(f"collect: wrote {res['written']} (measured cells {res['measured']}, pending cells {res['pending']})")
        return True
    except Blocked as exc:
        say(f"collect: not ready: {exc}")
        return False


# ------------------------------------------------------------------ worker loop
GPU_STAGES = ("x1", "x2", "x5", "x3", "x4")


def work(stages):
    attempted, failed = set(), []
    progressed_once = False
    while not STOP["flag"]:
        progressed = False
        for stage in stages:
            if STOP["flag"]:
                break
            if stage == "freeze-tau":
                try:
                    run_freeze_tau(quiet=progressed_once)
                except Exception as exc:  # noqa: BLE001
                    record_stage_failure(stage, f"{type(exc).__name__}: {exc}")
                    failed.append(stage)
                    say(f"freeze-tau FAILED: {type(exc).__name__}: {exc}")
                continue
            if stage == "collect":
                continue
            try:
                items = plan(stage)
            except Exception as exc:  # noqa: BLE001 - this stage only
                record_stage_failure(stage, f"{type(exc).__name__}: {exc}")
                failed.append(stage)
                say(f"{stage} planning FAILED (stage skipped): {type(exc).__name__}: {exc}")
                continue
            ready = [it for it in items if it["state"] == "ready" and it.get("task") and
                     task_id(it["task"]) not in attempted]
            for it in ready:
                if STOP["flag"]:
                    break
                tid = task_id(it["task"])
                attempted.add(tid)
                rc = launch(stage, it)
                if rc == 0:
                    progressed = progressed_once = True
                    break
                if rc == 75 or STOP["flag"]:
                    say(f"{stage}: {tid} busy on another node or interrupted (resumable)")
                    continue
                record_failure(tid, f"exit code {rc}")
                failed.append(f"{stage}:{tid}")
                say(f"{stage}: task {tid} FAILED rc={rc} (attempt kept; rerunning resumes)")
            if progressed:
                break
        if not progressed:
            break
    if "collect" in stages and not STOP["flag"]:
        try:
            run_collect()
        except Exception as exc:  # noqa: BLE001
            record_stage_failure("collect", f"{type(exc).__name__}: {exc}")
            failed.append("collect")
            say(f"collect FAILED: {type(exc).__name__}: {exc}")
    return failed


# ------------------------------------------------------------------ status
def status():
    from .status_view import render
    render(plan, task_id)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    word = argv[0] if argv else "all"
    if word in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if word == "status":
        status()
        return 0
    if word == "retry-failed":
        retry_failed()
        return 0
    if word not in ("all", *env.STAGES):
        print(f"unknown stage {word!r}; use one of: status {' '.join(env.STAGES)} retry-failed", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    stages = list(env.STAGES) if word == "all" else [word]
    say(f"worker start host={socket.gethostname()} stages={','.join(stages)} v8_root={env.v8_root()} "
        f"v9_root={env.v9_root()} code_root={env.code_root()}")
    if word == "freeze-tau":
        ok = run_freeze_tau()
        return 0 if ok else 3
    if word == "collect":
        return 0 if run_collect() else 3
    failed = work(stages)
    if STOP["flag"]:
        say("stopped by signal; running tasks resume from their latest.pt on the next start")
        return 75
    say("nothing more is ready on this node" + (f"; failures this run: {sorted(set(failed))}" if failed else ""))
    say("status:")
    status()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
