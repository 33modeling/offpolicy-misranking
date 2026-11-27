"""V9 worker: `sh v9/run.sh [stage]` -> python -m v9lib.orchestrate [stage].

No stage: freeze-tau -> x1 -> x2 -> x5 -> x3 -> x4 -> x8 -> x9 -> x10 -> collect. GPU tasks are taken in that priority
order, one at a time; after each finished task the worker starts again from the first stage, so a
newly unlocked high-priority task goes first. Stages with nothing ready are skipped. A failing
stage or task is recorded and the worker continues with the rest. When nothing is ready anywhere
the worker runs collect and exits. The same command on every node: a task is leased (flock) while
it runs; a run resumes from its latest.pt; finished outputs are never redone.

Words: status | freeze-tau | x1 | x2 | x3 | x4 | x5 | x8 | x9 | x10 | collect | retry-failed
"""

from __future__ import annotations

import json
import re
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

from . import env

env.activate()

from .common import Blocked, item, v9c  # noqa: E402

MAX_FAILURES = 2
STOP = {"flag": False, "child": None}
# A worker with nothing ready waits while work can still open up (another node runs a V9 task, or a live V8 worker
# is producing the V8 results a V9 task waits for), instead of exiting and leaving the opened task to nobody.
WAIT_SECONDS = float(os.environ.get("V9_WAIT_SECONDS", "300"))
BEAT_SECONDS = 60
LIVE_SECONDS = 300          # a heartbeat older than this is not a live worker
V8_LIVE_SECONDS = 900       # V8 worker heartbeat age that still counts as "V8 is running"
# When nothing is ready or running anywhere and only stopped tasks remain, each stopped task restarts from its
# checkpoint with a fresh failure count, at most AUTO_RETRIES times (V8 idle-retry grant, ead4399).
AUTO_RETRIES = int(os.environ.get("V9_AUTO_RETRIES", "3"))


# ------------------------------------------------------------------ heartbeat
def workers_dir():
    return env.v9_root() / ".state" / "workers"


class Beat:
    """`.state/workers/<host>-<pid>.json`, rewritten every BEAT_SECONDS: what this worker does now."""

    def __init__(self):
        self.path = workers_dir() / f"{socket.gethostname()}-{os.getpid()}.json"
        self.rec = {"host": socket.gethostname(), "pid": os.getpid(), "started": time.time(), "state": "starting",
                    "doing": "starting"}
        self.stop = threading.Event()
        self.thread = None

    def set(self, **fields):
        for k in ("task_id", "stage", "cohort", "key", "task_started"):
            self.rec.pop(k, None)
        self.rec.update(fields)
        self.write()

    def write(self):
        from switching.common import atomic_write_json
        try:
            self.rec.update(at=time.time(), log=str(say.path) if say.path else None)
            atomic_write_json(self.path, self.rec)
        except OSError:
            pass   # a status aid only; never stops the worker

    def start(self):
        self.write()

        def loop():
            while not self.stop.wait(BEAT_SECONDS):
                self.write()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


BEAT = {"beat": None}


def beat(**fields):
    if BEAT["beat"] is not None:
        BEAT["beat"].set(**fields)


def live_workers(now=None):
    """Heartbeat records of live V9 workers (this one included)."""
    now = time.time() if now is None else now
    out = []
    d = workers_dir()
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if now - float(r.get("at", 0)) < LIVE_SECONDS:
            out.append(r)
    return out


def others_working():
    me = (socket.gethostname(), os.getpid())
    return [r for r in live_workers() if r.get("state") == "working" and (r.get("host"), r.get("pid")) != me]


def v8_live():
    """Cohorts whose V8 queue has a worker heartbeat younger than V8_LIVE_SECONDS."""
    out = []
    now = time.time()
    for d in sorted(env.v8_root().glob("*/.multi-node/workers")) if env.v8_root().is_dir() else []:
        try:
            if any(now - p.stat().st_mtime < V8_LIVE_SECONDS for p in d.glob("*.json")):
                out.append(d.parent.parent.name)
        except OSError:
            continue
    return out


def done_times_dir():
    return env.v9_root() / ".state" / "done-times"


def record_done_time(stage, it, tid, seconds):
    """Wall time of a finished task (the status estimates time left from these)."""
    from switching.common import atomic_write_json
    try:
        atomic_write_json(done_times_dir() / f"{tid}.json", {"task_id": tid, "stage": stage, "cohort": it["cohort"],
                                                             "key": it.get("key"), "seconds": round(seconds, 1),
                                                             "host": socket.gethostname(), "at": time.time()})
    except OSError:
        pass


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
    # a cleared record keeps its history_count and auto_retries (else the restart limit would never be reached)
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


def auto_retry(items):
    """Give each stopped task (failed MAX_FAILURES times) one more fresh failure budget, at most AUTO_RETRIES times
    per task. Only called when nothing is ready or running anywhere. Returns the task ids granted."""
    from switching.common import atomic_write_json, utc_now
    if AUTO_RETRIES <= 0:
        return []
    granted = []
    for it in items:
        if it["state"] != "failed" or not it.get("task"):
            continue
        tid = task_id(it["task"])
        f = failures(tid)
        grants = f.setdefault("auto_retries", [])
        if len(grants) >= AUTO_RETRIES or f.get("count", 0) < MAX_FAILURES:
            continue
        grants.append({"at": utc_now(), "host": socket.gethostname(), "count_before": f["count"]})
        f["history_count"] = f.get("history_count", 0) + f["count"]
        f["count"], f["cleared_at"] = 0, utc_now()
        atomic_write_json(failure_file(tid), f)
        granted.append(tid)
    return granted


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
                left = max(0, AUTO_RETRIES - len(f.get("auto_retries", [])))
                it.update(state="failed", reason=(f"failed {f['count']} times · restarts by itself from its checkpoint when "
                                                  f"no other V9 work is left ({left} automatic restart(s) left)" if left else
                                                  f"failed {f['count']} times, no more automatic retries")
                          + f" · last error: {f['errors'][-1]['error'][:400]}")
            elif f.get("count", 0) > 0 and f.get("errors"):
                it["retry_note"] = (f"failed {f['count']} time(s); the next free node retries it from its checkpoint "
                                    f"(attempt {f['count'] + 1} of {MAX_FAILURES}) · last error: "
                                    f"{f['errors'][-1]['error'][:400]}")
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
            t0 = time.time()
            beat(state="working", doing="running a task", task_id=tid, stage=stage, cohort=cohort, key=it.get("key"),
                 task_started=t0)
            try:
                rc = _run_child(stage, cohort, tid, tfile)
                if rc == 0:
                    record_done_time(stage, it, tid, time.time() - t0)
                return rc
            finally:
                record.unlink(missing_ok=True)
                beat(state="choosing", doing="choosing the next task")
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
    token = f"{socket.gethostname()}-{os.getpid()}-{time.time_ns()}"
    child_env = dict(os.environ, V8_EXIT_FILE=str(exit_file), PYTHONUNBUFFERED="1", V8_FAILURE_TOKEN=token,
                     SRGC_PROGRESS_DIR=str(env.v9_root() / ".state" / "progress" / tid),
                     CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"))
    say(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {socket.gethostname()} stage={stage} cohort={cohort} task={tid}")
    with open(say.path or log_path(), "a") as fh:
        proc = subprocess.Popen(cmd, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        STOP["child"] = proc
        crashes, saved = [], None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
            m = RANK_SIGNAL.search(line)
            if m:
                crashes.append(m.group(2))
            m = SAVED_UPDATE.search(line)
            if m:
                saved = int(m.group(1))
        rc = proc.wait()
        STOP["child"] = None
    child_code = None
    if rc != 0 and exit_file.is_file():
        try:
            child_code = json.loads(exit_file.read_text()).get("exit_code")
        except ValueError:
            pass
    exit_file.unlink(missing_ok=True)
    rc, CRASH["why"] = classify_exit(rc, child_code, crashes, STOP["flag"])
    if CRASH["why"]:
        where = crash_place(env.v9_root() / ".state" / "progress" / tid, token)
        CRASH["why"] += (f"; on {socket.gethostname()}" + (f", last saved update {saved}" if saved is not None else "")
                         + (f"; {where}" if where else "; no fault dump was written"))
    return rc


SAVED_UPDATE = re.compile(r"resumes from saved update (\d+)")


def crash_place(directory, token):
    """'rank N <Fatal Python error kind> in f (file:line) <- ...' from this attempt's native fault logs
    (v8_task_entry.install_native_fault_handler files, read with V8 status's parser), else None."""
    import hashlib
    key = hashlib.sha256(token.encode()).hexdigest()[:24]
    try:
        sys.path.insert(0, str(env.REPO / "scripts"))
        from v8_status import fault_text, native_crash_frames
    except Exception:  # noqa: BLE001 - no parser: say nothing rather than guess
        return None
    for path in sorted(Path(directory).glob(f"native-fault-rank-*-{key}.log")):
        dump = fault_text(path)
        if dump and "Fatal Python error" in dump:
            kind = re.search(r"Fatal Python error: ([A-Za-z ]+)", dump)
            frames = native_crash_frames(dump)
            rank = path.name.split("-")[3]
            return (f"rank {rank} {kind.group(1).strip() if kind else 'native fault'} in "
                    + (" <- ".join(frames) if frames else "a thread with no Python frame (CUDA/NCCL/native library)")
                    + f" ({path})")
    return None


# torchrun's failure summary: "traceback : Signal 11 (SIGSEGV) received by PID 620590"
RANK_SIGNAL = re.compile(r"Signal (\d+) \((SIG[A-Z0-9]+)\) received by PID")
STOP_SIGNALS = ("SIGTERM", "SIGINT", "SIGHUP")
CRASH = {"why": None}


def classify_exit(rc, child_code, crashes, stopping):
    """(rc, reason). A rank that died by its own signal (SIGSEGV, SIGABRT, SIGKILL from the OOM killer, ...)
    is a failure even when rank 0 then wrote 75: torchrun stops the other ranks with SIGTERM after the first
    crash, and rank 0 reports that SIGTERM as a resumable interruption. Without this the task never counts a
    failure and a node can crash on it again and again. A stop this worker asked for, or a SIGTERM/SIGINT/
    SIGHUP from outside (pod shutdown), stays resumable (75)."""
    own = [c for c in crashes if c not in STOP_SIGNALS]
    if rc == 0:
        return 0, None
    if own and not stopping:
        sig = own[0]
        num = getattr(signal, sig, None)
        return 128 + int(num) if num is not None else 1, (
            f"a GPU process crashed with {sig}; torchrun then stopped the other ranks (resumes from its last "
            f"saved update on the next attempt)")
    if child_code == 75:
        return 75, None
    return rc, None


def failure_text(rc):
    why = CRASH.get("why")
    return f"exit code {rc}" + (f": {why}" if why else "")


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
GPU_STAGES = ("x1", "x2", "x5", "x3", "x4", "x8", "x9", "x10")


def wait_reason(items):
    """Why this worker should wait instead of exiting (None: nothing can open up)."""
    # ready items reach here only when this worker already tried them (busy elsewhere or failed once)
    blocked = [it for it in items if it["state"] in ("blocked", "ready")]
    if not blocked:
        return None
    others = others_working()
    if others:
        return (f"{len(blocked)} task(s) cannot start yet; {len(others)} other node(s) run V9 tasks: "
                + ", ".join(sorted({o['host'] for o in others})[:3]))
    v8 = [it for it in blocked if "V8" in (it.get("reason") or "")]
    live = v8_live()
    if v8 and live:
        return f"{len(v8)} task(s) wait for V8 results; V8 workers are live on {', '.join(live[:4])}"
    return None


def nap(seconds):
    end = time.time() + seconds
    while not STOP["flag"] and time.time() < end:
        time.sleep(min(5.0, max(0.0, end - time.time())))


def work(stages):
    attempted, failed = set(), []
    progressed_once = False
    last_wait, passes, rounds = None, 0, -1
    while not STOP["flag"]:
        rounds += 1
        progressed = False
        seen, late = [], []
        beat(state="choosing", doing="choosing the next task")
        for stage in stages:
            if STOP["flag"]:
                break
            if stage == "freeze-tau":
                try:
                    run_freeze_tau(quiet=progressed_once or rounds > 0)
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
            seen += items
            ready = [it for it in items if it["state"] == "ready" and it.get("task") and
                     task_id(it["task"]) not in attempted]
            late += [(stage, it) for it in ready if it.get("late")]
            ready = [it for it in ready if not it.get("late")]
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
                record_failure(tid, failure_text(rc))
                failed.append(f"{stage}:{tid}")
                say(f"{stage}: task {tid} FAILED rc={rc}{(' (' + CRASH['why'] + ')') if CRASH.get('why') else ''} (attempt kept; rerunning resumes)")
            if progressed:
                break
        # late tasks (X1 Random) only when no other stage had a ready task this round
        for stage, it in ([] if progressed or STOP["flag"] else late):
            if STOP["flag"]:
                break
            tid = task_id(it["task"])
            if tid in attempted:
                continue
            attempted.add(tid)
            rc = launch(stage, it)
            if rc == 0:
                progressed = progressed_once = True
                break
            if rc == 75 or STOP["flag"]:
                say(f"{stage}: {tid} busy on another node or interrupted (resumable)")
                continue
            record_failure(tid, failure_text(rc))
            failed.append(f"{stage}:{tid}")
            say(f"{stage}: task {tid} FAILED rc={rc}{(' (' + CRASH['why'] + ')') if CRASH.get('why') else ''} (attempt kept; rerunning resumes)")
        if progressed or STOP["flag"]:
            last_wait, passes = None, 0
            continue
        why = wait_reason(seen)
        if why is None and passes < MAX_FAILURES and any(it["state"] == "ready" for it in seen):
            passes += 1          # a task that failed once gets its second try now (bounded: no endless loop)
            attempted.clear()
            continue
        if why is None and not others_working() and not any(it["state"] == "ready" for it in seen):
            granted = auto_retry(seen)
            if granted:
                say(f"nothing else is left; {len(granted)} stopped task(s) restart from their checkpoints "
                    f"(automatic restart, at most {AUTO_RETRIES} per task): {', '.join(granted[:6])}")
                attempted.clear()
                passes = 0
                continue
        if why is None:
            break
        if why != last_wait:
            say(f"waiting (checks again every {int(WAIT_SECONDS)} s): {why}")
            last_wait = why
        beat(state="waiting", doing=why)
        nap(WAIT_SECONDS)
        attempted.clear()   # busy or failed-once tasks may be ready again
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
        from .status_view import main as status_main
        return status_main(plan, task_id, argv[1:])
    if word == "retry-failed":
        retry_failed()
        return 0
    if word not in ("all", *env.STAGES):
        print(f"unknown stage {word!r}; use one of: status {' '.join(env.STAGES)} retry-failed", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    stages = list(env.STAGES) if word == "all" else [word]
    BEAT["beat"] = Beat()
    BEAT["beat"].start()
    try:
        return _main_work(word, stages)
    finally:
        BEAT["beat"].close()
        BEAT["beat"] = None


def _main_work(word, stages):
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
