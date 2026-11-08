"""Run lifecycle (plan Code §6): one continuation from a parent state, resumable and idempotent.

runs/<run_id>/
  manifest.json            identity, hashes, parent, status (initialized -> active -> completed_* | failed | interrupted)
  attempts/<attempt>/events.jsonl   cost ledger events of that attempt (physical expenditure)
  attempts/<attempt>/phase-starts.jsonl  durable phase starts (a start without an event = killed mid-phase)
  attempts/<attempt>/updates.jsonl  per-update records as they happened (append-only)
  latest.pt                mutable resume checkpoint (v8 payload)
  snapshots/step-XXXXXX.pt immutable retained snapshots (branch points / budget endpoints)
  snapshots.jsonl          index: completed_update, sha256, state identity, cumulative own cost
  history.jsonl, decisions.jsonl, signals.jsonl, training_stats.jsonl, cache_log.jsonl
                           canonical lineage written at completion (one record per key)

Distributed contract (cluster child, one process per GPU): only rank 0 writes; every collective
(ctx.primary / ctx.broadcast) is reached by every rank in the same order. Error and stop paths
contain no collective, so a rank-local fault ends every rank promptly (torchrun stops the peers)
with the original error, as in the frozen trainer.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal as _signal
import socket
import sys
import threading
import time

from . import snapshots
from .budget import should_continue
from .common import (Busy, JsonlLog, atomic_write_json, atomic_write_text, canonical_json, file_lock, new_id,
                     read_json, read_jsonl, sha256_file, utc_now, write_once_json)
from .engine import fork_arm_for
from .ledger import REQUIRED, Ledger, component_totals, lineage_events
from .policies import build_policy

TERMINAL = ("completed_horizon", "completed_budget")
EXIT_BUSY_OR_INTERRUPTED = 75


class ConfigChanged(RuntimeError):
    pass


class Interrupted(RuntimeError):
    pass


class StopRequested(BaseException):
    """Raised by the SIGTERM/SIGINT handler outside critical sections.

    BaseException so that no `except Exception` (verifier, cache policy, frozen code) can swallow it.
    torchrun SIGKILLs its workers 30 s after it is signalled (the v8 supervisor after 10 s), while an
    update takes minutes: the in-flight update is abandoned and the run resumes from latest.pt.
    """


# Process-wide stop state of this child. `critical` > 0 defers the stop until the section ends
# (checkpoint, snapshot+index and manifest writes run to completion).
_STOP = {"requested": False, "raised": False, "critical": 0}
# The run in progress, for the stop watchdog: manifest path, run id, update held by latest.pt.
ACTIVE = {}
_MANIFEST_LOCK = threading.RLock()
_WATCHDOG = {}  # keeps the wakeup socketpair alive for the life of the process


def _on_stop_signal(signum, frame):  # noqa: ARG001 - signal handler signature
    _STOP["requested"] = True
    _STOP.setdefault("signal", signum)
    if _STOP["critical"] == 0 and not _STOP["raised"]:
        _STOP["raised"] = True
        raise StopRequested(_signal.Signals(signum).name)


def install_stop_handlers():
    """SIGTERM/SIGINT raise StopRequested (deferred inside critical sections). Returns previous handlers."""
    previous = {}
    for sig in (_signal.SIGTERM, _signal.SIGINT):
        try:
            previous[sig] = _signal.signal(sig, _on_stop_signal)
        except ValueError:  # not the main thread
            pass
    return previous


def restore_handlers(previous):
    for sig, handler in previous.items():
        try:
            _signal.signal(sig, handler)
        except (ValueError, TypeError):
            pass


@contextmanager
def critical():
    _STOP["critical"] += 1
    try:
        yield
    finally:
        _STOP["critical"] -= 1
    if _STOP["critical"] == 0 and _STOP["requested"] and not _STOP["raised"]:
        _STOP["raised"] = True
        raise StopRequested(_signal.Signals(_STOP.get("signal", _signal.SIGTERM)).name + " (deferred)")


def exit_marker_path():
    """Where the child reports 'busy'/'interrupted' (exit 75): torchrun turns every nonzero worker exit
    into its own exit 1, so the code alone never reaches the launcher."""
    explicit = os.environ.get("V8_EXIT_FILE")
    if explicit:
        return Path(explicit)
    progress = os.environ.get("SRGC_PROGRESS_DIR")
    return Path(progress) / "exit.json" if progress else None


def write_exit_marker(code, status, **details):
    path = exit_marker_path()
    if path is None:
        return None
    atomic_write_text(path, canonical_json({"exit_code": int(code), "status": status, "written_at": utc_now(),
                                            "pid": os.getpid(), **details}) + "\n")
    return path


def mark_interrupted(manifest_path, saved_update, reason, *, timeout=-1):
    """Record a stop at the update latest.pt holds (never a terminal run). No collective."""
    if not _MANIFEST_LOCK.acquire(timeout=timeout):
        return False
    try:
        m = read_json(manifest_path)
        if m.get("status") in TERMINAL:
            return False
        m.update(status="interrupted", last_completed_update=saved_update, resume_from_update=saved_update,
                 updated_at=utc_now())
        m.setdefault("interruptions", []).append({"at": utc_now(), "reason": str(reason)[:500],
                                                  "resume_from_update": saved_update})
        atomic_write_json(manifest_path, m)
        return True
    finally:
        _MANIFEST_LOCK.release()


def start_stop_watchdog(*, is_primary, grace=None, exit_code=EXIT_BUSY_OR_INTERRUPTED):
    """Bound the stop when the main thread cannot run its handler (blocked in a collective or I/O).

    The C-level signal handler writes the signal number to the wakeup fd immediately, whichever
    thread is running; this daemon thread then waits `grace` seconds (default V8_STOP_GRACE_SECONDS=7,
    below the supervisor's 10 s and torchrun's 30 s SIGKILL) for the normal exit, records the
    interruption on rank 0 and exits with `exit_code`. Call from the main thread after the handlers.
    """
    grace = float(os.environ.get("V8_STOP_GRACE_SECONDS", "7")) if grace is None else float(grace)
    reader, writer = socket.socketpair()
    writer.setblocking(False)
    _signal.set_wakeup_fd(writer.fileno(), warn_on_full_buffer=False)
    wanted = {int(_signal.SIGTERM), int(_signal.SIGINT)}

    def watch():
        try:
            while True:
                data = reader.recv(256)
                if not data:
                    return
                if wanted & set(data):
                    break
        except OSError:
            return
        time.sleep(max(0.0, grace))
        reason = f"stop signal: main thread did not exit within {grace:g}s (blocked in a collective or I/O)"
        try:
            if is_primary:
                if ACTIVE.get("manifest"):
                    mark_interrupted(ACTIVE["manifest"], ACTIVE.get("saved_update"), reason, timeout=2)
                write_exit_marker(exit_code, "interrupted", detail=reason, run_id=ACTIVE.get("run_id"),
                                  resume_from_update=ACTIVE.get("saved_update"))
            print(f"[v8] {reason}; exiting {exit_code}", file=sys.stderr, flush=True)
        finally:
            os._exit(exit_code)

    thread = threading.Thread(target=watch, name="v8-stop-watchdog", daemon=True)
    thread.start()
    _WATCHDOG.update(thread=thread, reader=reader, writer=writer)
    return thread


def run_child_task(call, *, is_primary, task_id, run_id=None):
    """The torchrun child's exit contract around one task.

    Interrupted (stop at the saved update) / StopRequested (stop outside a training update) and a 'busy'
    result exit 75 on every rank, after rank 0 wrote the exit marker; any other exception propagates
    (nonzero exit, torchrun stops the peers). Returns the task result otherwise.
    """
    try:
        res = call()
    except Interrupted as exc:
        if is_primary:
            print(f"INTERRUPTED {exc}", flush=True)
            write_exit_marker(EXIT_BUSY_OR_INTERRUPTED, "interrupted", detail=str(exc), task=task_id, run_id=run_id)
        raise SystemExit(EXIT_BUSY_OR_INTERRUPTED) from None
    except StopRequested as exc:
        if is_primary:
            print(f"INTERRUPTED stop signal {exc} outside a training update; no run state changed", flush=True)
            write_exit_marker(EXIT_BUSY_OR_INTERRUPTED, "interrupted", detail=f"stop signal {exc}", task=task_id,
                              run_id=run_id)
        raise SystemExit(EXIT_BUSY_OR_INTERRUPTED) from None
    if is_primary:
        print("RESULT " + json.dumps(res, default=str), flush=True)
    if isinstance(res, dict) and res.get("status") == "busy":
        if is_primary:
            write_exit_marker(EXIT_BUSY_OR_INTERRUPTED, "busy", task=task_id, run_id=run_id or res.get("run_id"))
        raise SystemExit(EXIT_BUSY_OR_INTERRUPTED)
    return res


class Context:
    """Execution environment. Defaults are single-process; the cluster child overrides
    primary/broadcast so that only rank 0 writes and every rank takes the same decisions."""

    def __init__(self, *, srgc, engine_class, backend, data, config, out_root, meter=None,
                 measurement_status="measured", gpu_type=None, is_primary=True,
                 primary=None, broadcast=None, cache_store=None, cache_context=None, verifier_identity=None):
        self.srgc, self.engine_class, self.backend, self.data, self.config = srgc, engine_class, backend, data, config
        self.out_root, self.meter = Path(out_root), meter
        self.measurement_status, self.gpu_type, self.is_primary = measurement_status, gpu_type, is_primary
        self._primary = primary or (lambda fn: fn())
        self._broadcast = broadcast or (lambda value: value)
        self.cache_store, self.cache_context = cache_store, cache_context or {}
        self.verifier_identity = verifier_identity
        self.event_sink = None

    def primary(self, fn):
        return self._primary(fn)

    def broadcast(self, value):
        return self._broadcast(value)

    def on_meter_event(self, event):
        if self.event_sink is not None:
            self.event_sink(event)


def run_dir(ctx, run_id):
    return ctx.out_root / "runs" / run_id


def run_status(path):
    m = Path(path) / "manifest.json"
    return read_json(m).get("status") if m.exists() else None


def run_attempts(rdir):
    m = read_json(Path(rdir) / "manifest.json")
    return [(a["attempt_id"], a["start_update"]) for a in m.get("attempts", []) if a.get("start_update") is not None]


def run_events(rdir, *, lineage=True):
    rdir = Path(rdir)
    events = []
    for p in sorted((rdir / "attempts").glob("*/events.jsonl")):
        events.extend(read_jsonl(p))
    return lineage_events(events, run_attempts(rdir)) if lineage else events


def run_starts(rdir):
    """Durable phase starts of every attempt (ledger.physical_gpu_seconds(..., starts=...))."""
    starts = []
    for p in sorted((Path(rdir) / "attempts").glob("*/phase-starts.jsonl")):
        starts.extend(read_jsonl(p))
    return starts


def _phase_cost(ctx, rdir, family, offset, upto=None):
    """Own continuation algorithm cost (+ parent offset); None if any required event is unmeasured.
    `upto`: only work through that completed update (re-indexing a snapshot after a restart)."""
    totals, missing = component_totals(run_events(rdir), max_update=upto)
    required = REQUIRED[family] - {"prefix", "cache_build"}
    unknown = [m for m in missing if m.get("component") in required]
    if unknown:
        return None, missing
    own = sum(v for k, v in totals.items() if k in required)
    return own + (offset or 0.0), missing


def execute_train(task: dict, ctx: Context) -> dict:
    rid = task["run_id"]
    rdir = run_dir(ctx, rid)
    lock = ctx.primary(lambda: _acquire(rdir))
    if lock == "busy":
        return {"run_id": rid, "status": "busy"}
    try:
        return _execute_train(task, ctx, rdir)
    finally:
        # Only rank 0 holds the lock. No collective here: after a rank-local exception the other
        # ranks are inside an update collective, and a broadcast would deadlock them all.
        if ctx.is_primary:
            _release(rdir)


_LOCKS = {}


def _acquire(rdir):
    rdir.mkdir(parents=True, exist_ok=True)
    try:
        cm = file_lock(rdir / ".execution.lock")
        cm.__enter__()
    except Busy:  # held by another process; every other error (EACCES, ENOLCK, ESTALE) is reported as is
        return "busy"
    _LOCKS[str(rdir)] = cm
    return "ok"


def _release(rdir):
    cm = _LOCKS.pop(str(rdir), None)
    if cm is not None:
        cm.__exit__(None, None, None)


def _snapshot_index(rdir):
    return {r["completed_update"]: r for r in read_jsonl(Path(rdir) / "snapshots.jsonl")}


def _execute_train(task, ctx, rdir):
    srgc, c = ctx.srgc, ctx.config
    method = task["method"]
    manifest_path = rdir / "manifest.json"

    def admit():
        with critical(), _MANIFEST_LOCK:
            if manifest_path.exists():
                m = read_json(manifest_path)
                if m["config_hash"] != task["config_hash"]:
                    raise ConfigChanged(f"{task['run_id']}: config hash changed ({m['config_hash']} -> "
                                        f"{task['config_hash']}); start a new study version")
                return m["status"]
            atomic_write_json(manifest_path, {**{k: task[k] for k in ("run_id", "study_id", "cohort_id", "method_id",
                              "training_seed", "role", "config_hash")}, "task": task, "status": "initialized",
                              "created_at": utc_now(), "attempts": []})
            return "initialized"

    status = ctx.broadcast(ctx.primary(admit))
    if status in TERMINAL:
        return {"run_id": task["run_id"], "status": "skipped_complete"}
    attempt = ctx.broadcast(ctx.primary(new_id))
    adir = rdir / "attempts" / attempt
    ledger = Ledger(adir / "events.jsonl", run_id=task["run_id"], attempt_id=attempt, gpu_type=ctx.gpu_type,
                    measurement_status=ctx.measurement_status,
                    starts_path=adir / "phase-starts.jsonl") if ctx.is_primary else None
    updates_log = JsonlLog(adir / "updates.jsonl", ("completed_updates",)) if ctx.is_primary else None
    ctx.event_sink = ledger.on_meter_event if ledger else None
    stop = {"flag": False}  # test hook (debug_stop_after_update)
    _STOP.update(requested=False, raised=False)
    _STOP.pop("signal", None)
    previous = install_stop_handlers()
    policy = build_policy(method["policy"])
    engine = ctx.engine_class(ctx.backend, ctx.data["candidate_ids"], ctx.data["ranking_validation_ids"],
                              ctx.data["cached_rewards"], method=method, policy=policy, config=c,
                              monitor=method.get("monitor", False), cache_store=ctx.cache_store,
                              cache_context=ctx.cache_context, run_id=task["run_id"], primary=ctx.primary)
    latest = rdir / "latest.pt"
    ACTIVE.clear()
    ACTIVE.update(manifest=manifest_path, run_id=task["run_id"], saved_update=None)
    try:
        def note_attempt(kind, start_update):
            with critical(), _MANIFEST_LOCK:
                m = read_json(manifest_path)
                m["attempts"].append({"attempt_id": attempt, "started_at": utc_now(), "start": kind,
                                      "start_update": start_update})
                m["status"] = "active"
                atomic_write_json(manifest_path, m)
        if latest.exists():
            payload = snapshots.load(latest)
            engine.load_state_dict(payload["engine"])
            snapshots.restore_rng(payload["rng"])
            ctx.primary(lambda: note_attempt(f"resume@{engine.step}", engine.step))
        elif task["parent"]["kind"] == "initial":
            # fresh prefix for a new seed: initial model (LoRA initialized by the child from the seed)
            ctx.primary(lambda: write_once_json(rdir / "branch_start.json", {
                "parent": task["parent"], "parent_identity": snapshots.state_identity(engine.state_dict()),
                "fork_arm": None, "first_update_of_branch": 1}))
            ctx.primary(lambda: note_attempt("initial", 0))
        else:
            parent = task["parent"]
            parent_payload = snapshots.load(parent["path"], parent.get("sha256"))
            state = parent_payload["engine"] if "engine" in parent_payload else parent_payload
            if state["step"] != parent["completed_update"]:
                raise ValueError("parent snapshot update count differs from the task")
            ident = snapshots.state_identity(state)
            if parent.get("state_hash") and parent["state_hash"] != ident["state_hash"]:
                raise ValueError("parent state identity differs from the matrix record")
            engine.load_state_dict(state, fork_arm=fork_arm_for(method))
            if "rng" in parent_payload:
                snapshots.restore_rng(parent_payload["rng"])
            branch_ident = snapshots.branch_identity(parent_payload)
            ctx.primary(lambda: write_once_json(rdir / "branch_start.json", {
                "parent": parent, "parent_identity": ident, "parent_branch_identity": branch_ident,
                "fork_arm": fork_arm_for(method), "first_update_of_branch": state["step"] + 1}))
            ctx.primary(lambda: note_attempt("fork", engine.step))
        # latest.pt (resume), or the parent itself (fork/initial: an interrupted attempt forks again)
        ACTIVE["saved_update"] = engine.step
        if method["policy"]["policy"] == "sampled_time":
            def record_schedule():
                with critical(), _MANIFEST_LOCK:
                    m = read_json(manifest_path)
                    m.setdefault("sampled_schedule", method["policy"])  # recorded once; never re-drawn
                    atomic_write_json(manifest_path, m)
            ctx.primary(record_schedule)
        retained = set(task.get("retain_updates", []))
        ledger_paths = rdir  # costs are always recomputed from the run's lineage ledger
        # A kill between a retained snapshot file and its snapshots.jsonl line leaves the file unindexed:
        # index it from its own bytes (never regenerate), then write a snapshot still missing at this step.
        with critical():
            ctx.primary(lambda: _repair_snapshot_index(ctx, rdir, task, retained, engine.step))
        if ctx.broadcast(ctx.primary(lambda: engine.step in retained and engine.step not in _snapshot_index(rdir)
                                     and latest.exists())):
            _save_retained(ctx, engine, rdir, task, ledger_paths)
        deltas, last_cost = [], None
        attempt_start = engine.step
        logged = {"signals": len(engine.signal_log), "decisions": len(engine.decision_log)}
        while True:
            def decide():
                if engine.step < task["horizon"]:
                    return True, None  # the matched-update horizon is always reached; no ledger read needed
                cost, _ = _phase_cost(ctx, rdir, method["family"], task.get("cost_offset_gpu_seconds"))
                nxt = sum(deltas[-5:]) / len(deltas[-5:]) if deltas else task.get("est_update_gpu_seconds")
                return should_continue(engine.step, horizon=task["horizon"], hard_cap=task.get("hard_cap"),
                                       budget_cap=task.get("budget_cap_gpu_seconds"), cumulative_cost=cost,
                                       next_update_cost=nxt), cost
            go, cost_now = ctx.broadcast(ctx.primary(decide))
            if last_cost is not None and cost_now is not None:
                deltas.append(cost_now - last_cost)
            last_cost = cost_now
            if not go:
                break
            if task.get("debug_stop_after_update") == engine.step and attempt_start < engine.step:
                stop["flag"] = True  # test hook: behaves like SIGTERM at a safe boundary
            if ctx.broadcast(stop["flag"] or _STOP["requested"]):
                _save_latest(ctx, engine, latest)
                ctx.primary(lambda: _set_status(manifest_path, "interrupted", engine.step))
                raise Interrupted(f"{task['run_id']} stopped at a safe boundary (update {engine.step})")
            record = engine.update()
            if ctx.is_primary:
                updates_log.append({"completed_updates": engine.step, "selector": record.get("selector"),
                                    "train_ids": record.get("train_ids"), "metrics": record.get("metrics"),
                                    "d": record.get("d"), "transition_update": engine.transition_update,
                                    "signals": engine.signal_log[logged["signals"]:],
                                    "decisions": engine.decision_log[logged["decisions"]:]})
            logged = {"signals": len(engine.signal_log), "decisions": len(engine.decision_log)}
            # latest first: a crash between the two writes resumes AT this step, and the
            # missing retained snapshot is then written from the identical state (never regenerated).
            if engine.step % task.get("latest_every", 1) == 0 or engine.step in retained:
                _save_latest(ctx, engine, latest)
            if engine.step in retained:
                _save_retained(ctx, engine, rdir, task, ledger_paths)
        _save_latest(ctx, engine, latest)
        if engine.step not in retained:
            _save_retained(ctx, engine, rdir, task, ledger_paths)
        with critical():
            ctx.primary(lambda: _finalize(rdir, task, engine))
        return {"run_id": task["run_id"], "status": read_json(manifest_path)["status"] if ctx.is_primary else "done",
                "completed_updates": engine.step, "transition_update": engine.transition_update}
    except Interrupted:
        raise
    except StopRequested as exc:
        # Stop signal inside an update: abandon it; latest.pt holds the last completed update.
        saved = ACTIVE.get("saved_update")
        if ctx.is_primary and manifest_path.exists():
            mark_interrupted(manifest_path, saved, f"stop signal {exc}")
        raise Interrupted(f"{task['run_id']} stopped by {exc}; resumes from saved update {saved}") from None
    except Exception as exc:
        if _STOP["requested"]:
            # A collective failed because the stopped peers left it (stop signal on every rank).
            saved = ACTIVE.get("saved_update")
            if ctx.is_primary and manifest_path.exists():
                mark_interrupted(manifest_path, saved, f"stop signal; collective ended: {type(exc).__name__}: {exc}")
            raise Interrupted(f"{task['run_id']} stopped; resumes from saved update {saved}") from exc
        if ctx.is_primary and manifest_path.exists():
            kind = "failed_algorithm_nonfinite" if "finite" in str(exc).lower() else "failed"
            _set_status(manifest_path, kind, engine.step, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if ledger is not None:
            ledger.close_interrupted()
        ctx.event_sink = None
        ACTIVE.clear()
        restore_handlers(previous)


def _set_status(path, status, step, error=None):
    with critical(), _MANIFEST_LOCK:
        m = read_json(path)
        m.update(status=status, last_completed_update=step, updated_at=utc_now())
        if error:
            m.setdefault("errors", []).append({"at": utc_now(), "error": error[:2000]})
        atomic_write_json(path, m)


def _payload(engine):
    return {"engine": engine.state_dict(), "rng": snapshots.capture_rng(), "saved_at": time.time()}


def _save_latest(ctx, engine, latest):
    state = _payload(engine)
    with critical():
        ctx.primary(lambda: snapshots.save(latest, state, immutable=False))
        if ACTIVE:
            ACTIVE["saved_update"] = engine.step


def _index_record(ctx, rdir, task, step, path, sha, payload, *, phase, transition_update, upto=None):
    cost, missing = _phase_cost(ctx, rdir, task["method"]["family"], 0.0, upto=upto)
    return {"completed_update": step, "path": str(path.relative_to(rdir)), "sha256": sha,
            "identity": snapshots.state_identity(payload["engine"]),
            "branch_identity": snapshots.branch_identity(payload),
            "payload_identity": snapshots.payload_identity(payload), "own_continuation_gpu_seconds": cost,
            "cost_missing_events": len(missing), "phase": phase,
            "transition_update": transition_update, "eligible": True}


def _save_retained(ctx, engine, rdir, task, ledger_paths):
    state = _payload(engine)
    path = rdir / "snapshots" / f"step-{engine.step:06d}.pt"

    def write():
        sha = snapshots.save(path, state, immutable=True)
        JsonlLog(rdir / "snapshots.jsonl", ("completed_update",)).append(
            _index_record(ctx, rdir, task, engine.step, path, sha, state, phase=engine.phase,
                          transition_update=engine.transition_update))
    with critical():
        ctx.primary(write)


def _repair_snapshot_index(ctx, rdir, task, retained, upto):
    """Rank 0, before an attempt continues: index snapshot files that a kill left without their line.

    Only retained updates (or the attempt's current update, a final non-grid endpoint) are indexed,
    from the file's own state; the cost is the lineage cost through that update, as when it was saved.
    """
    index = _snapshot_index(rdir)
    repaired = []
    for path in sorted((Path(rdir) / "snapshots").glob("step-*.pt")):
        try:
            step = int(path.stem.split("-", 1)[1])
        except ValueError:
            continue
        if step in index or step > upto or (step not in retained and step != upto):
            continue
        payload = snapshots.load(path)
        state = payload["engine"] if "engine" in payload else payload
        if state.get("step") != step:
            raise ValueError(f"{path}: snapshot holds update {state.get('step')}, not {step}")
        v8 = state.get("v8") or {}
        JsonlLog(Path(rdir) / "snapshots.jsonl", ("completed_update",)).append(
            _index_record(ctx, Path(rdir), task, step, path, sha256_file(path), payload, phase=v8.get("phase"),
                          transition_update=v8.get("transition_update"), upto=step))
        repaired.append(step)
    if repaired:
        print(f"[v8] {task['run_id']}: indexed retained snapshot(s) {repaired} written just before an interruption",
              flush=True)
    return repaired


def _finalize(rdir, task, engine):
    start = (task.get("parent") or {}).get("completed_update") or 0
    needed = {u for u in task.get("retain_updates", []) if start < u <= engine.step} | {engine.step}
    missing = sorted(needed - set(_snapshot_index(rdir)))
    if missing:
        raise ValueError(f"{task['run_id']}: retained snapshots missing from snapshots.jsonl: {missing}")
    hist = JsonlLog(rdir / "history.jsonl", ("completed_updates",))
    for r in engine.history:
        hist.append({k: v for k, v in r.items() if k not in ("ranking_scores",)})
    dec = JsonlLog(rdir / "decisions.jsonl", ("completed_update", "rule_id", "policy_id"))
    for d in engine.decision_log:
        dec.append({"run_id": task["run_id"], **d})
    sig = JsonlLog(rdir / "signals.jsonl", ("completed_update", "signal_definition_version"))
    for s in engine.signal_log:
        sig.append({"run_id": task["run_id"], **s})
    stats = JsonlLog(rdir / "training_stats.jsonl", ("completed_update",))
    for u, (total, count) in sorted(engine.stats.rows.items()):
        stats.append({"completed_update": u, "reward_sum": total, "valid_response_count": count})
    cl = JsonlLog(rdir / "cache_log.jsonl", ("completed_update", "event", "attempt"))
    for row in engine.cache_log:
        cl.append(row)
    status = "completed_horizon" if engine.step == task["horizon"] else "completed_budget"
    with _MANIFEST_LOCK:
        m = read_json(rdir / "manifest.json")
        m.update(status=status, completed_updates=engine.step, transition_update=engine.transition_update,
                 final_phase=engine.phase, finished_at=utc_now(),
                 no_switch=engine.transition_update is None and task["method"]["initial_selector"] == "on_policy")
        atomic_write_json(rdir / "manifest.json", m)


# ---------------------------------------------------------------- evaluation
def eval_record_path(ctx, snapshot_sha, spec_hash):
    return ctx.out_root / "evaluations" / f"{snapshot_sha[:16]}__{spec_hash[:12]}.json"


COUNT_TOLERANCE = 1e-9


def integer_counts(per_question, responses, *, tolerance=COUNT_TOLERANCE):
    """Convert per-question mean rewards to integer correct counts, or raise.

    A mean is accepted only if finite, inside [0, 1] and within `tolerance` of k/responses
    for an integer k; no count is ever invented by rounding (0.37 with 8 responses fails).
    """
    import math
    counts = {}
    for q, v in per_question.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{q}: non-finite evaluation reward {v!r}")
        if v < 0 or v > 1:
            raise ValueError(f"{q}: evaluation reward {v!r} outside [0, 1]")
        k = round(v * responses)
        if abs(k - v * responses) > tolerance * max(1, responses):
            raise ValueError(f"{q}: reward {v!r} is not an integer count over {responses} responses")
        counts[q] = int(k)
    return counts


def split_ids(data, split):
    return data["evaluation_ids"] if split == "final" else data["dev_evaluation_ids"]


def execute_eval(task: dict, ctx: Context) -> dict:
    """Immutable checkpoint evaluation, run OFFLINE from retained snapshots (never inside training).

    Never selects checkpoints or tunes rules. Records integer per-question counts.
    """
    from .common import sha256_json
    path = eval_record_path(ctx, task["snapshot_sha256"], task["evaluation_spec_hash"])

    def existing():
        if not path.exists():
            return False
        recorded, current = read_json(path).get("verifier"), ctx.verifier_identity
        if recorded is not None and current is not None and recorded != current:
            raise ValueError(f"{path.name}: this checkpoint/spec was evaluated with verifier {recorded!r}, but this "
                             f"child scores with {current!r}; refusing to reuse a different reward function")
        if recorded is None and current is not None:
            print(f"[v8] WARN {path.name}: existing evaluation predates verifier identities; reused as recorded",
                  flush=True)
        return True
    if ctx.broadcast(ctx.primary(existing)):
        return {"status": "skipped_complete", "path": str(path)}
    ids = list(split_ids(ctx.data, task["split"]))
    if sha256_json(sorted(ids)) != task["question_ids_hash"]:
        raise ValueError("evaluation question set differs from the matrix (split changed)")
    state = snapshots.load_engine_state(task["snapshot_path"], task["snapshot_sha256"])
    ctx.backend.load_state_dict(state["backend"])
    attempt = ctx.broadcast(ctx.primary(new_id))
    ledger = Ledger(ctx.out_root / "evaluations" / "events" / f"{attempt}.jsonl", run_id=task["run_id"],
                    attempt_id=attempt, gpu_type=ctx.gpu_type, measurement_status=ctx.measurement_status,
                    starts_path=ctx.out_root / "evaluations" / "phase-starts" / f"{attempt}.jsonl"
                    ) if ctx.is_primary else None
    ctx.event_sink = ledger.on_meter_event if ledger else None
    seed = ctx.srgc.stream_seed(task["training_seed"], task["completed_update"], task["seed_purpose"])
    counting = getattr(ctx.backend, "evaluate_counts", None)

    def run():
        if counting is not None:
            return dict(counting(ids, seed=seed, responses=task["responses"])), "integer_counts"
        return dict(ctx.backend.evaluate(ids, seed=seed, responses=task["responses"])), "per_question_mean"
    try:
        if ctx.meter is not None:
            with ctx.meter.phase("evaluation", task["completed_update"], ctx.backend.gpu_count):
                values, source = run()
        else:
            values, source = run()
    finally:
        ctx.event_sink = None
    if set(values) != set(ids):
        raise ValueError("evaluation returned a different question set")
    if source == "integer_counts":
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= task["responses"] for v in values.values()):
            raise ValueError("backend counts must be integers in [0, responses]")
        correct = dict(values)
    else:
        correct = integer_counts(values, task["responses"])
    record = {"run_id": task["run_id"], "checkpoint_hash": task["snapshot_sha256"],
              "completed_update": task["completed_update"], "evaluation_spec_hash": task["evaluation_spec_hash"],
              "split": task["split"], "n_questions": len(correct), "responses_per_question": task["responses"],
              "reward_fraction": sum(correct.values()) / (len(correct) * task["responses"]),
              "per_question_correct": correct, "count_source": source,
              "generation_seed_purpose": task["seed_purpose"],
              "verifier": ctx.verifier_identity if ctx.verifier_identity is not None else task.get("verifier"),
              "sample_level_rewards": False, "measurement_status": ctx.measurement_status,
              "attempt_id": attempt, "evaluated_at": utc_now()}
    ctx.primary(lambda: write_once_json(path, record))
    return {"status": "evaluated", "path": str(path), "reward_fraction": record["reward_fraction"]}
