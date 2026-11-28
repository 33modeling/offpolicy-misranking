"""`sh v9/run.sh status`: one screen that says what V9 is doing, what is stuck and what to do.

Read-only (the V9 locator copy written by the planners on first use aside). Sections:
header (code commit, time, roots) -> tau -> totals -> running now -> FAILED (needs you) ->
per stage and cohort done/total with the waiting reasons -> what a free node takes next -> result files.
Nodes and "running" come from the heartbeat each worker rewrites every minute (.state/workers/, what it does
now: working on a task, waiting and why, choosing); workers of older code without a heartbeat are read from their
worker log and the .state/running record, and a log silent for STALE_SECONDS is never counted as running.
Time left comes from the wall time of finished V9 tasks (.state/done-times/), never from a guess.
Every count on the screen is read back by check_screen(); a disagreement prints a red line.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from . import env

KST = timezone(timedelta(hours=9))
STALE_SECONDS = 1800
BEAT_LIVE_SECONDS = 300
STATES = ("done", "running", "ready", "retrying", "blocked", "failed")   # V8: retrying = restarts by itself, failed = needs you
COLOR = False   # set by main() only when stdout is a terminal (never in logs, pipes or tests)
STYLE = {"done": "34", "ok": "32", "run": "1;32", "bad": "1;31", "warn": "33", "dim": "2", "head": "1", "info": "36"}


def paint(text, style):
    """V8 status colours."""
    return f"\033[{STYLE[style]}m{text}\033[0m" if COLOR and style and str(text).strip() else text


def kst(t):
    return datetime.fromtimestamp(t, KST).strftime("%m-%d %H:%M")


def now_text(at=None):
    return datetime.fromtimestamp(time.time() if at is None else at, KST).strftime("%m-%d %H:%M KST")


def span(seconds):
    seconds = max(0, int(seconds))
    h, m = divmod(seconds // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def code_commit():
    try:
        out = subprocess.run(["git", "-C", str(env.REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or "?"
    except (OSError, subprocess.SubprocessError):
        return "?"


def running_dir():
    return env.v9_root() / ".state" / "running"


LOG_NAME = re.compile(r"worker-(?P<host>.+)-(?P<stamp>\d{8}T\d{6})-(?P<pid>\d+)\.log$")
TASK_LINE = re.compile(r"^\[v9\] (?P<at>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d) (?P<host>\S+) stage=(?P<stage>\S+) "
                       r"cohort=(?P<cohort>\S+) task=(?P<task>\S+)\s*$")
# Worker lines that mean its current task is over (V9 workers of every version print these).
TASK_OVER = ("busy on another node", " FAILED rc=", "nothing more is ready", "stopped by signal", "[v9] collect:",
             "[v9] freeze-tau:", "planning FAILED")


def _tail_lines(path, limit=4 << 20):
    """Lines of the last `limit` bytes of a file (worker logs carry the child's whole output)."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - limit))
        return fh.read().decode(errors="replace").splitlines()


def worker_logs():
    """Every worker log under $V9/logs written within STALE_SECONDS: host, pid, current task (or None)."""
    d = env.v9_root() / "logs"
    rows = []
    for p in sorted(d.glob("worker-*.log")) if d.is_dir() else []:
        m = LOG_NAME.match(p.name)
        try:
            quiet = time.time() - p.stat().st_mtime
        except OSError:
            continue
        if not m or quiet >= STALE_SECONDS:
            continue
        current, ended = None, False
        try:
            lines = _tail_lines(p)
        except OSError:
            continue
        for line in lines:
            t = TASK_LINE.match(line.strip())
            if t:
                current, ended = t.groupdict(), False
            elif line.startswith("[v9] worker start"):
                current, ended = None, False
            elif line.startswith("[v9] ") and ("nothing more is ready" in line or "stopped by signal" in line):
                current, ended = None, True   # this worker exited
            elif current is not None and line.startswith("[v9] ") and any(x in line for x in TASK_OVER):
                current = None
        if ended:
            continue
        rec = dict(host=m["host"], pid=int(m["pid"]), log=str(p), quiet=quiet, live=True, task=current)
        if current is not None:
            try:
                rec["started_at"] = time.mktime(time.strptime(current["at"], "%Y-%m-%dT%H:%M:%S"))
            except ValueError:
                rec["started_at"] = p.stat().st_mtime
            rec.update(task_id=current["task"], stage=current["stage"], cohort=current["cohort"])
        rows.append(rec)
    return rows


def heartbeats(now=None):
    """Live V9 worker heartbeats (.state/workers/<host>-<pid>.json younger than BEAT_LIVE_SECONDS)."""
    now = time.time() if now is None else now
    d = env.v9_root() / ".state" / "workers"
    out = []
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            r = json.loads(p.read_text())
            age = now - float(r.get("at", 0))
        except (OSError, ValueError, TypeError):
            continue
        if age < BEAT_LIVE_SECONDS:
            r.update(beat_age=age, live=True)
            out.append(r)
    return out


def workers():
    """One row per live worker: heartbeat if it writes one, else its worker log (older V9 code)."""
    rows = {}
    for b in heartbeats():
        log = b.get("log") or ""
        try:
            quiet = time.time() - Path(log).stat().st_mtime if log else b["beat_age"]
        except OSError:
            quiet = b["beat_age"]
        rec = dict(host=b.get("host"), pid=b.get("pid"), log=log, quiet=quiet, live=True, beat=b["beat_age"],
                   state=b.get("state"), doing=b.get("doing"), task=None)
        if b.get("state") == "working" and b.get("task_id"):
            rec.update(task_id=b["task_id"], stage=b.get("stage"), cohort=b.get("cohort"),
                       started_at=float(b.get("task_started") or b.get("at")), task=True)
        rows[(rec["host"], rec["pid"])] = rec
    for w in worker_logs():
        if (w["host"], w["pid"]) in rows:
            continue
        w.update(beat=None, state="working" if w.get("task_id") else "idle",
                 doing=None if w.get("task_id") else "no heartbeat (older V9 code): choosing the next task or finishing")
        rows[(w["host"], w["pid"])] = w
    return list(rows.values())


def running_records():
    """task id -> record of the worker running it, with 'live'.

    Main source: the worker logs (every V9 version writes a `stage=… cohort=… task=…` line when it starts a task
    and streams the child's output into the same log), so nodes started before the running records existed are
    seen too. The .state/running records (written since 66d20b7) add tasks whose log is not readable here."""
    out = {}
    for rec in workers():
        if rec.get("task_id"):
            out[rec["task_id"]] = rec
    d = running_dir()
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        tid = r.get("task_id") or p.stem
        if tid in out:
            continue
        log = Path(r.get("log") or "")
        try:
            quiet = time.time() - log.stat().st_mtime
        except OSError:
            quiet = time.time() - float(r.get("started_at", 0))
        live = quiet < STALE_SECONDS
        if r.get("host") == socket.gethostname() and r.get("pid"):
            try:
                os.kill(int(r["pid"]), 0)
                live = True
            except PermissionError:
                live = True
            except (OSError, ValueError):
                live = False
        r.update(quiet=quiet, live=live)
        out[tid] = r
    return out


def gpu_rows():
    """This node's GPU utilisation in % (V8 status: `GPU 98% 97% 0% 0%`); [] when nvidia-smi is not there."""
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=8)
        return [int(float(r)) for r in res.stdout.split() if r.strip()] if res.returncode == 0 else []
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def gpu_text(gpus=None):
    gpus = gpu_rows() if gpus is None else gpus
    return "GPU " + (" ".join(f"{u}%" for u in gpus) if gpus else "unknown")


def last_output(log, width=110):
    """The last non-empty line of a worker log (progress text from the child), shortened."""
    try:
        with open(log, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 16384))
            lines = [ln.strip() for ln in fh.read().decode(errors="replace").splitlines() if ln.strip()]
    except OSError:
        return ""
    text = lines[-1] if lines else ""
    return text if len(text) <= width else text[:width - 1] + "…"


def collect_items(plan, task_id):
    """All planner items, with ready items that a live worker holds marked 'running'. Planning errors kept."""
    running = running_records()
    items, errors = [], []
    for stage in env.STAGES:
        try:
            got = plan(stage)
        except Exception as exc:  # noqa: BLE001 - one stage must not hide the others
            errors.append((stage, f"{type(exc).__name__}: {exc}"))
            continue
        for it in got:
            tid = task_id(it["task"]) if it.get("task") else None
            rec = running.get(tid) if tid else None
            if rec is not None and it["state"] != "done" and rec["live"]:
                it = dict(it, state="running", run=rec)
            elif rec is not None and it["state"] != "done":
                it = dict(it, stale=rec)
            items.append(it)
    return items, errors, running


def stage_failures(limit=3):
    p = env.v9_root() / ".state" / "stage-failures.jsonl"
    try:
        rows = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    except (OSError, ValueError):
        return []
    return rows[-limit:]


STAGE_TITLES = {
    "freeze-tau": "tau",
    "x1": "X1 Staged vs SR vs Alignment at 275",
    "x2": "X2 equal budget",
    "x5": "X5 measurements",
    "x3": "X3 horizon 400",
    "x4": "X4 fresh seeds 10-14",
    "x8": "X8 refresh-frequency control (hold arms)",
    "x9": "X9 online SR baseline",
    "x10": "X10 budget sensitivity of the tau rule",
    "x11": "X11 dynamic-sampling baseline",
    "x13": "X13 Random then SR (warm-up control)",
    "x14": "X14 SR then Alignment (order reversed)",
    "x15": "X15 16 prompts per update",
    "x12": "X12 public benchmarks",
    "collect": "result files",
}
COHORT_NAMES = {"olmo_math": "OLMo MATH", "olmo_mbpp": "OLMo MBPP", "qwen_math": "Qwen MATH",
                "qwen_mbpp": "Qwen MBPP", "llama_math": "Llama MATH", "llama_mbpp": "Llama MBPP", "all": "all"}
CELL = {"done": "✓", "running": "▶", "ready": "ready", "retrying": "retry", "blocked": "wait", "failed": "FAIL"}
WIDTH = 118


def wrap(text, first, rest, out):
    """V8 status wrapping: lines up to WIDTH, continuation indented."""
    words, line = str(text).split(" "), first
    for w in words:
        if len(line) + len(w) + 1 > WIDTH and line.strip():
            out(line.rstrip());line = rest
        line += w + " "
    if line.strip():
        out(line.rstrip())


def cname(cohort):
    return COHORT_NAMES.get(cohort, cohort)


def split_key(key):
    """'s5.on.train' -> ('s5', 'on', 'train'); 's5.switch050@B' -> ('s5', 'switch050@B', ''); 'all' -> ('', 'all', '')."""
    parts = str(key).split(".")
    seed = parts[0] if parts and parts[0][:1] == "s" and parts[0][1:].isdigit() else ""
    rest = parts[1:] if seed else parts
    tail = rest[-1] if len(rest) > 1 and rest[-1] in ("train", "eval", "prefix") else ""
    if tail:
        rest = rest[:-1]
    return seed, ".".join(rest) or (tail or "-"), tail


def task_name(i):
    return f"{cname(i['cohort'])} {i['key']}"


PROGRESS = re.compile(r"(question|update|step|prompt|batch)\D{0,4}?(\d+)\s*(?:/|of)\s*(\d+)", re.I)


def progress(rec):
    """'update 120 of 275 · ~3h10m left' from the newest progress line in the worker log (V8 status wording)."""
    try:
        lines = _tail_lines(rec.get("log", ""), 262144)
    except OSError:
        return ""
    for line in reversed(lines):
        found = PROGRESS.findall(line)
        if found:
            word, n, m = found[-1][0].lower(), int(found[-1][1]), int(found[-1][2])
            if not 0 < n <= m:
                continue
            text = f"{word} {n} of {m}"
            elapsed = time.time() - rec.get("started_at", time.time())
            if n < m and elapsed > 60:
                text += f" · ~{span(elapsed * (m - n) / n)} left"
            return text
    return ""


def done_times():
    """Finished-task wall times: (stage, cohort) -> [seconds], from .state/done-times/ (written since this version)."""
    d = env.v9_root() / ".state" / "done-times"
    out = {}
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            r = json.loads(p.read_text())
            out.setdefault((r["stage"], r["cohort"]), []).append(float(r["seconds"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if not n else (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)


def left_seconds(i):
    """Seconds left of a running task from its newest 'update n/m' style line, or None."""
    m = re.search(r"~(\d+)h(\d\d)m left|~(\d+)m left", progress(i["run"]))
    if not m:
        return None
    return (int(m.group(1)) * 60 + int(m.group(2))) * 60 if m.group(1) else int(m.group(3)) * 60


def stage_eta(its, times, nodes):
    """(low, high seconds, reason) for the unfinished tasks of one stage, or (None, None, why unknown).

    Measured: the median wall time of finished V9 tasks of the same stage and cohort (else the same stage).
    low = running tasks' own estimate + the rest spread over `nodes` working nodes; high = the rest on one node."""
    rest = [i for i in its if i["state"] in ("ready", "retrying", "blocked", "failed")]
    run = [i for i in its if i["state"] == "running"]
    if not rest and not run:
        return 0.0, 0.0, "finished"
    if not run and not any(i["state"] == "ready" for i in rest):
        v8 = sum(1 for i in rest if "V8" in (i.get("reason") or ""))
        failed = sum(1 for i in rest if i["state"] == "failed")
        again = sum(1 for i in rest if i["state"] == "retrying")
        parts = ([f"{v8} wait for V8 results"] if v8 else []) + \
                ([f"{len(rest) - v8 - failed - again} wait for an earlier step"] if len(rest) - v8 - failed - again else []) + \
                ([f"{again} restart by themselves"] if again else []) + ([f"{failed} stopped, need you"] if failed else [])
        return None, None, "no task of this stage can start now (" + ", ".join(parts) + ")"
    work, unknown = 0.0, []
    stage_all = [x for (st, _), v in times.items() if st == (its[0]["stage"] if its else "") for x in v]
    for i in rest:
        per = median(times.get((i["stage"], i["cohort"]), [])) or median(stage_all)
        if per is None:
            unknown.append(i)
        else:
            work += per
    run_left = [left_seconds(i) for i in run]
    if unknown or any(x is None for x in run_left) and not times:
        n = len(unknown) or len(run)
        return None, None, f"no finished V9 task of this stage measured yet ({n} task(s) without a time)"
    run_left = [x if x is not None else (median(stage_all) or 0) * 0.5 for x in run_left]
    head = max(run_left) if run_left else 0.0
    k = max(1, nodes)
    return head + work / k, head + work, f"{len(rest)} more task(s) at a measured median"


def eta_text(lo, hi, now):
    a, b = span(lo), span(hi)
    short = "about " + a + ("" if a == b else "–" + b)
    when = kst(now + lo) if a == b else f"{kst(now + lo)} – {kst(now + hi)[6:] if kst(now + lo)[:5] == kst(now + hi)[:5] else kst(now + hi)}"
    return short, f"{short} left, done around {when} KST"


def check_screen(by_stage, tasks, working_rows, counted):
    """Counts on the screen read back against the items: a disagreement is printed, nothing is changed."""
    broken = []
    for s, (shown_total, shown_by_state) in counted.items():
        its = by_stage.get(s, [])
        if shown_total != len(its):
            broken.append(f"{s}: total {shown_total} on screen, {len(its)} tasks")
        for st, n in shown_by_state.items():
            real = sum(1 for i in its if i["state"] == st)
            if n != real:
                broken.append(f"{s}: {st} {n} on screen, {real} tasks")
    run = {i["run"].get("task_id") or i.get("key") for i in tasks if i["state"] == "running"}
    on_nodes = {w.get("task_id") for w in working_rows if w.get("task_id")}
    if not run <= on_nodes:
        broken.append(f"running tasks without a working node: {', '.join(sorted(map(str, run - on_nodes))[:4])}")
    return broken


def render(plan, task_id, out=print, gpus=None):
    now = time.time()
    items, errors, running = collect_items(plan, task_id)
    for i in items:   # V8 rule: a task the worker restarts by itself is not drawn as a dead end
        if (i["state"] == "failed" and "no more automatic retries" not in (i.get("reason") or "")) or \
                (i["state"] == "ready" and i.get("retry_note")):
            i["state"], i["reason"] = "retrying", i.get("retry_note") or i.get("reason")
    stages = [s for s in env.STAGES if s not in ("freeze-tau", "collect")]
    by_stage = {s: [i for i in items if i["stage"] == s] for s in env.STAGES}
    tasks = [i for s in stages for i in by_stage[s]]
    here = socket.gethostname()
    gpus = gpu_rows() if gpus is None else gpus
    nodes = workers()
    working = [i for i in tasks if i["state"] == "running"]
    have = {w.get("task_id") for w in nodes if w.get("task_id")}
    for i in working:   # a task seen only through its .state/running record (its log is not readable here)
        r = i["run"]
        if r.get("task_id") not in have:
            nodes.append(dict(r, beat=None, state="working"))
    working_rows = [w for w in nodes if w.get("task_id")]
    times = done_times()

    # header, as in V8 status
    out(paint(f"V9 status · {code_commit()} · {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')}", "head"))
    out(paint(str(env.v9_root()), "dim"))
    mine = [i for i in working if i["run"].get("host") == here]
    mine_text = ", ".join(f"{task_name(i)} ({STAGE_TITLES[i['stage']].split(' ')[0]})"
                          + (f" · {progress(i['run'])}" if progress(i["run"]) else "") for i in mine)
    out(f"this node ({here}): " + (paint(mine_text, "run") if mine else "no V9 task here") + f" · {gpu_text(gpus)}")

    # 1. time left first (V8: '남은 시간도 모르겠어')
    out("")
    out(paint(" time left", "head"))
    lows, highs, unknown = [], [], []
    n_work = len({(w["host"], w["pid"]) for w in working_rows})
    for s in stages:
        its = by_stage[s]
        if not its:
            continue
        k = sum(1 for i in its if i["state"] == "running")
        lo, hi, why = stage_eta(its, times, max(n_work, 1))
        if why == "finished":
            out(f"   {STAGE_TITLES[s]}: finished")
            continue
        tail = f" ({k} node{'s' if k != 1 else ''} working on it)"
        if lo is None:
            unknown.append(STAGE_TITLES[s].split(" ")[0])
            out(f"   {STAGE_TITLES[s]}: unknown — {why}{tail}")
        else:
            lows.append(lo); highs.append(hi)
            out(f"   {STAGE_TITLES[s]}: {eta_text(lo, hi, now)[1]}{tail}")
            if span(lo) != span(hi):
                out(paint(f"     longer end if only one node works on it · {why}", "dim"))
    if unknown:
        wrap(f"All experiments: unknown until {', '.join(unknown)} can start and has a measured task time"
             + (f" (the rest: {eta_text(sum(lows), sum(highs), now)[0]})" if lows else ""), "   ", "     ", out)
    elif lows:
        wrap(f"All experiments: {eta_text(sum(lows), sum(highs), now)[1]}", "   ", "     ", out)
    else:
        out("   All experiments: finished")

    # 2. one block per stage: % done, per-cohort counts
    out("")
    out(paint(" experiments", "head"))
    tau = (by_stage.get("freeze-tau") or [None])[0]
    if tau is None:
        out(paint("   tau · unknown (planning error below)", "bad"))
    elif tau["state"] == "done":
        out(f"   tau · frozen · {tau['reason']}")
    elif tau["state"] == "ready":
        wrap("tau · NOT FROZEN yet · the first `sh v9/run.sh` freezes it (CPU, seconds); X1-X4 and the result files "
             "wait for it", "   ", "       ", lambda t: out(paint(t, "warn")))
    else:
        wrap(f"tau · NOT FROZEN · cannot be frozen yet: {tau['reason']}", "   ", "       ", lambda t: out(paint(t, "warn")))
    counted = {}
    for s in stages:
        its = by_stage[s]
        if not its:
            continue
        done = sum(1 for i in its if i["state"] == "done")
        n = sum(1 for i in its if i["state"] == "running")
        pct = int(100 * done / len(its)) if its else 0
        lo, _, why = stage_eta(its, times, max(n_work, 1))
        when = " · finished" if why == "finished" else (f" · {eta_text(lo, lo, now)[0]} left" if lo is not None else "")
        out(paint(f"   {STAGE_TITLES[s]}", "head") + f" · {pct}% of the work done{when} · "
            f"{n} node{'s' if n != 1 else ''} working on it")
        cohorts, shown = {}, {st: 0 for st in STATES}
        for i in its:
            cohorts.setdefault(i["cohort"], []).append(i)
        total = 0
        for cohort, rows in cohorts.items():
            c = {st: sum(1 for i in rows if i["state"] == st) for st in STATES}
            for st in STATES:
                shown[st] += c[st]
            total += len(rows)
            parts = [f"done {c['done']} of {len(rows)}"] + [paint(f"{c[st]} {w}", sty) for st, w, sty in
                     (("running", "running", "run"), ("ready", "can start", "info"),
                      ("retrying", "restart by themselves", "warn"), ("blocked", "waiting", "dim"),
                      ("failed", "STOPPED", "bad")) if c[st]]
            out(f"     {cname(cohort):<12}" + " · ".join(parts))
        counted[s] = (total, shown)
    col = (by_stage.get("collect") or [None])[0]
    if col is not None:
        wrap(f"result files · {col['state']} · {col['reason'] or ''} · {env.v9_root() / 'paper'}", "   ", "       ", out)

    # 3. every task that is not finished, by what happens next
    for s in stages:
        its = [i for i in by_stage[s] if i["state"] != "done"]
        if not its:
            continue
        out("")
        out(paint(f" {STAGE_TITLES[s]}: not finished {len(its)} of {len(by_stage[s])}", "head"))
        stuck = [i for i in its if i["state"] == "failed"]
        auto = [i for i in its if i["state"] == "retrying"]
        if auto:
            out(paint(f"   failed, restart by themselves ({len(auto)}):", "warn"))
            for i in auto[:12]:
                wrap(f"{task_name(i)}: {i['reason']}", "     ", "       ", lambda t: out(paint(t, "warn")))
        if stuck:
            out(paint(f"   STOPPED — needs you ({len(stuck)}):", "bad"))
            for i in stuck[:12]:
                wrap(f"{task_name(i)}: {i['reason']}", "     ", "       ", lambda t: out(paint(t, "bad")))
            if len(stuck) > 12:
                wrap("… %d more: %s" % (len(stuck) - 12, ", ".join(task_name(i) for i in stuck[12:])), "     ", "       ", out)
            out("     → fix the cause, then run `sh v9/run.sh retry-failed` on one node (the next free node restarts them)")
        run = [i for i in its if i["state"] == "running"]
        if run:
            out(paint(f"   running now ({len(run)}):", "run"))
            for i in run:
                r = i["run"]
                prog = progress(r)
                wrap(f"{task_name(i)} · on {r.get('host')} · for {span(now - r.get('started_at', now))}"
                     + (f" · {prog}" if prog else "") + f" · last output {span(r['quiet'])} ago", "     ", "       ", out)
                line = last_output(r.get("log", ""))
                if line:
                    out(paint(f"       {line}", "dim"))
        for i in [i for i in its if i.get("stale")]:
            r = i["stale"]
            wrap(f"possibly stopped: {task_name(i)} on {r.get('host')} (no output for {span(r['quiet'])}); a free node "
                 "takes it again and resumes from latest.pt", "     ", "       ", lambda t: out(paint(t, "warn")))
        ready = [i for i in its if i["state"] == "ready"]
        if ready:
            first = next((i for i in ready if not i.get("late")), None)
            n_late = sum(1 for i in ready if i.get("late"))
            out(paint(f"   can start now ({len(ready)}):", "info") +
                (f" the next free node takes {task_name(first)}" if first else
                 " a free node takes them when no other stage has a ready task") +
                (f" · {n_late} Random task(s) wait until no other stage has a ready task" if first and n_late else ""))
            wrap(", ".join(task_name(i) for i in ready[:10]) + (f" … +{len(ready) - 10}" if len(ready) > 10 else ""),
                 "     ", "       ", out)
        wait = [i for i in its if i["state"] == "blocked"]
        if wait:
            out(f"   waiting for an earlier step ({len(wait)}):")
            why = {}
            for i in wait:
                why.setdefault(i["reason"] or "waiting for an earlier step", []).append(i)
            for reason, rows in sorted(why.items(), key=lambda kv: -len(kv[1])):
                names = ", ".join(task_name(i) for i in rows[:6]) + (f" … +{len(rows) - 6}" if len(rows) > 6 else "")
                wrap(f"{reason} ({len(rows)}): {names}", "     ", "       ", lambda t: out(paint(t, "dim")))

    fails = stage_failures()
    if errors or fails:
        out("")
        out(paint(f" STOPPED — needs you: problems with a stage itself ({len(errors) + len(fails)})", "bad"))
        for stage, why in errors:
            wrap(f"{stage}: planning error: {why}", "   ", "     ", lambda t: out(paint(t, "bad")))
        for row in fails:
            wrap(f"{row.get('stage', '?')}: {row.get('at', '?')} on {row.get('host', '?')}: {str(row.get('error', ''))[:300]}",
                 "   ", "     ", lambda t: out(paint(t, "bad")))

    # 4. seed x arm grids
    grids = False
    for s in stages:
        its = [i for i in by_stage[s] if split_key(i["key"])[0]]
        if not its:
            continue
        for cohort in dict.fromkeys(i["cohort"] for i in its):
            rows = [i for i in its if i["cohort"] == cohort]
            seeds = sorted({split_key(i["key"])[0] for i in rows}, key=lambda x: int(x[1:]))
            cols = list(dict.fromkeys(split_key(i["key"])[1] for i in rows))
            cell = {}
            for i in rows:
                sd, c, _ = split_key(i["key"])
                mark = CELL.get(i["state"], i["state"])
                if i["state"] == "running":   # V8: 187▶ = running at update 187
                    m = re.match(r"(?:update|step|question|prompt|batch) (\d+) of (\d+)", progress(i["run"]))
                    if m:
                        mark = (f"{m.group(1)}▶" if m.group(0).startswith(("update", "step")) else
                                f"{100 * int(m.group(1)) // int(m.group(2))}%▶")
                cell.setdefault((sd, c), []).append(mark)
            texts = {k: ",".join(v) for k, v in cell.items()}
            width = {c: max([len(c)] + [len(texts.get((sd, c), "-")) for sd in seeds]) + 2 for c in cols}
            out("")
            out(paint(f" {cname(cohort)} {STAGE_TITLES[s].split(' ')[0]} by seed and arm", "head"))
            out(" " + f"{'':<12}" + "".join(f"{c:<{width[c]}}" for c in cols))
            for sd in seeds:
                out("   " + f"{sd:<10}" + "".join(f"{texts.get((sd, c), '-'):<{width[c]}}" for c in cols))
            grids = True
    if grids:
        wrap("cells: ✓ done · 187▶ running at update 187 (eval 36%▶ = questions answered) · ready = can start now · "
             "wait = needs an earlier step or a V8 result · retry = restarts by itself · FAIL = stopped, needs you · "
             "- = no such task", "   ", "     ", lambda t: out(paint(t, "dim")))

    # 5. nodes: live heartbeats (worker logs for older code)
    out("")
    hosts = sorted({w["host"] for w in nodes})
    work_hosts = {w["host"] for w in working_rows}
    waiting = [w for w in nodes if not w.get("task_id")]
    from collections import Counter
    why_idle = Counter(w.get("doing") or "choosing the next task" for w in waiting)
    head = paint(f" nodes: {len(hosts)} live", "head") + f" — {len(work_hosts)} working"
    if waiting:
        head += f", {len({w['host'] for w in waiting} - work_hosts)} idle — why: " + "; ".join(w for w, _ in why_idle.most_common(3))
    wrap(head, "", "   ", out)
    known = {i["run"].get("task_id") for i in working}
    by_tid = {i["run"].get("task_id"): i for i in working}
    for w in sorted(working_rows, key=lambda w: w["host"]):
        i = by_tid.get(w["task_id"])
        if i is not None:
            prog = progress(i["run"])
            name = task_name(i)
            stage = STAGE_TITLES[i["stage"]].split(" ")[0]
        else:
            prog, name, stage = "", f"{cname(w.get('cohort') or '?')} task {w['task_id']}", w.get("stage") or "?"
        seen = f"seen {span(w['beat'])} ago" if w.get("beat") is not None else f"last log {span(w['quiet'])} ago"
        out(f"   {w['host']:<34} " + paint("working", "run") + f"  {stage:<3} {name:<28} for "
            f"{span(now - w.get('started_at', now))}" + (f"  {prog}" if prog else "") + f"  · {seen}")
    for w in sorted(waiting, key=lambda w: w["host"]):
        seen = f"seen {span(w['beat'])} ago" if w.get("beat") is not None else f"last log {span(w['quiet'])} ago"
        word = "waiting" if w.get("state") == "waiting" else "idle"
        wrap(f"{w['host']:<34} {word:<8} {w.get('doing') or 'choosing the next task'} · {seen}", "   ", "       ",
             lambda t: out(paint(t, "dim")))
    if not hosts:
        n_ready = sum(1 for i in tasks if i["state"] == "ready")
        out("   no V9 worker is live" + (f" · {n_ready} tasks could start now" if n_ready else "")
            + " — command on every 4xH100 node: `sh v9/run.sh`")

    # invariants and this node's idle GPUs
    broken = check_screen(by_stage, tasks, working_rows, counted)
    if broken:
        out(paint(f" ✗ screen counts disagree ({len(broken)}): report this; the work itself is not changed", "bad"))
        for line in broken[:8]:
            out(paint(f"   {line}", "bad"))
    if gpus and not any(u >= 10 for u in gpus):
        for i in mine:
            if i["run"].get("quiet", 0) > STALE_SECONDS:
                out(paint(f" WARNING {task_name(i)}: no output for {span(i['run']['quiet'])} and this node's GPUs are "
                          f"idle (log: {i['run'].get('log')})", "bad"))


def main(plan, task_id, argv=()):
    """`sh v9/run.sh status [--watch [SECONDS]]`."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog="sh v9/run.sh status")
    ap.add_argument("--watch", nargs="?", const=3.0, type=float, default=None, metavar="SECONDS",
                    help="refresh every SECONDS (default 3) until Ctrl-C")
    args = ap.parse_args(list(argv))
    global COLOR
    COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"
    global WIDTH
    if sys.stdout.isatty():   # V8: the terminal width, kept between 80 and 160
        import shutil
        WIDTH = max(80, min(shutil.get_terminal_size((120, 40)).columns, 160))
    if args.watch is None:
        render(plan, task_id)
        return 0
    try:
        while True:
            lines = []
            render(plan, task_id, out=lines.append)
            sys.stdout.write("\033[H\033[2J\033[3J" + "\n".join(lines) +
                             f"\n\n(refresh every {int(args.watch)}s · Ctrl-C to stop)\n")
            sys.stdout.flush()
            time.sleep(max(1.0, args.watch))
    except KeyboardInterrupt:
        return 0
