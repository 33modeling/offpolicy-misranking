"""`sh v9/run.sh status`: one screen that says what V9 is doing, what is stuck and what to do.

Read-only (the V9 locator copy written by the planners on first use aside). Sections:
header (code commit, time, roots) -> tau -> totals -> running now -> FAILED (needs you) ->
per stage and cohort done/total with the waiting reasons -> what a free node takes next -> result files.
"Running" comes from the record each worker writes when it starts a task (.state/running/); a record whose
worker log has been silent for STALE_SECONDS is shown as possibly stopped, never counted as running.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from . import env

KST = timezone(timedelta(hours=9))
STALE_SECONDS = 1800
STATES = ("done", "running", "ready", "blocked", "failed")


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


def running_records():
    """task id -> record, with 'live' (worker log written within STALE_SECONDS, or a live pid on this host)."""
    out = {}
    d = running_dir()
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError):
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
            except (OSError, ValueError):
                live = False
        r.update(quiet=quiet, live=live)
        out[r.get("task_id") or p.stem] = r
    return out


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
            if rec is not None and it["state"] == "ready" and rec["live"]:
                it = dict(it, state="running", run=rec)
            elif rec is not None and it["state"] == "ready":
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
    "collect": "result files",
}
COHORT_NAMES = {"olmo_math": "OLMo MATH", "olmo_mbpp": "OLMo MBPP", "qwen_math": "Qwen MATH",
                "qwen_mbpp": "Qwen MBPP", "llama_math": "Llama MATH", "llama_mbpp": "Llama MBPP", "all": "all"}
CELL = {"done": "✓", "running": "▶", "ready": "ready", "blocked": "wait", "failed": "FAIL"}
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


def render(plan, task_id, out=print):
    items, errors, running = collect_items(plan, task_id)
    stages = [s for s in env.STAGES if s not in ("freeze-tau", "collect")]
    by_stage = {s: [i for i in items if i["stage"] == s] for s in env.STAGES}
    tasks = [i for s in stages for i in by_stage[s]]
    here = socket.gethostname()
    out(f"V9 status · {code_commit()} · {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')}")
    out(str(env.v9_root()))
    mine = [i for i in tasks if i["state"] == "running" and i["run"].get("host") == here]
    out(f"this node ({here}): " + (", ".join(f"{STAGE_TITLES[i['stage']].split(' ')[0]} {task_name(i)} running"
                                             for i in mine) or "no V9 task here"))

    live_nodes = sorted({i["run"].get("host") for i in tasks if i["state"] == "running"})
    out("\n time left")
    for s in stages:
        its = by_stage[s]
        if not its:
            continue
        n = sum(1 for i in its if i["state"] == "running")
        left = sum(1 for i in its if i["state"] != "done")
        if not left:
            out(f"   {STAGE_TITLES[s]}: finished")
        else:
            out(f"   {STAGE_TITLES[s]}: unknown — V9 has no measured task time yet ({n} node{'s' if n != 1 else ''} "
                f"working on it)")

    out("\n experiments")
    tau = (by_stage.get("freeze-tau") or [None])[0]
    if tau is None:
        out("   tau · unknown (planning error below)")
    elif tau["state"] == "done":
        out(f"   tau · frozen · {tau['reason']}")
    elif tau["state"] == "ready":
        wrap("tau · NOT FROZEN yet · the first `sh v9/run.sh` freezes it (CPU, seconds); X1-X4 and the result files "
             "wait for it", "   ", "       ", out)
    else:
        wrap(f"tau · NOT FROZEN · cannot be frozen yet: {tau['reason']}", "   ", "       ", out)
    for s in stages:
        its = by_stage[s]
        if not its:
            continue
        done = sum(1 for i in its if i["state"] == "done")
        n = sum(1 for i in its if i["state"] == "running")
        pct = int(100 * done / len(its)) if its else 0
        out(f"   {STAGE_TITLES[s]} · {pct}% of the work done · {n} node{'s' if n != 1 else ''} working on it")
        cohorts = {}
        for i in its:
            cohorts.setdefault(i["cohort"], []).append(i)
        for cohort, rows in cohorts.items():
            c = {st: sum(1 for i in rows if i["state"] == st) for st in STATES}
            parts = [f"done {c['done']} of {len(rows)}"] + [f"{c[st]} {w}" for st, w in
                     (("running", "running"), ("ready", "can start"), ("blocked", "waiting"), ("failed", "STOPPED")) if c[st]]
            out(f"     {cname(cohort):<12}" + " · ".join(parts))
    col = (by_stage.get("collect") or [None])[0]
    if col is not None:
        wrap(f"result files · {col['state']} · {col['reason'] or ''} · {env.v9_root() / 'paper'}", "   ", "       ", out)

    for s in stages:
        its = [i for i in by_stage[s] if i["state"] != "done"]
        if not its:
            continue
        out(f"\n {STAGE_TITLES[s]}: not finished {len(its)} of {len(by_stage[s])}")
        failed = [i for i in its if i["state"] == "failed"]
        if failed:
            out(f"   STOPPED — needs you ({len(failed)}):")
            for i in failed[:12]:
                wrap(f"{task_name(i)}: {i['reason']}", "     ", "       ", out)
            if len(failed) > 12:
                wrap("… %d more: %s" % (len(failed) - 12, ", ".join(task_name(i) for i in failed[12:])), "     ", "       ", out)
            out("     → fix the cause, then run `sh v9/run.sh retry-failed` on one node (the next free node restarts them)")
        run = [i for i in its if i["state"] == "running"]
        if run:
            out(f"   running now ({len(run)}):")
            for i in run:
                r = i["run"]
                wrap(f"{task_name(i)} · on {r.get('host')} · for {span(time.time() - r.get('started_at', 0))} · "
                     f"last output {span(r['quiet'])} ago", "     ", "       ", out)
                line = last_output(r.get("log", ""))
                if line:
                    out(f"       {line}")
        stale = [i for i in its if i.get("stale")]
        for i in stale:
            r = i["stale"]
            wrap(f"possibly stopped: {task_name(i)} on {r.get('host')} (no output for {span(r['quiet'])}); a free node "
                 "takes it again and resumes from latest.pt", "     ", "       ", out)
        ready = [i for i in its if i["state"] == "ready"]
        if ready:
            out(f"   can start now ({len(ready)}): the next free node takes {task_name(ready[0])}")
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
                wrap(f"{reason} ({len(rows)}): {names}", "     ", "       ", out)

    if errors or stage_failures():
        out(f"\n STOPPED — needs you: problems with a stage itself ({len(errors) + len(stage_failures())})")
        for stage, why in errors:
            wrap(f"{stage}: planning error: {why}", "   ", "     ", out)
        for row in stage_failures():
            wrap(f"{row.get('stage', '?')}: {row.get('at', '?')} on {row.get('host', '?')}: {str(row.get('error', ''))[:300]}",
                 "   ", "     ", out)

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
                sd, col, _ = split_key(i["key"])
                cell.setdefault((sd, col), []).append(CELL.get(i["state"], i["state"]))
            texts = {k: ",".join(v) for k, v in cell.items()}
            width = {c: max([len(c)] + [len(texts.get((sd, c), "-")) for sd in seeds]) + 2 for c in cols}
            out(f"\n {cname(cohort)} {STAGE_TITLES[s].split(' ')[0]} by seed and arm")
            out(" " + f"{'':<12}" + "".join(f"{c:<{width[c]}}" for c in cols))
            for sd in seeds:
                out("   " + f"{sd:<10}" + "".join(f"{texts.get((sd, c), '-'):<{width[c]}}" for c in cols))
    out("   cells: ✓ done · ▶ running · ready = can start now · wait = needs an earlier step or V8 result · "
        "FAIL = stopped, needs you · - = no such task")

    out(f"\n nodes: {len(live_nodes)} working on V9" + (" — command on every 4xH100 node: `sh v9/run.sh`" if not live_nodes else ""))
    for i in sorted((i for i in tasks if i["state"] == "running"), key=lambda i: i["run"].get("host", "")):
        r = i["run"]
        out(f"   {r.get('host'):<34} working  {STAGE_TITLES[i['stage']].split(' ')[0]}  {task_name(i)}  "
            f"for {span(time.time() - r.get('started_at', 0))}")
