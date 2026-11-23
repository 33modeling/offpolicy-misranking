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


def render(plan, task_id, out=print):
    items, errors, running = collect_items(plan, task_id)
    tasks = [i for i in items if i["stage"] not in ("freeze-tau", "collect")]
    count = {s: sum(1 for i in tasks if i["state"] == s) for s in STATES}
    out(f"V9 status · {code_commit()} · {now_text()}")
    out(f"  code {env.REPO}   results {env.v9_root()}   V8 (read only) {env.v8_root()}")

    tau = next((i for i in items if i["stage"] == "freeze-tau"), None)
    if tau is None:
        out("\n tau: unknown (planning error below)")
    elif tau["state"] == "done":
        out(f"\n tau: frozen, {tau['reason']}")
    elif tau["state"] == "ready":
        out("\n tau: NOT FROZEN yet. OLMo MATH V8 bank is complete; the first `sh v9/run.sh` freezes it (CPU, seconds),"
            "\n      then x1-x4 and collect unlock. Until then those stages show 'tau is not frozen'.")
    else:
        out(f"\n tau: NOT FROZEN and cannot be frozen yet: {tau['reason']}")

    total = len(tasks)
    out(f"\n tasks: {count['done']} of {total} done · {count['running']} running · {count['ready']} can start now · "
        f"{count['blocked']} waiting · {count['failed']} FAILED")

    run = [i for i in tasks if i["state"] == "running"]
    out(f"\n running now ({len(run)}):" if run else "\n running now: nothing")
    for i in run:
        r = i["run"]
        out(f"   {i['stage']} {i['cohort']} {i['key']}  on {r.get('host')}  for {span(time.time() - r.get('started_at', 0))}"
            f"  last output {span(r['quiet'])} ago")
        line = last_output(r.get("log", ""))
        if line:
            out(f"      {line}")
    stale = [i for i in tasks if i.get("stale")]
    for i in stale:
        r = i["stale"]
        out(f"   possibly stopped: {i['stage']} {i['cohort']} {i['key']} on {r.get('host')} (no output for {span(r['quiet'])}); "
            "a free node takes it again and resumes from latest.pt")

    failed = [i for i in tasks if i["state"] == "failed"]
    if failed or errors:
        out(f"\n FAILED — needs you ({len(failed) + len(errors)}):")
        for i in failed:
            out(f"   {i['stage']} {i['cohort']} {i['key']}: {i['reason']}")
        for stage, why in errors:
            out(f"   {stage}: planning error: {why}")
    for row in stage_failures():
        out(f"   recent stage error {row.get('at', '?')} {row.get('host', '?')} {row.get('stage', '?')}: "
            f"{str(row.get('error', ''))[:160]}")

    out("\n stages (done / tasks · waiting reasons):")
    for stage in env.STAGES:
        its = [i for i in items if i["stage"] == stage]
        if stage in ("freeze-tau", "collect"):
            if its:
                out(f"   {stage:<10} {its[0]['state']}: {its[0]['reason'] or ''}")
            continue
        cohorts = {}
        for i in its:
            cohorts.setdefault(i["cohort"], []).append(i)
        if not cohorts:
            out(f"   {stage:<10} (no items)")
        for cohort, rows in cohorts.items():
            n = {s: sum(1 for i in rows if i["state"] == s) for s in STATES}
            parts = [f"{n['done']}/{len(rows)} done"] + [f"{n[s]} {w}" for s, w in
                     (("running", "running"), ("ready", "can start"), ("blocked", "waiting"), ("failed", "FAILED")) if n[s]]
            out(f"   {stage:<10} {cohort:<11} " + " · ".join(parts))
            why = {}
            for i in rows:
                if i["state"] == "blocked":
                    why.setdefault(i["reason"] or "waiting for an earlier step", []).append(i["key"])
            for reason, keys in sorted(why.items(), key=lambda kv: -len(kv[1])):
                shown = ", ".join(keys[:4]) + (f" +{len(keys) - 4}" if len(keys) > 4 else "")
                out(f"{'':26}waiting ({len(keys)}): {reason} [{shown}]")

    ready = [i for i in tasks if i["state"] == "ready"]
    if ready:
        nxt = ready[0]
        out(f"\n next: a free node takes {nxt['stage']} {nxt['cohort']} {nxt['key']} "
            f"({len(ready)} task(s) can start now; one per 4xH100 node, command `sh v9/run.sh`)")
    elif run:
        out("\n next: nothing can start now; the running tasks unlock the next ones")
    elif count["blocked"]:
        out("\n next: nothing can start now; see the waiting reasons above")
    else:
        out("\n next: no GPU work left")

    col = next((i for i in items if i["stage"] == "collect"), None)
    if col is not None:
        out(f" result files: {env.v9_root() / 'paper'} ({col['state']}: {col['reason'] or ''})")
