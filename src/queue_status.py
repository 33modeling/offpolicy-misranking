"""One-screen overview of the remaining queue (scripts/run_queue.sh status).

One line per queue step, in queue order, with a single state word:

    DONE      every artifact of the step exists
    RUNNING   a lease of the step is held right now (on this or another node)
    PARTIAL   some artifacts exist, nothing holds a lease (stopped or between steps)
    WAITING   a prerequisite step is not finished
    TODO      nothing started

Seed-level detail follows on indented lines; a held lease is marked
``*[node]`` with the node that wrote the lease note (``*[?]`` when the lease
predates the notes). A ``nodes`` section lists every node that ran the
queue (note + heartbeat under $OM_WORK/queue) and the leases it holds, and
``this node`` lists the queue processes on the current machine. Everything
is read from the filesystem; no GPU, no lease is taken (the lease probe
releases immediately). The only write is this node's own process view,
$OM_WORK/queue/<host>.seen.json, so that the overview on any node knows what
every node where status has run was doing (and can attribute leases taken
before the lease notes existed). Run status once on each allocated node.

    python src/queue_status.py [--work $OM_WORK] [--root $OM_OLMO3_ROOT] [--tag TAG] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ARM_LABELS = {"before": "before", "random": "random", "passrate_beta": "difficulty", "fresh_r": "fresh",
              "g11": "reused", "g00": "reused00", "g10": "reused10", "g01": "reused01", "gate_passrate": "gate"}
MIX_ARMS = ("random", "passrate_beta", "fresh_r", "g11")
STATES = ("DONE", "RUNNING", "PARTIAL", "WAITING", "TODO")
HEARTBEAT_ALIVE_SECONDS = 300
NOTE_MAX_AGE_SECONDS = 3 * 86400

# host -> list of "what" strings, filled while the rows are built; hosts whose
# identity was inferred (launcher log or a status run there) are in INFERRED
NODES: dict[str, list[str]] = {}
INFERRED: set[str] = set()
# host -> {'age', 'jobs', 'lock'}: what each node was running when `status` last ran there
SEEN: dict[str, dict] = {}
# hosts busy in a seed directory whose leases cannot be paired to them one by one
# (several launcher logs active there): host -> ["<label> s0: one of random, fresh", ...]
SHARED: dict[str, list[str]] = {}
LAUNCHER_ACTIVE_SECONDS = 3 * 3600
SEEN_FRESH_SECONDS = 300   # the node watcher reports every minute


# ---------------------------------------------------------------- helpers

def read_json(path: Path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def age(path: Path) -> str:
    try:
        seconds = max(0, int(time.time() - path.stat().st_mtime))
    except OSError:
        return "-"
    return age_text(seconds)


def age_text(seconds: int) -> str:
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d{(seconds % 86400) // 3600}h"


def lease_held(path: Path) -> bool:
    """True when another process holds flock on ``path``. Never creates the file."""
    if not path.is_file():
        return False
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError):
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def parse_note(text: str) -> dict:
    return dict(item.split("=", 1) for item in text.split() if "=" in item)


def active_launcher_hosts(logs_dir: Path, prefix: str, within: int = LAUNCHER_ACTIVE_SECONDS) -> list[str]:
    """Hosts of the '<prefix>-<host>-<UTC>.log' files in logs_dir written within `within` seconds,
    newest first (one launcher log per node and pass; a node writes it while its pass runs)."""
    hosts: dict[str, float] = {}
    for log in logs_dir.glob(f"{prefix}-*-*Z.log"):
        m = re.fullmatch(rf"{re.escape(prefix)}-(.+)-\d{{8}}T\d{{6}}Z\.log", log.name)
        if not m:
            continue
        try:
            mtime = log.stat().st_mtime
        except OSError:
            continue
        if time.time() - mtime <= within:
            hosts[m.group(1)] = max(hosts.get(m.group(1), 0.0), mtime)
    return [h for h, _ in sorted(hosts.items(), key=lambda kv: -kv[1])]


def launcher_host(logs_dir: Path, prefix: str) -> tuple[str, int] | None:
    """(host, 0) of the only active launcher log in logs_dir; None when none or several are active."""
    hosts = active_launcher_hosts(logs_dir, prefix)
    return (hosts[0], 0) if len(hosts) == 1 else None


def share_leases(logs_dir: Path, prefix: str, where: str, leases: list[str]) -> None:
    """Several nodes are active in one seed directory: mark each as busy there without pairing."""
    hosts = active_launcher_hosts(logs_dir, prefix)
    if len(hosts) >= 2 and leases:
        for host in hosts:
            SHARED.setdefault(host, []).append(f"{where}: one of {', '.join(leases)}")


def seen_host_for(job: str | None) -> str | None:
    """The one node whose recent `status` run saw this job among its processes, if unique."""
    if not job:
        return None
    hosts = [h for h, s in SEEN.items() if job in s["jobs"] and s["age"] < SEEN_FRESH_SECONDS]
    return hosts[0] if len(hosts) == 1 else None


def lease_holder(path: Path, logs_dir: Path | None = None, prefix: str | None = None, job: str | None = None) -> str | None:
    """None when the lease is free; otherwise the host from the lease note; else (jobs started
    before the notes, marked '~host') the host of the newest launcher log of that step, or the
    node where `status` recently saw the job running; else '?'."""
    if not lease_held(path):
        return None
    try:
        first = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        first = []
    host = parse_note(first[0]).get("host") if first else None
    if host:
        return host
    if logs_dir is not None and prefix:
        found = launcher_host(logs_dir, prefix)
        if found:
            return f"~{found[0]}"
    seen = seen_host_for(job)
    return f"~{seen}" if seen else "?"


def short_host(host: str) -> str:
    tilde = "~" if host.startswith("~") else ""
    parts = host.lstrip("~").split("-")
    return tilde + ("-".join(parts[-2:]) if len(parts) > 2 else host.lstrip("~"))


def since_text(stamp: str) -> str:
    m = re.match(r"\d{4}-(\d{2}-\d{2})T(\d{2}:\d{2}):\d{2}Z", stamp)
    return f"{m.group(1)} {m.group(2)}Z" if m else stamp


def note_lease(holder: str | None, what: str) -> str:
    """Register a held lease under its node and return the inline marker."""
    if holder is None:
        return ""
    inferred = holder.startswith("~")
    host = holder.lstrip("~")
    NODES.setdefault(host, []).append(what)
    if inferred:
        INFERRED.add(host)
    return f"*[{'~' if inferred else ''}{short_host(host)}]"


def newest_write(path: Path, pattern: str = "*") -> tuple[float, int]:
    """(newest mtime, total bytes) of the files under path matching pattern (recursive); (0, 0) when none."""
    latest, total = 0.0, 0
    try:
        for entry in path.rglob(pattern):
            if entry.is_file():
                st = entry.stat()
                latest = max(latest, st.st_mtime)
                total += st.st_size
    except OSError:
        pass
    return latest, total


def write_note(path: Path, pattern: str = "*") -> str:
    """' (write 3m ago)' for the newest file under path, or ' (no write yet)'."""
    latest, _ = newest_write(path, pattern)
    return f" (write {age_text(max(0, int(time.time() - latest)))} ago)" if latest else " (no write yet)"


def row_count(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def last_progress(run: Path) -> str:
    """'k/8 <stage> +<min>' from the point runner's own [progress] lines."""
    log = run / "logs" / "main.log"
    if not log.is_file():
        return ""
    try:
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 65536))
            lines = handle.read().decode(errors="replace").splitlines()
    except OSError:
        return ""
    text = ""
    for line in lines:
        if "[progress]" in line:
            text = line.split("[progress]", 1)[1].strip()
    if not text:
        return ""
    fields = [f.strip() for f in re.split(r"\s{2,}", text) if f.strip()]
    if len(fields) >= 2:
        fields = fields[1:]
    stage = re.sub(r"\s*\(.*?\)", "", fields[0]).strip() if fields else text
    words = stage.split()
    if len(words) >= 2 and re.fullmatch(r"\d+/\d+", words[0]):
        stage = f"{words[0]} {words[1]}"
    elapsed = fields[-1] if len(fields) > 1 and fields[-1].startswith("+") else ""
    return f"{stage} {elapsed}".strip()


PROBLEM = re.compile(r"^\s*\[(abort|failed|busy|skip|error)\]", re.I)


def tail_lines(path: Path, limit: int = 400) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 65536))
            return handle.read().decode(errors="replace").splitlines()[-limit:]
    except OSError:
        return []


def last_problem(path: Path) -> str:
    """The last '[abort]/[failed]/[busy]/[skip]' line of a log, shortened; '' when none."""
    lines = [l.strip() for l in tail_lines(path) if PROBLEM.match(l)]
    return lines[-1][:110] if lines else ""


def newest_launcher_log(logs_dir: Path, prefix: str) -> Path | None:
    logs = [p for p in logs_dir.glob(f"{prefix}-*-*Z.log") if p.is_file()]
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def queue_log_state(work: Path, host: str) -> str:
    """Last step marker and last problem line of a node's queue console log on the shared filesystem."""
    log = work / "queue" / f"{host}.log"
    if not log.is_file():
        return ""
    lines = tail_lines(log)
    steps = [l for l in lines if l.startswith("=====")]
    problems = [l.strip() for l in lines if PROBLEM.match(l)]
    text = f"queue log: {steps[-1][:80]}" if steps else "queue log present"
    if problems:
        text += f" | last problem: {problems[-1][:110]}"
    return text


def git_short() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parents[1])
        return out.stdout.strip() or "?"
    except (OSError, subprocess.SubprocessError):
        return "?"


# ---------------------------------------------------------------- E5 branch arms

def arms_of(out: Path, contract: dict) -> list[str]:
    arms = list(contract.get("selectors", []))
    extra = out / "arms.json"
    if extra.is_file():
        try:
            for arm in read_json(extra).get("arms", []):
                if arm not in arms:
                    arms.append(arm)
        except (OSError, ValueError, AttributeError):
            pass
    return arms


def arm_state(out: Path, arm: str, contract: dict) -> tuple[str, bool]:
    """(short state, done). Words: ok / eval n/4 / trained / train n/N / pilot ... / decided ... / -"""
    evaluation = out / arm / "evaluation"
    done = sum((evaluation / f"shard-{s}.done.json").is_file() for s in range(4))
    if done == 4:
        return "ok", True
    if done or any((evaluation / f"shard-{s}.jsonl.partial").is_file() for s in range(4)):
        return f"eval {done}/4", False
    if arm == "before":
        return "-", False
    policy = out / arm / "policy"
    steps = int(contract.get("steps", 0) or 0)
    if (policy / "policy_train.json").is_file():
        return "trained", False
    if policy.is_dir():
        logged = row_count(policy / "grpo_stats.jsonl")
        return f"train {logged}/{steps}", False
    if arm.startswith("gate"):
        decision = out / arm / "decision.json"
        pilot = out / arm / "pilot"
        if decision.is_file():
            try:
                d = read_json(decision)
                return f"decided {d.get('decision', '?')}", False
            except (OSError, ValueError):
                return "decided", False
        if (pilot / "policy_train.json").is_file():
            return "pilot trained", False
        if pilot.is_dir():
            rule_steps = 0
            try:
                rule_steps = int(read_json(out / "gate_rule.json").get("pilot_steps", 0) or 0)
            except (OSError, ValueError, AttributeError):
                pass
            logged = row_count(pilot / "grpo_stats.jsonl")
            return f"pilot {logged}/{rule_steps}" if rule_steps else f"pilot {logged}", False
    return "-", False


def branch_seed(out: Path, arms=None, label: str = "") -> dict:
    """State of one E5 seed directory: {'prepared', 'done', 'running', 'started', 'line'}."""
    if not (out / "experiment.json").is_file():
        return {"prepared": False, "done": False, "running": False, "started": False, "line": "not prepared"}
    try:
        contract = read_json(out / "experiment.json")
    except (OSError, ValueError):
        return {"prepared": True, "done": False, "running": False, "started": False, "line": "experiment.json unreadable"}
    names = list(arms) if arms else arms_of(out, contract)
    parts, all_done, started, running, unpaired = [], True, False, False, []
    for arm in ["before", *names]:
        word, done = arm_state(out, arm, contract)
        all_done = all_done and done
        started = started or word != "-"
        holder = lease_holder(out / f".{arm}.lock", out / "logs", "launcher", f"E5 arms {out.parent.name}")
        if holder is not None:
            running = True
            if holder == "?":
                unpaired.append(ARM_LABELS.get(arm, arm))
            word += note_lease(holder, f"{label} {out.name} {ARM_LABELS.get(arm, arm)} ({word})".strip())
            word += write_note(out / arm)
        parts.append(f"{ARM_LABELS.get(arm, arm)} {word}")
    share_leases(out / "logs", "launcher", f"{label} {out.name}", unpaired)
    line = "DONE" if all_done else " | ".join(parts)
    if not all_done and not running:
        log = newest_launcher_log(out / "logs", "launcher")
        problem = last_problem(log) if log else ""
        if problem:
            line += f"  <- last: {problem}"
    if all_done and not (out / "downstream_results.csv").is_file():
        line = "evaluated, summary pending"
        all_done = False
    return {"prepared": True, "done": all_done, "running": running, "started": started, "line": line}


def branch_state(root: Path, seeds, arms=None, label: str = "") -> tuple[str, list[str]]:
    infos = {seed: branch_seed(root / f"s{seed}", arms, label) for seed in seeds}
    lines = [f"s{seed}: {info['line']}" for seed, info in infos.items()]
    if all(info["done"] for info in infos.values()):
        return "DONE", []
    if any(info["running"] for info in infos.values()):
        return "RUNNING", lines
    if any(info["started"] or info["prepared"] for info in infos.values()):
        return "PARTIAL", lines
    return "TODO", []


# ---------------------------------------------------------------- benchmarks

def bench_seed(out: Path, label: str = "") -> dict:
    if not (out / "experiment.json").is_file():
        return {"done": False, "running": False, "started": False, "line": "E5 seed not prepared"}
    if not (out / "benchmarks.json").is_file():
        return {"done": False, "running": False, "started": False, "line": "-"}
    try:
        contract = read_json(out / "experiment.json")
        sets = list(read_json(out / "benchmarks.json").get("sets", {}))
    except (OSError, ValueError, AttributeError):
        return {"done": False, "running": False, "started": True, "line": "contract unreadable"}
    parts, all_done, started, running, unpaired = [], True, False, False, []
    for arm in ["before", *arms_of(out, contract)]:
        trained = arm == "before" or (out / arm / "policy" / "policy_train.json").is_file()
        finished = sum(all((out / arm / "benchmark" / name / f"shard-{s}.done.json").is_file() for s in range(4)) for name in sets)
        partial = any((out / arm / "benchmark" / name / f"shard-{s}.done.json").is_file() for name in sets for s in range(4))
        if finished == len(sets):
            word = "ok"
        elif not trained and not partial:
            word = "wait"          # policy not trained yet; the pass skips it
            all_done = False
        else:
            word = f"{finished}/{len(sets)}"
            all_done = False
        started = started or partial
        holder = lease_holder(out / f".bench-{arm}.lock", out / "logs", "bench-launcher", f"benchmarks {out.parent.name}")
        if holder is not None:
            running = True
            if holder == "?":
                unpaired.append(ARM_LABELS.get(arm, arm))
            word += note_lease(holder, f"{label} {out.name} {ARM_LABELS.get(arm, arm)} ({word})".strip())
            word += write_note(out / arm / "benchmark")
        parts.append(f"{ARM_LABELS.get(arm, arm)} {word}")
    share_leases(out / "logs", "bench-launcher", f"{label} {out.name}", unpaired)
    line = "DONE" if all_done else " | ".join(parts)
    if not all_done and not running:
        log = newest_launcher_log(out / "logs", "bench-launcher")
        problem = last_problem(log) if log else ""
        if problem:
            line += f"  <- last: {problem}"
    return {"done": all_done, "running": running, "started": started, "line": line}


def bench_state(root: Path, seeds, label: str = "") -> tuple[str, list[str]]:
    infos = {seed: bench_seed(root / f"s{seed}", label) for seed in seeds}
    lines = [f"s{seed}: {info['line']}" for seed, info in infos.items()]
    if all(info["done"] for info in infos.values()):
        return "DONE", []
    if any(info["running"] for info in infos.values()):
        return "RUNNING", lines
    if any(info["started"] for info in infos.values()):
        return "PARTIAL", lines
    if all(info["line"] == "E5 seed not prepared" for info in infos.values()):
        return "WAITING", ["E5 branch not prepared"]
    return "TODO", []


# ---------------------------------------------------------------- reuse split-half

def stale_state(run_dir, seeds, drift: int, label: str = "") -> tuple[str, list[str]]:
    parts, all_done, running, started = [], True, False, False
    for seed in seeds:
        run = run_dir(seed, drift)
        if (run / "scores_stale_splithalf.json").is_file():
            parts.append(f"s{seed} ok")
            started = True
            continue
        all_done = False
        if not (run / "DONE").is_file():
            parts.append(f"s{seed} no point")
            continue
        shards = len(list(run.glob("scores_stale_splithalf.shard*.json")))
        word = f"s{seed} {shards}/4 shards" if shards else f"s{seed} -"
        started = started or shards > 0
        holder = lease_holder(run / ".stale-splithalf.lock", job=f"reuse split-half d{drift}")
        if holder is not None:
            running = True
            word += note_lease(holder, f"{label} s{seed}")
            word += write_note(run / "logs", "stale-splithalf-*.log")
        parts.append(word)
    if all_done:
        return "DONE", []
    state = "RUNNING" if running else ("PARTIAL" if started else "TODO")
    return state, ["  ".join(parts)]


# ---------------------------------------------------------------- mixed pool

def mixed_states(work: Path, root: Path, tag: str, other: str, seed: int, steps: int):
    pool = work / "inputs" / "mixed" / f"pool-math500-{other}-s{seed}.jsonl"
    name = "math500mix"
    point = root / f"family-{name}-s{seed}" / f"{tag}-s{seed}-{name}-d0"
    branch = work / "runs" / "e5-reduced" / f"{name}-d0"
    rows = []
    # pool
    if pool.is_file() and pool.stat().st_size > 0:
        rows.append(("mixed pool: pool", "DONE", []))
    else:
        math_run = root / f"family-math500-s{seed}" / f"{tag}-s{seed}-math500-d0"
        other_run = root / f"family-{other}-s{seed}" / f"{tag}-s{seed}-{other}-d0"
        missing = [r.name for r in (math_run, other_run) if not (r / "DONE").is_file()]
        rows.append(("mixed pool: pool", "WAITING" if missing else "TODO",
                     [f"source point not complete: {m}" for m in missing]))
    pool_ready = rows[-1][1] == "DONE"
    # point
    if (point / "DONE").is_file():
        rows.append(("mixed pool: point", "DONE", []))
    else:
        progress = last_progress(point)
        log = point / "logs" / "main.log"
        latest, _ = newest_write(point)
        _, rollout_bytes = newest_write(point, "rollouts_*")
        activity = (f"last file write {age_text(max(0, int(time.time() - latest)))} ago" if latest else "no file written yet") \
            + (f", rollouts {rollout_bytes / 1e6:.0f} MB" if rollout_bytes else "")
        holder = lease_holder(Path(str(point) + ".lease"), job=f"point {point.name}")
        if holder is not None:
            mark = note_lease(holder, f"mixed pool point ({progress or 'started'})")
            rows.append(("mixed pool: point", "RUNNING", [f"{progress or 'started'} {mark}", activity,
                                                          "(the stage line changes only at stage boundaries; a stage takes hours)"]))
        elif log.is_file():
            rows.append(("mixed pool: point", "PARTIAL", [f"stopped at {progress or '?'}", activity]))
        elif pool_ready:
            rows.append(("mixed pool: point", "TODO", []))
        else:
            rows.append(("mixed pool: point", "WAITING", ["pool not built"]))
    point_done = rows[-1][1] == "DONE"
    # arms and gate
    out = branch / f"s{seed}"
    if not point_done:
        rows.append(("mixed pool: arms", "WAITING", ["point not done"]))
        rows.append(("mixed pool: gate", "WAITING", ["point not done"]))
        return rows
    state, lines = branch_state(branch, [seed], MIX_ARMS, "mixed pool arms")
    rows.append((f"mixed pool: arms ({steps} upd)", state, lines))
    if (out / "experiment.json").is_file():
        try:
            contract = read_json(out / "experiment.json")
        except (OSError, ValueError):
            contract = {}
        word, done = arm_state(out, "gate_passrate", contract)
        holder = lease_holder(out / ".gate_passrate.lock", out / "logs", "launcher", f"E5 arms {branch.name}")
        if done:
            rows.append(("mixed pool: gate", "DONE", []))
        elif holder is not None:
            mark = note_lease(holder, f"mixed pool gate ({word})")
            rows.append(("mixed pool: gate", "RUNNING", [f"s{seed}: gate {word}{mark}"]))
        elif word == "-":
            rows.append(("mixed pool: gate", "TODO" if state == "DONE" else "WAITING", []))
        else:
            rows.append(("mixed pool: gate", "PARTIAL", [f"s{seed}: gate {word}"]))
    else:
        rows.append(("mixed pool: gate", "WAITING", ["arms not prepared"]))
    return rows


# ---------------------------------------------------------------- exports

def exports_state(work: Path) -> tuple[str, list[str]]:
    exports = work / "exports"
    lines = []
    bundles = sorted(exports.glob("e5-results-*.txt"), key=lambda p: p.stat().st_mtime) if exports.is_dir() else []
    if bundles:
        lines.append(f"last bundle {age(bundles[-1])} ago: {bundles[-1].name}")
    for kind in ("gate-decision", "gain-law", "cost-accounting"):
        files = sorted(exports.glob(f"{kind}-*.txt"), key=lambda p: p.stat().st_mtime) if exports.is_dir() else []
        lines.append(f"{kind}: {age(files[-1]) + ' ago' if files else 'none'}")
    return ("PARTIAL" if bundles else "TODO"), lines


# ---------------------------------------------------------------- nodes

def queue_notes(work: Path, now: float | None = None) -> dict[str, dict]:
    """host -> {'step', 'since', 'alive', 'beat_age'} from $OM_WORK/queue/<host>.txt and .beat."""
    notes = {}
    directory = work / "queue"
    if not directory.is_dir():
        return notes
    now = time.time() if now is None else now
    for note in sorted(directory.glob("*.txt")):
        try:
            fields = parse_note(note.read_text(encoding="utf-8", errors="replace"))
            note_age = now - note.stat().st_mtime
        except OSError:
            continue
        beat = note.with_suffix(".beat")
        try:
            beat_seconds = max(0, int(now - beat.stat().st_mtime)) if beat.is_file() else None
        except OSError:
            beat_seconds = None
        if note_age > NOTE_MAX_AGE_SECONDS and (beat_seconds is None or beat_seconds > NOTE_MAX_AGE_SECONDS):
            continue
        host = fields.get("host", note.stem)
        notes[host] = {"step": fields.get("step", "?"), "since": fields.get("since", "?"),
                       "alive": beat_seconds is not None and beat_seconds < HEARTBEAT_ALIVE_SECONDS,
                       "beat_age": age_text(beat_seconds) if beat_seconds is not None else None}
    return notes


def seen_nodes(work: Path, now: float | None = None) -> dict[str, dict]:
    """host -> {'age', 'jobs', 'lock', 'gpu_busy', 'gpu'} from $OM_WORK/queue/<host>.seen.json
    (written by the node watchers); ages against `now` (the shared filesystem clock when given)."""
    import json
    seen = {}
    directory = work / "queue"
    if not directory.is_dir():
        return seen
    now = time.time() if now is None else now
    for path in directory.glob("*.seen.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            seconds = int(now - path.stat().st_mtime)
        except (OSError, ValueError):
            continue
        if seconds > NOTE_MAX_AGE_SECONDS or not isinstance(record, dict):
            continue
        util = [int(u) for u in record.get("gpu_util", []) if isinstance(u, int)]
        procs = [str(x) for x in record.get("gpu_procs", [])]
        seen[str(record.get("host", path.name.split(".")[0]))] = {
            "age": max(0, seconds), "jobs": [str(j) for j in record.get("jobs", [])], "lock": bool(record.get("lock")),
            "gpu_busy": bool(record.get("gpu_busy")),
            "gpu": (f"GPU util {'/'.join(str(u) for u in util)}%" if util else "") + (f", {'; '.join(procs)[:100]}" if procs else "")}
    return seen


def record_seen(work: Path) -> float | None:
    """Leave this node's view (queue steps, node lock, nvidia-smi) under $OM_WORK/queue so the overview
    on any node knows it (best effort). Returns the written file's mtime: the shared filesystem's
    clock, which the freshness of other nodes' reports is measured against (node clocks may differ)."""
    import json
    directory = Path(os.environ.get("OM_LOCAL_LOCK_DIR", f"/tmp/offpolicy-misranking-{os.getuid()}"))
    gpu = gpu_snapshot()
    record = {"host": socket.gethostname(), "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "jobs": node_jobs(), "lock": lease_held(directory / "primary.lock"),
              "gpu_util": gpu.get("util", []), "gpu_procs": sorted({p["cmd"] for p in gpu.get("procs", [])}),
              "gpu_busy": gpu_busy(gpu)}
    try:
        (work / "queue").mkdir(parents=True, exist_ok=True)
        target = work / "queue" / f"{record['host']}.seen.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(record) + "\n", encoding="utf-8")
        tmp.replace(target)
        return target.stat().st_mtime
    except OSError as exc:
        print(f"[warn] could not write this node's report under {work / 'queue'}: {exc}", file=sys.stderr)
        return None


WORK: list[Path] = []


def node_lines(notes: dict[str, dict], seen: dict[str, dict] | None = None) -> list[str]:
    seen = seen if seen is not None else SEEN
    unknown = NODES.pop("?", [])
    for host in SHARED:
        INFERRED.add(host)
    shared_leases = {w for items in SHARED.values() for w in items}
    unknown = [w for w in unknown if not any(w.split(" (")[0].rsplit(" ", 1)[0] in s for s in shared_leases)]
    hosts = sorted(set(notes) | set(NODES) | set(seen) | set(SHARED), key=short_host)
    fresh = lambda h: seen.get(h, {}).get("age", 10**9) < SEEN_FRESH_SECONDS  # noqa: E731
    disp = lambda h: ("~" if h in INFERRED else "") + short_host(h)  # noqa: E731
    queue_live = lambda h: notes.get(h, {}).get("alive") and notes[h]["step"] not in ("done", "stopped")  # noqa: E731
    busy = [h for h in hosts if h in NODES or h in SHARED or queue_live(h)
            or (fresh(h) and (seen[h]["jobs"] or seen[h].get("gpu_busy") or seen[h]["lock"]))]
    idle = [h for h in hosts if h not in busy and fresh(h)]
    gone = [h for h in hosts if h not in busy and h not in idle]
    reporting = [h for h in hosts if fresh(h)]
    lines = [f"  nodes reporting now: {len(reporting)}  (only nodes where run_queue.sh ran once; others are invisible)",
             f"  BUSY {len(busy)}: {', '.join(disp(h) for h in busy) or 'none'}"
             + (f"  + {len(unknown)} lease(s) on an unidentified node" if unknown else ""),
             f"  IDLE {len(idle)} (alive, nothing running; start the queue there): {', '.join(disp(h) for h in idle) or 'none'}",
             f"  GONE {len(gone)} (no report for {SEEN_FRESH_SECONDS // 60}+ min: allocation ended or killed): {', '.join(disp(h) for h in gone) or 'none'}"]
    for host in hosts:
        note = notes.get(host)
        if note is None:
            head = ("lease holder inferred (job started before the lease notes)"
                    if host in INFERRED else "no queue running here")
        elif note["step"] == "done":
            head = f"queue finished {since_text(note['since'])}"
        elif note["step"] == "stopped":
            head = f"queue stopped {since_text(note['since'])}"
        elif note["alive"]:
            head = f"{note['step']}  since {since_text(note['since'])}  alive (heartbeat {note['beat_age']} ago)"
        else:
            beat = f"no heartbeat for {note['beat_age']}" if note["beat_age"] else "no heartbeat"
            head = f"{note['step']}  since {since_text(note['since'])}  {beat.upper()} (killed?)"
        lines.append(f"  {disp(host):<10} {head}")
        if WORK:
            state = queue_log_state(WORK[0], host)
            if state:
                lines.append(f"  {'':<10}   {state}")
        if host in seen:
            s = seen[host]
            view = f"running {', '.join(s['jobs'])}" if s["jobs"] else ("GPU lock held by a non-queue job" if s["lock"] else "idle")
            if s.get("gpu"):
                view += f" | {s['gpu']}"
            if s.get("gpu_busy") and not s["jobs"]:
                view += " | GPUs busy but no queue step visible: job started outside the queue or from another shell"
            lines.append(f"  {'':<10}   reported {age_text(s['age'])} ago from that node: {view}")
        for what in NODES.get(host, []):
            lines.append(f"  {'':<10}   holds: {what}")
        for what in SHARED.get(host, []):
            lines.append(f"  {'':<10}   busy in {what} (its launcher log is active there)")
    for what in unknown:
        lines.append(f"  {'?':<10} holds: {what}  (node unknown: lease taken before the notes, no launcher log with a host name)")
    return lines


def job_label(marker: str) -> str:
    path = marker.rstrip("/")
    name = path.split("/")[-1]
    parent = path.split("/")[-2] if "/" in path else ""
    if name == ".bench":
        return f"benchmarks {parent}"
    if name.startswith(".stale-splithalf-"):
        return f"reuse split-half {name.split('-')[-1]}"
    if "/e5-reduced/" in path:
        return f"E5 arms {name}"
    if name in SUITES:
        return SUITES[name]
    if "/family-" in path:
        return f"point {name}"
    return path


def marked_processes() -> list[tuple[int, str]]:
    """(pid, OUT_ROOT) of every process on this machine that carries the queue marker (own processes)."""
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        if proc.name == str(os.getpid()):
            continue
        try:
            environ = (proc / "environ").read_bytes()
        except OSError:
            continue
        for item in environ.split(b"\0"):
            if item.startswith(b"OUT_ROOT="):
                marker = item[9:].decode(errors="replace")
                if marker:
                    found.append((int(proc.name), marker))
                break
    return found


def node_jobs() -> list[str]:
    """Labels of the queue processes on this machine (every one carries OUT_ROOT in its environment)."""
    return sorted({job_label(m) for _, m in marked_processes()})


def kill_orphans() -> list[str]:
    """Stop marked GPU processes on this node when no live driver holds the node lock: their driver
    died (for example with the terminal of the old queue pipeline), their leases are released, and
    they would collide with the work a fresh queue starts. Returns the labels stopped."""
    import signal
    directory = Path(os.environ.get("OM_LOCAL_LOCK_DIR", f"/tmp/offpolicy-misranking-{os.getuid()}"))
    if lease_held(directory / "primary.lock"):
        print("[orphans] node lock held: a live driver runs on this node; nothing stopped")
        return []
    procs = marked_processes()
    if not procs:
        print("[orphans] none")
        return []
    labels = sorted({job_label(m) for _, m in procs})
    print(f"[orphans] stopping {len(procs)} process(es) of a dead driver: {', '.join(labels)}")
    for pid, _ in procs:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + 20
    while time.time() < deadline and any(Path(f"/proc/{pid}").exists() for pid, _ in procs):
        time.sleep(1)
    for pid, _ in procs:
        if Path(f"/proc/{pid}").exists():
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    return labels


def gpu_snapshot() -> dict:
    """What nvidia-smi sees on this node: per-GPU utilisation (%) and the compute processes.
    {} when nvidia-smi is unavailable. A process whose /proc entry is not readable from this
    shell is reported as 'not visible' (different container or namespace)."""
    try:
        util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return {}
    if util.returncode != 0:
        return {}
    snapshot = {"util": [], "mem_mb": [], "procs": []}
    for line in util.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            snapshot["util"].append(int(parts[0]))
            snapshot["mem_mb"].append(int(parts[1]) if parts[1].isdigit() else 0)
    if apps.returncode == 0:
        for line in apps.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            pid, name = parts[0], parts[1].split("/")[-1]
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
                cmd = " ".join(w.split("/")[-1] for w in cmd.split()[:3]) or name
            except OSError:
                cmd = f"{name} (pid {pid}, not visible from this shell)"
            snapshot["procs"].append({"pid": int(pid), "cmd": cmd})
    return snapshot


def gpu_text(snapshot: dict) -> str:
    if not snapshot:
        return "nvidia-smi unavailable"
    util = "/".join(str(u) for u in snapshot.get("util", [])) or "?"
    procs = snapshot.get("procs", [])
    names = sorted({p["cmd"] for p in procs})
    return f"GPU util {util}%, {len(procs)} GPU process(es)" + (f": {'; '.join(names)[:120]}" if names else "")


def gpu_busy(snapshot: dict) -> bool:
    return bool(snapshot) and (bool(snapshot.get("procs")) or max(snapshot.get("util", [0]) or [0]) >= 10)


def node_line() -> str:
    directory = Path(os.environ.get("OM_LOCAL_LOCK_DIR", f"/tmp/offpolicy-misranking-{os.getuid()}"))
    lock = directory / "primary.lock"
    jobs = node_jobs()
    held = lease_held(lock)
    gpu = gpu_snapshot()
    host = socket.gethostname()
    markers = f"queue steps here: {', '.join(jobs)}" if jobs else "no queue step visible from this shell"
    if jobs or gpu_busy(gpu) or held:
        return f"this node ({host}): BUSY. {gpu_text(gpu)}. {markers}" + ("; node lock held" if held else "")
    return f"this node ({host}): IDLE. {gpu_text(gpu)}. {markers} -> bash scripts/run_queue.sh"


# ---------------------------------------------------------------- report

def build_rows(work: Path, root: Path, tag: str, seeds, other: str, mix_seed: int, mix_steps: int):
    NODES.clear()
    INFERRED.clear()
    SHARED.clear()

    def run_dir(seed, drift):
        return root / f"family-math500-s{seed}" / f"{tag}-s{seed}-math500-d{drift}"
    e5 = work / "runs" / "e5-reduced"
    rows = list(mixed_states(work, root, tag, other, mix_seed, mix_steps))
    rows.append(("reuse split-half d400",) + stale_state(run_dir, seeds, 400, "reuse split-half d400"))
    rows.append(("reuse split-half d0",) + stale_state(run_dir, seeds, 0, "reuse split-half d0"))
    rows.append(("benchmarks d0",) + bench_state(e5 / "math500-d0", seeds, "benchmarks d0"))
    rows.append(("benchmarks d400",) + bench_state(e5 / "math500-d400", seeds, "benchmarks d400"))
    rows.append(("d100 continuation",) + branch_state(e5 / "math500-d100", seeds, None, "d100"))
    rows.append(("analyses + export",) + exports_state(work))
    SUITE_LINES[:] = suite_lines(work)
    return rows


SUITE_LINES: list[str] = []


def action_lines(rows, notes: dict[str, dict], seen: dict[str, dict]) -> list[str]:
    """What to restart, where, how: steps with artifacts but no driver, and the nodes to use."""
    stalled = [name for name, state, _ in rows if state == "PARTIAL" and name != "analyses + export"]
    fresh = lambda h: seen.get(h, {}).get("age", 10**9) < SEEN_FRESH_SECONDS  # noqa: E731
    disp = lambda h: ("~" if h in INFERRED else "") + short_host(h)  # noqa: E731
    live_queue = {h for h, n in notes.items() if n["alive"] and n["step"] not in ("done", "stopped")}
    idle = [h for h in seen if fresh(h) and not seen[h]["jobs"] and not seen[h].get("gpu_busy") and h not in NODES and h not in live_queue]
    orphaned = [h for h in seen if fresh(h) and seen[h].get("gpu_busy") and not seen[h]["jobs"] and h not in NODES and h not in live_queue]
    lines = []
    if stalled:
        lines.append("  stalled (artifacts exist, no driver holds them):")
        for name in stalled:
            lines.append(f"      {name}")
    else:
        lines.append("  stalled: none (every started step has a driver)")
    if orphaned:
        lines.append(f"  first, on {', '.join(disp(h) for h in orphaned)} (GPU busy, no driver): bash scripts/run_queue.sh   -> stops the orphans, resumes")
    if idle:
        lines.append(f"  then, on {', '.join(disp(h) for h in idle)} (idle): bash scripts/run_queue.sh   -> each takes what is unclaimed")
    if not idle and not orphaned:
        lines.append("  no idle or orphaned node has reported yet: on every node you hold, run")
        lines.append("      git pull --ff-only && bash scripts/run_queue.sh")
        lines.append("  (a node with a live driver answers 'already running' or skips busy steps; nothing runs twice)")
    return lines


# ---------------------------------------------------------------- other GPU suites (same work directory)

SUITES = {"fixed-checkpoint-gate-v1": "fixed-checkpoint gate", "selection-gate-light-v2": "light selection gate",
          "selection-gate-one-shot-v1": "one-shot selection gate"}


def suite_lines(work: Path) -> list[str]:
    """Runs of the other GPU suites under $OM_WORK/runs (their progress.json records host, phase, state);
    a run in state 'started' updated within an hour marks its host busy."""
    lines = []
    for name, label in SUITES.items():
        root = work / "runs" / name
        if not root.is_dir():
            continue
        cells = []
        for progress in sorted(root.glob("d*/s*/progress.json")):
            try:
                rec = read_json(progress)
            except (OSError, ValueError):
                continue
            cell = f"{progress.parent.parent.name}/{progress.parent.name}"
            failure = progress.parent / "baseline-failure.json"
            state = str(rec.get("state", "?"))
            phase = str(rec.get("phase", "?"))
            host = str(rec.get("host", "?"))
            updated = rec.get("updated")
            fresh = isinstance(updated, (int, float)) and time.time() - updated < 3600
            if state == "started" and fresh:
                NODES.setdefault(host, []).append(f"{label} {cell} {phase}")
                cells.append(f"{cell} {phase} RUNNING on {short_host(host)}")
            elif failure.is_file():
                try:
                    err = str(read_json(failure).get("error", "failed"))[:60]
                except (OSError, ValueError):
                    err = "failed"
                cells.append(f"{cell} FAILED ({err})")
            else:
                cells.append(f"{cell} {phase} {state}")
        if cells:
            lines.append(f"  {label} ({name}):")
            for cell in cells:
                lines.append(f"    {cell}")
    return lines


def render(rows, header: str, node: str, notes: dict[str, dict] | None = None) -> str:
    out = [header, node, ""]
    out.append("NODES (queue note + heartbeat under $OM_WORK/queue, held leases, and what status saw on each node)")
    out.extend(node_lines(notes or {}))
    out.append("")
    out.append("ACTION")
    out.extend(action_lines(rows, notes or {}, SEEN))
    out.append("")
    if SUITE_LINES:
        out.append("OTHER GPU SUITES in the same work directory (not queue steps)")
        out.extend(SUITE_LINES)
        out.append("")
    out.append("STEPS")
    out.append(f"{'#':>2}  {'step':<26} state")
    counts = {s: 0 for s in STATES}
    for i, (name, state, lines) in enumerate(rows, 1):
        counts[state] = counts.get(state, 0) + 1
        out.append(f"{i:>2}  {name:<26} {state}")
        for line in lines:
            out.append(f"      {line}")
    out.append("")
    out.append("  ".join(f"{s} {counts[s]}" for s in STATES if counts.get(s)))
    out.append("RUNNING/*[node] = lease held now by that node (~ = node inferred); PARTIAL = artifacts exist, nothing running")
    out.append("details: bash scripts/run_e5.sh status | run_e5_bench.sh status [d0] | run_stale_splithalf.sh status [d0]")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=Path(os.environ.get("OM_WORK", "")))
    parser.add_argument("--tag", default=os.environ.get("OM_OLMO3_MODEL_TAG", "olmo3-1025-7b-base-rlzero-grpo-h100-v2"))
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[int(s) for s in os.environ.get("E5_SEEDS", "0 1 2").split()])
    parser.add_argument("--mix-other", default=os.environ.get("MIX_OTHER", "mbpp"))
    parser.add_argument("--mix-seed", type=int, default=int(os.environ.get("MIX_SEED", "0")))
    parser.add_argument("--mix-steps", type=int, default=int(os.environ.get("MIX_STEPS", "200")))
    parser.add_argument("--record", action="store_true", help="only write this node's report under $OM_WORK/queue (used by the watcher)")
    parser.add_argument("--kill-orphans", action="store_true", help="stop marked GPU processes on this node when no live driver holds the node lock")
    args = parser.parse_args(argv)
    if not str(args.work):
        print("[abort] OM_WORK not set", file=sys.stderr)
        return 2
    if args.record:
        record_seen(args.work)
        return 0
    if args.kill_orphans:
        kill_orphans()
        return 0
    root = args.root or Path(os.environ.get("OM_OLMO3_ROOT") or (args.work / "runs" / args.tag))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"QUEUE STATUS  {stamp}  code={git_short()}  work={args.work}"
    WORK[:] = [args.work]
    fs_now = record_seen(args.work)
    SEEN.clear()
    SEEN.update(seen_nodes(args.work, fs_now))
    rows = build_rows(args.work, root, args.tag, args.seeds, args.mix_other, args.mix_seed, args.mix_steps)
    print(render(rows, header, node_line(), queue_notes(args.work, fs_now)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
