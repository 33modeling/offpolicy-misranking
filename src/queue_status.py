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
is read from the filesystem; no GPU, nothing is written, no lease is taken
(the lease probe releases immediately).

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

# host -> list of "what" strings, filled while the rows are built
NODES: dict[str, list[str]] = {}


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


def lease_holder(path: Path) -> str | None:
    """None when the lease is free; otherwise the host from the lease note, or '?'."""
    if not lease_held(path):
        return None
    try:
        first = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return "?"
    return parse_note(first[0]).get("host", "?") if first else "?"


def short_host(host: str) -> str:
    parts = host.split("-")
    return "-".join(parts[-2:]) if len(parts) > 2 else host


def since_text(stamp: str) -> str:
    m = re.match(r"\d{4}-(\d{2}-\d{2})T(\d{2}:\d{2}):\d{2}Z", stamp)
    return f"{m.group(1)} {m.group(2)}Z" if m else stamp


def note_lease(holder: str | None, what: str) -> str:
    """Register a held lease under its node and return the inline marker."""
    if holder is None:
        return ""
    NODES.setdefault(holder, []).append(what)
    return f"*[{short_host(holder)}]"


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
    parts, all_done, started, running = [], True, False, False
    for arm in ["before", *names]:
        word, done = arm_state(out, arm, contract)
        all_done = all_done and done
        started = started or word != "-"
        holder = lease_holder(out / f".{arm}.lock")
        if holder is not None:
            running = True
            word += note_lease(holder, f"{label} {out.name} {ARM_LABELS.get(arm, arm)} ({word})".strip())
        parts.append(f"{ARM_LABELS.get(arm, arm)} {word}")
    line = "DONE" if all_done else " | ".join(parts)
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
    parts, all_done, started, running = [], True, False, False
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
        holder = lease_holder(out / f".bench-{arm}.lock")
        if holder is not None:
            running = True
            word += note_lease(holder, f"{label} {out.name} {ARM_LABELS.get(arm, arm)} ({word})".strip())
        parts.append(f"{ARM_LABELS.get(arm, arm)} {word}")
    line = "DONE" if all_done else " | ".join(parts)
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
        holder = lease_holder(run / ".stale-splithalf.lock")
        if holder is not None:
            running = True
            word += note_lease(holder, f"{label} s{seed}")
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
        holder = lease_holder(Path(str(point) + ".lease"))
        if holder is not None:
            mark = note_lease(holder, f"mixed pool point ({progress or 'started'})")
            rows.append(("mixed pool: point", "RUNNING", [f"{progress or 'started'}, last write {age(log)} ago {mark}"]))
        elif log.is_file():
            rows.append(("mixed pool: point", "PARTIAL", [f"stopped at {progress or '?'}, last write {age(log)} ago"]))
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
        holder = lease_holder(out / ".gate_passrate.lock")
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

def queue_notes(work: Path) -> dict[str, dict]:
    """host -> {'step', 'since', 'alive', 'beat_age'} from $OM_WORK/queue/<host>.txt and .beat."""
    notes = {}
    directory = work / "queue"
    if not directory.is_dir():
        return notes
    now = time.time()
    for note in sorted(directory.glob("*.txt")):
        try:
            fields = parse_note(note.read_text(encoding="utf-8", errors="replace"))
            note_age = now - note.stat().st_mtime
        except OSError:
            continue
        beat = note.with_suffix(".beat")
        try:
            beat_seconds = int(now - beat.stat().st_mtime) if beat.is_file() else None
        except OSError:
            beat_seconds = None
        if note_age > NOTE_MAX_AGE_SECONDS and (beat_seconds is None or beat_seconds > NOTE_MAX_AGE_SECONDS):
            continue
        host = fields.get("host", note.stem)
        notes[host] = {"step": fields.get("step", "?"), "since": fields.get("since", "?"),
                       "alive": beat_seconds is not None and beat_seconds < HEARTBEAT_ALIVE_SECONDS,
                       "beat_age": age_text(beat_seconds) if beat_seconds is not None else None}
    return notes


def node_lines(notes: dict[str, dict]) -> list[str]:
    hosts = sorted(set(notes) | set(NODES), key=short_host)
    if not hosts:
        return ["  (no queue note and no held lease; nothing is running)"]
    lines = []
    for host in hosts:
        note = notes.get(host)
        if note is None:
            head = "no queue note (started by hand or before this version)"
        elif note["step"] == "done":
            head = f"queue finished {since_text(note['since'])}"
        elif note["step"] == "stopped":
            head = f"queue stopped {since_text(note['since'])}"
        elif note["alive"]:
            head = f"{note['step']}  since {since_text(note['since'])}  alive (heartbeat {note['beat_age']} ago)"
        else:
            beat = f"no heartbeat for {note['beat_age']}" if note["beat_age"] else "no heartbeat"
            head = f"{note['step']}  since {since_text(note['since'])}  {beat.upper()} (killed?)"
        lines.append(f"  {short_host(host):<10} {head}")
        for what in NODES.get(host, []):
            lines.append(f"  {'':<10}   holds: {what}")
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
    if "/family-" in path:
        return f"point {name}"
    return path


def node_jobs() -> list[str]:
    """Labels of the queue processes on this machine (every one carries OUT_ROOT in its environment)."""
    markers = set()
    for proc in Path("/proc").glob("[0-9]*"):
        if proc.name == str(os.getpid()):
            continue
        try:
            environ = (proc / "environ").read_bytes()
        except OSError:
            continue
        for item in environ.split(b"\0"):
            if item.startswith(b"OUT_ROOT="):
                markers.add(item[9:].decode(errors="replace"))
    return sorted({job_label(m) for m in markers if m})


def node_line() -> str:
    directory = Path(os.environ.get("OM_LOCAL_LOCK_DIR", f"/tmp/offpolicy-misranking-{os.getuid()}"))
    lock = directory / "primary.lock"
    jobs = node_jobs()
    held = lease_held(lock)
    if jobs:
        return f"this node ({socket.gethostname()}): running {', '.join(jobs)}" + ("" if held else " (node lock free)")
    return f"this node ({socket.gethostname()}): " + ("GPU job running (node lock held, not a queue step)" if held else "no GPU job (node lock free)")


# ---------------------------------------------------------------- report

def build_rows(work: Path, root: Path, tag: str, seeds, other: str, mix_seed: int, mix_steps: int):
    NODES.clear()

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
    return rows


def render(rows, header: str, node: str, notes: dict[str, dict] | None = None) -> str:
    out = [header, node, ""]
    out.append(f"{'#':>2}  {'step':<26} state")
    counts = {s: 0 for s in STATES}
    for i, (name, state, lines) in enumerate(rows, 1):
        counts[state] = counts.get(state, 0) + 1
        out.append(f"{i:>2}  {name:<26} {state}")
        for line in lines:
            out.append(f"      {line}")
    out.append("")
    out.append("  ".join(f"{s} {counts[s]}" for s in STATES if counts.get(s)))
    out.append("RUNNING/*[node] = lease held now by that node; PARTIAL = artifacts exist, nothing running")
    out.append("")
    out.append("nodes (queue note + heartbeat under $OM_WORK/queue, and held leases)")
    out.extend(node_lines(notes or {}))
    out.append("")
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
    args = parser.parse_args(argv)
    if not str(args.work):
        print("[abort] OM_WORK not set", file=sys.stderr)
        return 2
    root = args.root or Path(os.environ.get("OM_OLMO3_ROOT") or (args.work / "runs" / args.tag))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"QUEUE STATUS  {stamp}  code={git_short()}"
    rows = build_rows(args.work, root, args.tag, args.seeds, args.mix_other, args.mix_seed, args.mix_steps)
    print(render(rows, header, node_line(), queue_notes(args.work)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
