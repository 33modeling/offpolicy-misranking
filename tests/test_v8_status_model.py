"""One task model behind every V8 status number (user 11-16 '표랑 아래 요약된거랑 하나도 안맞는다니까').

Random fake queues in the real job-file shapes (srgc_v8_nodes jobs/attempts/heartbeats, matrix.py train/extension/
evaluation tasks, run manifests, rank progress records, sealed plans) must always satisfy the screen invariants, and
every view must print the same numbers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import v8_status as status  # noqa: E402
from srgc_v8_nodes import task_key  # noqa: E402

H, P, CAP = 275, 25, 400
GRID = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
SEEDS = [5, 6, 7, 8, 9]
NOW = 1_800_000_000.0


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def h(*parts):
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()


class Queue:
    """Writes one cohort queue the way srgc_v8_nodes / the runner do."""

    def __init__(self, base, cohort="olmo_math", phase="train"):
        self.base, self.cohort = Path(base), cohort
        self.root = self.base / cohort
        self.q = self.root / ".multi-node"
        self.expect = dict(train=0, ext=0, eval=0, running=0, hung=0)
        write(self.q / "state.json", {"phase": phase, "failure": None,
                                      "automatic_retry_policy": {"max_failed_attempts": 4, "retry_delay": 0.0},
                                      "retry_failed_requested_at": NOW - 9000})
        (self.q / "jobs").mkdir(parents=True, exist_ok=True)

    def beat(self, worker, age, task=None):
        key = hashlib.sha256(json.dumps(worker).encode()).hexdigest()
        write(self.q / "workers" / f"{key}.json", {"worker": worker, "at": NOW - age, "task": task})

    def snapshot(self, rid, update):
        path = self.root / "runs" / rid / "snapshots" / f"u{update}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return str(path)

    def train_task(self, seed, mid, start, parent_rid=None, ext_of=None):
        rid = f"{self.cohort}.rebuttal_bank.s{seed}.{mid}.{h(seed, mid, start, ext_of)[:10]}"
        parent_path = self.snapshot(parent_rid or f"prefix-{seed}", start)
        task = {"kind": "train", "stage": "budget-extension" if ext_of else "bank-branches", "run_id": rid,
                "method_id": mid, "training_seed": seed, "role": "rebuttal_bank", "config_hash": h(rid),
                "parent": {"kind": "snapshot", "run_id": parent_rid, "completed_update": start, "path": parent_path,
                           "sha256": h(parent_path)},
                "horizon": H, "hard_cap": CAP if ext_of else None, "method": {"extension_of": ext_of} if ext_of else {}}
        if ext_of:
            task.update(budget_cap_gpu_seconds=50000.0, cost_offset_gpu_seconds=40000.0, est_update_gpu_seconds=160.0)
        return task

    def eval_task(self, seed, rid, update, exists=True):
        path = self.root / "runs" / rid / "snapshots" / f"u{update}.pt"
        if exists:
            self.snapshot(rid, update)
        sha = h("snap", rid, update)
        spec = h("spec")
        return {"kind": "evaluate", "run_id": rid, "training_seed": seed, "completed_update": update,
                "snapshot_path": str(path), "snapshot_sha256": sha, "split": "final", "responses": 8,
                "evaluation_spec_hash": spec, "task_id": f"eval.{sha[:12]}.{spec[:8]}"}

    def job(self, task, phase, state, attempts=()):
        key = task_key(task)
        write(self.q / "jobs" / f"{key}.json", {"task": task, "phase": phase, "state": state, "attempts": list(attempts)})
        return key

    def run_manifest(self, rid, status_, done, start):
        write(self.root / "runs" / rid / "manifest.json", {"status": status_, "completed_updates": done,
                                                           "attempts": [{"attempt_id": "a-" + rid[-6:], "start_update": start}]})
        upd = self.root / "runs" / rid / "attempts" / ("a-" + rid[-6:]) / "updates.jsonl"
        upd.parent.mkdir(parents=True, exist_ok=True)
        upd.write_text(json.dumps({"completed_updates": done}) + "\n")


def attempt(aid, worker, started, ended=None, rc=None, outcome=None, host=None):
    a = {"id": aid, "worker": worker, "started_at": started}
    if ended is not None:
        a.update(ended_at=ended, exit_code=rc, outcome=outcome)
    if host:
        a["host"] = host
    return a


def add_task(qu, rnd, task, phase, kind, n_worker):
    """One job in a random state: done, running (live / silent-but-locked-elsewhere / dead node), pending, failed
    under or at the retry limit, interrupted. Returns the expected screen state."""
    roll = rnd.random()
    rid = task.get("run_id")
    start = task["parent"]["completed_update"] if kind == "train" else None
    end = (H if phase == "train" else CAP) if kind == "train" else None
    worker = f"h{n_worker:02d}.{1000 + n_worker}"
    if roll < 0.35:
        took = rnd.uniform(1800, 9000)
        qu.job(task, phase, "done", [attempt("d" + h(rid, task.get("task_id"))[:8], worker, NOW - 20000, NOW - 20000 + took, 0, "done")])
        if kind == "train":
            qu.run_manifest(rid, "completed_budget" if phase == "extension" else "completed_horizon",
                            rnd.randint(H + 5, CAP) if phase == "extension" else end, start)
        else:
            spec = task["evaluation_spec_hash"]
            write(qu.root / "evaluations" / f"{task['snapshot_sha256'][:16]}__{spec[:12]}.json", {"n_questions": 500})
        return "done"
    if roll < 0.55:
        aid = "r" + h(rid, task.get("task_id"))[:8]
        live = rnd.random() < 0.8
        qu.job(task, phase, "running", [attempt(aid, worker, NOW - rnd.uniform(600, 7000))])
        qu.beat(worker, rnd.uniform(1, 60) if live else rnd.uniform(400, 2000), "child:1")
        if kind == "train":
            qu.run_manifest(rid, "running", rnd.randint(start, (H if phase == "train" else CAP) - 1), start)
        else:
            for rank in range(4):
                write(qu.q / "progress" / aid / f"rank-{rank}.json",
                      {"stage": "generation", "updated": NOW - 3, "question_index": rnd.randint(1, 125),
                       **({"questions_total": 125} if rnd.random() < 0.7 else {})})
        if live:
            qu.expect["running"] += 1
            return "running"
        (qu.q / "leases").mkdir(exist_ok=True)
        (qu.q / "leases" / f"{task_key(task)}.lock").touch()   # lock free from here: the next node resumes it
        return "retrying"
    if roll < 0.7:
        qu.job(task, phase, "pending", [])
        return "ready"
    if roll < 0.9:
        failures = rnd.choice([1, 2, 3, 4, 5])
        atts = [attempt(f"f{i}" + h(rid, task.get("task_id"))[:6], worker, NOW - 9000 + i * 100, NOW - 8950 + i * 100, 1, "failed")
                for i in range(failures)]
        qu.job(task, phase, "failed", atts)
        return "person" if failures >= 4 else "retrying"
    qu.job(task, phase, "interrupted", [attempt("i" + h(rid, task.get("task_id"))[:8], worker, NOW - 500, NOW - 400, 75, "interrupted")])
    return "retrying"


def random_queue(base, rnd, cohort="olmo_math"):
    """A queue somewhere between start and the budget evaluation, following the planner's creation rules."""
    stage = rnd.choice(["early", "branches", "sealed_train", "extension", "evaluation"])
    qu = Queue(base, cohort, phase={"early": "train", "branches": "train", "sealed_train": "train",
                                    "extension": "extension", "evaluation": "evaluation"}[stage])
    expect = dict(train=0, ext=0, eval=0, states={s: 0 for s in status.STATES})
    n = [0]
    def put(task, phase, kind, forced=None):
        n[0] += 1
        if forced == "waiting":
            qu.job(task, phase, "pending", [])
            Path(task.get("snapshot_path") or task["parent"]["path"]).unlink(missing_ok=True)
            st = "waiting"
        else:
            st = add_task(qu, rnd, task, phase, kind, n[0])
        expect[{"train": "train", "extension": "ext", "evaluation": "eval"}[phase]] += 1
        expect["states"][st] += 1
        return st
    train_runs, plan_tasks = [], []
    any_u = set()
    for seed in SEEDS:
        on = qu.train_task(seed, "on_monitored", P)
        on_done = stage != "early" and rnd.random() < 0.8 or stage in ("sealed_train", "extension", "evaluation")
        if on_done:
            qu.job(on, "train", "done", [attempt("on" + str(seed), "h00.1", NOW - 90000, NOW - 60000, 0, "done")])
            qu.run_manifest(on["run_id"], "completed_horizon", H, P)
            expect["train"] += 1; expect["states"]["done"] += 1
            reached = H
        else:
            st = put(on, "train", "train")
            reached = rnd.choice([P, 60, 130, 210])
        train_runs.append((seed, on, "On"))
        plan_tasks.append(on)
        for tau in GRID:
            if tau > reached and tau != P:
                continue   # its parent snapshot is not saved yet: the planner has not created it
            t = qu.train_task(seed, f"bank_sr_t{tau:03d}", tau, on["run_id"])
            put(t, "train", "train"); train_runs.append((seed, t, f"SR{tau}")); plan_tasks.append(t)
        if on_done and rnd.random() < 0.6:
            tau = rnd.choice(GRID)
            t = qu.train_task(seed, f"bank_uniform_t{tau:03d}", tau, on["run_id"])
            put(t, "train", "train"); train_runs.append((seed, t, "U")); plan_tasks.append(t); any_u.add(seed)
    if stage != "early" and stage != "branches":
        write(qu.q / "plans" / "train.json", {"tasks": plan_tasks, "sealed": True})
    budget = stage in ("extension", "evaluation") or rnd.random() < 0.3
    jobs = {p.stem: json.loads(p.read_text()) for p in (qu.q / "jobs").glob("*.json")}
    ext_runs = []
    if budget:
        for seed, t, col in train_runs:
            if col == "On" or jobs[task_key(t)]["state"] != "done":
                continue
            if rnd.random() < 0.8:
                e = qu.train_task(seed, t["method_id"], H, t["run_id"], ext_of=t["run_id"])
                put(e, "extension", "train"); ext_runs.append((seed, e, col))
    for seed, t, col in train_runs:
        if jobs[task_key(t)]["state"] == "done" and rnd.random() < 0.7:
            put(qu.eval_task(seed, t["run_id"], H), "evaluation", "evaluate")
    for seed, e, col in ext_runs:   # budget-endpoint evaluations of extension runs
        if rnd.random() < 0.5:
            put(qu.eval_task(seed, e["run_id"], rnd.choice([300, 325, 350])), "evaluation", "evaluate",
                forced="waiting" if rnd.random() < 0.3 else None)
    if stage == "evaluation" and rnd.random() < 0.5:
        tasks = [json.loads(p.read_text())["task"] for p in (qu.q / "jobs").glob("*.json")
                 if json.loads(p.read_text())["phase"] == "evaluation"]
        write(qu.q / "plans" / "evaluation.json", {"tasks": tasks, "sealed": True})
        write(qu.q / "plans" / "extension.json", {"tasks": [e for _, e, _ in ext_runs], "sealed": True})
    for i in range(rnd.randint(0, 3)):
        qu.beat(f"idle{i:02d}.{i}", rnd.uniform(1, 80))
    # What the planner is certain to add: 11 train tasks per seed until the train plan is sealed, one extension per
    # SR/U branch, one matched evaluation per training run (until each plan is sealed).
    sealed_train = (qu.q / "plans" / "train.json").exists()
    sealed_eval = (qu.q / "plans" / "evaluation.json").exists()
    certain_train = expect["train"] if sealed_train else len(SEEDS) * (1 + len(GRID)) + len(any_u)
    branches = certain_train - len(SEEDS)
    exts = expect["ext"] if sealed_eval else branches
    evals = expect["eval"] if sealed_eval else (certain_train + sum(1 for p in (qu.q / "jobs").glob("*.json")
                                                if json.loads(p.read_text())["phase"] == "evaluation"
                                                and json.loads(p.read_text())["task"]["completed_update"] != H))
    expect["planned"] = dict(train=certain_train, ext=exts, eval=evals)
    return qu, expect


def table_row(text, label="OLMo MATH"):
    bottom = text.split("━━ V8 status · ")[1]
    line = next(l for l in bottom.splitlines() if l.startswith(f" {label} "))
    head = next(l for l in bottom.splitlines() if l.split()[:2] == ["phase", "train"])
    return dict(zip(head.split(), line[len(label) + 1:].split()[-len(head.split()):]))


def test_random_queues_keep_every_invariant(tmp_path):
    rnd = random.Random(1116)
    checked = 0
    for i in range(300):
        base = tmp_path / f"q{i}" / "v8"
        qu, expect = random_queue(base, rnd)
        text = status.render_clean([base], now=NOW, gpus=[], host="viewer", width=200)
        assert "screen counts disagree" not in text, text.split("━━ V8 status")[1]
        row = table_row(text)
        for ph in ("train", "ext", "eval"):
            assert int(row[ph].split("/")[1]) == expect["planned"][ph], (i, ph, row, expect)
        model = status.task_model(base, "olmo_math", NOW)
        existing = {s: sum(1 for t in model["tasks"] if not t["planned"] and t["state"] == s) for s in status.STATES}
        assert existing == expect["states"], (i, existing, expect["states"])
        assert int(row["run"]) == qu.expect["running"]
        nodes = [l for l in text.splitlines() if l.startswith(" ▶ ")]
        assert len(nodes) == qu.expect["running"]
        assert "≥" not in text and "+ ext" not in text
        checked += 1
    assert checked == 300


def test_failure_under_the_retry_limit_restarts_by_itself_and_only_the_limit_needs_a_person(tmp_path):
    base = tmp_path / "v8"
    qu = Queue(base)
    on = qu.train_task(5, "on_monitored", P)
    qu.job(on, "train", "done", [attempt("a", "h.1", NOW - 900, NOW - 100, 0, "done")])
    qu.run_manifest(on["run_id"], "completed_horizon", H, P)
    for tau, fails in ((50, 1), (100, 3), (150, 4)):
        t = qu.train_task(5, f"bank_sr_t{tau:03d}", tau, on["run_id"])
        qu.job(t, "train", "failed", [attempt(f"f{tau}{i}", "h.1", NOW - 500, NOW - 400, 1, "failed") for i in range(fails)])
    text = status.render_clean([base], now=NOW, gpus=[], host="viewer", width=200)
    grid = text.split("━━")[0]
    s5 = next(l for l in grid.splitlines() if l.startswith("     5"))
    cols = next(l for l in grid.splitlines() if l.startswith(" train")).split()[1:]
    cells = dict(zip(cols, s5.split()[1:]))
    assert cells["SR50"] == "↻" and cells["SR100"] == "↻" and cells["SR150"] == "✗"
    row = table_row(text)
    assert row["retry"] == "2" and row["✗"] == "1"
    assert "failure 1/4" in text and "failure 3/4" in text and "stopped after 4 failures" in text


def test_every_view_prints_the_same_numbers(tmp_path, monkeypatch):
    """--simple / --classic / run_v8_nodes.sh status / v8_eval_time all come from the same model."""
    import v8_eval_time
    rnd = random.Random(7)
    base = tmp_path / "v8"
    random_queue(base, rnd)
    views = {style: status.render([base], now=NOW, gpus=[], host="viewer", style=style) for style in ("clean", "simple", "srgc")}
    assert len(set(views.values())) == 1
    row = table_row(views["clean"])
    model = status.task_model(base, "olmo_math", NOW)
    ev = [t for t in model["tasks"] if t["phase"] == "eval"]
    line = v8_eval_time.model_lines(model)[0]
    assert line.startswith(f"evaluations {row['eval']} done ")
    assert f"run {sum(t['state'] == 'running' for t in ev)} " in line
    out = tmp_path / "eval.txt"
    monkeypatch.setenv("V8_ROOT", str(base)); monkeypatch.setenv("V8_EVAL_TIME_OUT", str(out))
    v8_eval_time.main()
    assert f"evaluations {row['eval']} done " in out.read_text()
