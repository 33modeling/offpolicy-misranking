"""V8 time left must not grow for reasons already known (user 11-16 '어제 새벽 3시에 완료 된다며 ... 시간은 계속 늘어나고').

The old 'left' divided only the work that already existed by the tasks running at that moment, and an extension past
the planner's estimate jumped to its hard cap. Here a simulated run is stepped forward event by event on the random
fake queues of test_v8_status_model: every task runs at the measured speed of its kind, and new jobs appear exactly
when the planner would create them (a branch when the On run reaches its switch point, an extension once its branch
and the budget freeze are done, the matched evaluation of each finished run, one budget-endpoint evaluation per
finished extension and per On trajectory). Whatever order the free GPU groups take ready tasks in, the upper end of
the estimate must never rise; when the groups follow the longest-chain-first order the lower end must not rise
either."""
from __future__ import annotations

import json
import re
from pathlib import Path
import random
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v8_status as status  # noqa: E402
from srgc_v8_nodes import task_key  # noqa: E402
from test_v8_status_model import CAP, GRID, H, NOW, P, Queue, random_queue, write  # noqa: E402


def jobs(qu):
    return {p.stem: json.loads(p.read_text()) for p in (qu.q / "jobs").glob("*.json")}


def save(qu, key, job):
    write(qu.q / "jobs" / f"{key}.json", job)


class Sim:
    """Runs one fake queue forward. Every finished and running task is first re-timed to its kind's measured median
    speed, so 'measured speed' is one number per kind and the simulation can keep exactly to it."""

    def __init__(self, base, qu, rnd, order, uniform):
        self.base, self.qu, self.rnd, self.order, self.uniform = base, qu, rnd, order, uniform
        self.now, self.prog, self.ends = NOW, {}, {}
        m = status.task_model(base, "olmo_math", self.now)
        self.speed = {k: v[0] for k, v in m["speed"].items() if v}
        self.groups = m["groups"]
        shutil.rmtree(qu.q / "workers", ignore_errors=True)
        self.free = [f"sim{i:02d}.{100 + i}" for i in range(self.groups)]
        self.busy = {}
        alljobs = jobs(qu)
        for t in m["tasks"]:
            if t["row"] is None:
                continue
            key, job = t["key"], alljobs[t["key"]]
            att = job["attempts"][-1] if job["attempts"] else None
            kind = "eval" if t["phase"] == "eval" else "on" if t["col"] == "On" else "branch"
            if t["state"] == "done" and att:
                n = 1 if kind == "eval" else max(1, t["done_u"] - self.attempt_start(t))
                att["started_at"] = att["ended_at"] - n * self.speed[kind]
            elif t["state"] == "running":
                worker = self.free.pop()
                att["worker"] = worker
                self.busy[key] = worker
                if kind == "eval":
                    att["started_at"] = self.now - t["frac"] * self.speed["eval"]
                    self.prog[key] = t["frac"]
                else:
                    att["started_at"] = self.now - (t["done_u"] - self.attempt_start(t)) * self.speed[kind]
                    self.prog[key] = float(t["done_u"])
                    self.ends[key] = t["end"]
            elif job["state"] == "running":    # lost: its node stopped, the next free group resumes it
                job["state"] = "interrupted"
                att.update(ended_at=self.now - 1, exit_code=75, outcome="interrupted")
            save(qu, key, job)
        self.beat()

    @staticmethod
    def attempt_start(t):
        attempts = (t["row"].get("_manifest") or {}).get("attempts") or [{}]
        s = attempts[-1].get("start_update")
        return s if type(s) is int and s >= 0 else t["start"]

    def beat(self):
        for w in self.free:
            self.qu.beat(w, 0, None)
        for w in self.busy.values():
            self.qu.beat(w, 0, "child:1")
        for w in self.free + list(self.busy.values()):   # beat(age) is relative to NOW
            path = next(p for p in (self.qu.q / "workers").glob("*.json") if json.loads(p.read_text())["worker"] == w)
            data = json.loads(path.read_text()); data["at"] = self.now; path.write_text(json.dumps(data))

    def model(self):
        return status.task_model(self.base, "olmo_math", self.now)

    def sealed(self, name):
        return (self.qu.q / "plans" / f"{name}.json").exists()

    def create(self, m):
        """Jobs the planner adds now. Returns True if any was added."""
        qu, added = self.qu, False
        train = {(t["seed"], t["col"]): t for t in m["tasks"] if t["phase"] == "train"}
        ext = {(t["seed"], t["col"]): t for t in m["tasks"] if t["phase"] == "ext"}
        evals = [t for t in m["tasks"] if t["phase"] == "eval"]
        def rid(t):
            return (t.get("task") or {}).get("run_id") or (t["row"] or {}).get("run_id")
        def new(task, phase):
            qu.job(task, phase, "pending", [])
        for (s, c), t in train.items():
            on = train.get((s, "On"))
            if t["planned"] and str(t["key"]).startswith("planned.train.") and on is not None and not on["planned"] \
                    and on["done_u"] >= t["start"]:
                new(qu.train_task(s, f"bank_sr_t{t['start']:03d}", t["start"], rid(on)), "train"); added = True
        for s in m["open_u"]:
            on = train.get((s, "On"))
            if on is not None and on["state"] == "done" and self.uniform and self.rnd.random() < 0.5:
                tau = self.rnd.choice(GRID)
                new(qu.train_task(s, f"bank_uniform_t{tau:03d}", tau, rid(on)), "train"); added = True
        if added:
            return True
        u_tau = m["u_tau"]
        gate = m["frozen"] or any(t["phase"] == "ext" and not t["planned"] for t in m["tasks"]) or (
            all(t["state"] == "done" for (s, c), t in train.items() if c == "On") and
            all(train[(s, f"SR{v}")]["state"] == "done" for s, v in u_tau.items() if (s, f"SR{v}") in train))
        if gate and not self.sealed("extension"):
            for (s, c), t in train.items():
                e = ext.get((s, c))
                if c != "On" and t["state"] == "done" and e is not None and e["planned"] and \
                        str(e["key"]).startswith("planned.ext."):
                    task = qu.train_task(s, rid(t).split(".")[3], H, rid(t), ext_of=rid(t))
                    task["budget_cap_gpu_seconds"] = task["cost_offset_gpu_seconds"] + (e["end"] - H) * 160.0 + 80.0
                    new(task, "extension"); added = True
        if not self.sealed("evaluation"):
            have = {(t["seed"], t["col"]) for t in evals if not t["of_ext"] and t["update"] == H and not t["planned"]}
            budget = {(t["seed"], t["col"]) for t in evals if t["of_ext"] or t["update"] != H}
            for (s, c), t in train.items():
                if t["state"] == "done" and not t["planned"] and (s, c) not in have:
                    new(qu.eval_task(s, rid(t), H), "evaluation"); added = True
                if c == "On" and gate and t["state"] == "done" and (s, c) not in budget:
                    new(qu.eval_task(s, rid(t), 150), "evaluation"); added = True
            for (s, c), e in ext.items():
                if e["state"] == "done" and not e["planned"] and (s, c) not in budget:
                    new(qu.eval_task(s, rid(e), e["end"]), "evaluation"); added = True
        return added

    def start(self, m):
        eta = m["eta"]
        items = eta["low"]["items"]
        by = {}
        for t in m["tasks"]:
            iid = ("eval", t["key"]) if t["phase"] == "eval" else (t["phase"], t["seed"], t["col"])
            by[iid] = t
        def startable(i):
            t = by.get(i)
            return (t is not None and not t["planned"] and t["row"] is not None and t["state"] != "running"
                    and not any(d in items and lag is None for d, lag in items[i]["deps"]))
        if self.order == "chain":   # the waiting task on the longest remaining chain first
            succ = {i: [] for i in items}
            for i, it in items.items():
                for d, lag in it["deps"]:
                    if d in succ:
                        succ[d].append((i, lag))
            level = {}
            def bottom(i):
                if i not in level:
                    level[i] = max([items[i]["dur"] + bottom(j) for j, lag in succ[i] if lag is None] +
                                   [lag + bottom(j) for j, lag in succ[i] if lag is not None] + [items[i]["dur"]])
                return level[i]
            pick = sorted((i for i in items if startable(i)), key=lambda i: (-bottom(i), str(i)))
        else:
            pick = [i for i in items if startable(i)]
            self.rnd.shuffle(pick)
        alljobs = jobs(self.qu)
        for i in pick[:len(self.free)]:
            t = by[i]
            key, job = t["key"], alljobs[t["key"]]
            worker = self.free.pop()
            aid = f"sim{len(job['attempts'])}-{key[:10]}"
            job["state"] = "running"
            job["attempts"].append({"id": aid, "worker": worker, "started_at": self.now})
            save(self.qu, key, job)
            self.busy[key] = worker
            task = job["task"]
            if t["phase"] == "eval":
                Path(task["snapshot_path"]).parent.mkdir(parents=True, exist_ok=True)
                Path(task["snapshot_path"]).write_bytes(b"x")
                self.prog[key] = 0.0
                self.progress_eval(aid, 0.0)
            else:
                man_path = self.qu.root / "runs" / task["run_id"] / "manifest.json"
                man = json.loads(man_path.read_text()) if man_path.exists() else {"status": "running", "attempts": []}
                man["status"] = "running"
                man["attempts"].append({"attempt_id": aid, "start_update": t["done_u"]})
                write(man_path, man)
                self.prog[key] = float(t["done_u"])
                self.updates(task["run_id"], aid, t["done_u"])
                self.ends[key] = t["end"]
        self.beat()

    def updates(self, rid, aid, done):
        path = self.qu.root / "runs" / rid / "attempts" / aid / "updates.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"completed_updates": done}) + "\n")

    def progress_eval(self, aid, frac):
        for rank in range(4):
            write(self.qu.q / "progress" / aid / f"rank-{rank}.json",
                  {"stage": "generation", "updated": self.now, "question_index": int(frac * 125000 + 1e-9), "questions_total": 125000})

    def advance(self, m):
        """Move time to the next completion (or the On run reaching a switch point that has a branch to create)."""
        by_key = m["by_key"]
        alljobs = jobs(self.qu)
        def left(key):
            t = by_key[key]
            if t["phase"] == "eval":
                return (1 - self.prog[key]) * self.speed["eval"]
            sp = self.speed["on" if t["col"] == "On" else "branch"]
            return (self.ends[key] - self.prog[key]) * sp
        if not self.busy:
            return False
        dt = min(left(k) for k in self.busy)
        for t in m["tasks"]:
            if t["key"] in self.busy and t["col"] == "On" and t["phase"] == "train":
                taus = [p["start"] for p in m["tasks"] if p["phase"] == "train" and p["seed"] == t["seed"]
                        and p["planned"] and p["start"] > self.prog[t["key"]]]
                if taus:
                    dt = min(dt, (min(taus) - self.prog[t["key"]]) * self.speed["on"])
        dt = max(dt, 1e-6)
        self.now += dt
        for key in list(self.busy):
            t, job = by_key[key], alljobs[key]
            aid = job["attempts"][-1]["id"]
            task = job["task"]
            if t["phase"] == "eval":
                self.prog[key] = min(1.0, self.prog[key] + dt / self.speed["eval"])
                finished = self.prog[key] >= 1 - 1e-9
                self.progress_eval(aid, self.prog[key])
            else:
                sp = self.speed["on" if t["col"] == "On" else "branch"]
                self.prog[key] = min(self.ends[key], self.prog[key] + dt / sp)
                finished = self.prog[key] >= self.ends[key] - 1e-6
                done = self.ends[key] if finished else int(self.prog[key] + 1e-6)
                man = json.loads((self.qu.root / "runs" / task["run_id"] / "manifest.json").read_text())
                self.updates(task["run_id"], man["attempts"][-1]["attempt_id"], done)
                if t["col"] == "On":
                    self.qu.snapshot(task["run_id"], done)
            if finished:
                job["state"] = "done"
                job["attempts"][-1].update(ended_at=self.now, exit_code=0, outcome="done")
                save(self.qu, key, job)
                self.free.append(self.busy.pop(key))
                if t["phase"] == "eval":
                    write(self.qu.root / "evaluations" / f"{task['snapshot_sha256'][:16]}__{task['evaluation_spec_hash'][:12]}.json",
                          {"n_questions": 500})
                else:
                    man_path = self.qu.root / "runs" / task["run_id"] / "manifest.json"
                    man = json.loads(man_path.read_text())
                    man.update(status="completed_budget" if t["phase"] == "ext" else "completed_horizon",
                               completed_updates=self.ends[key])
                    write(man_path, man)
        self.beat()
        return True


def finish_times(m, now):
    eta = m["eta"]
    return now + eta["low"]["left"], now + eta["high"]["left"]


def run(tmp_path, seed, order, uniform, max_steps=90):
    rnd = random.Random(seed)
    base = tmp_path / f"s{seed}{order}" / "v8"
    qu, _ = random_queue(base, rnd)
    m = status.task_model(base, "olmo_math", NOW)
    if m["eta"] is None or not m["eta"].get("groups") or "left" not in m["eta"]["low"]:
        return None
    sim = Sim(base, qu, rnd, order, uniform)
    tol = 3 * max(sim.speed["on"], sim.speed["branch"]) + 60
    history, prev = [], None
    for step in range(max_steps):
        while sim.create(sim.model()):
            pass
        m = sim.model()
        if not any(t["state"] != "done" for t in m["tasks"]):
            break
        lo, hi = finish_times(m, sim.now)
        history.append((sim.now, lo, hi))
        if prev is not None:
            assert hi <= prev[1] + tol, (seed, order, step, "upper end rose", (hi - prev[1]) / 60, history[-3:])
            if order == "chain" and not uniform:
                assert lo <= prev[0] + tol, (seed, order, step, "lower end rose", (lo - prev[0]) / 60, history[-3:])
        prev = (lo, hi)
        sim.start(m)
        assert sim.advance(m), (seed, order, step, "work left but no task can run")
    shutil.rmtree(base.parent, ignore_errors=True)
    return history


def test_finish_time_never_rises_while_tasks_run_at_their_measured_speed(tmp_path):
    ran = {"chain": 0, "random": 0}
    for seed in range(40):
        for order, uniform in (("chain", False), ("random", True)):
            if ran[order] >= 4:
                continue
            history = run(tmp_path, 9000 + seed, order, uniform)
            if history:
                ran[order] += 1
                assert len(history) >= 20, (seed, order, len(history))
    assert ran == {"chain": 4, "random": 4}, ran


def test_extension_past_the_planned_endpoint_is_predicted_from_its_speed_not_the_cap(tmp_path):
    """Old code: an extension past the planner's endpoint (337 here) got end = hard_cap 400 at once."""
    base = tmp_path / "v8"
    qu = Queue(base, phase="extension")
    from test_v8_status_model import attempt
    on = qu.train_task(5, "on_monitored", P)
    qu.job(on, "train", "done", [attempt("on", "h.1", NOW - 90000, NOW - 60000, 0, "done")])
    qu.run_manifest(on["run_id"], "completed_horizon", H, P)
    br = qu.train_task(5, "bank_sr_t200", 200, on["run_id"])
    qu.job(br, "train", "done", [attempt("br", "h.1", NOW - 50000, NOW - 50000 + 75 * 100, 0, "done")])   # 100 s/update
    qu.run_manifest(br["run_id"], "completed_horizon", H, 200)
    e = qu.train_task(5, "bank_sr_t200", H, br["run_id"], ext_of=br["run_id"])   # planned end 275 + 10000/160 = 337
    qu.job(e, "extension", "running", [attempt("ex", "h.2", NOW - 65 * 50, host="h")])   # 50 s/update: half the cost
    qu.run_manifest(e["run_id"], "running", H + 65, H)
    qu.beat("h.2", 2, "child:1")
    m = status.task_model(base, "olmo_math", NOW)
    t = next(t for t in m["tasks"] if t["phase"] == "ext")
    assert t["end"] == H + 125 and t["progress"] == f"update {H + 65} · budget ?", t   # 10000 / (160 * 50/100) = 125
    qu.job(e, "extension", "running", [attempt("ex", "h.2", NOW - 65 * 90, host="h")])   # 90 s/update
    m = status.task_model(base, "olmo_math", NOW)
    t = next(t for t in m["tasks"] if t["phase"] == "ext")
    assert t["end"] == H + 69, t      # 10000 / (160 * 0.9) = 69.4 updates, not the cap


def test_time_left_counts_work_the_planner_has_not_created_yet(tmp_path):
    """Early in training only 'On' jobs exist; the estimate already holds every branch, extension and evaluation."""
    base = tmp_path / "v8"
    qu = Queue(base)
    from test_v8_status_model import attempt
    for seed in (5, 6, 7, 8, 9):
        on = qu.train_task(seed, "on_monitored", P)
        qu.job(on, "train", "running", [attempt(f"on{seed}", f"h{seed}.1", NOW - 100 * 50)])
        qu.run_manifest(on["run_id"], "running", P + 50, P)
        qu.beat(f"h{seed}.1", 2, "child:1")
        sr = qu.train_task(seed, "bank_sr_t025", P, f"prefix-{seed}")
        qu.job(sr, "train", "done", [attempt(f"sr{seed}", "h0.1", NOW - 90000, NOW - 90000 + 250 * 40, 0, "done")])
        qu.run_manifest(sr["run_id"], "completed_horizon", H, P)
        ev = qu.eval_task(seed, sr["run_id"], H)
        qu.job(ev, "evaluation", "done", [attempt(f"ev{seed}", "h0.1", NOW - 9000, NOW - 9000 + 1800, 0, "done")])
    m = status.task_model(base, "olmo_math", NOW)
    low = m["eta"]["low"]["items"]
    assert sum(1 for i in low if i[0] == "train") == 5 + 5 * 9          # On + 9 SR branches still to train
    assert sum(1 for i in low if i[0] == "ext") == 5 * 10               # one per SR branch (SR25 done included)
    assert sum(1 for i in low if i[0] in ("eval", "budget-eval")) == 5 * 10 + 5 * 11   # matched + budget endpoints
    high = m["eta"]["high"]["items"]
    assert sum(1 for i in high if i[2:3] == ("U?",)) >= 5 * 2           # a possible U branch per seed (train + ext)
    # 5 groups: the On runs are the long chain (200 updates x 100 s), then branches, extensions (cap: none measured)
    lo, hi = m["eta"]["low"]["left"], m["eta"]["high"]["left"]
    assert m["eta"]["low"]["work"] / 5 <= lo <= hi
    text = status.render_clean([base], now=NOW, gpus=[], host="viewer", width=200)
    rng = f"about {status.span(lo)}–{status.span(hi)}"
    head = next(l for l in text.splitlines() if l.startswith("   OLMo MATH · stage: "))
    assert f" · {rng} left · " in head, head
    # the first block of the screen says it as a sentence with the KST clock, and why the range is wide
    top = text.split(" time left\n")[1].split("\n\n")[0]
    line = next(l for l in top.splitlines() if l.startswith("   OLMo MATH: "))
    assert line.startswith(f"   OLMo MATH: {rng} left, done around ") and "KST" in line, line
    assert "longer end if U branches are added for s5, s6, s7, s8, s9" in re.sub(r"\s+", " ", top), top
    assert "   All experiments: " + rng in top, top


def test_budget_extension_shows_budget_spent_from_its_cost_ledger(tmp_path):
    """User 11-16 'status 300 이거 의미가 없잖아': 'done/300' showed the planner's estimated stop, the same for most
    branches. The run stops when its budget is spent (budget.should_continue), so the node line says
    'update 287 · budget 81%' from the runner's own ledger (events.jsonl of the surviving attempts), and the stop
    update and time left follow from the cost per update this run has measured."""
    from test_v8_status_model import attempt
    base = tmp_path / "v8"
    qu = Queue(base, phase="extension")
    br = qu.train_task(5, "bank_sr_t200", 200, "on-5")
    qu.job(br, "train", "done", [attempt("br", "h.1", NOW - 50000, NOW - 50000 + 75 * 100, 0, "done")])
    qu.run_manifest(br["run_id"], "completed_horizon", H, 200)
    e = qu.train_task(5, "bank_sr_t200", H, br["run_id"], ext_of=br["run_id"])
    e["method"]["family"] = "fixed_sr"
    e.update(budget_cap_gpu_seconds=50000.0, cost_offset_gpu_seconds=40000.0, est_update_gpu_seconds=160.0)
    qu.job(e, "extension", "running", [attempt("ex", "h.2", NOW - 12 * 100, host="h")])
    qu.run_manifest(e["run_id"], "running", H + 12, H)
    aid = "a-" + e["run_id"][-6:]
    events = [{"event_id": f"t{u}", "attempt_id": aid, "completed_update": u, "component": "training",
               "gpu_seconds": 400.0, "attribution": "interval", "source_phase": "update"} for u in range(H, H + 12)]
    (qu.root / "runs" / e["run_id"] / "attempts" / aid / "events.jsonl").write_text(
        "\n".join(json.dumps(x) for x in events) + "\n")
    qu.beat("h.2", 2, "child:1")
    m = status.task_model(base, "olmo_math", NOW)
    t = next(t for t in m["tasks"] if t["phase"] == "ext")
    # spent 12 x 400 = 4800 on top of 40000 of 50000 = 89%; 5200 left / 400 per update = 13 more updates
    assert t["progress"] == f"update {H + 12} · budget 89%", t["progress"]
    assert t["end"] == H + 12 + 13, t
    text = status.render_clean([base], now=NOW, gpus=[], host="viewer", width=200)
    node = next(l for l in text.split(" nodes: ")[1].splitlines() if l.startswith("   h "))
    assert f"update {H + 12}, budget 89%" in node and " left" in node and "/" not in node, node
    assert "89%▶" in text.split(" by seed and branch")[1]
