"""Prospective method-choice comparison using the existing GRPO/evaluation engines.

No changes to reduced E5, registered scoring, or its frozen scientific files.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import evidence_downstream as ed
from downstream_compare import ESTIMATORS, SELECTORS, selector_scores
from score_artifacts import load_complete_score_artifacts
from select_rules import jittered_topk, topk_count

SCHEMA = "offpolicy-method-choice/v1"
HERE = Path(__file__).resolve()
SOURCE_FILES = ("src/method_choice.py", "scripts/run_method_choice.sh")


@contextmanager
def lease(path, *, blocking=False):
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def choose(values, seed):
    best = max(values.values())
    tied = sorted(m for m, v in values.items() if v == best)
    winner = min(tied, key=lambda m: hashlib.sha256(f"{seed}:{m}".encode()).hexdigest())
    return {"method": winner, "tied_maxima": tied, "values": values}


def decisions(run):
    config = ed.read(run / "run_config.json")
    artifacts = load_complete_score_artifacts(run)
    scores = selector_scores(run, config["seed"])
    n = len(ed.read(run / "prompts.json")["train"])
    if set(artifacts.oracle) != set(range(n)):
        raise ValueError("score IDs must exactly index the candidate prompts")
    k = topk_count(n, config["topk_frac"])
    subsets = {m: sorted(jittered_topk(s, k, config["seed"] + 1000)) for m, s in scores.items()}
    ref = {i: (v["a"] + v["b"]) / 2 for i, v in artifacts.splithalf.items()}
    uniform = sum(ref.values()) / n
    overlap = {m: len(set(subsets[m]) & set(subsets["fresh_r"])) / k for m in ESTIMATORS}
    gain = {m: sum(ref[i] for i in subsets[m]) / k - uniform for m in ESTIMATORS}
    sensitivity = {}
    for half in ("a", "b"):
        half_scores = {i: s[half] for i, s in artifacts.splithalf.items()}
        center = sum(half_scores.values()) / n
        sensitivity[half] = choose({m: sum(half_scores[i] for i in subsets[m])/k-center
                                    for m in ESTIMATORS}, config["seed"])
    return {"alignment": choose(gain, config["seed"]), "overlap": choose(overlap, config["seed"]),
            "half_sensitivity": sensitivity, "subsets": subsets, "k": k,
            "scope": "A/B choose a method, so the winning A/B score is not its independent evaluation."}


def source_runs(matrix):
    runs = []
    for seed in range(5):
        matches = sorted(matrix.glob(f"family-math500-s{seed}/*-s{seed}-math500-d100"))
        if len(matches) != 1:
            raise ValueError(f"need exactly one MATH d100 source for seed {seed}; found {len(matches)}")
        runs.append(matches[0].resolve())
    return runs


def prepare(root, matrix, pool, pool_manifest, *, full_pool=True):
    runs = source_runs(matrix)
    root = ed.require_separate_output(root, [matrix, pool.parent])
    root.mkdir(parents=True, exist_ok=True)
    with lease(root / ".prepare.lock", blocking=True):
        suite_path = root / "suite.json"
        if suite_path.exists():
            suite = verify(root)
            expected = {"matrix": str(matrix.resolve()), "pool_sha256": ed.digest(pool),
                        "pool_manifest_sha256": ed.digest(pool_manifest), "full_pool": full_pool}
            if suite["inputs"] != expected:
                raise ValueError("method-choice inputs changed; original suite preserved")
            return suite
        if (root / "seeds").exists() and not (root / "preparation.json").exists():
            raise ValueError("unfrozen seed outputs exist; inspect them before using this root")
        if any((root / "seeds").glob("s*/*/policy")) or any((root / "seeds").glob("s*/*/evaluation")):
            raise ValueError("training/evaluation predates the frozen choices; cannot call this prospective")
        test = root / "inputs/test.json"
        manifest = ed.read(pool_manifest)
        ed.prepare_test(pool, runs, test, 500, 20260911, "EleutherAI/hendrycks_math",
                        manifest["source_revision"], "train")
        pending = []
        # Validate every source and freeze every decision before any GPU job starts.
        for seed, run in enumerate(runs):
            out = root / "seeds" / f"s{seed}"
            contract = ed.prepare(run, out, test, 200, 32, arms=SELECTORS, dry=True)
            if contract["seed"] != seed or contract["drift"] != 100:
                raise ValueError("source path and seed/drift metadata disagree")
            if len(ed.read(run / "prompts.json")["train"]) != 400:
                raise ValueError("method-choice protocol requires 400 candidates")
            contract["code_hashes"].update({p: ed.digest(ed.ROOT / p) for p in SOURCE_FILES})
            if full_pool:
                contract["selectors"] = list(SELECTORS) + ["full_pool"]
                contract["all_subsets"] = list(SELECTORS) + ["full_pool"]
            pending.append((out, run, contract, decisions(run)))
        ed.bind(root / "preparation.json", {"contracts": [p[2] for p in pending],
                                            "decisions": [p[3] for p in pending]})
        entries = []
        for out, run, contract, decision in pending:
            out.mkdir(parents=True, exist_ok=True)
            written = ed.write_subsets(run, out / "subsets", .1, contract["seed"])
            if full_pool:
                source = ed.read(run / "prompts.json")
                written["full_pool"] = out / "subsets/subset-full_pool.json"
                ed.atomic_json(written["full_pool"], {**source, "selector": "full_pool", "k": 400,
                                                     "selected_idx": list(range(400)), "source_run": run.name})
            for arm, path in written.items():
                if arm != "full_pool" and ed.read(path)["selected_idx"] != decision["subsets"][arm]:
                    raise ValueError("decision and training subsets disagree")
                args = ed.train_args(ed.read(run / "run_config.json"), run, out, arm, 200)
                (out / "subsets" / f"train-{arm}.args").write_bytes(b"\0".join(a.encode() for a in args)+b"\0")
            ed.atomic_json(out / "experiment.json", contract)
            evaluation = ed.read(test)
            ed.atomic_json(out / "evaluation.json", {"val": evaluation["test"], "provenance": evaluation["provenance"]})
            ed.atomic_json(out / "subsets_hashes.json", {m: ed.digest(p) for m, p in written.items()})
            ed.atomic_json(out / "decision.json", decision)
            bound = ("experiment.json", "evaluation.json", "subsets_hashes.json", "decision.json")
            entries.append({"seed": contract["seed"], "path": str(out),
                            "hashes": {name: ed.digest(out / name) for name in bound}})
        suite = {"schema": SCHEMA, "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "inputs": {"matrix": str(matrix.resolve()), "pool_sha256": ed.digest(pool),
                            "pool_manifest_sha256": ed.digest(pool_manifest), "full_pool": full_pool},
                 "seeds": entries, "prospective": True, "test_sha256": ed.digest(test),
                 "registered_matrix_changed": False, "power_verified": False,
                 "inference": "All five seed contrasts; prompt intervals conditional on trained policies. No overlap gate.",
                 "cost_scope": "Matched updates are not matched end-to-end cost. Missing generation/scoring costs stay missing."}
        ed.atomic_json(suite_path, suite)
        return suite


def verify(root, *, code=True):
    suite = ed.read(root / "suite.json")
    if suite["schema"] != SCHEMA or [e["seed"] for e in suite["seeds"]] != list(range(5)):
        raise ValueError("invalid method-choice suite")
    if ed.digest(root / "inputs/test.json") != suite["test_sha256"]:
        raise ValueError("frozen test input changed")
    for entry in suite["seeds"]:
        out = Path(entry["path"])
        if out.resolve() != (root / "seeds" / f"s{entry['seed']}").resolve():
            raise ValueError("seed output is outside the suite")
        for name, value in entry["hashes"].items():
            if ed.digest(out / name) != value:
                raise ValueError(f"frozen file changed: {out / name}")
        contract = ed.read(out / "experiment.json")
        for name, value in ed.read(out / "subsets_hashes.json").items():
            if ed.digest(out / "subsets" / f"subset-{name}.json") != value:
                raise ValueError("training subset changed")
        for name, value in contract["source_hashes"].items():
            if ed.digest(Path(contract["source_run"]) / name) != value:
                raise ValueError(f"source changed: {name}")
        if code:
            for name, value in contract["code_hashes"].items():
                if ed.digest(ed.ROOT / name) != value:
                    raise ValueError(f"scientific code changed: {name}; review before resuming")
    return suite


def append_cost(root, row):
    with (root / "cost.jsonl").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def phase(root, out, arm, name, commands, gpus, env):
    event = uuid.uuid4().hex
    base = {"event": event, "seed": ed.read(out / "experiment.json")["seed"], "arm": arm,
            "phase": name, "hostname": socket.gethostname(), "pid": os.getpid(), "gpus": len(gpus)}
    append_cost(root, {**base, "state": "started", "timestamp": time.time()})
    children = []
    start = time.monotonic()
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    print(f"[choice] seed={base['seed']} arm={arm} phase={name}", flush=True)
    rc = 1
    try:
        for i, command in enumerate(commands):
            child_env = dict(env)
            child_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus) if len(commands) == 1 else gpus[i]
            with (logs / f"{name}-{arm}-{i}.log").open("a") as log:
                children.append(subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True, close_fds=True))
        while any(c.poll() is None for c in children):
            time.sleep(1)
            elapsed = int(time.monotonic() - start)
            if elapsed and elapsed % 60 == 0:
                print(f"[choice] s{base['seed']} {arm} {name}: {elapsed}s; logs={logs}", flush=True)
        rc = max(abs(c.returncode) for c in children)
        return rc
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        seconds = time.monotonic() - start
        append_cost(root, {**base, "state": "finished", "timestamp": time.time(), "exit_code": rc,
                           "seconds": seconds, "allocated_gpu_seconds": seconds * len(gpus),
                           "scope": "Phase allocation including process startup and failed attempts; not all-in method cost."})


def work(root):
    suite = verify(root)
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(gpus) != 4 or len(set(gpus)) != 4 or not all(gpus):
        raise ValueError("requires exactly four allocated GPUs")
    if os.environ.get("OM_NODE_LOCK_HELD") != "1":
        raise ValueError("use run_method_choice.sh for node admission")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    failures = 0
    for entry in suite["seeds"]:
        out = Path(entry["path"])
        contract = ed.read(out / "experiment.json")
        config = ed.read(Path(contract["source_run"]) / "run_config.json")
        env = dict(os.environ)
        for name, field, default in (("OM_ATTN", "attn", "eager"), ("OM_GEN_BATCH", "gen_batch", 32),
                                     ("OM_LORA_TARGETS", "lora_targets", ""), ("OM_TOP_P", "top_p", 1.),
                                     ("OM_THINKING", "thinking", "off"), ("OM_PROMPT_FORMAT", "prompt_format", "olmo_rlzero_math")):
            env[name] = str(config.get(field) if config.get(field) is not None else default)
        for arm in ["before"] + contract["selectors"]:
            with lease(out / f".{arm}.lock") as acquired:
                if not acquired:
                    print(f"[busy] s{entry['seed']} {arm}; trying another arm", flush=True)
                    continue
                try:
                    if arm != "before":
                        try:
                            ed.arm_policy(out, arm)
                            ready = True
                        except (ValueError, OSError):
                            ready = False
                            if (out / arm / "policy/policy_train.json").exists():
                                from check_downstream_resume import (
                                    recoverable_checkpoint,
                                )
                                recoverable_checkpoint(out, arm)
                        if not ready:
                            args = ed.train_args(config, Path(contract["source_run"]), out, arm, contract["steps"])
                            if phase(root, out, arm, "training", [[sys.executable]+args], gpus, env):
                                raise ValueError("training failed; checkpoint retained")
                    # Validate completed artifacts before skipping; never label corrupt output complete.
                    if all((out / arm / "evaluation" / f"shard-{s}.done.json").exists() for s in range(4)):
                        ed.evaluation_means(out, arm)
                        print(f"[done] s{entry['seed']} {arm}", flush=True)
                        continue
                    commands = [[sys.executable, str(HERE), "evaluate", "--out", str(out), "--arm", arm,
                                 "--shard", str(s)] for s in range(4)]
                    if phase(root, out, arm, "test_evaluation", commands, gpus, env):
                        raise ValueError("evaluation failed; completed shards retained")
                except (OSError, ValueError, KeyError) as exc:
                    failures += 1
                    print(f"[failed] s{entry['seed']} {arm}: {exc}; continuing", flush=True)
    return int(bool(failures))


def summarize(root):
    suite = verify(root)
    rows, missing = [], []
    for entry in suite["seeds"]:
        out = Path(entry["path"])
        decision = ed.read(out / "decision.json")
        a, o = decision["alignment"]["method"], decision["overlap"]["method"]
        if any(not (out / m / "evaluation" / f"shard-{s}.done.json").exists()
               for m in {a, o} for s in range(4)):
            missing.append(entry["seed"])
            continue
        av, ov = ed.evaluation_means(out, a), ed.evaluation_means(out, o)
        delta = av - ov
        lo, hi = ed.paired_interval(delta, entry["seed"])
        row = {"seed": entry["seed"], "alignment_method": a, "overlap_method": o,
               "same_choice": a == o, "alignment_reward": float(av.mean()), "overlap_reward": float(ov.mean()),
               "difference": float(delta.mean()), "prompt_interval_lower": lo, "prompt_interval_upper": hi}
        for control in ("fresh_r", "random", "passrate_beta", "full_pool"):
            if all((out / control / "evaluation" / f"shard-{s}.done.json").exists() for s in range(4)):
                values = ed.evaluation_means(out, control)
                row[control + "_reward"] = float(values.mean())
            else:
                row[control + "_reward"] = None
        rows.append(row)
    report = {"schema": SCHEMA, "suite_sha256": ed.digest(root / "suite.json"), "rows": rows,
              "missing_seed_contrasts": missing, "complete_primary_contrast": not missing,
              "mean_difference": float(np.mean([r["difference"] for r in rows])) if not missing else None,
              "interval_scope": "Paired test prompts conditional on each trained seed; not training-seed uncertainty or equivalence.",
              "cost": {"end_to_end_complete": False,
                       "reason": "Source generation, verification and gradient-scoring costs require measured provenance; not assumed zero."}}
    if not missing:
        differences = np.array([r["difference"] for r in rows])
        half = 2.7764451051977987 * float(differences.std(ddof=1)) / np.sqrt(5)
        report["descriptive_seed_t_interval"] = [report["mean_difference"] - half, report["mean_difference"] + half]
        report["seed_interval_scope"] = "Five source seeds, conditional on the common test questions; not 35 or 40 independent arms."
    ledger = root / "cost.jsonl"
    if ledger.exists():
        events = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        finished = {e["event"] for e in events if e["state"] == "finished"}
        report["cost"]["unfinished_events"] = [e for e in events if e["state"] == "started" and e["event"] not in finished]
        report["cost"]["recorded_phase_gpu_hours"] = sum(e["allocated_gpu_seconds"] for e in events if e["state"] == "finished") / 3600
        report["cost"]["events"] = events
    ed.atomic_json(root / "comparison.json", report)
    fields = list(rows[0]) if rows else ["seed", "alignment_method", "overlap_method", "difference"]
    temporary = root / f"comparison.csv.tmp.{os.getpid()}"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "comparison.csv")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("root", "matrix", "pool", "pool-manifest"):
        prep.add_argument("--"+name, type=Path, required=True)
    prep.add_argument("--without-full-pool", action="store_true")
    for command in ("work", "status", "summarize"):
        sub.add_parser(command).add_argument("--root", type=Path, required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--arm", required=True)
    ev.add_argument("--shard", type=int, required=True)
    args = p.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.root, args.matrix, args.pool, args.pool_manifest, full_pool=not args.without_full_pool)
            print(f"[frozen] five seeds; full_pool={result['inputs']['full_pool']}; {args.root}/suite.json")
        elif args.command == "work":
            return work(args.root)
        elif args.command == "evaluate":
            ed.evaluate(args.out, args.arm, args.shard)
        elif args.command == "summarize":
            print(json.dumps(summarize(args.root), indent=2))
        else:
            if not (args.root / "suite.json").exists():
                print(f"[not prepared] {args.root}")
                return 0
            suite = verify(args.root, code=False)
            for entry in suite["seeds"]:
                out = Path(entry["path"])
                decision = ed.read(out / "decision.json")
                print(f"seed {entry['seed']}: alignment={decision['alignment']['method']} overlap={decision['overlap']['method']}")
                for arm in ["before"] + ed.read(out / "experiment.json")["selectors"]:
                    print(f"  {arm:14} {ed.arm_state(out, arm)}")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
