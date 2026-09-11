"""Resumable low-order scoring and independent-test GRPO continuation.

Scientific source points remain immutable. Multiple nodes lease distinct
points or training arms; a failed task is recorded once and other tasks run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import additive_experiment as ae
import evidence_downstream as ed
import low_order_backend as backend
from artifact_contract import validate_generation_contract
from experiment import read_rollouts
from grads import sequence_logprobs_batch
from low_order_reuse import METHODS, algebra_audit, approximation, reuse_score
from method_choice import append_cost, lease
from select_rules import jittered_topk, topk_count

SCHEMA = "offpolicy-low-order/v1"
HERE = Path(__file__).resolve()
ARMS = ("random", "pair_u2", "low_order", "passrate_beta")
CODE_FILES = (*ae.CODE_FILES, "src/low_order_reuse.py", "src/low_order_backend.py",
              "src/low_order_experiment.py", "src/evidence_downstream.py",
              "src/train_policy_grpo.py", "src/method_choice.py")


def read_groups(path, n, k):
    groups = read_rollouts(path)
    ae.validate_rows(groups, n)
    for idx, rows in groups.items():
        if len(rows) != k or any(float(r["reward"]) not in (0., 1.) for r in rows):
            raise ValueError(f"{path}: prompt {idx} needs {k} binary-reward responses")
    return groups


def prepare(root, runs, *, evaluation=None, pool=None, pool_manifest=None,
            steps=100, random_extra_steps=0, eval_k=8, test_count=300,
            micro_batch=1, derivative="finite", fd_step=.1, geometry="adam_rms",
            arms=("random", "pair_u2", "low_order")):
    if (not runs or len(set(map(str, runs))) != len(runs) or steps < 1
            or random_extra_steps < 0 or eval_k < 2 or micro_batch < 1
            or derivative not in ("finite", "autograd") or geometry not in ("adam_rms", "identity")
            or not math.isfinite(fd_step) or fd_step <= 0
            or not arms or len(set(arms)) != len(arms) or set(arms)-set(ARMS)):
        raise ValueError("invalid low-order suite arguments")
    root = ed.require_separate_output(root, runs)
    if any(root in run.resolve().parents for run in runs):
        raise ValueError("output must not contain a source point")
    if evaluation is None:
        if pool is None or pool_manifest is None:
            raise ValueError("provide independent evaluation or a math pool and manifest")
        manifest = ed.read(pool_manifest)
        evaluation = root / "inputs/test.json"
        ed.prepare_test(pool, runs, evaluation, test_count, 20260912,
                        "EleutherAI/hendrycks_math", manifest["source_revision"], "train")
    evaluation = evaluation.resolve()
    pending = []
    for run in runs:
        base = ae.source_contract(run, micro_batch=micro_batch)
        config = base["config"]
        if (config.get("behavior_k") != 8 or config.get("grpo_group_size") != 8
                or config.get("temperature") != 1. or config.get("top_p", 1.) != 1.):
            raise ValueError("initial protocol requires K=8, G=8, temperature=1, top-p=1")
        spec = approximation(epsilon=config["grpo_advantage_epsilon"])
        source = ed.read(run / "prompts.json")
        nval = len(source["val"])
        if nval < 4 or config.get("val_k", 0) < 2:
            raise ValueError("at least four validation prompts and two responses required")
        read_groups(run / "rollouts_behavior_train.jsonl", base["n"], 8)
        read_groups(run / "rollouts_fresh_val.jsonl", nval, config["val_k"])
        if set(ed.questions(source["train"])) & set(ed.questions(source["val"])):
            raise ValueError("candidate and validation prompts overlap")
        generation = validate_generation_contract(run, ("rollouts_behavior_train", "rollouts_fresh_val"))
        files = ["rollouts_fresh_val.jsonl"] + [str(p.relative_to(run)) for p in
                 sorted(run.glob("rollouts_fresh_val*.manifest.json"))]
        base["source_hashes"].update({name: ed.digest(run / name) for name in files})
        out = root / "points" / run.name
        downstream = ed.prepare(run, out / "training", evaluation, steps, eval_k, arms=("random",), dry=True)
        contract = {**base, "schema": SCHEMA, "validation_count": nval,
                    "derivative": derivative, "fd_step": fd_step, "geometry": geometry,
                    "normalization": spec.record(), "methods": list(METHODS), "arms": list(arms),
                    "steps": steps, "random_extra_steps": random_extra_steps,
                    "generation_validation": generation, "downstream_template": downstream,
                    "evaluation": {"val": ed.read(evaluation)["test"],
                                   "provenance": ed.read(evaluation)["provenance"]},
                    "code_hashes": {**downstream["code_hashes"],
                                    **{name: ed.digest(ed.ROOT / name) for name in CODE_FILES}},
                    "clipping": "none; overflowing moments fail instead of changing the estimator",
                    "no_generation": "selection only; continuation and independent evaluation generate responses",
                    "theory_scope": "Approximate initial GRPO loss-gradient score, not exact AdamW utility"}
        pending.append((out, contract))
    if len({out.name for out, _ in pending}) != len(pending):
        raise ValueError("source point names must be unique")
    root.mkdir(parents=True, exist_ok=True)
    with lease(root / ".prepare.lock", blocking=True):
        ed.bind(root / "preparation.json", {"contracts": [c for _, c in pending]})
        entries = []
        for out, contract in pending:
            out.mkdir(parents=True, exist_ok=True)
            ed.bind(out / "experiment.json", contract)
            entries.append({"name": out.name, "sha256": ed.digest(out / "experiment.json")})
        suite = {"schema": SCHEMA, "points": entries, "matched_total_cost": False,
                 "cost_scope": "Measured phase allocation plus separate historical-cost requirements; no automatic Pareto claim."}
        ed.bind(root / "suite.json", suite)
    return suite


def entries(root):
    suite = ed.read(root / "suite.json")
    if suite.get("schema") != SCHEMA:
        raise ValueError("unsupported low-order suite")
    for entry in suite["points"]:
        if Path(entry["name"]).name != entry["name"]:
            raise ValueError("invalid point name")
        out = root / "points" / entry["name"]
        if ed.digest(out / "experiment.json") != entry["sha256"]:
            raise ValueError("point contract changed")
        yield out


def verify(out, *, inputs=True):
    c = ed.read(out / "experiment.json")
    if c.get("schema") != SCHEMA:
        raise ValueError("unsupported scoring contract")
    for name, digest in c["code_hashes"].items():
        if ed.digest(ed.ROOT / name) != digest:
            raise ValueError(f"scientific code changed: {name}; retain this root and use a reviewed new root")
    if inputs:
        for name, digest in c["source_hashes"].items():
            if ed.digest(Path(c["source_run"]) / name) != digest:
                raise ValueError(f"source input changed: {name}")
        if ed.digest(Path(c["config"]["model"]) / "config.json") != c["model_config_sha256"]:
            raise ValueError("source model configuration changed")
    return c


def bound_tensor(path, binding):
    if not path.exists():
        return None
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("binding") != binding:
        raise ValueError(f"cached artifact belongs to a different contract: {path}")
    return value


def progress(out, shard, phase, done, total, prompt=None):
    row = {"host": socket.gethostname(), "pid": os.getpid(), "updated": time.time(),
           "phase": phase, "done": done, "total": total, "prompt": prompt}
    ed.atomic_json(out / f"progress-{shard}.json", row)
    print(f"[low-order] {phase} {done}/{total} prompt={prompt} shard={shard}", flush=True)


def validation_worker(out, shard):
    c = verify(out, inputs=False)
    binding = ed.digest(out / "experiment.json")
    target = out / "validation" / f"part-{shard}.pt"
    if bound_tensor(target, binding) is not None:
        return
    target.parent.mkdir(exist_ok=True)
    run = Path(c["source_run"])
    groups = read_groups(run / "rollouts_fresh_val.jsonl", c["validation_count"], c["config"]["val_k"])
    groups = {i: groups[i] for i in sorted(groups)[shard::4]}
    progress(out, shard, "load validation model", 0, len(groups))
    model, _ = backend.load_current(c["config"], run)
    value = backend.validation_gradient(model, groups, progress=lambda d, n, i: progress(out, shard, "validation", d, n, i))
    ae.atomic_torch(target, {**value, "binding": binding, "indices": sorted(groups)})
    progress(out, shard, "validation done", len(groups), len(groups))


def merge_direction(out):
    c = verify(out)
    binding = ed.digest(out / "experiment.json")
    paths = [out / "validation" / f"part-{s}.pt" for s in range(4)]
    partials = [bound_tensor(p, binding) for p in paths]
    if any(p is None for p in partials):
        raise ValueError("validation shards incomplete")
    for shard, partial in enumerate(partials):
        expected = list(range(c["validation_count"]))[shard::4]
        if partial.get("indices") != expected or partial["prompts"] != len(expected):
            raise ValueError("validation shard identities or coverage changed")
    hashes = {p.name: ed.digest(p) for p in paths}
    path = out / "direction.pt"
    cached = bound_tensor(path, binding)
    if cached is not None:
        if cached["shard_hashes"] != hashes:
            raise ValueError("validation shards changed")
        return
    optimizer = None
    if c["geometry"] == "adam_rms":
        optimizer = torch.load(Path(c["source_run"]) / f"policy_step_{c['config']['drift']}" / "optimizer.pt",
                               map_location="cpu", weights_only=True)
    value = backend.make_direction(partials, optimizer)
    ae.atomic_torch(path, {**value, "binding": binding, "shard_hashes": hashes})


def completed(out, idx, binding):
    path = out / "scores" / f"p{idx}.json"
    if not path.exists():
        return None
    row = ed.read(path)
    if (row.get("binding") != binding or row.get("prompt_idx") != idx
            or set(row.get("methods", {})) != set(METHODS)
            or not all(math.isfinite(v) for v in row["methods"].values())
            or row.get("direction_sha256") != ed.digest(out / "direction.pt")):
        raise ValueError(f"invalid completed score: {path}")
    return row


def score_worker(out, shard):
    c = verify(out, inputs=False)
    binding = ed.digest(out / "experiment.json")
    run = Path(c["source_run"])
    for folder in ("scores", "cache"):
        (out / folder).mkdir(exist_ok=True)
    groups = read_groups(run / "rollouts_behavior_train.jsonl", c["n"], 8)
    keys = [i for i in sorted(groups)[shard::4] if completed(out, i, binding) is None]
    if not keys:
        return
    progress(out, shard, "load current model", 0, len(keys))
    model, _ = backend.load_current(c["config"], run)
    saved = bound_tensor(out / "direction.pt", binding)
    direction = backend.device_direction(model, saved["direction"])
    direction_hash = ed.digest(out / "direction.pt")
    calibration = None
    if c["derivative"] == "finite":
        # Fixed validation examples, never independently held-out test responses.
        validation = read_groups(run / "rollouts_fresh_val.jsonl", c["validation_count"], c["config"]["val_k"])
        probes = []
        for rows in validation.values():
            if len({r["reward"] for r in rows}) == 2:
                probes.extend(next(r for r in rows if r["reward"] == reward) for reward in (0, 1))
            if len(probes) >= 4:
                break
        progress(out, shard, "finite-difference calibration", 0, len(probes))
        calibration = backend.calibrate(model, probes, direction, step=c["fd_step"], micro_batch=c["micro_batch"])
        ed.atomic_json(out / f"calibration-{shard}.json", {**calibration, "binding": binding,
                       "direction_sha256": direction_hash})
    for done, idx in enumerate(keys):
        progress(out, shard, "scoring", done, len(keys), idx)
        rows = groups[idx]
        cache_path = out / "cache" / f"beta-{idx}.pt"
        cache_binding = f"{binding}:prompt:{idx}"
        beta = ae.cached_logps(cache_path, cache_binding, rows)
        beta_computed = beta is None
        if beta is None:
            with model.disable_adapter():
                beta = sequence_logprobs_batch(model, rows, micro_batch=c["micro_batch"])
            ae.atomic_torch(cache_path, {"binding": cache_binding, "logps": beta})
        current = sequence_logprobs_batch(model, rows, micro_batch=c["micro_batch"])
        if c["derivative"] == "finite":
            derivatives = backend.finite_directional(model, rows, direction, calibration["step"], c["micro_batch"])
        else:
            derivatives = backend.exact_directional(model, rows, direction)
        log_ratios = [float((p.double()-b.double()).sum()) for p, b in zip(current, beta, strict=True)]
        value = reuse_score([r["reward"] for r in rows], derivatives.numpy(), log_ratios,
                            epsilon=c["config"]["grpo_advantage_epsilon"])
        input_tokens = sum(int(r["input_ids"].numel()) for r in rows)
        ed.atomic_json(out / "scores" / f"p{idx}.json", {**value, "binding": binding,
                       "prompt_idx": idx, "direction_sha256": direction_hash,
                       "derivative": c["derivative"], "calibration": calibration,
                       "directional": derivatives.tolist(), "log_ratios": log_ratios,
                       "cost": {"candidate_generation": 0, "input_tokens": input_tokens,
                                "beta_evaluated_now": beta_computed,
                                "teacher_forced_passes": (3 if c["derivative"] == "finite" else 2)+int(beta_computed),
                                "count_scope": "Logical successful passes; phase GPU time also includes startup and failed attempts.",
                                "candidate_backward_passes": 0 if c["derivative"] == "finite" else len(rows)}})
    progress(out, shard, "scoring done", len(keys), len(keys))


def merge(out):
    c = verify(out)
    binding = ed.digest(out / "experiment.json")
    values = [completed(out, i, binding) for i in range(c["n"])]
    if any(v is None for v in values):
        raise ValueError("scoring incomplete; no partial top-k is published")
    source = ed.read(Path(c["source_run"]) / "prompts.json")
    k = topk_count(c["n"], c["config"]["topk_frac"])
    scores = {m: {i: v["methods"][m] for i, v in enumerate(values)} for m in METHODS}
    scores["passrate_beta"] = {i: v["successes"]/v["responses"] for i, v in enumerate(values)}
    rng = np.random.default_rng(c["seed"]+907)
    scores["random"] = dict(enumerate(rng.random(c["n"]).tolist()))
    selected = {m: sorted(jittered_topk(s, k, c["seed"]+1000)) for m, s in scores.items()}
    ed.bind(out / "selection.json", {"binding": binding, "selected": selected,
            "scores": {m: {str(i): v for i, v in s.items()} for m, s in scores.items()},
            "direction_sha256": ed.digest(out / "direction.pt")})
    training_hashes = {}
    for arm in c["arms"]:
        arm_out = out / "training" / arm
        (arm_out / "subsets").mkdir(parents=True, exist_ok=True)
        payload = {**source, "train": [source["train"][i] for i in selected[arm]],
                   "selector": arm, "selected_idx": selected[arm], "k": k}
        subset = arm_out / "subsets" / f"subset-{arm}.json"
        ed.bind(subset, payload)
        extra = c["random_extra_steps"] if arm == "random" else 0
        contract = {**c["downstream_template"], "selectors": [arm], "all_subsets": [arm],
                    "steps": c["steps"]+extra, "selection_sha256": ed.digest(out / "selection.json")}
        ed.bind(arm_out / "experiment.json", contract)
        ed.bind(arm_out / "evaluation.json", c["evaluation"])
        ed.bind(arm_out / "subsets_hashes.json", {arm: ed.digest(subset)})
        for name in ("experiment.json", "evaluation.json", "subsets_hashes.json", f"subsets/subset-{arm}.json"):
            training_hashes[str((arm_out / name).relative_to(out))] = ed.digest(arm_out / name)
    ed.bind(out / "complete.json", {"binding": binding, "selection_sha256": ed.digest(out / "selection.json"),
            "training_hashes": training_hashes,
            "score_hashes": {str(i): ed.digest(out / "scores" / f"p{i}.json") for i in range(c["n"])}})


def verify_selection(out):
    c = verify(out)
    done = ed.read(out / "complete.json")
    if done["binding"] != ed.digest(out / "experiment.json") or done["selection_sha256"] != ed.digest(out / "selection.json"):
        raise ValueError("frozen selection changed")
    if set(done["score_hashes"]) != {str(i) for i in range(c["n"])}:
        raise ValueError("incomplete frozen score coverage")
    if ed.read(out / "selection.json")["direction_sha256"] != ed.digest(out / "direction.pt"):
        raise ValueError("frozen validation direction changed")
    for name, digest in done["training_hashes"].items():
        if ed.digest(out / name) != digest:
            raise ValueError(f"frozen training input changed: {name}")
    for idx, digest in done["score_hashes"].items():
        if ed.digest(out / "scores" / f"p{idx}.json") != digest:
            raise ValueError("completed candidate score changed")
    for arm in c["arms"]:
        arm_out = out / "training" / arm
        hashes = ed.read(arm_out / "subsets_hashes.json")
        if hashes != {arm: ed.digest(arm_out / "subsets" / f"subset-{arm}.json")}:
            raise ValueError("training subset changed")
    return c


def log_tail(path, count=8):
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size-8192))
        return handle.read().decode("utf-8", errors="replace").splitlines()[-count:]


def phase(root, out, arm, name, commands, gpus, env):
    """Terminate siblings promptly on failure; account for failed attempts too."""
    children = []
    start = time.monotonic()
    stamp = {"point": out.name, "arm": arm, "phase": name, "host": socket.gethostname(),
             "pid": os.getpid(), "started": time.time(), "gpus": len(gpus)}
    append_cost(root, {**stamp, "state": "started"})
    (out / "logs").mkdir(exist_ok=True)
    rc = 1
    try:
        for i, command in enumerate(commands):
            child_env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(gpus) if len(commands) == 1 else gpus[i]}
            with (out / "logs" / f"{name}-{arm}-{i}.log").open("a") as log:
                children.append(subprocess.Popen(command, env=child_env, stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True, close_fds=True))
        last = 0.
        while True:
            codes = [p.poll() for p in children]
            if any(code not in (None, 0) for code in codes):
                for i, code in enumerate(codes):
                    if code not in (None, 0):
                        path = out / "logs" / f"{name}-{arm}-{i}.log"
                        print(f"[worker failed] {path} exit={code}\n"+"\n".join(log_tail(path)), flush=True)
                return 1
            if all(code == 0 for code in codes):
                rc = 0
                return 0
            if time.monotonic()-last >= 15:
                print(f"[low-order] {out.name} {arm} {name}: {int(time.monotonic()-start)}s; logs={out / 'logs'}", flush=True)
                for i in range(len(children)):
                    tail = log_tail(out / "logs" / f"{name}-{arm}-{i}.log", 1)
                    if tail:
                        print(f"  worker {i}: {tail[0][:800]}", flush=True)
                last = time.monotonic()
            time.sleep(.5)
    finally:
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        seconds = time.monotonic()-start
        append_cost(root, {**stamp, "state": "finished", "exit_code": rc,
                          "seconds": seconds, "allocated_gpu_seconds": seconds*len(gpus)})


def record_failure(out, task, exc):
    print(f"[failed] {out.name} {task}: {exc}; trying other work", flush=True)
    ed.atomic_json(out / f"error-{task}.json", {"error": str(exc), "time": time.time(),
                   "host": socket.gethostname(), "historical": True})


def work(root, mode="run"):
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(gpus) != 4 or len(set(gpus)) != 4 or not all(gpus) or os.environ.get("OM_NODE_LOCK_HELD") != "1":
        raise ValueError("use run_low_order.sh on a four-GPU allocated node")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
    failures = 0
    for out in entries(root):
        if mode != "train":
            with lease(out / ".score.lock") as acquired:
                if acquired:
                    try:
                        c = verify(out)
                        if not (out / "complete.json").exists():
                            env = ae.model_environment(c["config"])
                            for stage in ("validation", "score"):
                                commands = [[sys.executable, str(HERE), "worker", "--out", str(out),
                                             "--stage", stage, "--shard", str(s)] for s in range(4)]
                                if phase(root, out, "selection", stage, commands, gpus, env):
                                    raise ValueError(f"{stage} failed; successful artifacts retained")
                                if stage == "validation":
                                    merge_direction(out)
                            merge(out)
                        verify_selection(out)
                    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                        failures += 1
                        record_failure(out, "selection", exc)
                else:
                    print(f"[leased] {out.name} scoring; trying other tasks", flush=True)
        if mode == "score" or not (out / "complete.json").exists():
            continue
        try:
            c = verify_selection(out)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            failures += 1
            record_failure(out, "selection", exc)
            continue
        env = ae.model_environment(c["config"])
        # Each arm gets its own existing-engine contract, including optional extra random updates.
        tasks = [(c["arms"][0], "before")] + [(a, a) for a in c["arms"]]
        for folder, arm in tasks:
            arm_out = out / "training" / folder
            with lease(arm_out / f".{arm}.lock") as acquired:
                if not acquired:
                    continue
                try:
                    if arm != "before":
                        target = arm_out / arm / "policy/policy_train.json"
                        if not target.exists():
                            steps = ed.read(arm_out / "experiment.json")["steps"]
                            command = [sys.executable, *ed.train_args(c["config"], Path(c["source_run"]), arm_out, arm, steps)]
                            if phase(root, out, arm, "train", [command], gpus, env):
                                raise ValueError("training failed; checkpoints retained")
                        ed.arm_policy(arm_out, arm)
                    commands = [[sys.executable, str(HERE), "worker", "--out", str(arm_out),
                                 "--stage", "evaluate", "--arm", arm, "--shard", str(s)] for s in range(4)]
                    if phase(root, out, arm, "evaluate", commands, gpus, env):
                        raise ValueError("evaluation failed; completed shards retained")
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    failures += 1
                    record_failure(out, arm, exc)
    status(root)
    return int(bool(failures))


def status(root):
    if not (root / "suite.json").exists():
        print(f"[not prepared] {root}")
        return
    results = []
    for out in entries(root):
        c = ed.read(out / "experiment.json")
        count = len(list((out / "scores").glob("p*.json")))
        print(f"{out.name}: scoring {count}/{c['n']}; frozen={(out / 'complete.json').exists()}")
        for path in sorted(out.glob("progress-*.json")):
            row = ed.read(path)
            print(f"  {row['host']} {row['phase']} {row['done']}/{row['total']} age={int(time.time()-row['updated'])}s")
        before_out = out / "training" / c["arms"][0]
        print(f"  before: {ed.arm_state(before_out, 'before')}")
        before = None
        if all((before_out / "before/evaluation" / f"shard-{s}.done.json").exists() for s in range(4)):
            before = float(ed.evaluation_means(before_out, "before").mean())
        point_results = []
        for arm in c["arms"]:
            arm_out = out / "training" / arm
            print(f"  {arm}: {ed.arm_state(arm_out, arm)}")
            if all((arm_out / arm / "evaluation" / f"shard-{s}.done.json").exists() for s in range(4)):
                means = ed.evaluation_means(arm_out, arm)
                result = {"point": out.name, "seed": c["seed"], "arm": arm,
                          "test_reward": float(means.mean()), "test_prompts": len(means),
                          "reward_before": before,
                          "reward_change": float(means.mean())-before if before is not None else None,
                          "steps": ed.read(arm_out / "experiment.json")["steps"]}
                results.append(result)
                point_results.append(result)
                print(f"    independent test reward={result['test_reward']:.6f}")
        random = next((r["test_reward"] for r in point_results if r["arm"] == "random"), None)
        for row in point_results:
            row["difference_vs_random"] = row["test_reward"]-random if random is not None else None
        for path in sorted(out.glob("error-*.json")):
            print(f"  historical failure: {ed.read(path)['error']}")
    ed.atomic_json(root / "results.json", {"schema": SCHEMA, "rows": results,
                   "matched_total_cost": False, "intervals": None,
                   "scope": "Observed independent test reward; no bootstrap or superiority claim. See cost.jsonl."})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "score", "train", "worker", "status", "check"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--runs", type=Path, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--drift", type=int, default=100)
    parser.add_argument("--eval-prompts", type=Path)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--pool-manifest", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--random-extra-steps", type=int, default=0)
    parser.add_argument("--eval-k", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=300)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--derivative", choices=("finite", "autograd"), default="finite")
    parser.add_argument("--fd-step", type=float, default=.1)
    parser.add_argument("--geometry", choices=("adam_rms", "identity"), default="adam_rms")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS[:3]))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stage", choices=("validation", "score", "evaluate"))
    parser.add_argument("--shard", type=int, choices=range(4), default=0)
    parser.add_argument("--arm", default="before")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            print(json.dumps(algebra_audit(), indent=2))
        elif args.command == "worker":
            if args.out is None or args.stage is None:
                parser.error("worker requires --out and --stage")
            if not torch.cuda.is_available() or os.environ.get("OM_NODE_LOCK_HELD") != "1":
                raise ValueError("GPU worker requires an admitted allocated GPU")
            if args.stage == "evaluate":
                ed.evaluate(args.out, args.arm, args.shard)
            elif args.stage == "validation":
                validation_worker(args.out, args.shard)
            else:
                score_worker(args.out, args.shard)
        else:
            if args.root is None:
                parser.error("--root is required")
            if args.command == "prepare":
                if not args.runs and args.matrix is None:
                    parser.error("prepare requires --runs or --matrix")
                runs = args.runs or ae.resolve_runs(args.matrix, args.seeds, args.drift)
                prepare(args.root, runs, evaluation=args.eval_prompts, pool=args.pool, pool_manifest=args.pool_manifest,
                        steps=args.steps, random_extra_steps=args.random_extra_steps, eval_k=args.eval_k,
                        test_count=args.test_count, micro_batch=args.micro_batch, derivative=args.derivative,
                        fd_step=args.fd_step, geometry=args.geometry, arms=args.arms)
                print(f"[prepared] {args.root}")
            elif args.command == "status":
                status(args.root)
            else:
                return work(args.root, args.command)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
