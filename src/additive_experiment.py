"""Rescore immutable response pools with additive and terminal TayPO-2 weights."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import torch

import evidence_downstream as ed
from additive_correction import METHODS, algebra_audit, correction_weights
from artifact_contract import validate_generation_contract
from downstream_compare import selector_scores
from experiment import read_rollouts, split_validation_directions
from grads import (
    ProjectionSpec,
    cosine,
    grad_params,
    loo_advantages,
    prompt_gradient,
    sequence_logprobs_batch,
)
from method_choice import lease, phase
from rollout import load_policy
from score_artifacts import load_complete_score_artifacts
from select_rules import jittered_topk, topk_count

SCHEMA = "offpolicy-additive-scoring/v1"
HERE = Path(__file__).resolve()
CODE_FILES = ("src/additive_correction.py", "src/additive_experiment.py", "src/grads.py",
              "src/experiment.py", "src/rollout.py", "src/rollout_contract.py",
              "src/artifact_contract.py", "src/select_rules.py", "src/fresh_validation.py",
              "src/downstream_compare.py", "src/score_artifacts.py")


def resolve_runs(matrix, seeds, drift):
    runs = []
    for seed in seeds:
        found = sorted(matrix.glob(f"family-math500-s{seed}/*-s{seed}-math500-d{drift}"))
        if len(found) != 1:
            raise ValueError(f"seed {seed}: expected one MATH d{drift} point under {matrix}; found {len(found)}")
        runs.append(found[0].resolve())
    return runs


def model_environment(config):
    env = dict(os.environ)
    for name, field, default in (("OM_ATTN", "attn", "eager"), ("OM_TOP_P", "top_p", 1.),
                                 ("OM_THINKING", "thinking", "off"),
                                 ("OM_PROMPT_FORMAT", "prompt_format", "olmo_rlzero_math"),
                                 ("OM_GEN_BATCH", "gen_batch", 32), ("OM_LORA_TARGETS", "lora_targets", "")):
        env[name] = str(config[field] if config.get(field) is not None else default)
    env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def source_contract(run, *, micro_batch):
    if not (run / "DONE").is_file():
        raise ValueError(f"source is not complete: {run}")
    config = ed.read(run / "run_config.json")
    for key in ("model", "seed", "drift", "dataset", "topk_frac", "proj_dim", "grad_layers", "clip_cap"):
        if key not in config:
            raise ValueError(f"source config lacks {key}")
    if micro_batch < 1 or config["proj_dim"] < 1 or config["grad_layers"] < 1:
        raise ValueError("micro-batch, projection dimension and gradient layers must be positive")
    if not math.isfinite(config["clip_cap"]) or config["clip_cap"] < 1:
        raise ValueError("invalid source clipping cap")
    model = Path(config["model"]).resolve()
    if not (model / "config.json").is_file():
        raise ValueError(f"source model snapshot unavailable: {model}")
    artifacts = load_complete_score_artifacts(run)
    n = len(ed.read(run / "prompts.json")["train"])
    if set(artifacts.oracle) != set(range(n)):
        raise ValueError("source score coverage does not match candidate identities")
    if any("r" not in halves for halves in artifacts.splithalf.values()):
        raise ValueError("source scores lack the matched R ranking split")
    val = torch.load(run / "val_groups.pt", map_location="cpu", weights_only=True)
    if val.ndim != 2 or val.shape[1] != config["proj_dim"] or not torch.isfinite(val).all():
        raise ValueError("validation gradients have invalid shape or values")
    split_validation_directions(val)
    topk_count(n, config["topk_frac"])
    generation = validate_generation_contract(run, ("rollouts_behavior_train",))
    files = ["run_config.json", "prompts.json", "rollouts_behavior_train.jsonl", "val_groups.pt",
             "scores_offpolicy.json", "scores_oracle.json", "scores_splithalf.json"]
    files += [str(p.relative_to(run)) for p in sorted(run.glob("rollouts_behavior_train*.manifest.json"))]
    if config["drift"]:
        adapter = run / f"policy_step_{config['drift']}"
        from train_policy_grpo import validate_policy_manifest
        validate_policy_manifest(adapter, target_steps=config["drift"],
                                 world_size=config["grpo_world_size"],
                                 training_objective="grpo", require_complete_hashes=True)
        files += [f"policy_step_{config['drift']}/{name}" for name in ed.POLICY_FILES]
    return {"schema": SCHEMA, "seed": config["seed"], "source_run": str(run.resolve()),
            "config": config, "n": n, "micro_batch": micro_batch, "methods": list(METHODS),
            "source_hashes": {name: ed.digest(run / name) for name in files},
            "model_config_sha256": ed.digest(model / "config.json"),
            "generation_validation": generation,
            "code_hashes": {name: ed.digest(ed.ROOT / name) for name in CODE_FILES},
            "clipping": "component clipping before composition; signed weights allowed",
            "theory_scope": "Raw remainder identities do not hold unchanged after component clipping.",
            "novelty_verified": False, "no_generation": True}


def prepare(root, runs, *, micro_batch=1):
    if not runs or len(set(map(str, runs))) != len(runs):
        raise ValueError("source points must be nonempty and unique")
    root = ed.require_separate_output(root, runs)
    if any(root in run.resolve().parents for run in runs):
        raise ValueError("output must not contain a source point")
    contracts = [source_contract(run, micro_batch=micro_batch) for run in runs]
    names = [Path(c["source_run"]).name for c in contracts]
    if len(set(names)) != len(names):
        raise ValueError("source point names must be unique")
    root.mkdir(parents=True, exist_ok=True)
    with lease(root / ".prepare.lock", blocking=True):
        ed.bind(root / "preparation.json", {"contracts": contracts})
        entries = []
        for name, contract in zip(names, contracts, strict=True):
            out = root / "points" / name
            out.mkdir(parents=True, exist_ok=True)
            ed.bind(out / "experiment.json", contract)
            entries.append({"name": name, "seed": contract["seed"],
                            "contract_sha256": ed.digest(out / "experiment.json")})
        suite = {"schema": SCHEMA, "points": entries, "methods": list(METHODS),
                 "scope": "Exploratory scoring extension; no existing matrix or E5 outputs modified."}
        ed.bind(root / "suite.json", suite)
    return suite


def verify_point(out, *, inputs=True, code=True):
    c = ed.read(out / "experiment.json")
    if c.get("schema") != SCHEMA:
        raise ValueError("unsupported additive scoring contract")
    if code:
        for name, value in c["code_hashes"].items():
            if ed.digest(ed.ROOT / name) != value:
                raise ValueError(f"scoring code changed: {name}; use a reviewed new output root")
    if inputs:
        run = Path(c["source_run"])
        for name, value in c["source_hashes"].items():
            if ed.digest(run / name) != value:
                raise ValueError(f"source input changed: {run / name}")
        if ed.digest(Path(c["config"]["model"]) / "config.json") != c["model_config_sha256"]:
            raise ValueError("source model configuration changed")
    return c


def atomic_torch(path, obj):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(obj, tmp)
    tmp.replace(path)


def validate_rows(rows, n):
    if set(rows) != set(range(n)):
        raise ValueError("behavior response coverage differs from candidate prompts")
    for idx, group in rows.items():
        seen = [r["rollout_idx"] for r in group]
        if len(group) < 2 or sorted(seen) != list(range(len(group))):
            raise ValueError(f"prompt {idx}: duplicate, missing, or insufficient responses")
        for row in group:
            if not math.isfinite(float(row["reward"])) or not 0 <= row["reward"] <= 1:
                raise ValueError(f"prompt {idx}: invalid reward")
            if not 0 < row["resp_start"] < row["input_ids"].numel():
                raise ValueError(f"prompt {idx}: invalid response boundary")
        group.sort(key=lambda row: row["rollout_idx"])


def cached_logps(path, binding, rows):
    if not path.exists():
        return None
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value["binding"] != binding or len(value["logps"]) != len(rows):
        raise ValueError(f"behavior cache binding mismatch: {path}")
    for logp, row in zip(value["logps"], rows, strict=True):
        if logp.ndim != 1 or logp.numel() != row["input_ids"].numel()-row["resp_start"] or not torch.isfinite(logp).all():
            raise ValueError(f"invalid behavior log-probability cache: {path}")
    return value["logps"]


def completed_prompt(out, idx, binding):
    path = out / "scores" / f"p{idx}.json"
    if not path.exists():
        return None
    value = ed.read(path)
    if value.get("binding") != binding or value.get("prompt_idx") != idx:
        raise ValueError(f"completed prompt binding mismatch: {path}")
    if set(value.get("methods", {})) != set(METHODS):
        raise ValueError(f"incomplete method coverage: {path}")
    for row in value["methods"].values():
        if not all(math.isfinite(row[key]) for key in ("score", "norm", "negative_fraction", "max_abs_weight")):
            raise ValueError(f"nonfinite prompt result: {path}")
        if not -1.00001 <= row["score"] <= 1.00001 or row["norm"] < 0:
            raise ValueError(f"invalid score or norm: {path}")
    if ed.digest(out / "scores" / f"p{idx}.pt") != value["gradient_sha256"]:
        raise ValueError(f"saved projected gradients changed: {path}")
    return value


class Progress:
    def __init__(self, out, shard):
        self.path = out / f"progress-{shard}.json"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = {"host": socket.gethostname(), "pid": os.getpid(), "shard": shard,
                      "phase": "loading", "done": 0, "total": 0, "prompt": None}
        self.start = time.monotonic()
        self.thread = threading.Thread(target=self.beat, daemon=True)

    def emit(self):
        with self.lock:
            elapsed = time.monotonic()-self.start
            state = {**self.state, "updated": time.time(), "elapsed": elapsed,
                     "eta_seconds": elapsed * (self.state["total"]-self.state["done"]) / self.state["done"]
                     if self.state["done"] else None}
            ed.atomic_json(self.path, state)
        eta = "warming up" if state["eta_seconds"] is None else f"{int(state['eta_seconds'])}s"
        print(f"[additive] shard={state['shard']} {state['phase']} {state['done']}/{state['total']} "
              f"prompt={state['prompt']} elapsed={int(elapsed)}s ETA={eta}", flush=True)

    def update(self, stage, done, total, prompt=None):
        with self.lock:
            if stage != self.state["phase"]:
                self.start = time.monotonic()
            self.state.update(phase=stage, done=done, total=total, prompt=prompt)
        self.emit()

    def beat(self):
        while not self.stop.wait(15):
            self.emit()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def worker(out, shard, shards):
    if not 0 <= shard < shards or shards < 1:
        raise ValueError("invalid scoring shard")
    c = verify_point(out, inputs=False)
    binding = ed.digest(out / "experiment.json")
    if os.environ.get("OM_NODE_LOCK_HELD") != "1" or not torch.cuda.is_available():
        raise ValueError("scoring needs an admitted GPU; use run_additive.sh")
    run, config = Path(c["source_run"]), c["config"]
    for directory in ("cache", "scores"):
        (out / directory).mkdir(exist_ok=True)
    rows = read_rollouts(run / "rollouts_behavior_train.jsonl")
    validate_rows(rows, c["n"])
    keys = [idx for idx in sorted(rows)[shard::shards] if completed_prompt(out, idx, binding) is None]
    with Progress(out, shard) as progress:
        if not keys:
            progress.update("done", 0, 0)
            return
        missing = [idx for idx in keys if cached_logps(out / "cache" / f"beta-{idx}.pt", binding, rows[idx]) is None]
        if missing:
            progress.update("load behavior model", 0, len(missing))
            beta, tok = load_policy(config["model"], None)
            for done, idx in enumerate(missing):
                progress.update("behavior logprobs", done, len(missing), idx)
                logps = sequence_logprobs_batch(beta, rows[idx], micro_batch=c["micro_batch"])
                atomic_torch(out / "cache" / f"beta-{idx}.pt", {"binding": binding, "logps": [p.cpu() for p in logps]})
            del beta, tok
            gc.collect()
            torch.cuda.empty_cache()
        progress.update("load current model", 0, len(keys))
        adapter = run / f"policy_step_{config['drift']}" if config["drift"] else None
        pi, _ = load_policy(config["model"], adapter)
        params = grad_params(pi, config["grad_layers"])
        spec = ProjectionSpec(dim=config["proj_dim"])
        val = torch.load(run / "val_groups.pt", map_location="cpu", weights_only=True)
        if val.ndim != 2 or val.shape[1] != spec.dim or not torch.isfinite(val).all():
            raise ValueError("validation gradients have invalid shape or values")
        selection_val, _, _ = split_validation_directions(val)
        for done, idx in enumerate(keys):
            progress.update("gradient scoring", done, len(keys), idx)
            group = rows[idx]
            beta_lp = cached_logps(out / "cache" / f"beta-{idx}.pt", binding, group)
            pi_lp = sequence_logprobs_batch(pi, group, micro_batch=c["micro_batch"])
            advantages = loo_advantages(torch.tensor([r["reward"] for r in group]))
            result, gradients = {}, {}
            for method in METHODS:
                bare = [correction_weights(lp, lb, method=method, cap=config["clip_cap"])
                        for lp, lb in zip(pi_lp, beta_lp, strict=True)]
                weights = [w * float(a) for w, a in zip(bare, advantages, strict=True)]
                g = prompt_gradient(pi, params, group, weights, spec, micro_batch=c["micro_batch"])
                if not torch.isfinite(g).all():
                    raise ValueError(f"nonfinite gradient: prompt {idx}, method {method}")
                all_weights = torch.cat(bare)
                result[method] = {"score": cosine(g, selection_val), "norm": float(g.norm()),
                                  "negative_fraction": float((all_weights < 0).float().mean()),
                                  "max_abs_weight": float(all_weights.abs().max())}
                gradients[method] = g
            target = out / "scores" / f"p{idx}.pt"
            atomic_torch(target, {"binding": binding, "prompt_idx": idx, "gradients": gradients})
            ed.atomic_json(out / "scores" / f"p{idx}.json", {"binding": binding, "prompt_idx": idx,
                           "gradient_sha256": ed.digest(target), "methods": result})
        progress.update("done", len(keys), len(keys))


def merge(out):
    c = verify_point(out)
    binding = ed.digest(out / "experiment.json")
    values = [completed_prompt(out, idx, binding) for idx in range(c["n"])]
    if any(row is None for row in values):
        raise ValueError("scoring incomplete; no partial selection is published")
    scores = {method: {str(row["prompt_idx"]): row["methods"][method] for row in values} for method in METHODS}
    ed.bind(out / "scores_additive.json", scores)
    source = ed.read(Path(c["source_run"]) / "prompts.json")
    k = topk_count(c["n"], c["config"]["topk_frac"])
    target = out / "subsets"
    target.mkdir(exist_ok=True)
    hashes = {}
    for method in METHODS:
        selected = sorted(jittered_topk({int(i): row["score"] for i, row in scores[method].items()},
                                       k, c["seed"] + 1000))
        payload = {**source, "train": [source["train"][idx] for idx in selected],
                   "selector": method, "k": k, "selected_idx": selected,
                   "source_run": Path(c["source_run"]).name}
        path = target / f"subset-{method}.json"
        ed.bind(path, payload)
        hashes[method] = ed.digest(path)
    report = score_report(c, scores)
    ed.bind(out / "comparison.json", report)
    temporary = out / f"comparison.csv.tmp.{os.getpid()}"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    temporary.replace(out / "comparison.csv")
    ed.bind(out / "complete.json", {"contract_sha256": binding,
                                    "scores_sha256": ed.digest(out / "scores_additive.json"), "subsets": hashes,
                                    "comparison_sha256": ed.digest(out / "comparison.json")})
    return scores


def score_report(contract, scores):
    run = Path(contract["source_run"])
    artifacts = load_complete_score_artifacts(run)
    methods = selector_scores(run, contract["seed"])
    methods.update({name: {int(i): row["score"] for i, row in values.items()}
                    for name, values in scores.items()})
    k = topk_count(contract["n"], contract["config"]["topk_frac"])
    selected = {name: set(jittered_topk(values, k, contract["seed"]+1000)) for name, values in methods.items()}
    reference = {i: (row["a"]+row["b"])/2 for i, row in artifacts.splithalf.items()}
    center = sum(reference.values()) / len(reference)
    return {"schema": SCHEMA, "seed": contract["seed"], "rows": [
        {"method": name, "k": k,
         "independent_ab_gain": sum(reference[i] for i in subset)/k-center,
         "overlap_with_fresh_r": len(subset & selected["fresh_r"])/k,
         "overlap_with_g11": len(subset & selected["g11"])/k}
        for name, subset in selected.items()],
        "scope": "Descriptive fixed-budget A/B cosine gain, not population cosine or downstream reward. No method choice or superiority test.",
        "cost": {"end_to_end_complete": False, "reason": "Historical generation and scoring costs are not included."}}


def suite_entries(root):
    suite = ed.read(root / "suite.json")
    if suite.get("schema") != SCHEMA:
        raise ValueError("unsupported additive suite")
    for entry in suite["points"]:
        if Path(entry["name"]).name != entry["name"]:
            raise ValueError("invalid point name")
        out = root / "points" / entry["name"]
        if ed.digest(out / "experiment.json") != entry["contract_sha256"]:
            raise ValueError("point contract changed")
        yield out


def work(root):
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(gpus) != 4 or len(set(gpus)) != 4 or not all(gpus) or os.environ.get("OM_NODE_LOCK_HELD") != "1":
        raise ValueError("use run_additive.sh on an allocated four-GPU node")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    failures = 0
    for out in suite_entries(root):
        with lease(out / ".work.lock") as acquired:
            if not acquired:
                print(f"[busy] {out.name}; trying another point", flush=True)
                continue
            try:
                c = verify_point(out)
                binding = ed.digest(out / "experiment.json")
                if not all(completed_prompt(out, i, binding) for i in range(c["n"])):
                    commands = [[sys.executable, str(HERE), "worker", "--out", str(out),
                                 "--shard", str(s), "--shards", "4"] for s in range(4)]
                    if phase(root, out, "additive", "rescoring", commands, gpus, model_environment(c["config"])):
                        raise ValueError("GPU rescoring failed; completed prompts retained")
                merge(out)
                print(f"[complete] {out.name}", flush=True)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                failures += 1
                ed.atomic_json(out / "last_error.json", {"message": str(exc), "timestamp": time.time(), "host": socket.gethostname()})
                print(f"[failed] {out.name}: {exc}; trying another point", flush=True)
    return int(bool(failures))


def status(root):
    if not (root / "suite.json").exists():
        print(f"[not prepared] {root}")
        return
    for out in suite_entries(root):
        c = verify_point(out, inputs=False, code=False)
        count = len(list((out / "scores").glob("p*.json")))
        print(f"{out.name}: scored {count}/{c['n']}; merged={'yes' if (out / 'complete.json').exists() else 'no'}")
        for path in sorted(out.glob("progress-*.json")):
            row = ed.read(path)
            age = int(time.time() - row["updated"])
            print(f"  {row['host']} shard {row['shard']}: {row['phase']} {row['done']}/{row['total']} "
                  f"prompt={row['prompt']} last_update={age}s ago")
        if (out / "last_error.json").exists():
            print(f"  last failure (historical): {ed.read(out / 'last_error.json')['message']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run", "worker", "status", "check"))
    p.add_argument("--root", type=Path)
    p.add_argument("--matrix", type=Path)
    p.add_argument("--runs", type=Path, nargs="+")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--drift", type=int, default=400)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--out", type=Path)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=4)
    args = p.parse_args(argv)
    try:
        if args.command == "check":
            print(json.dumps(algebra_audit(), indent=2))
        elif args.command == "worker":
            if args.out is None:
                p.error("worker requires --out")
            worker(args.out, args.shard, args.shards)
        else:
            if args.root is None:
                p.error("--root is required")
            if args.command == "prepare":
                if not args.runs and args.matrix is None:
                    p.error("prepare requires --matrix or --runs")
                runs = args.runs or resolve_runs(args.matrix, args.seeds, args.drift)
                prepare(args.root, runs, micro_batch=args.micro_batch)
                print(f"[prepared] {args.root}")
            elif args.command == "run":
                return work(args.root)
            else:
                status(args.root)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
