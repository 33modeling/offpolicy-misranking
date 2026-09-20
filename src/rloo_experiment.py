"""Isolated d0 RLOO comparison using frozen E5 selections, not a new H measurement."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

import evidence_downstream as ed

ARMS = ("random", "passrate_beta", "fresh_r")
SCHEMA = "rloo-frozen-selection/v1"
TAG = "olmo3-1025-7b-base-rlzero-grpo-h100-v2"
ROOT = Path(__file__).resolve().parents[1]
SCOPE = ("Native RLOO training from the base model on frozen GRPO-study selections. "
         "Matched updates; selection is not recomputed. Costs cover new training/evaluation "
         "only, not historical selection. No H, switching-time, or RLOO-native selector claim.")


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def disjoint(out, inputs):
    out = out.resolve()
    for item in inputs:
        item = item.resolve()
        if out == item or out in item.parents or item in out.parents:
            raise ValueError(f"output must be separate from input: {item}")


def runtime_env(config):
    if float(config.get("top_p", 1.0)) != 1.0:
        raise ValueError("RLOO requires source top_p=1.0")
    return {"OM_TOP_P": "1.0", "OM_PROMPT_FORMAT": config["prompt_format"],
            "OM_ATTN": str(config.get("attn") or "eager"),
            "OM_GEN_BATCH": str(config.get("gen_batch", 32)),
            "OM_LORA_TARGETS": str(config.get("lora_targets") or ""),
            "OM_THINKING": str(config.get("thinking") or "off"),
            "OM_SKIP_HYBRID": "1", "OM_ONLINE": "0",
            "OM_MATH_VERIFIER": "math_verify", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "TOKENIZERS_PARALLELISM": "false"}


def prepare(run, out, evaluation, *, smoke=False, dry=False):
    run, out, evaluation = run.resolve(), out.resolve(), evaluation.resolve()
    config = ed.read(run / "run_config.json")
    if config.get("drift") != 0:
        raise ValueError("d0 only: a GRPO parent cannot be resumed as RLOO; no implicit algorithm handoff")
    disjoint(out, [run, evaluation.parent, Path(config["model"])])
    steps, k = (2, 2) if smoke else (100, 8)
    source = ed.prepare(run, out, evaluation, steps, k, ARMS, dry=True)
    test = ed.read(evaluation)
    rows = test["test"][:4] if smoke else test["test"]
    if not smoke and len(rows) != 300:
        raise ValueError("full protocol requires the frozen 300-question independent test")
    contract = {"schema": SCHEMA, "objective": "rloo", "scope": SCOPE,
                "source": source, "evaluation_input": str(evaluation),
                "steps": steps, "eval_k": k, "eval_n": len(rows), "smoke": smoke,
                "arms": list(ARMS), "environment": runtime_env(config),
                "code_hashes": {str(p.relative_to(ROOT)): ed.digest(p)
                                for p in sorted((ROOT / "src").glob("*.py"))}}
    if dry:
        return contract
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / ".prepare.lock"):
        ed.bind(out / "experiment.json", contract)
        ed.bind(out / "evaluation.json", {"val": rows, "provenance": test["provenance"]})
        hashes = out / "inputs.json"
        if not hashes.exists():
            ed.write_subsets(run, out / "subsets", config["topk_frac"], config["seed"])
            paths = [out / "evaluation.json", *(out / "subsets").glob("subset-*.json")]
            ed.bind(hashes, {str(p.relative_to(out)): ed.digest(p) for p in paths})
        validate(out)
    return contract


def validate(out):
    c = ed.read(out / "experiment.json")
    if c["schema"] != SCHEMA or c["objective"] != "rloo" or c["arms"] != list(ARMS):
        raise ValueError("invalid RLOO contract")
    source = c["source"]
    for package, version in source["runtime"]["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"runtime package changed: {package}")
    run = Path(source["source_run"])
    for name, digest in source["source_hashes"].items():
        if ed.digest(run / name) != digest:
            raise ValueError(f"source changed: {name}")
    config = ed.read(run / "run_config.json")
    if ed.digest(Path(config["model"]) / "config.json") != source["model_config_sha256"]:
        raise ValueError("model configuration changed")
    for name, digest in c["code_hashes"].items():
        if ed.digest(ROOT / name) != digest:
            raise ValueError(f"code changed since preparation: {name}")
    for name, digest in ed.read(out / "inputs.json").items():
        if ed.digest(out / name) != digest:
            raise ValueError(f"prepared input changed: {name}")
    return c, config


def training_command(out, arm):
    c, config = validate(out)
    if arm not in ARMS:
        raise ValueError("unknown RLOO arm")
    args = ed.train_args(config, Path(c["source"]["source_run"]), out, arm, c["steps"])
    ed._replace_flag(args, "--objective", "rloo")
    if "--reliability-log" in args:
        args.remove("--reliability-log")
    return [sys.executable, *args]


def policy(out, arm):
    c, config = validate(out)
    if arm == "before":
        return None
    if arm not in ARMS:
        raise ValueError("unknown RLOO arm")
    from train_policy_grpo import validate_policy_lineage
    path = out / arm / "policy"
    validate_policy_lineage(
        path, target_steps=c["steps"], world_size=4, training_objective="rloo",
        expected_start_step=0, expected_parent=None, expected_model=Path(config["model"]),
        expected_seed=config["seed"], expected_max_new_tokens=config["max_new_tokens"],
        expected_prompt_format=config["prompt_format"], expected_config=ed._expected_config(config),
        expected_prompts=out / "subsets" / f"subset-{arm}.json", require_complete_hashes=True)
    return path


def binding(out, arm, shard):
    c, _ = validate(out)
    if shard not in range(4):
        raise ValueError("shard must be 0..3")
    adapter = policy(out, arm)
    indices = range(c["eval_n"] * shard // 4, c["eval_n"] * (shard + 1) // 4)
    b = {"experiment_sha256": ed.digest(out / "experiment.json"),
         "inputs_sha256": ed.digest(out / "inputs.json"), "arm": arm, "shard": shard,
         "adapter_sha256": ed.digest(adapter / "adapter_model.safetensors") if adapter else None,
         "manifest_sha256": ed.digest(adapter / "policy_train.json") if adapter else None}
    return b, adapter, indices


def checked_rows(out, arm, shard):
    c, _ = validate(out)
    b, _, indices = binding(out, arm, shard)
    target = out / arm / "evaluation"
    path = target / f"shard-{shard}.jsonl"
    expected = {"binding": b, "rollouts_sha256": ed.digest(path)}
    if ed.read(target / f"shard-{shard}.done.json") != expected:
        raise ValueError("evaluation seal mismatch")
    return ed.reward_rows(path, indices, c["eval_k"])


def evaluate(out, arm, shard):
    c, config = validate(out)
    os.environ.update(c["environment"])
    b, adapter, indices = binding(out, arm, shard)
    target = out / arm / "evaluation"
    with lock(target / f"shard-{shard}.lock"):
        done = target / f"shard-{shard}.done.json"
        if done.exists():
            checked_rows(out, arm, shard)
            return
        ed.bind(target / f"shard-{shard}.contract.json", b)
        from rollout import collect_rollouts, load_policy
        model, tokenizer = load_policy(config["model"], adapter)
        rows = ed.read(out / "evaluation.json")["val"][indices.start:indices.stop]
        path = target / f"shard-{shard}.jsonl"
        collect_rollouts(model, tokenizer, rows, c["eval_k"], config["max_new_tokens"],
                         float(config["temperature"]), path, idx_offset=indices.start,
                         sampling_seed_base=c["source"]["eval_seed"])
        ed.reward_rows(path, indices, c["eval_k"])
        ed.atomic_json(done, {"binding": b, "rollouts_sha256": ed.digest(path)})


def complete(out, arm):
    for shard in range(4):
        if not (out / arm / "evaluation" / f"shard-{shard}.done.json").exists():
            return False
        checked_rows(out, arm, shard)
    return True


def state(out, arm):
    if complete(out, arm):
        return "done"
    if arm != "before" and (out / arm / "policy/policy_train.json").exists():
        policy(out, arm)
        return "trained; evaluation incomplete"
    if any((out / arm / "policy").glob("checkpoint-*")):
        return "paused; checkpoint available"
    return "pending / incomplete"


def report(out):
    c, _ = validate(out)
    values = {}
    for arm in ("before", *ARMS):
        if not complete(out, arm):
            raise ValueError(f"incomplete arm: {arm}")
        rewards = [[] for _ in range(c["eval_n"])]
        for shard in range(4):
            for row in checked_rows(out, arm, shard):
                rewards[row["prompt_idx"]].append(row["reward"])
        values[arm] = np.array([np.mean(r) for r in rewards])
    rows = []
    for arm in ARMS:
        row = {"arm": arm, "mean_reward": float(values[arm].mean())}
        for reference in ("before", "random", "passrate_beta"):
            delta = values[arm] - values[reference]
            lo, hi = ed.paired_interval(delta, c["source"]["seed"])
            row["vs_" + reference] = {"mean": float(delta.mean()), "lower": lo, "upper": hi}
        rows.append(row)
    result = {"schema": SCHEMA, "scope": SCOPE, "smoke": c["smoke"],
              "experiment_sha256": ed.digest(out / "experiment.json"), "rows": rows,
              "interval_scope": "Paired prompt bootstrap, conditional on this training seed; not across-seed uncertainty."}
    ed.atomic_json(out / "results.json", result)
    return result


def run_arm(out, arm, seconds):
    from selection_gate_gpu import meter
    c, _ = validate(out)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(devices) != 4 or len(set(devices)) != 4 or any(not d.strip() for d in devices):
        raise ValueError("exactly four distinct allocated GPUs required")
    if os.environ.get("OM_NODE_LOCK_HELD") != "1":
        raise ValueError("use run_rloo.sh: node admission lock is required")
    env = {**os.environ, **c["environment"]}
    if arm != "before":
        # The trainer validates a completed publication or resumes its own checkpoint.
        if not (out / arm / "policy/policy_train.json").exists():
            meter(out / arm, "train", os.environ.get("RLOO_GPU_TYPE", "unrecorded"),
                  commands=[(training_command(out, arm), ",".join(devices))],
                  env=env, timeout=seconds, devices=len(devices))
        policy(out, arm)
    commands = []
    for shard, device in enumerate(devices):
        if (out / arm / "evaluation" / f"shard-{shard}.done.json").exists():
            checked_rows(out, arm, shard)
        else:
            commands.append(([sys.executable, str(Path(__file__).resolve()), "evaluate",
                              "--out", str(out), "--arm", arm, "--shard", str(shard)], device))
    if commands:
        meter(out / arm, "evaluation", os.environ.get("RLOO_GPU_TYPE", "unrecorded"), commands=commands,
              env=env, timeout=seconds, devices=len(devices))
    if not complete(out, arm):
        raise ValueError("evaluation did not complete")


def source_paths(work, source_root, seed):
    root = source_root or work / "runs" / TAG
    return root / f"family-math500-s{seed}" / f"{TAG}-s{seed}-math500-d0"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "prepare", "status", "smoke", "run", "report", "evaluate", "check"))
    p.add_argument("--root", type=Path, default=Path(os.environ.get("RLOO_ROOT", "/tmp/rloo-selector-v1")))
    p.add_argument("--work", type=Path, default=Path(os.environ.get("OM_WORK", "/group-volume/minsoo3.kim/offpolicy-misranking")))
    p.add_argument("--source-root", type=Path)
    p.add_argument("--eval-prompts", type=Path)
    p.add_argument("--max-phase-seconds", type=float)
    p.add_argument("--out", type=Path)
    p.add_argument("--arm", choices=("before", *ARMS))
    p.add_argument("--shard", type=int)
    args = p.parse_args()
    root = args.root.resolve()
    if args.command == "plan":
        print(json.dumps({"scope": SCOPE, "dataset": "math500", "drift": 0, "seeds": [0, 1, 2],
                          "arms": ARMS, "training_runs": 9, "updates": 100, "eval_prompts": 300,
                          "eval_k": 8, "d400": "blocked: needs an explicit algorithm-handoff protocol",
                          "smoke": "2 updates, 4 questions x 2 responses, isolated output",
                          "launch": "manual only; smoke required; phase timeout must be specified"}, indent=2))
        return
    if args.command == "evaluate":
        if args.out is None or args.arm is None or args.shard is None:
            p.error("evaluate requires --out, --arm and --shard")
        evaluate(args.out, args.arm, args.shard)
        return
    outs = [root / f"s{seed}" for seed in range(3)]
    if args.command == "prepare":
        evaluation = args.eval_prompts or args.work / "inputs/e5-reduced/test-math500-d0.json"
        runs = [source_paths(args.work, args.source_root, seed) for seed in range(3)]
        disjoint(root, [*runs, evaluation.parent])
        configs = [ed.read(run / "run_config.json") for run in runs]
        fields = ("model", "dataset", "max_new_tokens", "temperature", "prompt_format",
                  "attn", "gen_batch", "lora_targets", "thinking", "top_p",
                  "grpo_gradient_checkpointing", *ed.TRAIN_FLAGS.values())
        if any(any(config.get(key) != configs[0].get(key) for key in fields) for config in configs[1:]):
            raise ValueError("source model/runtime/training configuration differs across seeds")
        for seed, run in enumerate(runs):
            if ed.read(run / "run_config.json")["seed"] != seed:
                raise ValueError("source seed does not match matrix position")
            prepare(run, outs[seed], evaluation, dry=True)
        for seed, run in enumerate(runs):
            prepare(run, outs[seed], evaluation)
        prepare(runs[0], root / "smoke", evaluation, smoke=True)
        print("Prepared 9 RLOO training arms + isolated smoke; no GPU work launched.")
        return
    if args.command == "status":
        for out in [root / "smoke", *outs]:
            if not (out / "experiment.json").exists():
                print(f"{out.name}: not prepared")
                continue
            validate(out)
            for arm in (("random",) if out.name == "smoke" else ("before", *ARMS)):
                try:
                    with lock(out / arm / ".worker.lock"):
                        progress = state(out, arm)
                except BlockingIOError:
                    progress = "running (lock held)"
                print(f"{out.name}/{arm}: {progress}")
        return
    for out in [root / "smoke", *outs]:
        validate(out)
    if args.command == "check":
        print("RLOO inputs and code hashes verified.")
        return
    if args.command == "report":
        print(json.dumps([report(out) for out in outs], indent=2))
        return
    seconds = args.max_phase_seconds
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        p.error("GPU modes require positive --max-phase-seconds (timeout is failure, not completion)")
    if args.command == "smoke":
        with lock(root / "smoke/random/.worker.lock"):
            run_arm(root / "smoke", "random", seconds)
        print("Smoke training and evaluation passed; not included in benchmark results.")
        return
    if not complete(root / "smoke", "random"):
        raise ValueError("isolated GPU smoke must pass before the full experiment")
    busy = 0
    for out in outs:
        for arm in ("before", *ARMS):
            try:
                with lock(out / arm / ".worker.lock"):
                    if not complete(out, arm):
                        run_arm(out, arm, seconds)
            except BlockingIOError:
                busy += 1
    print(f"Worker finished; {busy} arms held by other workers. Use status for global completion.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f"[rloo-error] {exc}", file=sys.stderr)
        raise SystemExit(75 if isinstance(exc, BlockingIOError) else 1)
