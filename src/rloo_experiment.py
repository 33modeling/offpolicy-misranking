"""Matched E5 comparison changing only the continuation objective to RLOO."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

import evidence_downstream as ed

ARMS = ("random", "passrate_beta", "fresh_r")
POINTS = tuple((drift, seed) for drift in (0, 400) for seed in range(3))
SCHEMA = "rloo-frozen-selection/v2"
TAG = "olmo3-1025-7b-base-rlzero-grpo-h100-v2"
ROOT = Path(__file__).resolve().parents[1]
PRE_QUEUE_OBSERVATION_CODE = 'f23ccd63e564d1a9cbf65aa21de835b1317f5aa5bae9ad3530a4e01e6ca1ad92'
PAIR_OBSERVATION_UPGRADE = (
    'f02238e97e9d691e2e13491f33653916ab5a51db82f4c98a72fa299e5b9739bf',
    '042446a0513d8eaeba2dc93ad0b4401a85f8ae9013c3042f80691afa81901f0f')
SCOPE = ("Matched GRPO-study data, selections, checkpoints, optimizer state, updates and evaluation; "
         "only the continuation objective changes to RLOO. d0 starts from the base model; "
         "d400 inherits the real GRPO parent and optimizer. Selection is not recomputed. "
         "New costs exclude historical selection; no H or RLOO-native full-history claim.")


@contextlib.contextmanager
def lock(path, *, blocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
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


# Status display modules: read by the status and report tools only, imported by
# no other module under src/. Their drift cannot change training, selection,
# evaluation, checkpoints or costs, so it is recorded in the observation
# receipt instead of refusing to start. Every other src module stays frozen.
DISPLAY_MODULES = ('src/matrix_status.py', 'src/rlzero_status.py',
                   'src/downstream_status.py', 'src/queue_status.py')


def display_is_isolated(name):
    """True when no other src module imports or names the display module."""
    module = re.escape(Path(name).stem)
    reference = re.compile(rf"^\s*(?:import|from)\s+{module}\b|['\"]{module}['\"]", re.M)
    for path in sorted((ROOT / 'src').glob('*.py')):
        if str(path.relative_to(ROOT)) != name and reference.search(path.read_text()):
            return False
    return True


def reviewed_code_changes(recorded):
    changes = {}
    for name, digest in recorded.items():
        current = ed.digest(ROOT / name)
        if current == digest:
            continue
        reviewed = ((name == 'src/rloo_experiment.py' and digest == PRE_QUEUE_OBSERVATION_CODE)
                    or (name == 'src/selector_pair_gpu.py' and (digest, current) == PAIR_OBSERVATION_UPGRADE)
                    or (name in DISPLAY_MODULES and display_is_isolated(name)))
        if not reviewed:
            raise ValueError(f"code changed since preparation: {name}")
        changes[name] = {'frozen_sha256': digest, 'runtime_sha256': current}
    return changes


def observation_receipt(out, changes):
    return {'schema': 'rloo-queue-observation-runtime/v1',
            'experiment_sha256': ed.digest(out / 'experiment.json'), 'changes': changes,
            'change': 'reviewed queue/status compatibility only; training objective, inputs, optimizer, '
                      'steps, evaluations, checkpoints and costs unchanged'}


def prepare(run, out, evaluation, *, dry=False):
    run, out, evaluation = run.resolve(), out.resolve(), evaluation.resolve()
    config = ed.read(run / "run_config.json")
    disjoint(out, [run, evaluation.parent, Path(config["model"])])
    steps, k = 100, 8
    source = ed.prepare(run, out, evaluation, steps, k, ARMS, dry=True)
    from train_policy_rloo import handoff_tree, lineage_tree
    handoff_tree()
    lineage_tree()
    test = ed.read(evaluation)
    rows = test["test"]
    if len(rows) != 300:
        raise ValueError("full protocol requires the frozen 300-question independent test")
    contract = {"schema": SCHEMA, "objective": "rloo", "scope": SCOPE,
                "source": source, "evaluation_input": str(evaluation),
                "steps": steps, "eval_k": k, "eval_n": len(rows),
                "arms": list(ARMS), "environment": runtime_env(config),
                "code_hashes": {str(p.relative_to(ROOT)): ed.digest(p)
                                for p in sorted((ROOT / "src").glob("*.py"))}}
    changes = {}
    if (out / 'experiment.json').exists():
        frozen = ed.read(out / 'experiment.json')
        changes = reviewed_code_changes(frozen['code_hashes'])
        contract['code_hashes'] = frozen['code_hashes']
        if contract != frozen:
            raise ValueError(f'contract changed: {out / "experiment.json"}')
    if dry:
        return contract
    out.mkdir(parents=True, exist_ok=True)
    with lock(out / ".prepare.lock"):
        ed.bind(out / "experiment.json", contract)
        if changes:
            ed.bind(out / 'queue-observation-runtime.json', observation_receipt(out, changes))
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
    changes = reviewed_code_changes(c['code_hashes'])
    if changes:
        receipt = out / 'queue-observation-runtime.json'
        if not receipt.is_file() or ed.read(receipt) != observation_receipt(out, changes):
            raise ValueError('reviewed runtime receipt missing or changed; use the RLOO launcher to prepare safely')
    for name, digest in ed.read(out / "inputs.json").items():
        if ed.digest(out / name) != digest:
            raise ValueError(f"prepared input changed: {name}")
    return c, config


def training_command(out, arm):
    c, config = validate(out)
    if arm not in ARMS:
        raise ValueError("unknown RLOO arm")
    args = ed.train_args(config, Path(c["source"]["source_run"]), out, arm, c["steps"])
    args[args.index(str(ed.ROOT / "src/train_policy_grpo.py"))] = str(ROOT / "src/train_policy_rloo.py")
    ed._replace_flag(args, "--objective", "rloo")
    if "--reliability-log" in args:
        args.remove("--reliability-log")
    return [sys.executable, *args]


def policy(out, arm):
    c, config = validate(out)
    drift = c["source"]["drift"]
    parent = Path(c["source"]["source_run"]) / f"policy_step_{drift}" if drift else None
    if arm == "before":
        return parent
    if arm not in ARMS:
        raise ValueError("unknown RLOO arm")
    from train_policy_rloo import validate_policy_lineage
    path = out / arm / "policy"
    validate_policy_lineage(
        path, target_steps=drift + c["steps"], world_size=4, training_objective="rloo",
        expected_start_step=drift, expected_parent=parent, expected_model=Path(config["model"]),
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
    result = {"schema": SCHEMA, "scope": SCOPE, "seed": c["source"]["seed"],
              "drift": c["source"]["drift"],
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


def source_paths(work, source_root, seed, drift=0):
    root = source_root or work / "runs" / TAG
    return root / f"family-math500-s{seed}" / f"{TAG}-s{seed}-math500-d{drift}"


def prepare_matrix(args, root, outs):
    existing = [ed.read(out / "experiment.json") if (out / "experiment.json").exists() else None for out in outs]
    evaluations = [args.eval_prompts or (Path(c["evaluation_input"]) if c else
                   args.work / f"inputs/e5-reduced/test-math500-d{drift}.json")
                   for (drift, _), c in zip(POINTS, existing, strict=True)]
    for drift in (0, 400):
        if len({e.resolve() for (d, _), e in zip(POINTS, evaluations, strict=True) if d == drift}) != 1:
            raise ValueError("prepared seeds use different evaluation inputs within a checkpoint")
    runs = [Path(c["source"]["source_run"]) if c and args.source_root is None else
            source_paths(args.work, args.source_root, seed, drift)
            for (drift, seed), c in zip(POINTS, existing, strict=True)]
    disjoint(root, [*runs, *(e.parent for e in evaluations)])
    configs = [ed.read(run / "run_config.json") for run in runs]
    fields = ("model", "dataset", "max_new_tokens", "temperature", "prompt_format",
              "attn", "gen_batch", "lora_targets", "thinking", "top_p",
              "grpo_gradient_checkpointing", *ed.TRAIN_FLAGS.values())
    if any(any(config.get(key) != configs[0].get(key) for key in fields) for config in configs[1:]):
        raise ValueError("source model/runtime/training configuration differs across seeds")
    for (drift, seed), config, run, out, evaluation in zip(POINTS, configs, runs, outs, evaluations, strict=True):
        if config["seed"] != seed or config["drift"] != drift:
            raise ValueError("source seed/checkpoint does not match matrix position")
        prepare(run, out, evaluation, dry=True)
    for run, out, evaluation in zip(runs, outs, evaluations, strict=True):
        prepare(run, out, evaluation)
    print("Prepared 18 matched RLOO training arms; no smoke stage.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "prepare", "ensure-prepared", "status", "run", "report", "evaluate", "check"))
    p.add_argument("--root", type=Path, default=Path(os.environ.get("RLOO_ROOT", "/tmp/rloo-selector-v2")))
    p.add_argument("--work", type=Path, default=Path(os.environ.get("OM_WORK", "/group-volume/minsoo3.kim/offpolicy-misranking")))
    p.add_argument("--source-root", type=Path)
    p.add_argument("--eval-prompts", type=Path)
    p.add_argument("--max-phase-seconds", type=float, default=float(os.environ.get("RLOO_MAX_PHASE_SECONDS", 86400)))
    p.add_argument("--out", type=Path)
    p.add_argument("--arm", choices=("before", *ARMS))
    p.add_argument("--shard", type=int)
    args = p.parse_args()
    root = args.root.resolve()
    if args.command == "plan":
        print(json.dumps({"scope": SCOPE, "dataset": "math500", "drifts": [0, 400], "seeds": [0, 1, 2],
                          "arms": ARMS, "training_runs": 18, "updates": 100, "eval_prompts": 300,
                          "eval_k": 8,
                          "launch": "default launcher prepares inputs then runs GPU experiment directly; no smoke",
                          "default_phase_timeout_seconds": 86400}, indent=2))
        return
    if args.command == "evaluate":
        if args.out is None or args.arm is None or args.shard is None:
            p.error("evaluate requires --out, --arm and --shard")
        evaluate(args.out, args.arm, args.shard)
        return
    outs = [root / f"math500-d{drift}" / f"s{seed}" for drift, seed in POINTS]
    if args.command in ("prepare", "ensure-prepared"):
        with lock(root / ".matrix-prepare.lock", blocking=True):
            prepare_matrix(args, root, outs)
        return
    if args.command == "status":
        for out in outs:
            if not (out / "experiment.json").exists():
                print(f"{out.parent.name}/{out.name}: not prepared")
                continue
            validate(out)
            for arm in ("before", *ARMS):
                try:
                    with lock(out / arm / ".worker.lock"):
                        progress = state(out, arm)
                except BlockingIOError:
                    progress = "running (lock held)"
                print(f"{out.parent.name}/{out.name}/{arm}: {progress}")
        return
    for out in outs:
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
    busy = 0
    for out in outs:
        for arm in ("before", *ARMS):
            try:
                with lock(out / arm / ".worker.lock"):
                    if not complete(out, arm):
                        run_arm(out, arm, seconds)
            except BlockingIOError:
                busy += 1
        if all(complete(out, arm) for arm in ("before", *ARMS)):
            try:
                with lock(out / ".report.lock"):
                    report(out)
            except BlockingIOError:
                pass
    print(f"Worker finished; {busy} arms held by other workers. Use status for global completion.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f"[rloo-error] {exc}", file=sys.stderr)
        raise SystemExit(75 if isinstance(exc, BlockingIOError) else 1)
