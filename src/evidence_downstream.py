"""Independent-test downstream comparison (extension E5, reduced design).

Ported from the v2 workspace (2026-09-10) and generalised: any completed
MATH-500 point can be the source (drift 0 branches from the base model), and
the trained arms are a chosen subset of the seven selectors; arms may be added
to a prepared seed later (arms.json) without changing the frozen contract. Invoked through scripts/run_downstream_independent.sh.

Each arm starts from the source point's policy_step_<drift> adapter and
optimizer, receives the same number of further GRPO updates on its selected
prompt subset, and is evaluated on an independent test set that shares no
question with the candidate pool or the ranking validation set.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import unicodedata
from pathlib import Path

import numpy as np

from downstream_compare import SELECTORS, write_subsets

SCHEMA = "offpolicy-downstream-independent/v2"
ROOT = Path(__file__).resolve().parents[1]
POLICY_FILES = ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "policy_train.json", "grpo_stats.jsonl")
TRAIN_FLAGS = {
    "group-size": "grpo_group_size", "clip-epsilon": "grpo_clip_epsilon",
    "learning-rate": "grpo_learning_rate", "epochs-per-batch": "grpo_epochs_per_batch",
    "max-grad-norm": "grpo_max_grad_norm", "advantage-epsilon": "grpo_advantage_epsilon",
    "lora-rank": "grpo_lora_rank", "lora-alpha": "grpo_lora_alpha",
    "logprob-micro-batch": "grpo_logprob_micro_batch",
}
DEFAULT_ARMS = ("random", "passrate_beta", "fresh_r", "g11")


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def bind(path: Path, value) -> None:
    if path.exists():
        if read(path) != value:
            raise ValueError(f"contract changed: {path}; use a new output root, do not mix runs")
    else:
        atomic_json(path, value)


INFORMATIONAL = ("code_hashes", "runtime", "selectors")


def bind_experiment(path: Path, contract: dict) -> dict:
    """Freeze the experiment contract once; later launches must match it.

    Code and package hashes are recorded for provenance but not enforced, so a
    launcher or driver fix does not orphan a seed that is already half done.
    Everything that defines the experiment (source point, policy, test set,
    steps, arms) is enforced.
    """
    if not path.exists():
        atomic_json(path, contract)
        return contract
    existing = read(path)
    strip = lambda c: {k: v for k, v in c.items() if k not in INFORMATIONAL}  # noqa: E731
    if strip(existing) != strip(contract):
        raise ValueError(f"contract changed: {path}; use a new output root, do not mix runs")
    if any(existing.get(k) != contract.get(k) for k in INFORMATIONAL):
        print("[note] recorded contract differs only in informational fields (code hashes, "
              "packages, arm list); continuing with the recorded contract", file=sys.stderr)
    return existing


def arms_of(out: Path, contract: dict | None = None) -> list[str]:
    """Frozen arms plus any added later through arms.json."""
    contract = contract or read(out / "experiment.json")
    arms = list(contract["selectors"])
    extra = out / "arms.json"
    if extra.exists():
        arms += [arm for arm in read(extra)["selectors"] if arm not in arms]
    return arms


def extend_arms(out: Path, contract: dict, arms) -> list[str]:
    """Record arms added after preparation without touching the frozen contract,
    so completed shard bindings (which hash experiment.json) stay valid."""
    current = arms_of(out, contract)
    new = [arm for arm in arms if arm not in current]
    if new:
        extra = [arm for arm in current + new if arm not in contract["selectors"]]
        atomic_json(out / "arms.json", {"selectors": extra})
        print(f"[arms] added after preparation: {' '.join(new)}", file=sys.stderr)
    return current + new


def require_separate_output(output: Path, inputs) -> Path:
    """The output root must not lie inside any input (source point, test file)."""
    destination = output.resolve()
    for item in inputs:
        root = Path(item).resolve()
        if destination == root or root in destination.parents:
            raise ValueError(f"refusing to write into an input location {root}: {destination}")
    return destination


def questions(rows: list[dict]) -> list[str]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("prompt list must be nonempty")
    keys = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("every prompt needs a nonempty question string")
        if not isinstance(row.get("answer"), str) or not row["answer"].strip():
            raise ValueError("every prompt needs a nonempty answer string")
        keys.append(" ".join(unicodedata.normalize("NFKC", row["question"]).split()))
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate question in prompt split")
    return keys


def independent_test(source: dict, evaluation: dict) -> list[dict]:
    if not isinstance(evaluation.get("provenance"), dict) or not all(
        isinstance(evaluation["provenance"].get(k), str) and evaluation["provenance"][k].strip()
        for k in ("dataset", "revision", "split")
    ):
        raise ValueError("evaluation needs provenance.dataset/revision/split")
    test = evaluation.get("test")
    test_keys = set(questions(test))
    used = set(questions(source["train"])) | set(questions(source["val"]))
    if used & test_keys:
        raise ValueError("evaluation overlaps candidate/training or ranking-validation prompts")
    return test


def prepare_test(candidates: Path, runs: list[Path], out: Path, count: int, seed: int,
                 dataset: str, revision: str, split: str) -> dict:
    out = require_separate_output(out, [candidates, *runs])
    if count < 4 or not all((dataset.strip(), revision.strip(), split.strip())):
        raise ValueError("test count must be at least four; provenance fields must be nonempty")
    used = set()
    for run in runs:
        source = read(run / "prompts.json")
        used.update(questions(source["train"]))
        used.update(questions(source["val"]))
    if candidates.suffix == ".jsonl":
        raw = [json.loads(line) for line in candidates.read_text().splitlines() if line.strip()]
    else:
        raw = read(candidates)
        if isinstance(raw, dict):
            raw = raw["test"]
    unique = {}
    for row in raw:
        item = {"question": row.get("question", row.get("problem")), "answer": row.get("answer")}
        key = questions([item])[0]
        if key in used:
            continue
        if key in unique and unique[key]["answer"] != item["answer"]:
            raise ValueError("duplicate evaluation question has conflicting answers")
        unique.setdefault(key, item)
    pool = [unique[key] for key in sorted(unique)]
    if len(pool) < count:
        raise ValueError(f"only {len(pool)} disjoint unique questions remain; requested {count}")
    random.Random(seed).shuffle(pool)
    result = {"test": pool[:count], "provenance": {"dataset": dataset, "revision": revision, "split": split,
              "candidate_sha256": digest(candidates), "selection_seed": seed,
              "exclusions": {str(run): digest(run / "prompts.json") for run in runs},
              "eligible_questions": len(pool), "overlap_check": "NFKC/whitespace-normalized exact questions, not semantic deduplication"}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_name(out.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.exists():
            atomic_json(out, result)
            return result
        # A frozen test set is reused when its content and pool provenance match;
        # the exclusion record is informational, but the frozen questions must
        # still be disjoint from every run given now.
        existing = read(out)
        if used & set(questions(existing["test"])):
            raise ValueError(f"frozen test set overlaps the prompts of a given run: {out}")
        core = lambda d: (d["test"], {k: d["provenance"].get(k) for k in  # noqa: E731
                          ("dataset", "revision", "split", "candidate_sha256", "selection_seed")})
        if core(existing) != core(result):
            raise ValueError(f"contract changed: {out}; use a new output root, do not mix runs")
    return existing


def train_args(config: dict, run: Path, out: Path, selector: str, steps: int) -> list[str]:
    drift = config["drift"]
    args = ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
            str(ROOT / "src/train_policy_grpo.py"), "--model", config["model"],
            "--prompts", str(out / "subsets" / f"subset-{selector}.json"),
            "--output", str(out / selector / "policy"), "--target-steps", str(drift + steps),
            "--start-step", str(drift)]
    if drift > 0:  # drift 0 trains the base model directly; no adapter or optimizer to resume
        parent = run / f"policy_step_{drift}"
        args += ["--resume-adapter", str(parent), "--resume-optimizer", str(parent / "optimizer.pt")]
    args += ["--objective", "grpo", "--expected-world-size", "4", "--checkpoint-every", "5",
             "--max-new-tokens", str(config["max_new_tokens"]), "--seed", str(config["seed"])]
    for flag, field in TRAIN_FLAGS.items():
        args += ["--" + flag, str(config[field])]
    if str(config["grpo_gradient_checkpointing"]).strip().lower() in {"0", "false", "no", "off", ""}:
        args += ["--disable-gradient-checkpointing"]
    return args


def check_arms(arms) -> tuple[str, ...]:
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("arms must be a nonempty list without repeats")
    for arm in arms:
        if arm not in SELECTORS:
            raise ValueError(f"unknown selector: {arm}")
    return arms


def prepare(run: Path, out: Path, eval_path: Path, steps: int, eval_k: int,
            arms=DEFAULT_ARMS, *, dry: bool = False) -> dict:
    run, eval_path = run.resolve(), eval_path.resolve()
    out = require_separate_output(out, [run, eval_path.parent])
    arms = check_arms(arms)
    if steps <= 0 or eval_k < 2:
        raise ValueError("steps must be positive and eval-k at least two")
    if not (run / "DONE").is_file():
        raise ValueError("source point is not complete")
    config = read(run / "run_config.json")
    drift = config.get("drift")
    if config.get("dataset") != "math500" or not isinstance(drift, int) or drift < 0:
        raise ValueError("E5 requires a completed MATH-500 point (drift 0 branches from the base model)")
    if config.get("grpo_world_size") != 4 or config.get("grpo_epochs_per_batch") != 1:
        raise ValueError("E5 requires the four-rank, one-epoch GRPO source protocol")
    if config.get("seed") not in range(5) or config.get("topk_frac") != 0.1 or config.get("prompt_format") != "olmo_rlzero_math":
        raise ValueError("E5 requires seeds 0..4, top 10%, and the OLMo math prompt format")
    source = read(run / "prompts.json")
    evaluation = read(eval_path)
    test = independent_test(source, evaluation)
    if len(test) < 4:
        raise ValueError("independent evaluation needs at least four prompts")
    questions(source["train"])
    from score_artifacts import load_complete_score_artifacts
    from train_policy_grpo import validate_policy_manifest

    load_complete_score_artifacts(run)
    if drift > 0:
        parent = run / f"policy_step_{drift}"
        policy = validate_policy_manifest(parent, target_steps=drift, world_size=4,
                                          training_objective="grpo", require_complete_hashes=True)
        if policy.get("seed") != config["seed"] or policy.get("prompt_format") != config["prompt_format"]:
            raise ValueError("source policy seed/prompt format differs from the point config")
    model = Path(config["model"]).resolve()
    if not (model / "config.json").is_file():
        raise ValueError(f"source model snapshot is unavailable: {model}")
    input_files = ["run_config.json", "prompts.json", "scores_offpolicy.json",
                   "scores_splithalf.json", "scores_oracle.json", "rollouts_behavior_train.jsonl"]
    if drift > 0:
        input_files += [f"policy_step_{drift}/{name}" for name in POLICY_FILES]
    contract = {
        "schema": SCHEMA, "source_run": str(run), "dataset": config["dataset"],
        "seed": config["seed"], "drift": drift, "steps": steps, "eval_k": eval_k,
        "eval_seed": 701_000_003 + config["seed"] * 1_000_003,
        "eval_prompts": len(test), "eval_input_sha256": digest(eval_path),
        "evaluation_payload_sha256": hashlib.sha256((json.dumps(
            {"val": test, "provenance": evaluation["provenance"]}, indent=2, allow_nan=False) + "\n").encode()).hexdigest(),
        "eval_provenance": evaluation["provenance"], "selectors": list(arms),
        "all_subsets": list(SELECTORS),
        "source_hashes": {name: digest(run / name) for name in input_files},
        "model_config_sha256": digest(model / "config.json"),
        "runtime": {"attention": os.environ.get("OM_ATTN", "eager"),
                    "generation_batch": os.environ.get("OM_GEN_BATCH", "32"),
                    "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")}},
        "code_hashes": {name: digest(ROOT / name) for name in (
            "src/evidence_downstream.py", "src/downstream_compare.py", "src/train_policy_grpo.py",
            "src/rollout.py", "src/rollout_contract.py", "src/data.py", "src/select_rules.py",
            "src/bootstrap_math_verify.py", "src/artifact_contract.py")},
        "inference": "descriptive; prompt-bootstrap intervals conditional on each trained seed",
        "power_verified": False, "registered_matrix_changed": False,
    }
    # Validate command fields before creating any output or loading a GPU model.
    train_args(config, run, out, arms[0], steps)
    if dry:
        return contract
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        contract = bind_experiment(out / "experiment.json", contract)
        extend_arms(out, contract, arms)
        bind(out / "evaluation.json", {"val": test, "provenance": evaluation["provenance"]})
        if (out / "subsets_hashes.json").exists():
            written = {arm: out / "subsets" / f"subset-{arm}.json" for arm in SELECTORS}
            if read(out / "subsets_hashes.json") != {key: digest(path) for key, path in written.items()}:
                raise ValueError("saved selection subsets have changed")
        else:
            written = write_subsets(run, out / "subsets", float(config["topk_frac"]), config["seed"])
        for selector in written:
            args_path = out / "subsets" / f"train-{selector}.args"
            args_path.write_bytes(b"\0".join(arg.encode() for arg in train_args(config, run, out, selector, steps)) + b"\0")
        bind(out / "subsets_hashes.json", {key: digest(path) for key, path in written.items()})
    return contract


def arm_policy(out: Path, arm: str) -> Path | None:
    """Adapter directory of an arm; None means the base model (the d0 'before' policy)."""
    contract = read(out / "experiment.json")
    drift = contract["drift"]
    parent = Path(contract["source_run"]) / f"policy_step_{drift}" if drift > 0 else None
    if arm == "before":
        return parent
    if arm not in arms_of(out, contract):
        raise ValueError(f"unknown arm: {arm}")
    from dataclasses import asdict

    from train_policy_grpo import GrpoConfig, validate_policy_lineage
    config = read(Path(contract["source_run"]) / "run_config.json")
    path = out / arm / "policy"
    expected_config = asdict(GrpoConfig(**{field.removeprefix("grpo_"): config[field]
                            for field in TRAIN_FLAGS.values() if field != "grpo_logprob_micro_batch"}, checkpoint_every=5))
    validate_policy_lineage(path, target_steps=drift + contract["steps"], world_size=4,
                            training_objective="grpo", expected_start_step=drift,
                            expected_parent=parent,
                            expected_model=Path(config["model"]), expected_seed=contract["seed"],
                            expected_max_new_tokens=config["max_new_tokens"],
                            expected_prompt_format=config["prompt_format"],
                            expected_config=expected_config,
                            expected_prompts=out / "subsets" / f"subset-{arm}.json",
                            require_complete_hashes=True)
    return path


def reward_rows(path: Path, indices: range, k: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = {(i, j) for i in indices for j in range(k)}
    seen = set()
    for row in rows:
        key = (row["prompt_idx"], row["rollout_idx"])
        reward = row["reward"]
        if key not in expected or key in seen or not isinstance(reward, (int, float)) or not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError(f"invalid, duplicate, or unexpected evaluation row: {path}")
        seen.add(key)
    if seen != expected:
        raise ValueError(f"incomplete evaluation coverage: {path}")
    return rows


def eval_binding(out: Path, arm: str, shard: int) -> tuple[dict, Path, range]:
    contract = read(out / "experiment.json")
    if contract.get("schema") != SCHEMA or shard not in range(4):
        raise ValueError("unsupported contract or evaluation shard")
    if digest(out / "evaluation.json") != contract["evaluation_payload_sha256"]:
        raise ValueError("independent evaluation prompts changed after preparation")
    source = Path(contract["source_run"])
    drift = contract["drift"]
    names = ["run_config.json"]
    if drift > 0:
        names += [f"policy_step_{drift}/adapter_model.safetensors", f"policy_step_{drift}/policy_train.json"]
    for name in names:
        if digest(source / name) != contract["source_hashes"][name]:
            raise ValueError("source configuration or policy changed after preparation")
    policy = arm_policy(out, arm)
    n = contract["eval_prompts"]
    indices = range(n * shard // 4, n * (shard + 1) // 4)
    if not indices:
        raise ValueError("evaluation requires at least four prompts for four GPU shards")
    if policy is None:  # base model at drift 0: bind the model snapshot instead of an adapter
        binding = {"experiment_sha256": digest(out / "experiment.json"),
                   "prompts_sha256": digest(out / "evaluation.json"),
                   "adapter_sha256": None, "policy_manifest_sha256": None,
                   "base_model_config_sha256": contract["model_config_sha256"],
                   "arm": arm, "shard": shard, "shards": 4}
    else:
        binding = {"experiment_sha256": digest(out / "experiment.json"),
                   "prompts_sha256": digest(out / "evaluation.json"),
                   "adapter_sha256": digest(policy / "adapter_model.safetensors"),
                   "policy_manifest_sha256": digest(policy / "policy_train.json"),
                   "arm": arm, "shard": shard, "shards": 4}
    return binding, policy, indices


def evaluate(out: Path, arm: str, shard: int) -> None:
    binding, policy, indices = eval_binding(out, arm, shard)
    contract = read(out / "experiment.json")
    config = read(Path(contract["source_run"]) / "run_config.json")
    target = out / arm / "evaluation"
    target.mkdir(parents=True, exist_ok=True)
    with (target / f"shard-{shard}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bind(target / f"shard-{shard}.contract.json", binding)
        path = target / f"shard-{shard}.jsonl"
        completed = target / f"shard-{shard}.done.json"
        if completed.exists():
            if read(completed) != {"binding": binding, "rollouts_sha256": digest(path)}:
                raise ValueError("completed evaluation hash mismatch")
            reward_rows(path, indices, contract["eval_k"])
            print(f"[reuse] {arm} shard {shard}", flush=True)
            return
        from rollout import collect_rollouts, load_policy
        prompts = read(out / "evaluation.json")["val"][indices.start:indices.stop]
        model, tokenizer = load_policy(config["model"], policy)
        collect_rollouts(model, tokenizer, prompts, contract["eval_k"], config["max_new_tokens"],
                         float(config["temperature"]), path, idx_offset=indices.start,
                         sampling_seed_base=contract["eval_seed"])
        reward_rows(path, indices, contract["eval_k"])
        atomic_json(completed, {"binding": binding, "rollouts_sha256": digest(path)})


def evaluation_means(out: Path, arm: str) -> np.ndarray:
    contract = read(out / "experiment.json")
    rewards = [[] for _ in range(contract["eval_prompts"])]
    for shard in range(4):
        binding, _, indices = eval_binding(out, arm, shard)
        target = out / arm / "evaluation"
        path = target / f"shard-{shard}.jsonl"
        if read(target / f"shard-{shard}.done.json") != {"binding": binding, "rollouts_sha256": digest(path)}:
            raise ValueError("evaluation result does not match its contract")
        for row in reward_rows(path, indices, contract["eval_k"]):
            rewards[row["prompt_idx"]].append(row["reward"])
    return np.array([np.mean(values) for values in rewards])


def paired_interval(values: np.ndarray, seed: int, reps: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for start in range(0, reps, 256):
        size = min(256, reps - start)
        draws = rng.integers(len(values), size=(size, len(values)))
        means[start:start + size] = values[draws].mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def arm_state(out: Path, arm: str) -> str:
    """Durable progress word for status displays: missing / policy / partial / done."""
    if all((out / arm / "evaluation" / f"shard-{s}.done.json").is_file() for s in range(4)):
        return "done"
    shards = sum((out / arm / "evaluation" / f"shard-{s}.done.json").is_file() for s in range(4))
    if shards:
        return f"eval {shards}/4"
    if arm == "before":
        return "not evaluated"
    if (out / arm / "policy" / "policy_train.json").is_file():
        return "trained, not evaluated"
    if (out / arm / "policy").is_dir():
        return "training"
    return "not started"


def summarize(out: Path, *, allow_partial: bool = False) -> dict:
    contract = read(out / "experiment.json")
    subset_paths = {arm: out / "subsets" / f"subset-{arm}.json" for arm in contract["all_subsets"]}
    if read(out / "subsets_hashes.json") != {arm: digest(path) for arm, path in subset_paths.items()}:
        raise ValueError("saved selection subsets have changed")
    baseline_complete = all((out / "before/evaluation" / f"shard-{s}.done.json").is_file() for s in range(4))
    if not baseline_complete and not allow_partial:
        raise ValueError("baseline evaluation is incomplete")
    before = evaluation_means(out, "before") if baseline_complete else None
    values, missing = {}, []
    for arm in arms_of(out, contract):
        if not all((out / arm / "evaluation" / f"shard-{s}.done.json").is_file() for s in range(4)):
            missing.append(arm)
            continue
        values[arm] = evaluation_means(out, arm)
    if missing and not allow_partial:
        raise ValueError(f"incomplete arms: {missing}")
    subsets = {arm: set(read(path)["selected_idx"]) for arm, path in subset_paths.items()}
    rows = []
    for arm, after in values.items():
        delta = after - before if before is not None else None
        lo, hi = paired_interval(delta, contract["seed"]) if delta is not None else (None, None)
        row = {"dataset": contract["dataset"], "seed": contract["seed"], "drift": contract["drift"],
               "selector": arm,
               "reward_before": float(before.mean()) if before is not None else None, "reward_after": float(after.mean()),
               "reward_change": float(delta.mean()) if delta is not None else None, "change_lower": lo, "change_upper": hi,
               "overlap_with_fresh": len(subsets[arm] & subsets["fresh_r"]) / len(subsets[arm]),
               "difference_vs_fresh": None, "difference_lower": None, "difference_upper": None}
        if "fresh_r" in values:
            difference = after - values["fresh_r"]
            lo, hi = paired_interval(difference, contract["seed"])
            row.update(difference_vs_fresh=float(difference.mean()), difference_lower=lo, difference_upper=hi)
        rows.append(row)
    report = {"schema": SCHEMA, "experiment_sha256": digest(out / "experiment.json"),
              "complete": not missing and baseline_complete, "missing_baseline": not baseline_complete,
              "missing_selectors": missing, "rows": rows,
              "interval_scope": "paired prompt bootstrap, 10000 draws, conditional on this trained seed; not equivalence or a registered gate"}
    atomic_json(out / "downstream_results.json", report)
    if rows:
        with (out / "downstream_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-test")
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260910)
    for name in ("dataset", "revision", "split"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eval-prompts", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--eval-k", type=int, default=8)
    p.add_argument("--selectors", nargs="+", default=list(DEFAULT_ARMS))
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", choices=("before",) + SELECTORS, required=True)
    p.add_argument("--shard", type=int, required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    p = sub.add_parser("policy-ready")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", choices=SELECTORS, required=True)
    p = sub.add_parser("status")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare-test":
            result = prepare_test(args.candidates, args.runs, args.out, args.count, args.seed,
                                  args.dataset, args.revision, args.split)
            print(f"[test] {len(result['test'])} disjoint questions frozen at {args.out}")
            return 0
        if args.command == "prepare":
            result = prepare(args.run, args.out, args.eval_prompts, args.steps, args.eval_k,
                             args.selectors, dry=args.dry_run)
        elif args.command == "evaluate":
            evaluate(args.out.resolve(), args.arm, args.shard)
            return 0
        elif args.command == "policy-ready":
            if not (args.out / args.arm / "policy/policy_train.json").is_file():
                return 1
            arm_policy(args.out.resolve(), args.arm)
            return 0
        elif args.command == "status":
            out = args.out.resolve()
            if not (out / "experiment.json").is_file():
                print("not prepared")
                return 0
            contract = read(out / "experiment.json")
            print(f"seed {contract['seed']} d{contract['drift']} steps={contract['steps']} eval_k={contract['eval_k']} test={contract['eval_prompts']}")
            for arm in ["before", *arms_of(out, contract)]:
                print(f"  {arm:14s} {arm_state(out, arm)}")
            return 0
        else:
            result = summarize(args.out.resolve(), allow_partial=args.allow_partial)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
