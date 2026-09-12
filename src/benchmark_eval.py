"""Public-benchmark evaluation of the trained E5 policies.

The reduced E5 measures every arm on 300 held-out MATH-train problems. This
module evaluates the same policies (the source checkpoint ``before`` and every
finished arm, gate arms included) on public sets that share no problem with
the candidate pool: AIME 2024, AIME 2025, AMC 2023, a GSM8K test subsample and
a subsample of the MATH test problems outside MATH-500. Sampling, prompt
format, verifier and reward are those of the E5 evaluation.

    python src/benchmark_eval.py fetch --datasets-dir DIR          # online, once
    python src/benchmark_eval.py prepare --out SEED_DIR --datasets-dir DIR [--sets ...] [--count 200] [--eval-k 8]
    python src/benchmark_eval.py evaluate --out SEED_DIR --arm ARM --shard S [--sets ...]   # one GPU
    python src/benchmark_eval.py summarize --out SEED_DIR [--allow-partial]
    python src/benchmark_eval.py status --out SEED_DIR

Outputs under SEED_DIR: benchmarks.json (frozen selection), benchmarks/<set>.json
(the prompts), <arm>/benchmark/<set>/shard-<s>.{jsonl,done.json}, and
benchmark_results.{csv,json}: per arm and set the mean verifier reward, the
paired prompt-bootstrap difference against the random arm and against the
source checkpoint, and a macro average over sets.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import random
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

import evidence_downstream as ed

SCHEMA = "offpolicy-e5-benchmark/v1"
SETS = ("aime24", "aime25", "amc23", "gsm8k", "math_rest")
SUBSAMPLED = {"gsm8k", "math_rest"}
SOURCES = {
    "aime24": {"repo": "HuggingFaceH4/aime_2024", "config": None, "split": "train", "question": "problem", "answer": "answer"},
    "aime25": {"repo": "math-ai/aime25", "config": None, "split": "test", "question": "problem", "answer": "answer"},
    "amc23": {"repo": "math-ai/amc23", "config": None, "split": "test", "question": "question", "answer": "answer"},
    "gsm8k": {"repo": "openai/gsm8k", "config": "main", "split": "test", "question": "question", "answer": "answer"},
    "math_rest": {"repo": "EleutherAI/hendrycks_math", "config": "algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus",
                  "split": "test", "question": "problem", "answer": "solution"},
}
MATH500 = "HuggingFaceH4/MATH-500"


# ---------------------------------------------------------------- conversion
def normalize_question(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def boxed(solution: str) -> str | None:
    i = solution.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 1, i + len("\\boxed{")
    while j < len(solution) and depth:
        depth += {"{": 1, "}": -1}.get(solution[j], 0)
        j += 1
    return solution[i + len("\\boxed{"):j - 1].strip() if depth == 0 else None


def gsm8k_answer(solution: str) -> str | None:
    if "####" not in solution:
        return None
    value = solution.rsplit("####", 1)[1].strip().replace(",", "").replace("$", "")
    return value or None


def convert(name: str, row: dict) -> dict | None:
    source = SOURCES[name]
    question = row.get(source["question"])
    raw = row.get(source["answer"])
    if not isinstance(question, str) or not question.strip() or raw is None:
        return None
    if name == "gsm8k":
        answer = gsm8k_answer(str(raw))
    elif name == "math_rest":
        answer = boxed(str(raw))
    else:
        answer = str(raw).strip()
    if not answer:
        return None
    return {"question": question, "answer": answer, "source_id": str(row.get("id", row.get("unique_id", "")))}


def write_set(rows: list[dict], target: Path) -> str:
    tmp = target.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(target)
    return ed.digest(target)


def fetch(datasets_dir: Path, sets=SETS) -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    def revision_of(repo: str) -> str:
        try:
            return HfApi().dataset_info(repo).sha
        except Exception as exc:  # the manifest records what could be resolved
            return f"unresolved({type(exc).__name__})"

    datasets_dir.mkdir(parents=True, exist_ok=True)
    for name in sets:
        target, manifest_path = datasets_dir / f"{name}.jsonl", datasets_dir / f"{name}.manifest.json"
        if target.is_file() and manifest_path.is_file():
            print(f"[fetch] {name} already present ({sum(1 for _ in target.open())} rows); delete it to refetch")
            continue
        source = SOURCES[name]
        excluded = 0
        rows = []
        if name == "math_rest":
            math500 = load_dataset(MATH500, split="test")
            banned = {normalize_question(r["problem"]) for r in math500}
            for config in source["config"].split():
                for row in load_dataset(source["repo"], config, split=source["split"]):
                    item = convert(name, row)
                    if item is None:
                        continue
                    if normalize_question(item["question"]) in banned:
                        excluded += 1
                        continue
                    item["subject"] = config
                    item["level"] = row.get("level")
                    rows.append(item)
        else:
            dataset = load_dataset(source["repo"], source["config"], split=source["split"]) if source["config"] \
                else load_dataset(source["repo"], split=source["split"])
            rows = [item for item in (convert(name, row) for row in dataset) if item is not None]
        if not rows:
            raise ValueError(f"{name}: no usable rows")
        sha = write_set(rows, target)
        manifest = {"schema_version": 1, "dataset": name, "source_repository": source["repo"],
                    "source_config": source["config"], "split": source["split"],
                    "source_revision": revision_of(source["repo"]), "rows": len(rows),
                    "excluded_math500": excluded if name == "math_rest" else None,
                    "math500_revision": revision_of(MATH500) if name == "math_rest" else None,
                    "sha256": sha, "fetched_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[manifest] {name}: rows={len(rows)} excluded={excluded} sha256={sha[:12]}")


# ------------------------------------------------------------------- prepare
def load_set(datasets_dir: Path, name: str) -> tuple[list[dict], dict]:
    path, manifest = datasets_dir / f"{name}.jsonl", datasets_dir / f"{name}.manifest.json"
    if not path.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"benchmark set missing: {path} (run once online: bash scripts/fetch_benchmarks.sh)")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows, ed.read(manifest)


def prepare(out: Path, datasets_dir: Path, sets=SETS, count: int = 200, eval_k: int = 8) -> dict:
    contract = ed.read(out / "experiment.json")
    if contract.get("schema") != ed.SCHEMA:
        raise ValueError("not an E5 seed directory")
    if count < 4 or eval_k < 2:
        raise ValueError("count must be at least four and eval-k at least two")
    source = ed.read(Path(contract["source_run"]) / "prompts.json")
    (out / "benchmarks").mkdir(parents=True, exist_ok=True)
    selection = {}
    for index, name in enumerate(sets):
        if name not in SETS:
            raise ValueError(f"unknown benchmark set: {name}")
        rows, manifest = load_set(datasets_dir, name)
        rows = [{"question": r["question"], "answer": r["answer"]} for r in rows]
        keys = ed.questions(rows)
        unique = {}
        for key, row in zip(keys, rows):
            unique.setdefault(key, row)
        pool = [unique[k] for k in sorted(unique)]
        seed = contract["eval_seed"] + 7_919 * (index + 1)
        if name in SUBSAMPLED and len(pool) > count:
            random.Random(seed).shuffle(pool)
            pool = sorted(pool[:count], key=lambda r: normalize_question(r["question"]))
        provenance = {"dataset": manifest["source_repository"], "revision": str(manifest["source_revision"]),
                      "split": manifest["split"], "set": name, "source_sha256": manifest["sha256"],
                      "source_rows": manifest["rows"], "selection_seed": seed, "selected": len(pool)}
        test = ed.independent_test(source, {"test": pool, "provenance": provenance})
        payload = {"val": test, "provenance": provenance}
        ed.bind(out / "benchmarks" / f"{name}.json", payload)
        selection[name] = {"prompts": len(test), "sha256": ed.digest(out / "benchmarks" / f"{name}.json"),
                           "provenance": provenance}
    frozen = {"schema": SCHEMA, "experiment_sha256": ed.digest(out / "experiment.json"),
              "eval_k": eval_k, "eval_seed": contract["eval_seed"] + 100_003, "sets": selection,
              "prompt_format": "the source point's format (OM_PROMPT_FORMAT), as in the E5 evaluation"}
    existing = out / "benchmarks.json"
    if existing.exists():  # later launches may add sets; frozen sets must match
        recorded = ed.read(existing)
        if recorded["eval_k"] != frozen["eval_k"] or recorded["eval_seed"] != frozen["eval_seed"] or \
                recorded["experiment_sha256"] != frozen["experiment_sha256"]:
            raise ValueError(f"contract changed: {existing}; use a new output root, do not mix runs")
        for name, spec in recorded["sets"].items():
            if name in selection and selection[name] != spec:
                raise ValueError(f"contract changed for benchmark set {name}: {existing}")
        recorded["sets"].update({k: v for k, v in selection.items() if k not in recorded["sets"]})
        ed.atomic_json(existing, recorded)
        return recorded
    ed.atomic_json(existing, frozen)
    return frozen


# ------------------------------------------------------------------ evaluate
def sets_of(out: Path) -> list[str]:
    return list(ed.read(out / "benchmarks.json")["sets"])


def binding_for(out: Path, arm: str, name: str, shard: int) -> tuple[dict, Path | None, range]:
    frozen = ed.read(out / "benchmarks.json")
    if frozen.get("schema") != SCHEMA or shard not in range(4) or name not in frozen["sets"]:
        raise ValueError("unsupported benchmark contract, set or shard")
    if ed.digest(out / "experiment.json") != frozen["experiment_sha256"]:
        raise ValueError("experiment contract changed after benchmark preparation")
    if ed.digest(out / "benchmarks" / f"{name}.json") != frozen["sets"][name]["sha256"]:
        raise ValueError(f"benchmark prompts changed after preparation: {name}")
    policy = ed.arm_policy(out, arm)
    n = frozen["sets"][name]["prompts"]
    indices = range(n * shard // 4, n * (shard + 1) // 4)
    binding = {"experiment_sha256": frozen["experiment_sha256"], "benchmarks_sha256": ed.digest(out / "benchmarks.json"),
               "prompts_sha256": frozen["sets"][name]["sha256"], "set": name, "arm": arm, "shard": shard, "shards": 4,
               "eval_k": frozen["eval_k"], "eval_seed": frozen["eval_seed"],
               "adapter_sha256": ed.digest(policy / "adapter_model.safetensors") if policy else None,
               "policy_manifest_sha256": ed.digest(policy / "policy_train.json") if policy else None,
               "base_model_config_sha256": ed.read(out / "experiment.json")["model_config_sha256"] if policy is None else None}
    return binding, policy, indices


def shard_done(out: Path, arm: str, name: str, shard: int) -> bool:
    return (out / arm / "benchmark" / name / f"shard-{shard}.done.json").is_file()


def evaluate(out: Path, arm: str, shard: int, sets=None) -> None:
    """One process per (arm, shard): loads the policy once and runs every set."""
    contract = ed.read(out / "experiment.json")
    config = ed.read(Path(contract["source_run"]) / "run_config.json")
    frozen = ed.read(out / "benchmarks.json")
    names = list(sets or frozen["sets"])
    pending = []
    for name in names:
        binding, policy, indices = binding_for(out, arm, name, shard)
        target = out / arm / "benchmark" / name
        target.mkdir(parents=True, exist_ok=True)
        completed = target / f"shard-{shard}.done.json"
        path = target / f"shard-{shard}.jsonl"
        if completed.exists():
            record = ed.read(completed)
            if record["binding"] != binding or record["rollouts_sha256"] != ed.digest(path):
                raise ValueError(f"completed benchmark shard does not match its contract: {completed}")
            ed.reward_rows(path, indices, frozen["eval_k"])
            print(f"[reuse] {arm} {name} shard {shard}", flush=True)
            continue
        if not indices:
            atomic_done(completed, binding, path, 0.0, 0)
            continue
        pending.append((name, binding, policy, indices, target, path, completed))
    if not pending:
        return
    from rollout import collect_rollouts, load_policy
    policy = pending[0][2]
    model, tokenizer = load_policy(config["model"], policy)
    for name, binding, _, indices, target, path, completed in pending:
        with (target / f"shard-{shard}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ed.bind(target / f"shard-{shard}.contract.json", binding)
            prompts = ed.read(out / "benchmarks" / f"{name}.json")["val"][indices.start:indices.stop]
            started = time.perf_counter()
            collect_rollouts(model, tokenizer, prompts, frozen["eval_k"], config["max_new_tokens"],
                             float(config["temperature"]), path, idx_offset=indices.start,
                             sampling_seed_base=frozen["eval_seed"] + 1_000 * (list(frozen["sets"]).index(name) + 1))
            ed.reward_rows(path, indices, frozen["eval_k"])
            atomic_done(completed, binding, path, time.perf_counter() - started, len(prompts))
            print(f"[done] {arm} {name} shard {shard}: {len(prompts)} prompts", flush=True)


def atomic_done(completed: Path, binding: dict, path: Path, seconds: float, prompts: int) -> None:
    if not path.exists():
        path.write_text("")
    ed.atomic_json(completed, {"binding": binding, "rollouts_sha256": ed.digest(path),
                               "elapsed_seconds": seconds, "prompts": prompts,
                               "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})


def set_means(out: Path, arm: str, name: str) -> np.ndarray:
    frozen = ed.read(out / "benchmarks.json")
    rewards = [[] for _ in range(frozen["sets"][name]["prompts"])]
    seconds = 0.0
    for shard in range(4):
        binding, _, indices = binding_for(out, arm, name, shard)
        target = out / arm / "benchmark" / name
        path = target / f"shard-{shard}.jsonl"
        record = ed.read(target / f"shard-{shard}.done.json")
        if record["binding"] != binding or record["rollouts_sha256"] != ed.digest(path):
            raise ValueError(f"benchmark result does not match its contract: {arm} {name} shard {shard}")
        seconds += float(record.get("elapsed_seconds", 0.0))
        for row in ed.reward_rows(path, indices, frozen["eval_k"]):
            rewards[row["prompt_idx"]].append(row["reward"])
    return np.array([np.mean(v) for v in rewards]), seconds


def summarize(out: Path, *, allow_partial: bool = False) -> dict:
    contract = ed.read(out / "experiment.json")
    frozen = ed.read(out / "benchmarks.json")
    arms = ["before", *ed.arms_of(out, contract)]
    values, seconds, missing = {}, {}, []
    for arm in arms:
        for name in frozen["sets"]:
            if all(shard_done(out, arm, name, s) for s in range(4)):
                values[(arm, name)], seconds[(arm, name)] = set_means(out, arm, name)
            else:
                missing.append(f"{arm}:{name}")
    if missing and not allow_partial:
        raise ValueError(f"incomplete benchmark evaluations: {missing}")
    rows = []
    for arm in arms:
        per_set = []
        for name in frozen["sets"]:
            if (arm, name) not in values:
                continue
            after = values[(arm, name)]
            row = {"dataset": contract["dataset"], "seed": contract["seed"], "drift": contract["drift"],
                   "selector": arm, "benchmark": name, "prompts": len(after), "eval_k": frozen["eval_k"],
                   "reward": float(after.mean()), "gpu_seconds": seconds[(arm, name)],
                   "vs_before": None, "before_lower": None, "before_upper": None,
                   "vs_random": None, "random_lower": None, "random_upper": None}
            if ("before", name) in values and arm != "before":
                d = after - values[("before", name)]
                lo, hi = ed.paired_interval(d, contract["seed"] + 21)
                row.update(vs_before=float(d.mean()), before_lower=lo, before_upper=hi)
            if ("random", name) in values and arm != "random":
                d = after - values[("random", name)]
                lo, hi = ed.paired_interval(d, contract["seed"] + 23)
                row.update(vs_random=float(d.mean()), random_lower=lo, random_upper=hi)
            rows.append(row)
            per_set.append(row)
        if len(per_set) == len(frozen["sets"]) and per_set:
            rows.append({"dataset": contract["dataset"], "seed": contract["seed"], "drift": contract["drift"],
                         "selector": arm, "benchmark": "macro", "prompts": sum(r["prompts"] for r in per_set),
                         "eval_k": frozen["eval_k"], "reward": float(np.mean([r["reward"] for r in per_set])),
                         "gpu_seconds": sum(r["gpu_seconds"] for r in per_set),
                         "vs_before": (float(np.mean([r["vs_before"] for r in per_set])) if all(r["vs_before"] is not None for r in per_set) else None),
                         "before_lower": None, "before_upper": None,
                         "vs_random": (float(np.mean([r["vs_random"] for r in per_set])) if all(r["vs_random"] is not None for r in per_set) else None),
                         "random_lower": None, "random_upper": None})
    report = {"schema": SCHEMA, "experiment_sha256": frozen["experiment_sha256"], "complete": not missing,
              "missing": missing, "rows": rows,
              "interval_scope": "paired prompt bootstrap, 10000 draws, conditional on this trained seed; "
                                "macro rows average the per-set means and carry no interval"}
    ed.atomic_json(out / "benchmark_results.json", report)
    if rows:
        with (out / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


def status(out: Path) -> str:
    if not (out / "benchmarks.json").is_file():
        return "benchmarks not prepared"
    frozen = ed.read(out / "benchmarks.json")
    contract = ed.read(out / "experiment.json")
    lines = [f"benchmarks: {' '.join(f'{n}({s['prompts']})' for n, s in frozen['sets'].items())} eval_k={frozen['eval_k']}"]
    for arm in ["before", *ed.arms_of(out, contract)]:
        parts = []
        for name in frozen["sets"]:
            done = sum(shard_done(out, arm, name, s) for s in range(4))
            parts.append(f"{name} {'done' if done == 4 else f'{done}/4'}")
        lines.append(f"  {arm:14s} " + " | ".join(parts))
    if (out / "benchmark_results.csv").is_file():
        lines.append(f"  results: {out / 'benchmark_results.csv'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--datasets-dir", type=Path, required=True)
    p.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--datasets-dir", type=Path, required=True)
    p.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--eval-k", type=int, default=8)
    p = sub.add_parser("evaluate")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--sets", nargs="*")
    p = sub.add_parser("summarize")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    p = sub.add_parser("status")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            fetch(args.datasets_dir.resolve(), args.sets)
        elif args.command == "prepare":
            frozen = prepare(args.out.resolve(), args.datasets_dir.resolve(), args.sets, args.count, args.eval_k)
            print(f"[benchmarks] {' '.join(f'{n}={s['prompts']}' for n, s in frozen['sets'].items())} eval_k={frozen['eval_k']}")
        elif args.command == "evaluate":
            evaluate(args.out.resolve(), args.arm, args.shard, args.sets or None)
        elif args.command == "summarize":
            report = summarize(args.out.resolve(), allow_partial=args.allow_partial)
            print(json.dumps({k: report[k] for k in ("complete", "missing")}))
        else:
            print(status(args.out.resolve()))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
